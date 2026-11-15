"""Split an article into chunks exactly as the running system's chunks were made.

The chunks in the two live collections were produced by n8n's Recursive Character Text Splitter (chunk size 1,000, overlap 200),
which is LangChain's JavaScript splitter. The Python LangChain splitter is close but not identical, so this module ports the
JavaScript algorithm: lengths count UTF-16 code units, empty pieces are dropped, separators are not kept, and `trim()` strips the
JavaScript whitespace set. `build_collections.py check` compares the output with the chunks of the live collections.
"""
import re

SEPARATORS = ["\n\n", "\n", " ", ""]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

_JS_WHITESPACE = "\t\n\v\f\r                  　﻿"


def js_len(s):
    """String length the way JavaScript counts it (UTF-16 code units)."""
    return len(s.encode("utf-16-le")) // 2


def _join(docs, sep):
    text = sep.join(docs).strip(_JS_WHITESPACE)
    return text or None


def _merge(splits, sep, size, overlap):
    docs, current, total = [], [], 0
    sep_len = js_len(sep)
    for piece in splits:
        n = js_len(piece)
        if total + n + len(current) * sep_len > size and current:
            doc = _join(current, sep)
            if doc is not None:
                docs.append(doc)
            while total > overlap or (total + n + len(current) * sep_len > size and total > 0):
                total -= js_len(current[0])
                current.pop(0)
        current.append(piece)
        total += n
    doc = _join(current, sep)
    if doc is not None:
        docs.append(doc)
    return docs


def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, separators=None):
    separators = SEPARATORS if separators is None else separators
    sep, rest = separators[-1], None
    for i, s in enumerate(separators):
        if s == "":
            sep = s
            break
        if s in text:
            sep, rest = s, separators[i + 1:]
            break
    pieces = [p for p in (list(text) if sep == "" else text.split(sep)) if p != ""]
    chunks, good = [], []
    for piece in pieces:
        if js_len(piece) < size:
            good.append(piece)
            continue
        if good:
            chunks.extend(_merge(good, sep, size, overlap))
            good = []
        chunks.extend([piece] if not rest else split_text(piece, size, overlap, rest))
    if good:
        chunks.extend(_merge(good, sep, size, overlap))
    return chunks


def title_of(dataset, text):
    """The article title: the heading of a Wix article, the `Title:` line of a TechQA technote (without the trailing country)."""
    first = text.lstrip("﻿").split("\n", 1)[0].strip()
    if dataset == "wix":
        return re.sub(r"^#\s*", "", first).strip()
    return re.sub(r"\s+-\s+United States$", "", re.sub(r"^Title:\s*", "", first).strip())


def chunk_article(dataset, text):
    """Chunks of one article as dicts: text, title, line (the line of the article where the chunk starts, counted from 1)."""
    title, out, start = title_of(dataset, text), [], 0
    for chunk in split_text(text):
        at = text.find(chunk, start)
        if at < 0:
            at = text.find(chunk)
        start = max(at, 0) + 1
        out.append({"text": chunk, "title": title, "line": text[:max(at, 0)].count("\n") + 1})
    return out
