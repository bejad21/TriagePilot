from ingest.chunking import chunk_article, js_len, split_text, title_of


def test_short_text_is_one_chunk():
    assert split_text("Hello world.") == ["Hello world."]


def test_paragraphs_are_merged_up_to_the_size_limit_with_overlap():
    paragraphs = [("word " * 90).strip() for _ in range(6)]
    chunks = split_text("\n\n".join(paragraphs))
    assert all(js_len(c) <= 1000 for c in chunks)
    assert len(chunks) > 1
    assert chunks[0].split("\n\n")[-1] == chunks[1].split("\n\n")[0]


def test_separators_are_dropped_and_pieces_rejoined_with_the_separator():
    text = "alpha\n\nbeta\n\ngamma"
    assert split_text(text) == [text]


def test_overlong_word_run_is_split_further_down_the_separator_list():
    chunks = split_text("x" * 2500)
    assert [js_len(c) for c in chunks] == [1000, 1000, 900]


def test_length_counts_utf16_units_like_javascript():
    assert js_len("a") == 1
    assert js_len("\U0001F600") == 2


def test_empty_text_gives_no_chunks():
    assert split_text("") == []
    assert split_text("\n\n\n") == []


def test_titles_for_both_datasets():
    assert title_of("wix", "# About Domains\n\nBody") == "About Domains"
    assert title_of("techqa", "Title: IBM Fix PI1: crash - United States\r\n\r\nText:\r\nbody") == "IBM Fix PI1: crash"


def test_chunk_article_reports_the_start_line_of_each_chunk():
    text = "# Title\n\n" + "\n\n".join(("line " * 60).strip() for _ in range(8))
    chunks = chunk_article("wix", text)
    assert [c["line"] for c in chunks] == [1, 3, 5]
    assert {c["title"] for c in chunks} == {"Title"}
