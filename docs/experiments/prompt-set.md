# Reader, writer and checker prompts

Experiment 5 in [experiments](../experiments.md). The question: does splitting the answer into a reader step, a writer step and a checker step beat the single prompt? It did not. This page holds the final tables from the saved run outputs. The scripts and raw files of this run are not part of this repository.

## The three prompts

- A **reader** finds the evidence and flags ambiguity, wrong premises and missing key terms.
- A **writer** answers only from that evidence, covers every part and says "not in the documentation" when needed.
- A **checker** looks at parts covered, unsupported claims, product match and the headline, then allows one rewrite or a hand-off.

They were tested on 40 Wix questions fixed before any run: the 20 wrong answers of the whole-article version plus 20 good ones as a control. Both arms replay the same saved passages with the same model (`ministral-14b`, temperature 0). Arm A is the baseline prompt and arm B the three new prompts. The panel is the majority of three judges.

## Result (out of 20 questions per group)

| | A, baseline prompt | B, new prompts |
|---|---|---|
| The 20 wrong ones: good and sent | 4 | 4 |
| The 20 wrong ones: still wrong and sent | 15 | 13 |
| The 20 wrong ones: handed off | 1 | 3 |
| The 20 good ones: good and sent | 19 | 18 |
| All 40: good sent / wrong sent / handed off | 23 / 15 / 2 | 22 / 13 / 5 |
| Score on the 40 | 0.20 | 0.23 |
| Tokens per question | 4,551 / 3,533 | 9,277 / 8,233 (3.5 calls) |

- No gain. Re-running the baseline prompt fixed 4 of the 20 by chance, and the new prompts fixed 4, three of them the same questions. The net difference is one question, at about twice the tokens.
- The prompts were written after reading these 20 failures, so any gain would have been optimistic. There is none.
- The reader step makes the same selection mistakes the writer would make, and the checker compares the draft with the reader's own notes, so it passes them. The same model cannot see an error it made earlier.
- 8 of the 20 cannot be fixed by any prompt: 5 where the search never returned the article and 3 test-set problems.

## Failure types of the wrong answers sent

Every wrong answer sent (Wix tuning set: 13 and 5 in two versions, Wix held-out set: 15, TechQA: 8 and 8) was read against its reference and its article. Wix has 30 distinct wrong questions and TechQA 16. A question can carry two tags.

| Type | What went wrong | Questions |
|---|---|---|
| Incomplete | Several parts, or steps for several products, and the answer covers only some | 11 |
| Missed the headline | The article's key statement is left out and the answer lists side issues | 5 |
| Wrong product, feature, direction or version | Look-alike products or screens | 10 |
| Invented content | Steps, icons or limits the articles do not state | 7 |
| False premise accepted | The customer's assumption is wrong or narrower than the article | 2 |
| Answered a neighbouring problem | Key terms are not in the articles and the answer fills in with a related article | 11 |
| Garbage output | The reply is the output instructions | 3 |
| Test-set problem | Reference not in the source article or answers only half the question | 10 |

A prompt can address the first six types and the garbage output. It cannot fix test-set problems, or the cases where the search never returned the article.

## Limits

40 questions, one run, AI judges, temperature 0.
