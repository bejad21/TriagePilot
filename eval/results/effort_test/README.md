# Reasoning effort test

Built by `effort_test.py report`.

## Wix: 43 questions (20 wrong and 10 handed off in the live run, 13 random good)

| Setup | Good sent | Wrong sent | Handed off | Score on these rows | Not graded |
|---|---|---|---|---|---|
| Live run (shared free endpoint, low) | 13 | 20 | 10 | -0.16 | 0 |
| Mistral API, low | 16 | 15 | 12 | +0.02 | 0 |
| Mistral API, medium | 16 | 15 | 10 | +0.02 | 2 |
| Mistral API, high | 12 | 18 | 11 | -0.14 | 2 |

| Compared with low | Wrong turned good | Good turned wrong | Wrong turned hand-off | Hand-off turned good | Hand-off turned wrong |
|---|---|---|---|---|---|
| medium | 0 | 0 | 0 | 1 | 1 |
| high | 0 | 2 | 0 | 0 | 1 |

## TechQA: 57 questions (13 wrong and 31 handed off in the live run, 13 random good)

| Setup | Good sent | Wrong sent | Handed off | Score on these rows | Not graded |
|---|---|---|---|---|---|
| Live run (shared free endpoint, low) | 13 | 13 | 31 | +0.00 | 0 |
| Mistral API, low | 13 | 11 | 33 | +0.04 | 0 |
| Mistral API, medium | 14 | 13 | 30 | +0.02 | 0 |
| Mistral API, high | 14 | 11 | 31 | +0.05 | 1 |

| Compared with low | Wrong turned good | Good turned wrong | Wrong turned hand-off | Hand-off turned good | Hand-off turned wrong |
|---|---|---|---|---|---|
| medium | 0 | 1 | 1 | 2 | 2 |
| high | 0 | 0 | 1 | 2 | 1 |

## Time and tokens per request (all 100 rows; Wix and TechQA combined only here because the call itself does not differ by dataset)

| Effort | Time, median | Time, mean | Time, 90th pct | Input tokens | Output tokens | of which reasoning | Total vs low |
|---|---|---|---|---|---|---|---|
| low | 7.7 s | 10.4 s | 18.0 s | 2,272 | 967 | 801 | 1.00x |
| medium | 11.7 s | 14.4 s | 27.7 s | 2,272 | 1,515 | 1,341 | 1.17x |
| high | 22.2 s | 30.5 s | 48.9 s | 2,272 | 2,725 | 2,530 | 1.54x |

Cost per 1,000 requests at the two quoted list prices ($0.70 / $2.10 and $1.36 / $4.18 per million input / output tokens):

| Effort | Lower price | Higher price |
|---|---|---|
| low | $3.62 | $7.13 |
| medium | $4.77 | $9.42 |
| high | $7.31 | $14.48 |
