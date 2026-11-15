# Speed and token usage of the running workflows

Read from n8n's execution records by `cost_speed.py`. Each row of a table is one successful request that reached the answer step (Wix and TechQA are separate). Times are in seconds.

## Wix: 171 requests

| Step | Seconds per request |
|---|---|
| Whole request | median 172.4, mean 227.7, 90th pct 480.4, max 731.5 |
| guard check on the question | median 0.3, mean 0.3, 90th pct 0.4, max 1.2 |
| embed question | median 0.7, mean 0.7, 90th pct 0.9, max 1.2 |
| vector search | median 0.0, mean 0.0, 90th pct 0.1, max 0.2 |
| answer call (mistral-large-4) | median 170.8, mean 225.9, 90th pct 478.7, max 730.0 |
| guard check on the reply | median 0.5, mean 0.5, 90th pct 0.6, max 1.3 |
| Everything except the answer call | median 1.7, mean 1.8, 90th pct 2.1, max 2.9 |

| Tokens per request | Input (prompt) | Output (completion) |
|---|---|---|
| Median | 2,697 | 649 |
| Mean | 2,903 | 728 |
| 90th percentile | 4,418 | 1,183 |
| Requests with usage recorded | 165 of 171 | 165 of 171 |

## TechQA: 237 requests

| Step | Seconds per request |
|---|---|
| Whole request | median 194.7, mean 281.2, 90th pct 724.3, max 731.4 |
| guard check on the question | median 0.3, mean 0.4, 90th pct 0.4, max 4.2 |
| embed question | median 0.7, mean 0.7, 90th pct 0.8, max 1.3 |
| vector search | median 0.0, mean 0.0, 90th pct 0.1, max 0.2 |
| answer call (mistral-large-4) | median 193.2, mean 279.5, 90th pct 722.4, max 730.0 |
| guard check on the reply | median 0.5, mean 0.5, 90th pct 0.6, max 1.5 |
| Everything except the answer call | median 1.6, mean 1.7, 90th pct 2.1, max 6.7 |

| Tokens per request | Input (prompt) | Output (completion) |
|---|---|---|
| Median | 1,798 | 636 |
| Mean | 1,803 | 746 |
| 90th percentile | 2,083 | 1,190 |
| Requests with usage recorded | 214 of 237 | 214 of 237 |
