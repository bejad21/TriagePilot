# Final numbers of the live run

Source: `../live_best/work/` (200 Wix and 200 TechQA questions sent through the live webhooks, answers and judge grades saved). Score = (good answers sent - wrong answers sent) / questions. Good = at least 2 of 3 judges give 4 or more. Run-to-run noise is about 0.04 to 0.05.

## Headline: current panel (Nemotron-3.5-super, GLM-4.5-flash, Laguna-xs-2.1)

| Dataset | Questions | Answered | Handed off | Good sent | Wrong sent | Wrong share of sent | Score (95% interval) |
|---|---|---|---|---|---|---|---|
| Wix | 200 | 190 | 10 | 170 | 20 | 10.5% | +0.750 (+0.660 to +0.835) |
| TechQA | 200 | 169 | 31 | 156 | 13 | 7.7% | +0.715 (+0.635 to +0.795) |

## Same panel without the 8 unclear-label rows (4 Wix, 4 TechQA)

| Dataset | Questions | Answered | Handed off | Good sent | Wrong sent | Wrong share of sent | Score (95% interval) |
|---|---|---|---|---|---|---|---|
| Wix | 196 | 186 | 10 | 170 | 16 | 8.6% | +0.786 (+0.704 to +0.862) |
| TechQA | 196 | 165 | 31 | 156 | 9 | 5.5% | +0.750 (+0.673 to +0.821) |
