# Draft check and clarifying-question tests

Experiments 8 and 9 in [docs/experiments.md](../../../docs/experiments.md). Neither change was added to the live system.

| File | Contents |
|---|---|
| `draft_check.py` | Experiment 8: a second model (Laguna) reads the question, the retrieved passages and the draft, without the reference, and rates 1 to 5 how safe the draft is to send. `report` prints what blocking at each strictness would have gained and lost. |
| `checker.jsonl` | The saved ratings of 343 answers: all 93 wrong ones and 250 random good ones, which the report weights back up. |
| `clarify_check.py` | Experiment 9: Laguna rates how clear each of the 400 live questions is (1 vague, 5 clear), and the live outcomes of the vague ones are compared. |
| `clarify.jsonl` | The saved clarity ratings. |

```bash
cd eval
python results/extra_step_tests/draft_check.py report
python results/extra_step_tests/clarify_check.py report
```
