# Benchmark data

The questions, reference answers and gold sources the evaluation uses. The files are copies of two public benchmarks, with fields added by this project (ids, truncation flags, the mapping from gold articles to file names, and the cross-dataset sets below).

| Set | Source | Licence |
|---|---|---|
| Wix | WixQA (`Wix/WixQA` on Hugging Face): expert-written questions and answers, and the help-center article snapshot | MIT |
| TechQA | TechQA-RAG-Eval (`nvidia/TechQA-RAG-Eval` on Hugging Face): IBM technotes and forum questions | Apache 2.0 |

Check the original licences before reuse.

| File | Contents |
|---|---|
| `wix_full.json` | The 200 Wix questions with their complete reference answers and gold article names. Built by `build_full_references.py`, which restores the full expert-written answers. |
| `wix_v2.json` | The same 200 questions, built by `build_eval_sets.py`. 85 reference answers are cut short (`answer_truncated`). |
| `techqa_full.json` | 608 TechQA questions with complete reference answers. The live evaluation uses 200 of them (ids 31 to 230). |
| `techqa_v2.json` | The same 608 questions from `build_eval_sets.py`. 19 reference answers are cut short. |
| `techqa_unanswerable.json` | 294 TechQA questions that have no answer in the technotes. The system should hand them off. |
| `wix_to_techqa_cross.json`, `techqa_to_wix_cross.json` | 100 questions each, asked of the other dataset's workflow. The system should hand them off. |
