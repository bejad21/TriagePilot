# Rows with an unclear label

Eight of the 400 questions have a reference answer that does not fit the question well. They are **kept** in every headline number; the table in `README.md` shows the scores with them left out as a side line. The datasets were not edited.
The judges only see the reference and the system answer, so a good answer that differs from an unfair reference is graded as wrong.

| Row | Why the label is unclear |
|---|---|
| wix:59 | The answer names the cart and sales reports that match the question; the reference is a broader list of sales reports. |
| wix:144 | The question is vague ("not responsive"); no gold article exists, and the reference text is mostly absent from the Wix docs. |
| wix:185 | The answer explains entering and formatting Rich Text, which the cited article contains; the reference only covers adding the field. |
| wix:197 | The reference describes Editor X, while the customer uses the Wix Editor; the reference text is almost absent from its own gold article. |
| techqa:31 | The customer asks how to remove compressed references; the answer gives `-Xnocompressedrefs`, a direct answer. The reference gives a different workaround. |
| techqa:36 | The reference ("contact your vendor") only fits a Student Version buyer; the question never says that, so the right document cannot be found from it. |
| techqa:179 | The reference is a list of Mac OS support, unrelated to authorizing SPSS 19 on a new computer. |
| techqa:216 | The reference is a CVE list from a sibling bulletin; the answer correctly states what the matching bulletin says. |

Related notes: TechQA 52 changes between runs (the live search finds the right document; the model cited a look-alike once and the right document on a rerun), so it is counted as ordinary run-to-run noise, not as a label problem.
