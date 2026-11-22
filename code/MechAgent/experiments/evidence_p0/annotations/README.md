# P0 人工标注说明

对象：`papers.csv` 中 `hand_annotate=1` 的 10 篇。**标注前不要打开这 10 篇的 LLM 输出**
（`outputs/evidence_p0_20261101/<model>/<PMCID>.json`），以免被锚定。

每篇一个文件 `annotations/<PMCID>.json`：

```json
{
  "paper_id": "PMC12780792",
  "annotator": "annotator_initials",
  "minutes_spent": 40,
  "cards": [
    {
      "experiment_type": "radical_trap",
      "perturbation": "TEMPO (2.0 equiv) added under standard conditions",
      "observable": "yield of 3a",
      "outcome_raw": "3a not detected; TEMPO-alkyl adduct detected by HRMS",
      "outcome_bin": "suppressed",
      "location": {"file": "si", "page": 23, "label": "Scheme S5a"},
      "author_interpretation": "alkyl radical involved",
      "competing_hypotheses_discussed": false
    }
  ]
}
```

规则：

1. 一个实验 = 一张卡。同一次实验测了两个量（TEMPO：产率被抑制 + 检测到加合物）→ 两张卡。
2. 正文与 SI 报告同一实验 → 只标一张，`location` 写信息更全的那处。
3. `experiment_type` 与 `outcome_bin` 用 `../schema.json` 的取值；`outcome_bin` 写括号前的词（`suppressed`、`primary`）。
   标准产率的参照取同一张表/Scheme 的标准条件。
4. `outcome_raw` 只写观察（数值、检测到的物种），不写“表明/支持/自由基途径”之类。
5. 不标：底物范围、条件优化（除非是“无光/无催化剂/空气下”这类显式对照）、DFT。
6. 可选字段 `competing_hypotheses_discussed`：作者是否明确讨论并排除了另一机理（为 P4 选文用）。
7. 记录用时 `minutes_spent`，用于估算扩到 100–200 篇的人工成本。

标完后运行 `python score.py --models gemini-3.5-flash-lite`，`outputs/.../matches.jsonl` 中逐对核查匹配是否正确。
