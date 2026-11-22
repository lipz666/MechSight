# 机理预测测试例（11 条）

从 `outputs/gemini37_oracle_steps/`（Gemini 3.7、`structures_only`、180 rollouts / 60 反应类）里挑出的一组测试例，
带 SMILES 输入，可以直接拿去喂别的模型、画图或做人工核对。

- `testcases.csv` —— 表格视图，每行一个反应
- `testcases.jsonl` —— 同样 11 条，额外带完整的 `initial_state_pgl`、`target_state_pgl`、`gt_actions_pgl`
- 生成脚本见本文件末尾

## 怎么挑的

三条硬性条件，一条软性目标：

1. **GT 与预测路径都能干净回放**。用 `standalone_chemistry.apply_action` 从初态逐步施加，中途不能报错——否则这条例子没法评分、没法画图、也没法做差异对比。60 个反应里只有 13 条同时满足。
2. **规模可读**：重原子 ≤ 34，画出来一页能看清。
3. **`repeat_index=1`**，避免同一反应的三次重复互相干扰。
4. 软性目标：**难度成梯度**（1 步到 6 步），且**覆盖确定性指标区分的每一种失败模式**。

结果是 5 条模型完全做对（L1/L2/L3 全过）、5 条 `content_mismatch` 或 `missing_events`、1 条 `split_steps`。
这个比例比全量偏乐观（全量 180 条里只有 68 条 `match`），因为"能干净回放"本身就是一道过滤。

## 11 条

| 反应类 | GT 步数 | 原子 | L1/L2/L3 | 诊断 | 输入 SMILES |
|---|---:|---:|:---:|---|---|
| SN2 | 1 | 17 | ✓✓✓ | match | `O=[N+]([O-])c1cccc(CBr)c1.N` |
| alkene_epoxidation | 1 | 32 | ✗✗✗ | content_mismatch | `O=C(OO)c1cccc(Cl)c1.C=CCC1(O)CCN(C(=O)OC(C)(C)C)CC1` |
| SN1 | 2 | 22 | ✓✓✓ | match | `CC(C)(C)OC(=O)C(Br)Oc1cc(Cl)cc(Cl)c1.C[O-]` |
| SNAr(ortho) | 2 | 31 | ✗✗✗ | missing_events | `O=C(Nc1cccc(C(F)(F)F)c1)c1ccccc1Br.NCCc1ccncc1` |
| alcohol_attack_to_carbonyl_or_sulfonyl | 3 | 25 | ✗✗✗ | content_mismatch | `CS(=O)(=O)Cl.CC(=O)OC(COCCO)c1cccc(Cl)c1` |
| O_demethylation | 3 | 31 | ✗✗✗ | content_mismatch | `COc1ccc(-c2nn(C(C)C)c3c(Cl)cccc23)cc1` |
| double_SN2 | 4 | 24 | ✓✓✓ | match | `ClCCNCCCl.Nc1cccc2c1CCCC2` |
| imine_formation | 4 | 28 | ✗✗✗ | content_mismatch | `O=Cc1ccccn1.Cc1ccccc1C(N)c1ccccc1C` |
| isothiocyanate_synthesis | 5 | 18 | ✓✓✓ | match | `S=C(Cl)Cl.CNS(=O)(=O)c1ccc(N)cc1` |
| acetal_formation | 6 | 28 | ✓✓✗ | split_steps | `OCCO.O=Cc1ccc2nc(C(=O)Nc3ccccc3)cn2c1` |
| Paal_Knorr_pyrrole_synthesis | 6 | 31 | ✓✓✓ | match | `CC1CCC(=O)C(CC(=O)c2ccccc2)C1.CCN(CC)CCN` |

几条值得单独看的：

- **SN2 / SN1** —— 最小的正例，适合当冒烟测试。
- **imine_formation** —— 模型和标注都给出化学上说得通的路径（模型先质子化再加成，GT 先加成再转质子），但电子转移集合对不上，环境判 success、指标判 `content_mismatch`。「环境成功 ≠ 机理正确」的标准案例。
- **SNAr(ortho)** —— `missing_events`：模型漏了 GT 里的事件，没有多出来的。
- **acetal_formation** —— `split_steps`：电子和顺序都对（L1、L2 过），只是分步边界比标注多切了一刀。MechFinder 按反应类给固定步数，这一类失败衡量的是标注约定，不是化学。

## 字段

| 列 | 含义 |
|---|---|
| `reaction_id` | `mechfinder::<反应类>::<source_row_index>`，末段就是 `core/mechfinder_merged_pgl_paths.core.jsonl` 里的行号 |
| `gt_steps` / `predicted_steps` | 标注的 / 模型的 elementary step 数 |
| `L1_transfer_set` | 电子转移集合正确（忽略分步与顺序） |
| `L2_transfer_sequence` | L1 + 因果顺序正确 |
| `L3_step_partition` | L2 + 分步边界与标注一致 |
| `diagnosis` | `match` / `content_mismatch` / `missing_events` / `extra_events` / `reordered` / `split_steps` / `merged_steps` |
| `matched_transfers` / `gt_transfers` | 命中的电子转移数 / 标注总数 |
| `env_success`、`stop_reason` | 环境层面的结局，注意它和上面的指标经常不一致 |
| **`input_smiles`** | **反应物 SMILES（去原子映射，已规范化）——要喂给别的工具就用这一列** |
| `product_smiles` | 产物 SMILES（去原子映射） |
| `uspto_reaction_smiles` | 原始 USPTO 反应 SMILES，带原子映射，`反应物>>产物` |
| `pgl_initial_smiles` / `pgl_target_smiles` | 从 Agent 实际看到的 PGL 初态/目标态反推的 SMILES |
| `data_notes` | 该条数据的已知标注瑕疵，见下 |

## 两个必须知道的数据瑕疵

跑之前请先看 `data_notes` 这一列。这两个问题出在 MechFinder 标注本身，不是转换脚本引入的：

1. **形式电荷会丢**（11 条里有 5 条）。例如 SN2 那条，硝基在 USPTO SMILES 里是 `[N+](=O)[O-]`，PGL 初态却只保留了 `N7` 的 +1，氧上的 −1 没了，整体净电荷从 0 变成 +1。后果是任何按价键补氢的工具（包括 RDKit）都会给那个氧补上一个并不存在的 H——`pgl_initial_smiles` 里能看到 `O=[N+](O)c1...`。**做定量对照请用 `input_smiles`/`uspto_reaction_smiles`，它们的电荷是对的。**
2. **目标态里有桥连氢**（3 条）。标注用「一个 H 同时连着给体和受体」来表示质子转移的终态，例如 `Cl[H]Cl`。这在价键模型里非法，`pgl_target_smiles` 那一格因此不能被 RDKit 解析。画图和回放不受影响（PGL 层面只是一条键），但别把它当成可解析的 SMILES。

## 跑其中一条

`run_agent.py` 支持按 id 精确或子串筛选：

```bash
cd <PROJECT_ROOT>/MechAgent && python -m develop.run_agent --data-source core --core-subset mechfinder --reaction-id "mechfinder::imine_formation::136" --repeats-per-task 1 --task-input-mode structures_only --analysis-output outputs/probe_imine.jsonl
```

评分和画图：

```bash
cd <PROJECT_ROOT>/MechAgent && python -m develop.evaluate_gt --result outputs/probe_imine.jsonl
```

想把某条画成论文图，把它加进 `develop/viz/build_payload.py` 顶部的 `CASES`，重建后用 `develop/viz/export_figure.py` 导出。

## 重新生成

```bash
/tmp/vizenv/bin/python <PROJECT_ROOT>/MechAgent/testcases/build_testcases.py
```

需要 RDKit（仅用于 SMILES 转换与规范化）。改选取规则见脚本末尾的 `take(...)` 部分；换 run 改脚本顶部的 `RUN`。
