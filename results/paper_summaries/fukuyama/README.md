# FukuyamaBench，转换为本仓库的 PGL 任务格式

源：[HaCTang/ReactionMechanismReasoning](https://github.com/HaCTang/ReactionMechanismReasoning)
的 `fukuyama_bench/mechanisms`，319 例，取自福山透《Advanced Organic Reaction
Mechanism》习题册。转换脚本：`develop/fukuyama_to_pgl.py`（需要 RDKit，离线跑一次即可；
agent 运行时只会看到 PGL 文本）。

## 为什么需要转换

源数据有两处与本仓库不兼容：

1. **它记录的是状态，不是箭头。** 相邻状态能可靠给出成键/断键/键级/电荷变化，但给不出
   箭头方向（成键 A–B 是 A 进攻 B 还是反过来需要化学判断）。因此本转换产出的 GT 是
   **逐步的键变化**，而不是 `ReactionTransformation` 序列——两边都能机械计算，绕开了
   臆造箭头方向的问题。
2. **原子映射是逐步独立跑的。** 实测 1635 对相邻步中 **38.2%** 的同一物种在前一步产物与
   后一步反应物之间被重新编号。转换器按去映射规范 SMILES 匹配物种、再用子图同构做原子级
   对应，缝合出一套全局标号。

此外源数据把试剂放在用到它的那一步、把副产物在下一步丢弃。本转换将初态与终态**累积**：
初态含全路径用到的所有物种，终态含最终产物与所有副产物。否则 Planner 会被要求向它从未
见过的原子成键。

## 文件

| 文件 | 内容 |
|---|---:|
| `all.jsonl` | 310 例 / 1943 基元步（9 例因 SMILES 解析失败跳过） |
| `clean.jsonl` | **60 例 / 278 步**（A 18 / B 24 / C 18）：映射无断链、质子来源可追溯、没有缺映射号的重原子，且最低映射置信度 ≥0.5 |
| `core/` | `clean.jsonl` 的初态和目标态，采用现有加载器可读的 core 格式 |
| `core_forward/` | 与 `core/` 相同，额外附带反应条件文本，供隐藏目标态的正向模式使用 |

`setA_runnable.jsonl`（37 例）是加入质子账本和无映射重原子过滤之前的产物，已作废并删除。

每行字段：

- `initial_state_pgl` / `target_state_pgl` — 累积后的 PGL `ReactionState`，可直接喂给现有 Planner/Executor
- `states_pgl` — 全部中间态（比 mechfinder 多给的信息）
- `gt_step_changes` — 逐步的 `bonds_formed` / `bonds_broken` / `bond_order_changes` / `charge_changes`
- `quality` — 映射置信度、断链数、零变化步、显式化的可移动氢数、来源不明的质子数、无映射号重原子数、`initial_state_complete`
- `conditions` — 原始试剂/溶剂文本

## 该用哪个子集

**用 `clean.jsonl`。** 全量 310 例中，有 125 例初态不完整，原因如下：

- 117 例存在映射断链：某个原子在逐步映射中追溯不到反应物，因此无法出现在累积初态里。
- 7 例有来源不明的质子：质子转移用到的氢，在前面的状态中找不到出处。
- 1 例含没有映射号的重原子。

这些都是源数据的映射缺陷，不是转换问题，这些案例无法作为可运行的任务。

剩下 185 例初态完整。再去掉最低映射置信度低于 0.5 的 125 例，最终得到 60 例。这 60 例的平均映射置信度为 0.84，最低置信度的平均值为 0.71。

## 与 mechfinder 的关系

- **源数据几乎无重叠。** 295 个 Fukuyama 终产物物种与 mechfinder 前 8000 条的终产物对撞，
  只命中 1 条。一个是日文教科书习题，一个是 USPTO 专利。
- **任务粒度不同。** mechfinder 一条 = 一个命名反应（平均 3.9 步）；Fukuyama 一条 =
  一段合成序列（平均 6.3 步，最长 17 步），是把多个 mechfinder 式任务串起来。
- **反应类型有稀疏重合。** 1548 条条件文本中出现 Wittig 19、Grignard 10、aldol 10、
  还原 7、Boc 4、Friedel–Crafts 3 等。

## 需要知道的三点

1. **它的 GT 同样是机器生成的。** 构建流程是图像识别 → Gemini 生成机理 → 配平检查 → 修补
   （见源仓库 README）。原始图来自教科书，机理不是专家逐条标注。我们对 MechFinder 提出的
   那些问题（粒度惯例、共振式选择）同样适用。
2. **映射质量参差。** 全量 1997 步中 **35.2%** 的置信度低于 0.5。按置信度分层报告，或直接用
   `clean.jsonl`。
3. **原论文报的 SOTA 很低**：Qwen3-30B 经 SFT+GSPO 后 Set A exact pass@1 ≈ 8.3%。这个
   benchmark 要么更难，要么判定极严。作为**来源完全不同的泛化性检验**有价值，不要期待
   数字好看。

   用原论文自己的代码和协议零样本测试 Gemini 3.8，Set A exact pass@1 为 19.2%，结果见
   `outputs/expA_paper_protocol/` 和 `doc/rmr_benchmark_report.md` 第 9 节。

## 打分

不要硬凑箭头。逐步键变化两边都可机械计算：

```
GT_step_i  = gt_step_changes[i]
模型_step_i = diff(应用模型箭头前的状态, 应用后的状态)
```

分层思路可平移：L1 = 键变化集合是否一致（忽略步划分），L3 = 步划分也一致。
注意 `develop/state_diff.py` 只比较两状态**共有**的原子（那是给完成度闸门用的设计），
逐步 GT 需要比较全部键——这一点上两者口径不同，实测差异出现在 22% 的步骤上，全部由
物种进出解释。

实现见 `develop/score_external_benchmark.py`。它把键变化交给未改动的
`develop/metrics.py`，因此 L1/L2/L3、两种 F1 和失败诊断都沿用 MechFinder 的定义。测评结果见
`doc/rmr_benchmark_report.md`。
