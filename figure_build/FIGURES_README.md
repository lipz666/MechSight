# MechSight 论文图说明（2026-11-06）

所有图都由 Python 脚本直接生成 SVG，同时输出 PDF 和 PNG（4 倍分辨率）。宽 175 mm，正文文字 ≥ 5 pt，面板字母 8 pt。结构式用 `chemdraw.py` 绘制，为 ChemDraw 风格（ACS 1996 规范）；每个结构都附有可在 ChemDraw 中编辑的 `.cdxml`。图中数字全部由脚本从冻结的结果文件读取，并在脚本里断言核对。

## 图表一览

| 图 | 文件（`out/`） | 绘图脚本 | 内容 |
|---|---|---|---|
| Fig. 1（新） | `angew_fig1_concept_20261106/out/fig1_concept.*` | `build/fig1n.py` | a：溴鎓离子概念图（带推电子箭头），以及 251 个漏掉的隐式中间体拆分为惰性知识 / 盲区，分别对应 Auto / Guided；b：模板内外对比；c：M3 反事实三联体 + M2 等效；d：隐式中间体数量分层 + GT 注入 |
| Fig. 2（正文版） | `angew_fig2_mechagent_20261104/out/fig2_mechagent.*` | `build/fig2.py` | 方法示意、244 题确认集、分层、消融、argon 第二主干 |
| Fig. 2（方法图 v3） | `angew_fig_method_20261105/out/fig_method_v3.*` | `build/figm3.py` | a：Agent 循环；b：J3-B057 真实运行轨迹；c：完整 Favorskii 路径，每一步都有推电子箭头 |
| Fig. 3 | `angew_fig3_generalization_20261104/out/fig3_generalization.*` | `build/fig3.py` | a：76 题盲区救回（早期 Guided v7）；b：2026 文献，开发集 165 篇 + 预注册确认集 271 篇；c：两个案例（PMC13171995 Auto；PMC13321810 Guided），带推电子箭头 |
| ToC | `angew_toc_20261104/out/toc_graphic.*` | `toc.py` | 55 × 50 mm，文字 ≥ 6 pt，惰性知识 51.4 % / 盲区 47.8 % |

正文用到的 Fig. 2、Fig. 3 和 ToC 已复制到 `manuscript/figures/`，旧版本在 `manuscript/figures/_prev_20261106/`。

## 共享工具

- `figkit.py`：统一配色（灰 = Direct，青 = Auto，紫 = Guided，琥珀 = 隐式中间体，红 = 缺口）和绘图元件。推电子箭头用 `curly()`（`half=True` 为单钩箭头），按原子坐标定位用 `atom_xy()` / `bond_mid()`。
- `chemdraw.py`：结构渲染和 CDXML 导出，支持缩写 Ph、SiPh₃、C₃F₇、CO₂Me、Tol 等。
- `data/figdata.py` → `data/figdata.json`：Fig. 2 和 Fig. 3 的数据。Fig. 1 直接在脚本里读结果文件。

## 重新生成

```bash
cd publication
P=../external/venv-llm/bin/python
$P data/figdata.py
$P angew_fig1_concept_20261106/build/fig1n.py
$P angew_fig3_generalization_20261104/build/fig3.py   # 其余脚本同理
$P angew_fig_method_20261105/build/render.py <图.svg> <图.png> <图.pdf> 4
```

## 注意事项

1. "显式 70.6 % vs 隐式 20.3–25.4 %"和"+hidden GT +33.3 / +visible GT +2.2"找不到对应的结果文件，没有画进图里。
2. 文献评分器按 σ 骨架匹配，忽略电荷和自由基。Fig. 3c 案例 1 的 5/5 中，PhS⁻ 是通过骨架相同的 PhS• 计入的；按物种严格比对是 3/5 → 4/5。
3. Fig. 3c 下半的双环[2.1.1]己烷产物有 F 压键，需要在 ChemDraw 里手调 `angew_fig3_generalization_20261104/build/mols/ct_product.cdxml`。
4. 方法图 J3-B057 例子中，第 ③ 步保留的是候选池里本来就含溴鎓离子的那条路径，不是把第 ② 步的结果插进去。确认集上，候选池加修正贡献了大部分增益（合并结果中 MechSight 只比"仅候选池"对照高 +2.0 pp）。
