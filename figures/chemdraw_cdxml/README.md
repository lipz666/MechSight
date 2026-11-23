# ChemDraw 源文件（CDXML）

所有结构都按 **ACS Document 1996** 文档设置写入：键长 14.4 pt，线宽 0.6 pt，边距 1.6 pt，双键间距为键长的 18 %，链角 120°，标签 Arial 10 pt。

| 文件夹 | 内容 |
|---|---|
| `fig1/` | 4-甲基苯乙酮（同系物实验） |
| `fig2/` | 普列酮 → 溴鎓 → 二溴化物 → 烯醇盐 → 环丙酮 → 四面体中间体 → 酯（J3-B057） |
| `fig3/` | 碘代环己烷 → 自由基 → 酰基自由基 / 酰基正离子 → 酰基碘 → 四面体中间体 → 酰基铵 → 酰胺（*Chem. Sci.* 2026） |

每个文件夹里有每个结构的单独文件 `<name>.cdxml`，另有 `all_structures.cdxml`（全部结构排在一页上，结构下方标有名字）。

## 在 ChemDraw 里使用
1. 打开 `all_structures.cdxml`。
2. 全选，执行 Structure ▸ Clean Up Structure（可选；坐标已按 120° 排好）。
3. 如需和其他图统一样式：File ▸ Apply Document Settings from ▸ ACS Document 1996。
4. 每个结构导出一份 SVG（File ▸ Save As ▸ SVG），文件名与 `.cdxml` 相同，然后把文件交回来。ChemDraw 导出的 SVG 头部格式和现在的渲染器不同，需要加一小段适配代码才能嵌进 `fig*.py`。在那之前，图里用的是 `publication/chemdraw.py` 按同样规则画的 ChemDraw 风格结构。

## 校验
所有文件都用 RDKit 的 CDXML 读取器回读过，规范 SMILES 与原结构完全一致（16 / 16，包括电荷、H 数和自由基）。
**尚未在 ChemDraw 本身打开测试**，因为这台机器上没有安装 ChemDraw。
