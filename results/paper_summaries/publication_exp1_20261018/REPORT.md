# 实验 1：同模型三组对照

**状态：480 条轨迹全部完成。**

| 方法 | 完成/160 | L1 | L2 | L3 | 重原子终态匹配 | 平均生成 token | 平均总 token | 平均调用 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | 160 | 44.4% | 30.0% | 27.5% | 100.0% | 70778 | 130581 | 4.0 |
| agent_no_skills | 160 | 48.8% | 32.5% | 29.4% | 98.8% | 47214 | 150936 | 13.3 |
| agent_skills79 | 160 | 40.0% | 30.0% | 27.5% | 72.5% | 83024 | 239061 | 25.7 |

## 配对差值（按反应重采样的 95% 区间）

- direct_to_agent_no_skills，80 个反应 / 160 对轨迹：
  - transfer_set_match: +4.4 个百分点，95% CI [-0.6, +10.6]。
  - transfer_sequence_match: +2.5 个百分点，95% CI [-1.2, +6.9]。
  - step_partition_match: +1.9 个百分点，95% CI [-1.2, +5.6]。
  - heavy_final_match: -1.2 个百分点，95% CI [-3.1, +0.0]。
- agent_no_skills_to_agent_skills79，80 个反应 / 160 对轨迹：
  - transfer_set_match: -8.8 个百分点，95% CI [-16.9, -1.2]。
  - transfer_sequence_match: -2.5 个百分点，95% CI [-11.2, +5.6]。
  - step_partition_match: -1.9 个百分点，95% CI [-10.6, +6.2]。
  - heavy_final_match: -26.2 个百分点，95% CI [-35.6, -18.1]。
- direct_to_agent_skills79，80 个反应 / 160 对轨迹：
  - transfer_set_match: -4.4 个百分点，95% CI [-13.1, +4.4]。
  - transfer_sequence_match: +0.0 个百分点，95% CI [-8.8, +8.8]。
  - step_partition_match: +0.0 个百分点，95% CI [-8.8, +8.8]。
  - heavy_final_match: -27.5 个百分点，95% CI [-36.9, -19.4]。

## 解释边界

- 样本为已知反应家族的新记录，不能据此宣称未见类别泛化或排除预训练污染。
- 原始来源是模板标注的 MechFinder；L1/L2/L3 衡量与标注箭头的一致性，不等同于专家认定的唯一正确机理。
- 三组生成 token 上限相同，但实际 token 和调用量不同；不能省略成本直接声称等计算量优越。
- 网关可能在一次请求中产生超过 max_tokens 的思考 token，因此 98304 是按已返回用量停止后续调用的阈值，并非供应商可保证的硬上限；超额轨迹数另行统计。
- 重原子终态指标忽略氢归属，不能替代完整状态匹配。
- 网关请求模型为 gemini-3.8-flash-high，并显式传入 reasoning_effort=high；实际响应模型名另存 summary.json。
- 完整协议见 PROTOCOL.md，数据和代码哈希见 manifest.json。
