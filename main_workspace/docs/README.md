# main_workspace 文档索引

当前**建议优先阅读**的文档（按用途分类）。已归档的文档见 `archive/docs/`。

---

## 研究设计与可行性

| 文档 | 用途 |
|------|------|
| [Stratification_Plan_vs_Current_Implementation.md](Stratification_Plan_vs_Current_Implementation.md) | **分层与 personalization 研究计划**：假设 H1/H2、irregular 定义、B1/B2/M3、k-shot、两种 protocol、interaction 分析，与当前实现的对照与落地步骤 |
| [Personalization_and_Stratification_Feasibility.md](Personalization_and_Stratification_Feasibility.md) | Personalization 正式定义、四层次、论文可用的英文表述、可预测性分层与异质性收益的可行性 |
| [MetaLearning_Feasibility_and_Unlabeled_Cycles.md](MetaLearning_Feasibility_and_Unlabeled_Cycles.md) | 无 LH 标签周期成因、排卵→月经是否需 personal、Meta-learning 可行性结论 |

---

## 系统与实验

| 文档 | 用途 |
|------|------|
| [System_Architecture_v4.md](System_Architecture_v4.md) | 经期预测系统架构（数据处理、特征、模型、代码结构） |
| [Experiment_Design_Documentation.md](Experiment_Design_Documentation.md) | 实验设计：各 run_* 脚本职责、划分方式、评估指标 |
| [Data_Leakage_Audit_Report.md](Data_Leakage_Audit_Report.md) | 数据泄漏审计与 LOSO 划分正确性 |

---

## 实验报告与结果

| 文档 | 用途 |
|------|------|
| [Enhanced_Ovulation_Labels_Report.md](Enhanced_Ovulation_Labels_Report.md) | 增强排卵标签恢复（95→111 周期）、四级恢复策略 |
| [Unsupervised_Ovulation_Menses_Report.md](Unsupervised_Ovulation_Menses_Report.md) | 无监督排卵→月经 pipeline、固定 vs personal 黄体期 |
| [Advanced_Experiment_Report.md](Advanced_Experiment_Report.md) | 高级实验：LOSO、多方法、泄漏防护 |
| [MultiSignal_Ovulation_Experiment_Report.md](MultiSignal_Ovulation_Experiment_Report.md) | 多信号排卵检测实验 |
| [Ablation_Study_Results.md](Ablation_Study_Results.md) | Ablation 结果与特征重要性 |

---

## 已归档（可不再优先看）

见 **`../archive/docs/`**：明日 TODO、实验数据潜在问题清单、Meeting 报告、数据处理/模型实现等历史方案与计划。
