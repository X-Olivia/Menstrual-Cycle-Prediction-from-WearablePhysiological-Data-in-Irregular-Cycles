# Archive — 可不再优先查看的代码与文档

本目录存放**已 superseded、一次性或历史**的代码与文档，当前主研究（排卵→月经 pipeline、分层与 personalization）无需日常查看。若需复现早期实验或查历史设计可在此查找。

---

## 目录结构

```
archive/
├── README.md                 # 本说明
├── model/
│   ├── experiment/           # 已归档的实验脚本
│   │   ├── run_experiment.py
│   │   ├── run_final_experiment.py, run_final_experiment_v2.py
│   │   ├── run_advanced_ov_menses.py
│   │   ├── run_unsupervised_ov_menses.py, run_multisignal_ov.py
│   │   ├── run_ovulation_experiments.py
│   │   ├── run_highfreq_temp_experiment.py, run_leakage_check.py
│   │   └── tune_optuna.py
│   ├── run_ab_*.py, run_seq_*.py, seq_*.py, two_stage.py, compare_all.py, residual_experiment.py  # 原 archive
│   └── (无 experiment 子目录时的其他归档)
└── data_process/
    ├── build_features_v3.py, build_features_v5.py, build_features_v6.py
    └── baseline_ovulation_probe.py

archive/docs/                 # 已归档的文档
├── 明日TODO_实验数据复盘与问题记录.md
├── 实验数据潜在问题清单.md
├── Meeting_Report_v4.md
├── 数据处理改进方案_v2.md, 数据处理流程检查.md, 数据处理计划_基于中期报告.md
├── 模型代码实现计划_基于数据与设计方案.md
└── model_v7_plan.md
```

---

## 归档说明

| 类型 | 内容 | 原因 |
|------|------|------|
| **实验脚本** | run_experiment.py | 仅简单 LGB，已被 run_final_ov_menses* 等取代 |
| | run_final_experiment.py, run_final_experiment_v2.py | 排卵+直接预测的“final”系列，主故事已改为 run_final_ov_menses* |
| | run_advanced_ov_menses.py | V1 全方法扫描，保留 run_advanced_ov_menses_v2 做 ablation |
| | run_unsupervised_ov_menses.py | v2，已有 run_unsupervised_ov_menses_v3 |
| | run_multisignal_ov.py | V1，已有 run_multisignal_ov_v2 |
| | run_ovulation_experiments.py | 排卵综合对比大脚本，当前重心在月经+分层 |
| | run_highfreq_temp_experiment.py | 一次性高频温度实验 |
| | run_leakage_check.py | 一次性泄漏检查，已完成 |
| | tune_optuna.py | 超参搜索，需要时可再运行 |
| **文档** | 明日TODO、实验数据潜在问题清单 | 一次性复盘/清单 |
| | Meeting_Report_v4 | 会议记录 |
| | 数据处理*、模型代码实现计划、model_v7_plan | 历史方案/计划，实现已完成或已迭代 |

---

## 当前应优先使用的代码与文档（留在主目录）

- **主 pipeline**：`model/experiment/run_final_ov_menses.py`, `run_final_ov_menses_v2.py`
- **Oracle / 检测实验**：`run_oracle_experiment.py`, `run_detected_ov_experiment.py`
- **多信号 / 验证**：`run_advanced_ov_menses_v2.py`, `run_multisignal_ov_v2.py`, `run_unsupervised_ov_menses_v3.py`, `validate_enhanced_labels.py`
- **评估与配置**：`model/robust_eval.py`, `model/config.py`, `model/ovulation_detect.py`, `model/dataset.py`
- **文档**：`docs/Stratification_Plan_vs_Current_Implementation.md`, `docs/Personalization_and_Stratification_Feasibility.md`, `docs/System_Architecture_v4.md`, `docs/Experiment_Design_Documentation.md`, 及各实验报告（Enhanced_Ovulation_Labels, Unsupervised_Ovulation_Menses, Data_Leakage_Audit 等）

---

## 如何运行已归档的脚本

归档脚本已移出 `model/experiment/`，故 `python -m model.experiment.run_xxx` 会报错。需要时有两种方式：

1. **临时复制回主目录**：将 `archive/model/experiment/run_xxx.py` 拷回 `model/experiment/` 后，在 `main_workspace` 下执行 `python -m model.experiment.run_xxx`。
2. **直接以脚本运行**：在 `main_workspace` 下执行 `python archive/model/experiment/run_xxx.py`（脚本内若用 `from model.xxx`，需保证当前目录或 PYTHONPATH 包含 `main_workspace`）。
