# Multisignal Prefix Pipeline：方法清单与基准结果

本文档汇总 `new_workspace/record/experiment/multisignal_pipeline` 当前基准中**可能出现的所有方法类别**，并记录一次可复现的 **`run.py` fast 模式**实测输出（含基线与排序表）。  
完整逐日打印日志可重跑后自行保存：`python run.py 2>&1 | tee benchmark_log.txt`（在仓库内已配置 Python 环境下）。

---

## 1. 元数据（本次记录）

| 项目 | 值 |
|------|-----|
| 记录时间（UTC） | 2026-03-31T02:44Z |
| 仓库 `git`（工作区） | `540d7eb`（以你本机 `git rev-parse` 为准） |
| 基准脚本 | `run.py` → `benchmark_main.run_prefix_benchmark` + `report_utils` |
| 模式 | **fast**（`config/protocol_main.py` 中 `FAST_PREFIX_BENCHMARK = True`） |
| 运行耗时（本次） | 约 14s（含缓存命中） |
| 周期标签 | `new_workspace/processed_dataset/cycle_cleaned_ov.csv` |
| 信号目录 | `new_workspace/processed_dataset/signals` |
| Rule σ / ML σ | 2.0 / 1.5 |

---

## 2. 评估与排序在做什么（简要）

- **前缀约束**：仅在每个周期日 `d` 使用当日及之前可见数据；无未来泄漏。
- **主打印指标**：PostOvDays、AllDays、PostTrigger、多锚点经期误差等；详定义见 `menses.py` / `evaluate_prefix_*`。
- **候选排序规则**（相对 Calendar 的 PostOvDays MAE 优先，见终端 「Ranking rule」）：
  1. PostOvDays MAE 相对 Calendar 更低  
  2. AllDays MAE 相对 Calendar 更低  
  3. `first_detection_day_mean` 更早  
  4. `first_detection_ov_mae` 更低  
  5. `availability_rate` 更高  

---

## 3. 方法分类（本仓库实现的「这一套」）

### 3.1 基线（每次 `run.py` 都会评估）

| 名称 | 类型 | 说明 |
|------|------|------|
| **Calendar** | 基线 | 无个体排卵估计；用于相对排序参照。 |
| **Oracle-prefix** | 上界 | 自真实排卵日起在 prefix 内「告知」真值排卵日（不可部署，用于对照）。 |

### 3.2 Fast 模式候选（`FAST_PREFIX_BENCHMARK=True` 时）

| 名称 | 家族 | 说明 |
|------|------|------|
| Rule-TempOnly-ftt_prefix | rule-fused-tt | 温度组多信号 fused **前缀 t 检验**（`detectors_rule`）。 |
| Rule-HROnly-ftt_prefix | rule-fused-tt | 心率组 fused 前缀 t 检验。 |
| PhaseCls-HROnly | phasecls-rf | **监督** LOSO 相位分类（RF）+ 默认触发 + 确定性 localizer；特征组 HROnly。 |
| PhaseCls-TempOnly | phasecls-rf | 同上，特征组 TempOnly。 |
| PhaseCls-Temp+HR[RF-baseline] | phasecls-rf | RF 单模 + baseline 触发；主线对照。 |
| PhaseCls-ENS-Temp+HR[Champion] | phasecls-ens | RF+HGB 相位概率 **集成** + `score_smooth` 局部细化（实验性组合，fast 池内仍参与排名）。 |
| PhaseCls-Temp+HR[EvidenceSticky] | phasecls-rf | `trigger_mode=evidence` + `sticky` 稳定策略 + 证据门控。 |
| RuleState-Temp+HR | rule-state | **无监督式**局部证据触发：`prefix_rule_state_detect`（非 LOSO 分类器得分）。 |

### 3.3 Full 模式候选（`FAST_PREFIX_BENCHMARK=False`，或调用 `main(mode="full")`）

在 `benchmark_main._full_candidate_specs` 中额外包括（**本次未跑 full，下表无数值**）：

- **单信号规则**：每个 `PREFIX_SINGLE_SIGNAL_SPECS` 的 `{代号}-tt_prefix` 与 `{代号}-cusum_prefix`。  
- **分组规则**：每个 `PREFIX_RULE_SIGNAL_GROUPS` 的 `{组名}-ftt_prefix` 与 `{组名}-cusum_prefix`。  
- **PhaseCls**：`PHASECLS_MODEL_TYPES` × 每个 `PREFIX_ML_SIGNAL_GROUPS` 的相位分类组合（默认配置下模型类型见 `protocol`）。  
- **前缀 ML 回归**：`PREFIX_ML_MODELS` × 每个 ML 信号组的 `prefix_ml_detect_loso`（Ridge/RF 等标签与 `phasecls` 不同，族名 `ml-*`）。

### 3.4 可选研究分支（默认 `run.py` 不带参不执行）

通过 `run.py` 或 `main(...)` 标志可开启（实现于 `experimental/ablation_phase.py`）：

- Phase policy **网格搜索**（`search_phase_policy`）  
- Trigger **族对比**（`compare_trigger_families`）  
- Stateful localizer **消融**（`compare_stateful_localizer`）  
- Localizer refinement **消融**（`--localizer-refinement` / `compare_localizer_refinement`）  

上述分支**不计入**下列默认 fast 表，除非单次运行显式打开。

---

## 4. 实测结果：Fast 模式（本次运行）

### 4.1 D. PREFIX BENCHMARK SUMMARY（相对 Calendar 排名）

```
  Rank Method                       Group         AllMAE  PostMAE  PostΔCal  AllΔCal  FirstDet   OvMAE   Avail  TimeSec
  ----------------------------------------------------------------------------------------------------------------------------
  1    PhaseCls-ENS-Temp+HR[Champion] Temp+HR         3.96     2.92     -1.47    -0.64     23.74    3.75  25.6%     0.02
  2    PhaseCls-Temp+HR[RF-baseline] Temp+HR         3.96     2.93     -1.45    -0.64     23.96    3.58  24.7%     0.01
  3    PhaseCls-TempOnly            TempOnly        4.05     3.12     -1.27    -0.55     23.74    3.54  25.6%     0.01
  4    PhaseCls-HROnly              HROnly          4.04     3.13     -1.26    -0.56     23.91    3.66  24.8%     0.02
  5    PhaseCls-Temp+HR[EvidenceSticky] Temp+HR         4.35     3.79     -0.60    -0.25     24.72    3.68  20.5%     0.01
  6    Rule-TempOnly-ftt_prefix     TempOnly        4.60     4.03     -0.35     0.00     15.80    6.85  41.3%     6.46
  7    Rule-HROnly-ftt_prefix       HROnly          5.04     4.51      0.13     0.43     14.12    8.10  50.9%     7.80
  8    RuleState-Temp+HR            Temp+HR         5.03     4.95      0.56     0.43     18.32    5.49  31.7%     0.00
```

辅助最优（同次运行打印）：

- **Post-trigger MAE 最优**：PhaseCls-Temp+HR[RF-baseline]，MAE=2.93，n=598  
- **Anchor-post 聚合 MAE 最优**：PhaseCls-ENS-Temp+HR[Champion]，MAE=3.06  

**本 run 结论行**：Best valid prefix method: **PhaseCls-ENS-Temp+HR[Champion]**；Calendar PostOvDays MAE=4.39 | Oracle PostOvDays MAE=1.49 | Best PostOvDays MAE=2.92。

### 4.2 E. OPERATIONAL REPORTING（节选：All labeled）

```
  Method                        PostOv    ±2d    ±3d  PostTrig    ±2d    ±3d AnchorPost   Avail  FirstDet   Ov1st    Time
  Oracle-prefix                   1.49  77.7%  95.3%      1.49  77.7%  95.3%       1.44   42.7%     18.53    0.00    0.00
  Calendar                        4.39  37.4%  46.1%         -      -      -       4.34    0.0%         -       -    0.00
  PhaseCls-ENS-Temp+HR[Champion]    2.92  49.7%  62.8%      3.02  51.1%  64.8%       3.06   25.6%     23.74    3.75    0.02
  PhaseCls-Temp+HR[RF-baseline]    2.93  51.2%  63.8%      2.93  53.8%  67.4%       3.32   24.7%     23.96    3.58    0.01
  PhaseCls-TempOnly               3.12  50.2%  61.9%      3.32  50.6%  62.4%       3.54   25.6%     23.74    3.54    0.01
  PhaseCls-HROnly                 3.13  47.0%  59.8%      3.35  46.9%  60.3%       3.44   24.8%     23.91    3.66    0.02
  PhaseCls-Temp+HR[EvidenceSticky]    3.79  39.7%  51.0%      4.39  37.0%  46.1%       3.57   20.5%     24.72    3.68    0.01
  Rule-TempOnly-ftt_prefix        4.03  40.8%  54.9%      5.26  30.3%  42.2%       3.97   41.3%     15.80    6.85    6.46
  Rule-HROnly-ftt_prefix          4.51  36.7%  46.0%      5.44  30.6%  39.2%       4.46   50.9%     14.12    8.10    7.80
  RuleState-Temp+HR               4.95  30.6%  42.6%      5.90  25.7%  36.0%       5.42   31.7%     18.32    5.49    0.00
```

（Quality 子 cohort 的完整表见同次终端输出或 `benchmark_log.txt`。）

### 4.3 F. DETECTED-CYCLE（节选：All labeled）

```
  Method                       DetectRate   n_det  Latency   Ov1st    ±2d    ±3d  PostTrig    ±2d    ±3d AnchorPost
  Oracle-prefix                    100.0%      79     0.00    0.00 100.0% 100.0%      1.49  77.7%  95.3%       1.44
  Calendar                           0.0%       0        -       -      -      -         -      -      -          -
  PhaseCls-ENS-Temp+HR[Champion]      97.5%      77     5.04    3.75  37.7%  58.4%      3.02  51.1%  64.8%       2.99
  PhaseCls-Temp+HR[RF-baseline]      96.2%      76     5.25    3.58  44.7%  56.6%      2.93  53.8%  67.4%       3.24
  PhaseCls-TempOnly                 96.2%      76     4.95    3.54  42.1%  60.5%      3.32  50.6%  62.4%       3.52
  PhaseCls-HROnly                   93.7%      74     5.03    3.66  51.4%  58.1%      3.35  46.9%  60.3%       3.32
  PhaseCls-Temp+HR[EvidenceSticky]      87.3%      69     6.06    3.68  46.4%  59.4%      4.39  37.0%  46.1%       3.54
  Rule-TempOnly-ftt_prefix         100.0%      79    -2.73    6.85  26.6%  30.4%      5.26  30.3%  42.2%       3.97
  Rule-HROnly-ftt_prefix            98.7%      78    -4.47    8.10  11.5%  17.9%      5.44  30.6%  39.2%       4.45
  RuleState-Temp+HR                 93.7%      74    -0.23    5.49  27.0%  40.5%      5.90  25.7%  36.0%       5.50
```

---

## 5. 如何更新本文档中的数字

1. 在 `multisignal_pipeline` 目录下执行：`python run.py`。跑 full 池时：`python -c "from run import main; main(mode='full')"`（耗时明显更长）。  
2. 将终端完整输出追加保存，并把上表替换为新表中的 **D / E / F** 与「Best valid」行。  
3. 更新第 1 节元数据（时间、commit、耗时）。  

若合并了协议或候选列表变更，请同步修改 **第 3 节** 方法表（以 `benchmark_main._fast_candidate_specs` / `_full_candidate_specs` 为准）。

---

## 6. 相关源文件索引

| 模块 | 路径 |
|------|------|
| 入口 | `run.py` |
| 主基准 | `benchmark_main.py` |
| 报告与排序 | `report_utils.py` |
| 消融 / 搜索 | `experimental/ablation_phase.py` |
| ML / Phase / Localizer | `detectors_ml.py`, `core/localizer.py`, `core/stabilization.py` |
| 规则检测 | `detectors_rule.py` |
| 数据加载 | `data.py` |
| 协议常量 | `protocol.py`（re-export `config/protocol_*.py`） |
