# Multisignal Prefix Benchmark

这份文档只回答 4 个问题：

1. 当前默认 benchmark 在跑什么
2. 目前最好的组合是谁
3. 各类方法分别表现如何
4. 如何复现和更新这份结果

数据来源：`logs/post_reorg_validation.log`

## Current Snapshot

| 项目 | 当前值 |
|------|--------|
| 运行入口 | `python run.py` |
| 模式 | `fast` |
| Rule σ / ML σ | `2.0 / 1.5` |
| 基线 | `Calendar`, `Oracle-prefix` |
| 排名主指标 | `PostOvDays MAE` |
| 当前最佳方法 | `PhaseCls-ENS-Temp+HR[Champion]` |
| Calendar PostOvDays MAE | `3.19` |
| Oracle PostOvDays MAE | `1.65` |
| Best PostOvDays MAE | `2.80` |

## Best Combinations

### Main Answer

| 目标 | 当前最佳方法 | 结果 |
|------|--------------|------|
| 默认总排名最佳 | `PhaseCls-ENS-Temp+HR[Champion]` | `PostOvDays MAE = 2.80` |
| 非个性化 Phase 系列最佳 | `PhaseCls-ENS-Temp+HR[Champion]` | `AllDays = 3.95`, `PostOvDays = 2.80` |
| Post-trigger 最佳 | `PhaseCls-Temp+HR[RF-baseline]` | `PostTrigger MAE = 2.74` |
| Anchor-post 最佳 | `PhaseCls-Temp+HR[Bayesian]` | `AnchorPost = 3.02` |
| 个性化方法最佳 | `PhaseCls-Temp+HR[BayesianPersonalized]` | `PostOvDays = 2.92` |
| 个性化集成最佳 | `PhaseCls-ENS-Temp+HR[Champion-BayesianPersonalized]` | `PostOvDays = 2.92` |

### Practical Reading

- 如果你只想看当前默认 benchmark 的冠军，直接看 `PhaseCls-ENS-Temp+HR[Champion]`。
- 如果你想看“不用 personalization 的最好组合”，也是 `PhaseCls-ENS-Temp+HR[Champion]`。
- 如果你想看“个性化 prior 是否真的赢了”，当前答案是否定的：个性化候选进入了前列，但没有超过当前 champion。

## How To Read The Families

| 家族 | 含义 | 代表方法 |
|------|------|----------|
| `rule-fused-tt` | 无监督规则法，靠前缀 t 检验找变化点 | `Rule-TempOnly-ftt_prefix` |
| `phasecls-rf` | LOSO 监督相位分类 + localizer | `PhaseCls-Temp+HR[RF-baseline]` |
| `phasecls-ens` | 相位概率集成 + localizer refinement | `PhaseCls-ENS-Temp+HR[Champion]` |
| `Bayesian` | 在 localizer 上加入总体先验 | `PhaseCls-Temp+HR[Bayesian]` |
| `BayesianPersonalized` | 在 Bayesian localizer 上加入个体历史 prior | `PhaseCls-Temp+HR[BayesianPersonalized]` |
| `rule-state` | 规则状态机式触发 | `RuleState-Temp+HR` |

## Ranked Summary

排序规则：

1. 更低的 `PostOvDays MAE`
2. 更低的 `AllDays MAE`
3. 更早的 `FirstDet`
4. 更低的 `Ov1st MAE`
5. 更高的 `Avail`

### Full Fast-Mode Ranking

| Rank | Method | Group | AllMAE | PostMAE | FirstDet | Ov1st MAE | Avail | Notes |
|------|--------|-------|--------|---------|----------|-----------|-------|-------|
| 1 | `PhaseCls-ENS-Temp+HR[Champion]` | Temp+HR | 3.95 | 2.80 | 23.71 | 3.68 | 28.5% | 当前总冠军 |
| 2 | `PhaseCls-Temp+HR[RF-baseline]` | Temp+HR | 3.95 | 2.82 | 23.88 | 3.62 | 27.7% | 当前主线对照 |
| 3 | `PhaseCls-Temp+HR[Bayesian]` | Temp+HR | 3.97 | 2.85 | 23.88 | 3.45 | 27.7% | Anchor-post 最佳 |
| 4 | `PhaseCls-HROnly` | HROnly | 3.99 | 2.90 | 23.89 | 3.85 | 27.7% | 纯 HR 监督基线 |
| 5 | `PhaseCls-ENS-Temp+HR[Champion-BayesianPersonalized]` | Temp+HR | 4.00 | 2.92 | 23.71 | 3.19 | 28.5% | 个性化集成最佳 |
| 6 | `PhaseCls-Temp+HR[BayesianPersonalized]` | Temp+HR | 4.00 | 2.92 | 23.88 | 3.16 | 27.7% | 个性化单模最佳 |
| 7 | `PhaseCls-TempOnly` | TempOnly | 4.00 | 2.93 | 23.74 | 3.74 | 28.4% | 最强纯温度监督法 |
| 8 | `PhaseCls-Temp+HR+HRV` | Temp+HR+HRV | 4.00 | 2.94 | 23.99 | 3.68 | 27.4% | 加 HRV 后未超过 champion |
| 9 | `PhaseCls-AllSignals` | AllSignals | 4.01 | 2.98 | 23.97 | 3.73 | 27.5% | 全信号不优于更简组合 |
| 10 | `PhaseCls-Temp+HR[EvidenceSticky]` | Temp+HR | 4.07 | 3.13 | 24.56 | 3.71 | 23.1% | 更保守，召回变低 |
| 11 | `Rule-TempOnly-ftt_prefix` | TempOnly | 4.14 | 3.19 | 15.80 | 6.85 | 45.8% | 检测早，但误差大 |
| 12 | `Rule-HROnly-ftt_prefix` | HROnly | 4.31 | 3.35 | 14.12 | 8.10 | 56.4% | 检测更早，但排卵误差更大 |
| 13 | `RuleState-Temp+HR` | Temp+HR | 4.31 | 3.35 | 18.32 | 5.49 | 35.2% | 当前不如 phase 系列 |

## Operational View

这部分只保留最关键的 6 个代表方法，避免长表看不出结论。

### All Labeled

| Method | PostOv | PostTrig | AnchorPost | Avail | FirstDet | Ov1st |
|--------|--------|----------|------------|-------|----------|-------|
| `Oracle-prefix` | 1.65 | 1.65 | 1.82 | 47.4% | 18.53 | 0.00 |
| `Calendar` | 3.19 | - | 3.65 | 0.0% | - | - |
| `PhaseCls-ENS-Temp+HR[Champion]` | 2.80 | 2.77 | 3.19 | 28.5% | 23.71 | 3.68 |
| `PhaseCls-Temp+HR[RF-baseline]` | 2.82 | 2.74 | 3.32 | 27.7% | 23.88 | 3.62 |
| `PhaseCls-Temp+HR[Bayesian]` | 2.85 | 2.84 | 3.02 | 27.7% | 23.88 | 3.45 |
| `PhaseCls-Temp+HR[BayesianPersonalized]` | 2.92 | 2.94 | 3.10 | 27.7% | 23.88 | 3.16 |

### Quality Group

| Method | PostOv | PostTrig | AnchorPost | Avail | FirstDet | Ov1st |
|--------|--------|----------|------------|-------|----------|-------|
| `Oracle-prefix` | 1.69 | 1.69 | 1.93 | 46.9% | 18.95 | 0.00 |
| `Calendar` | 3.25 | - | 3.77 | 0.0% | - | - |
| `PhaseCls-ENS-Temp+HR[Champion]` | 2.72 | 2.85 | 3.23 | 30.4% | 23.60 | 3.68 |
| `PhaseCls-Temp+HR[RF-baseline]` | 2.74 | 2.79 | 3.33 | 29.4% | 23.78 | 3.62 |
| `PhaseCls-Temp+HR[Bayesian]` | 2.83 | 2.95 | 3.06 | 29.4% | 23.78 | 3.45 |
| `PhaseCls-Temp+HR[BayesianPersonalized]` | 2.88 | 3.02 | 3.14 | 29.4% | 23.78 | 3.16 |

## Detected-Cycle View

这里看“有检测输出的周期”上的表现，更接近 Apple-style reporting。

### All Labeled Detected Cycles

| Method | DetectRate | Latency | Ov1st | PostTrig | AnchorPost |
|--------|------------|---------|-------|----------|------------|
| `PhaseCls-ENS-Temp+HR[Champion]` | 97.5% | 5.01 | 3.68 | 2.77 | 3.10 |
| `PhaseCls-Temp+HR[RF-baseline]` | 96.2% | 5.17 | 3.62 | 2.74 | 3.21 |
| `PhaseCls-Temp+HR[Bayesian]` | 96.2% | 5.17 | 3.45 | 2.84 | 2.90 |
| `PhaseCls-Temp+HR[BayesianPersonalized]` | 96.2% | 5.17 | 3.16 | 2.94 | 2.98 |

## Conclusions

### What is clearly true now

- 当前默认 benchmark 的最佳组合是 `PhaseCls-ENS-Temp+HR[Champion]`。
- `Temp+HR` 依然是最稳的主信号组；继续加更多信号没有带来更好的主排名。
- `Bayesian` 会改善部分后段指标，尤其是 `AnchorPost`，但没有超过 champion 的主排名。
- `BayesianPersonalized` 已经进入前列，但还没有打赢当前非个性化 champion。
- 两个 rule-based 方法虽然更早给出检测，但排卵误差明显更大。

### Short recommendation

- 论文或对外主结果：优先报告 `PhaseCls-ENS-Temp+HR[Champion]`
- 主线对照：保留 `PhaseCls-Temp+HR[RF-baseline]`
- Bayesian comparator：保留 `PhaseCls-Temp+HR[Bayesian]`
- 个性化 comparator：保留 `PhaseCls-Temp+HR[BayesianPersonalized]`

## Reproduce

在目录 `multisignal_pipeline/` 下运行：

```bash
python run.py
```

日志建议保存到：

```bash
python run.py > logs/<your_log_name>.log 2>&1
```

## Update Checklist

每次更新本文档时，只需要同步下面这些内容：

1. `Current Snapshot`
2. `Best Combinations`
3. `Ranked Summary`
4. `Operational View`
5. `Detected-Cycle View`

如果 fast candidate 池发生变化，还要同步更新：

- `README.md`
- `docs/open_source_notes.md`
