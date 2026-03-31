# personalization_stratified

独立于现有稳定实验脚本的个性化分层评估目录（不修改原 `multisignal_*` 主流程）。


B1 Population-only
完全不用个人历史
acl = global_cycle_mean（如全人群均值或固定 28）

B2 Personalized-cycle
只用个人历史周期长度（前 k 个）
acl = personal_hist_cycle_len_k

M3 Global+bias
先用 B1 产生 baseline 预测，再用该用户前 k 个周期残差做 user-bias 校准
不依赖个人黄体期

## 目标

- 主结果：`LLCO + k-shot(0/1/2/3)` 下比较 `B1/B2/M3`
- 分层：按受试者 `cycle_len` 的 `CV` 标注 `is_irregular`；并补充 **median |CLD|**（相邻周期长度差）与文献式 `median_abs_cld > 9` 的 `is_irregular_cld_strict`
- 分析：H1（B1 在 irregular 更差）与 H2（个性化 gain 在 irregular 更大）

## 当前定义（第一阶段）

- 历史个人信息：仅历史周期长度（不用个人黄体期）
- 排卵检测输入：先用 `oracle`（`det = LH ov_dic`）固定，先隔离经期个性化收益
- irregular 分层（两套并行）：
  - **CV 轴**：`mean_shift_abs` × `cycle_cv`（分位阈值）→ `irregular_2d_stratum`
  - **CLD 轴**：`mean_shift_abs` × `median_abs_cld` → `irregular_2d_stratum_cld`

## 文件

- `data_prep.py`: 加载 `load_all_signals()` 并构建 irregular 元信息
- `cld_metrics.py`: CLD（|Δ周期长度|）与文献阈值辅助函数
- `full_chain_llco_kshot.py`: **全链路** LLCO+k-shot — LOSO ML 排卵检测 + 同一套 B1/B2/M3（排卵非 oracle；k-shot 仍作用于周期历史/偏置，见脚本头注释）
- `models_cyclelevel.py`: B1/B2/M3 的周期级预测定义
- `eval_llco_kshot.py`: 生成 `llco_kshot_long.csv` 和 `llco_kshot_agg.csv`
- `analysis_interaction.py`: 生成 gain 与 H1/H2 汇总
- `analysis_interaction.py`: 生成按 `pre/post`（pre=`ov-7/-3/-1`, post=`ov+2/+5/+10`）拆分的二维分层 gain 汇总
- `run_main.py`: 一键运行入口

## 运行

在 `new_workspace/record/experiment/personalization_stratified` 下：

```bash
python run_main.py --cv-threshold 0.15 --shots 0,1,2,3 --detector oracle
# 或：排卵用 LOSO 多信号回归（与 multisignal 实验一致），输出见 outputs/full_chain_llco_kshot_*.csv
python full_chain_llco_kshot.py --model ridge --shots 0,1,2,3
```

输出目录：`outputs/`

- `llco_kshot_long.csv`
- `llco_kshot_agg.csv`
- `interaction_gain_by_subject.csv`
- `interaction_gain_by_subject_prepost.csv`
- `gain_2d_strata_prepost_summary.csv`

## 后续扩展

- 将 `detector` 从 `oracle` 扩展到指定 `multisignal` 检测器输出
- 增加 Setting 1（LOSO）稳健性
- 增加 signal-level personalization（第二阶段）
