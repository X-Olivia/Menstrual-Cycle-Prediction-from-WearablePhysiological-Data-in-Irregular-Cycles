# Personalization 与 Irregular：代码实现与运行结果说明

本文档整理 `new_workspace/record/experiment/personalization_stratified/` 目录下与「不规则分层」「个性化（B1/B2/M3）」相关的代码、已实现内容与典型运行结果。  
**不修改**原 `multisignal_*` 主实验脚本；本目录为独立评估管线。

---

## 1. 目录定位

| 项目 | 说明 |
|------|------|
| 目的 | 在 **LLCO + k-shot** 协议下比较 B1/B2/M3，并对 **irregular / 二维分层** 做 gain 与 pre/post 拆分分析 |
| 数据入口 | `multisignal_data.load_all_signals()` → `cycle_cleaned_ov` 衍生周期、`LH` 排卵标签、`subj_order` |
| 主命令 | `python run_main.py --cv-threshold 0.15 --shots 0,1,2,3 --detector oracle` |
| 输出目录 | `outputs/` |

---

## 2. 相关文件一览

| 文件 | 作用 |
|------|------|
| `data_prep.py` | 加载数据；为每个 `id` 计算周期统计与 **irregular 相关元信息**（见下节） |
| `cld_metrics.py` | 相邻周期长度差的 **CLD**（`|L_{t+1}-L_t|`）的中位数/均值；文献式严格高波动 `median_abs_cld > 9` |
| `models_cyclelevel.py` | **B1/B2/M3** 的 `menses_start` 预测与 **M3 的 B1 历史残差 bias** |
| `eval_llco_kshot.py` | **Oracle 排卵**（`det = LH ov_dic`）+ LLCO + k-shot + 锚点误差；写出 long/agg CSV |
| `analysis_interaction.py` | 读 `llco_kshot_long.csv`，按 **pre/post × 二维 CV 分层** 汇总 **gain** |
| `run_main.py` | 依次调用 `eval_llco_kshot.py` 与 `analysis_interaction.py` |
| `full_chain_llco_kshot.py` | **全链路**：LOSO **非 oracle** 排卵（`ml_detect_loso`）+ 同一套 B1/B2/M3；独立输出 `full_chain_llco_kshot_*.csv` |
| `README.md` | 简短使用说明（可能与本文档并存；以本文档为「实现与结果」主文档） |

---

## 3. 「不规则」在代码里实现了什么

### 3.1 一维标签（兼容早期 H1/H2 思路）

- **`is_irregular`（CV 阈值）**  
  - 每人用其 **全部已完成周期** 的长度序列算 `CV = std/mean`。  
  - `CV > --cv-threshold`（默认 **0.15**）→ `is_irregular=1`。  
  - 定义见 `data_prep.py` 中 `build_subject_meta`。

- **`is_irregular_cld_strict`（文献式 CLD）**  
  - `median_abs_cld > 9`（天）→ 标记为严格高波动，见 `cld_metrics.is_high_vol_cld_strict`。

### 3.2 二维分层（主分析用）

在 **`eval_llco_kshot.py`** 内，对「当前可 LLCO 评估」的用户子集计算：

- **偏移**：`mean_shift_abs = |mean(cycle_len) - 28|`
- **波动（CV 轴）**：`cycle_cv`
- **阈值**：全体可评估用户的 `mean_shift_abs` **中位数**、`cycle_cv` **中位数** 作为切分（非固定常数）。

由此得到 **`irregular_2d_stratum`**（四象限）：

| 编码 | 含义 |
|------|------|
| `low_shift_low_vol` | 低偏移 & 低波动 |
| `high_shift_low_vol` | 高偏移 & 低波动 |
| `low_shift_high_vol` | 低偏移 & 高波动 |
| `high_shift_high_vol` | 高偏移 & 高波动 |

**第二套（CLD 轴）**：用 `median_abs_cld` 与中位数阈值切分，得到 **`irregular_2d_stratum_cld`**（`low_shift_low_cld` 等），与 CV 轴并行写在 long 表中，便于后续扩展分析。

### 3.3 Long 表中的关键列（与 irregular / 分层相关）

除 `id, test_sgk, shot_k, model, anchor_offset` 外，典型包括：

- `cycle_mean`, `cycle_cv`, `median_abs_cld`, `mean_abs_cld`, `mean_shift_abs`
- `shift_threshold`, `vol_threshold`, `cld_threshold`
- `is_irregular`, `is_irregular_cld_strict`
- `irregular_2d_stratum`, `irregular_2d_stratum_cld`
- `anchor_group`: `pre`（`ov-7,-3,-1`）/ `post`（`ov+2,+5,+10`）

---

## 4. 「Personalization」在代码里实现了什么

### 4.1 共同设定

- **预测对象**：在锚点日 `anchor_day = ov_true + offset` 上，预测 **剩余天数** `pred_remaining`，与 `true_remaining = cycle_len - anchor_day` 比较；误差 `signed_err = pred_remaining - true_remaining`，`acc_3d`: `|err| < 3.5`。
- **排卵**：默认管线中 **`ov_est = LH`（oracle）**，用于判断是否进入 **countdown**（`anchor_day >= ov_est + 2` 且 `ov_est > 3`）。
- **黄体长度（countdown）**：B1/B2 在 post 段均用 **人群估计** `pop_luteal_len`（由带标签周期上 `cycle_len - ov` 在 8–22 天内样本均值），**不把个人黄体期**作为个性化主信号（第一阶段设计）。

### 4.2 B1 — Population-only

- **不用**该用户历史周期长度。  
- **Pre（未进入 countdown）**：`pred_menses_start = pop_cycle_len`（全数据 `cycle_len` 均值）。  
- **Post（countdown）**：`pred_menses_start = ov_est + pop_luteal_len`。

### 4.3 B2 — Personalized cycle（仅周期长度个性化）

- **Pre**：`acl = weighted_cycle_len_mean(前 k 个历史周期长度)`（指数权重）；无历史时退回 `pop_cycle_len`。  
- **Post**：与 B1 相同，用 `ov_est + pop_luteal_len`（**不在此版用个人黄体期**）。

即：**个性化主要体现在 pre 段的 calendar 分支**；post 段 B2 与 B1 在公式上可相同，故 **post 上 `gain_b2_vs_b1` 常为 0** 属预期。

### 4.4 M3 — Global + user bias

- `pred_remaining_m3 = pred_remaining_b1 - bias_k`。  
- `bias_k`：在该用户 **前 k 个历史周期** 上，对所有可用锚点用 **B1** 算 `pred_remaining - true_remaining` 的 **均值**（`estimate_b1_history_bias`）。

### 4.5 k-shot 与 LLCO

- **LLCO**：每人取 **最后一个带 LH 标签的周期** 作为测试周期；其前若干周期为历史。  
- **k-shot**：`shot_k ∈ {0,1,2,3}` 表示用于 B2/M3 的历史周期条数（`hist_k = history_all[-k:]`）。

---

## 5. 分析脚本 `analysis_interaction.py` 产出什么

- 输入：**`llco_kshot_long.csv`**（须含 `anchor_group`, `irregular_2d_stratum` 等）。  
- 先按 `id × test_sgk × irregular_2d_stratum × shot_k × anchor_group × model` 聚合 **MAE**（对锚点内 `abs_err` 平均），再算：

  - `gain_b2_vs_b1 = MAE(B1) - MAE(B2)`（正表示 B2 更好）  
  - `gain_m3_vs_b1 = MAE(B1) - MAE(M3)`

- 输出：
  - **`interaction_gain_by_subject_prepost.csv`**：每人每分层每 phase 的 gain 长表。  
  - **`gain_2d_strata_prepost_summary.csv`**：在 **`low_shift_low_vol` … `high_shift_high_vol`** 四类上，对 **pre / post** 分别 **平均 gain**（跨 subject），并给出 `n_subjects`, `n_rows`。

**注意**：当前 `summarize_gain_by_stratum_phase` 仅汇总 **CV 轴** 四类名称；若需 CLD 轴汇总表，需在分析脚本中增加对 `irregular_2d_stratum_cld` 的分组（或另写小脚本）。

---

## 6. 全链路脚本 `full_chain_llco_kshot.py`

- 排卵日为 **`ml_detect_loso`**（与 `multisignal` 实验一致），**非 oracle**。  
- 经期预测仍用同一套 B1/B2/M3 + 锚点；k-shot 仍只作用于 **周期历史 / bias**，检测器不做 few-shot。  
- 输出：`outputs/full_chain_llco_kshot_long.csv`、`full_chain_llco_kshot_agg.csv`（具体以脚本为准）。

---

## 7. 运行方式

```bash
cd new_workspace/record/experiment/personalization_stratified

# Oracle 排卵 + 评估 + 二维 pre/post gain 汇总
python run_main.py --cv-threshold 0.15 --shots 0,1,2,3 --detector oracle

# 全链路（LOSO 检测排卵）
python full_chain_llco_kshot.py --model ridge --shots 0,1,2,3
```

---

## 8. 输出文件说明

| 文件 | 来源 | 内容摘要 |
|------|------|----------|
| `llco_kshot_long.csv` | `eval_llco_kshot.py` | 每行：某用户、某 shot、某模型、某锚点的误差与分层字段 |
| `llco_kshot_agg.csv` | 同上 | 按用户×shot×模型聚合后的 MAE 等 |
| `interaction_gain_by_subject_prepost.csv` | `analysis_interaction.py` | 分层 + pre/post 的 gain |
| `gain_2d_strata_prepost_summary.csv` | 同上 | 四类 strata × pre/post × shot 的 gain 汇总 |

若目录中仍存在 **`h1_b1_irregular_summary.csv` / `h2_gain_irregular_summary.csv`**，多为 **旧版一维 irregular 分析** 遗留；当前 **`analysis_interaction.py`** 主路径不再生成这两份，以 **`gain_2d_strata_prepost_summary.csv`** 为准。

---

## 9. 典型运行结果（Oracle 管线，一次完整跑通）

以下摘自一次成功运行日志（`--cv-threshold 0.15 --shots 0,1,2,3 --detector oracle`），数据规模与总体误差为：

- 数据：`Cycles: 111 | Labeled: 79 | Quality: 42`  
- Long 行数：**2616**（与 31 可评估用户 × 多锚点 × 模型 × shot 组合一致的量级）

### 9.1 全体平均 MAE（`abs_err`，按 shot × 模型）

| shot_k | B1 | B2 | M3 |
|--------|-----|-----|-----|
| 0 | 2.408 | 2.408 | 2.408 |
| 1 | 2.408 | 2.939 | 2.607 |
| 2 | 2.408 | 2.798 | 2.584 |
| 3 | 2.408 | 2.817 | 2.560 |

解读要点：

- **shot=0**：无历史，B1/B2/M3 行为对齐，三模型相同。  
- **shot≥1**：B2 总体劣于 B1；M3 介于 B1 与 B2 之间（bias 部分纠偏）。

### 9.2 二维分层 × pre/post 的 gain 摘要（`gain_2d_strata_prepost_summary.csv` 快速视图）

**Pre（`ov-7/-3/-1`）** 在 **shot=1,2,3** 下：

- **`high_shift_low_vol`**：`gain_b2_vs_b1` 多为 **正**（历史周期长度个性化在「高偏移、低波动」人群 pre 段有益）。  
- **`low_shift_high_vol` / `high_shift_high_vol`**：`gain_b2_vs_b1` 多为 **负**（高波动人群用历史均值反而有害）。

**Post（`ov+2/+5/+10`）**：

- **`gain_b2_vs_b1` 常为 0**：因 B1/B2 在 countdown 段使用相同 `ov + pop_luteal`，calendar 个性化不区分。  
- **`gain_m3_vs_b1`** 可非零：来自 **bias 校正**；部分分层（如 `high_shift_high_vol`）可能出现较大负值，需结合样本量解读。

四维分层在该次运行中的可评估用户数约为 **8 / 7 / 9 / 7**（四类合计 31），结论宜作 **探索性**，并配合全链路或非 oracle 实验做稳健性对照。

---

## 10. 小结

| 维度 | 已实现 |
|------|--------|
| Irregular | CV 阈值标签；CLD 与严格阈值；**二维 CV 分层** + **二维 CLD 分层字段** |
| Personalization | B1 无历史；B2 前 k 周期长度加权均值；M3 B1+历史 bias |
| 协议 | LLCO 最后周期为测试；k-shot 0–3；锚点 pre/post 拆分 |
| 全链路 | `full_chain_llco_kshot.py`：LOSO 排卵 + 同上经期模型 |
| 典型结果 | Oracle 下总体 B2 相对 B1 无优势；**二维分层下 pre 段高偏移低波动组可见正 gain，高波动组多为负 gain**；post 段 B2≈B1，差异主要在 M3 |

---

## 11. 文档维护

- 代码变更时：请同步更新 **第 2 节文件表**、**第 3–5 节定义** 与 **第 9 节典型数字**（或注明运行日期与 commit）。  
- 若增加 CLD 轴的汇总表或 H1/H2 正式检验，在本节追加小节即可。
