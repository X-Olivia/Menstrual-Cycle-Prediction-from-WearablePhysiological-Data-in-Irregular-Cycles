# 分层与 Personalization 研究计划 vs 当前实现

**目的**：对照你提出的 plan（假设 H1/H2、irregular 定义、预测任务、三个基线、k-shot、两种 protocol、interaction 分析），逐条分析哪些与当前实现一致、哪些需要调整、如何衔接。

---

## 一、你的 Plan 摘要（按步骤）

| 步骤 | 内容 |
|------|------|
| 核心假设 | H1: 周期越不规律，population model 表现越差；H2: personalization 对不规律人群的提升 > 对规律人群的提升 |
| 第一步 | 把 irregular 定义清楚（可用按人周期长度标准差） |
| 第二步 | 定义预测任务（与现在一致：生理信号→排卵→月经日） |
| 第三步 | 设计模型/基线：Baseline 1 Population-only；Baseline 2 Personalized mean；Model 3 Global + personalization |
| 第四步 | k-shot：0-shot（无该用户历史）、1-shot、2-shot、3-shot |
| 第五步 | Subject-wise protocol：Setting 1 LOSO / strict subject-wise；Setting 2 within-subject chronological |
| 第六步 | 核心分析：**interaction**（personalization gain 在 irregular 上是否显著更大），而非单纯“谁最准” |

---

## 二、与当前实现的对照与需调整点

### 2.1 预测任务（第二步）— 完全一致，无需改

- **当前**：先由生理信号（体温等）推测排卵日，再用「排卵日 + 黄体期」或日历回退预测月经日（等价于预测周期长度）。
- **Plan**：与现在任务一致。  
→ **无需调整**，继续用 `run_final_ov_menses` 这类 pipeline（排卵检测 → `pred_len = det_ov + luteal` 或 `pred_len = hist_cycle_len`）。

---

### 2.2 Irregular 定义（第一步）— 需在代码里显式实现

- **当前**：文档里写了用「按人 cycle_len 的 std 或 CV」，但脚本里**没有**预先算 per-user `std(cycle_len)` / CV 并打 regular/irregular 标签。
- **建议**：
  - 在加载 `cycle_series` / `subj_order` 后，对每个 `id` 用其**已完成周期**的 `cycle_len` 算 `std` 和（可选）`CV = std/mean`。
  - 定义例：**irregular = CV(cycle_len) > 0.15**（或按中位数/分位数二分类）。可先做敏感性分析（如 0.12 / 0.15 / 0.18）。
  - 输出：每个 `id` 的 `cycle_len_std`、`cycle_len_cv`、`is_irregular`，供后续分层和 interaction 用。

---

### 2.3 三个基线/模型（第三步）— 与当前的对应与缺口

| 你的命名 | 含义 | 当前实现对应 | 需补/改 |
|----------|------|----------------|----------|
| **Baseline 1: Population-only** | 完全不用该用户历史 | 固定黄体期 12–14d + 日历用**人群**平均周期长（如 28d） | 当前日历回退用的是 `hist_cycle_len`（该用户过去周期均值），即已用用户历史。要严格 Population-only，需在 0-shot 或「不用该用户历史」时强制用**全局**均值（如 28）和固定黄体期。 |
| **Baseline 2: Personalized mean** | 用该用户过去周期均值（周期长 + 黄体期） | `hist_cycle_len` + **personal luteal**（`past_cycle_len - past_detected_ov`） | 已有；需保证按 k-shot 只使用「前 k 个周期」算 mean，而不是用全部过去周期。 |
| **Model 3: Global + personalization** | 全局模型 + 个体化校准/适配 | 文档里的「Global + user calibration」（bias 校正）；`robust_eval.run_multi_seed_llco_bias` 有 per-subject bias | 排卵→月经 pipeline 当前是规则（det_ov + luteal），不是 LGB 日级预测。若「Global」指同一套排卵检测+规则，则 Model 3 = 该规则 + 用户 bias 校正；需在**同一任务**（预测周期长度/月经日）上实现并和 B1/B2 同口径评估。 |

**具体调整**：

- **Baseline 1 (Population-only)**：  
  - 当「该用户历史周期数 = 0」或显式 0-shot 时：`hist_cycle_len` 用**人群**均值（如 28），黄体期用固定 12 或 13。  
  - 当 k-shot 时：B1 仍**不**用该用户历史，始终用人群均值+固定黄体期（这样 B1 在各 k 下一致，便于和 B2/M3 比较）。
- **Baseline 2 (Personalized mean)**：  
  - 仅用该用户**前 k 个**已完成周期算 `hist_cycle_len` 和 personal luteal（k=0 时退化为人群均值，即与 B1 相同）。
- **Model 3 (Global + personalization)**：  
  - 在现有「排卵检测 + 固定/个人黄体期 + 日历」规则上，加一层 **user-level bias**：在训练/历史周期上算每用户残差均值，预测时 `pred_final = pred_global + bias(uid)`；bias 仅用该用户前 k 个周期估计（k=0 则无 bias）。

这样三个模型在「用不用、用多少该用户历史」上对齐，便于做 k-shot 和 interaction。

---

### 2.4 k-shot 定义（第四步）— 需与 protocol 对齐

- **0-shot**：预测该周期时，**完全不使用**该用户任何历史周期（B1 用人群均值+固定黄体期；B2 在 0-shot 下与 B1 相同；M3 无 bias）。
- **1/2/3-shot**：仅用该用户**前 1 / 2 / 3 个**已完成周期来算 personalized mean 或 bias，预测当前周期。

**与两种 protocol 的对应**：

| Protocol | 谁当测试 | k-shot 含义 | 当前实现 |
|----------|----------|-------------|----------|
| **Setting 1: LOSO** | 每折留 1 个 subject 做测试，其余训练 | 对**该测试用户**：0-shot = 预测其第 1 个周期（无历史）；1-shot = 用其第 1 个周期，预测第 2 个；2-shot = 用前 2 个周期，预测第 3 个；3-shot = 用前 3 个，预测第 4 个。 | 排卵检测已有 LOSO（如 `run_detected_ov_experiment`）；**月经预测**目前是「全量规则 + 按人按周期遍历」，没有「留一 subject 再按该人周期顺序做 0/1/2/3-shot」。需在 LOSO 循环内，对留出用户的周期按时间顺序做 k-shot 预测并记录。 |
| **Setting 2: Within-subject chronological (LLCO)** | 每人用**最后一个周期**做测试，前面做历史 | k-shot = 测试周期前该用户有 **k 个**已完成周期。例如 3 个周期的人，最后一个是 2-shot（前 2 个用于 mean/bias）。0-shot 仅出现在「只有 1 个周期」的用户（用该唯一周期当测试，无历史）。 | 当前 `run_final_ov_menses` 已是按人、按时间顺序遍历，`past_clens`/`past_luts` 即「该用户此前周期」。需加：对每个测试周期记录 `k = len(past_clens)`，并可选地只取前 k 个算 mean（已自然满足），且对 B1 在任意 k 下都强制用人群均值+固定黄体期。 |

**数据约束**：中位数约 3 个周期/人，3-shot 需要「至少 4 个周期」（3 个历史 + 1 个测试），人数会很少。建议：

- 主表报 0/1/2-shot（及 3-shot 若样本量允许）；
- 在文中说明 3-shot 的 n 较小，结果作敏感性分析。

---

### 2.5 Subject-wise protocol（第五步）— 两种 setting 的落地

**Setting 1: LOSO（strict subject-wise）**

- **含义**：训练（排卵模型 + 若有全局模型则全局模型）时**不含**测试 subject；测试时只在该 subject 上评估，且按时间顺序做 0/1/2/3-shot。
- **当前**：排卵检测有 LOSO（如 `run_detected_ov_experiment`）；月经预测在 `run_final_ov_menses` 里是**全量**数据做规则（无训练），没有「留一 subject」的评估循环。
- **需要**：  
  - 外层：对每个 subject 做 LOSO 折（或 42 折每折留 1 人）。  
  - 排卵：用其余 N−1 人训练/规则，在该折的测试 subject 上得到每周期 `det_ov`（若用 ML 检测，就已是 LOSO）。  
  - 月经：在该测试 subject 的周期上，按时间顺序做预测——第 1 个周期 0-shot（B1 用人群+固定，B2/M3 同 B1），第 2 个 1-shot，第 3 个 2-shot，第 4 个 3-shot；记录每条的 MAE、±2d、±3d、是否 irregular、以及 baseline/model 标识。  
  - 汇总时：按 (fold, shot, model) 和 (fold, shot, irregular) 聚合，为 interaction 准备。

**Setting 2: Within-subject chronological (LLCO)**

- **含义**：不按 subject 留出训练集，每人用**最后一个周期**做测试，前面周期仅当该用户的「历史」用于 personalized mean / bias，不做跨人训练集（排卵检测若用规则则仍可全量）。
- **当前**：`run_final_ov_menses` 已是按人、按时间顺序，且每个周期用该人**此前**周期做 `hist_cycle_len` 和 personal luteal，等价于 LLCO。
- **需要**：  
  - 显式把「测试周期」限定为每人**最后一个**周期（与当前一致）。  
  - 对每个测试周期记录 `k = 该用户在此测试周期前的周期数`（0/1/2/3），以及该用户的 irregular 标签。  
  - B1 在 LLCO 下也**不看**该用户历史：预测该用户最后周期时用人群均值+固定黄体期（即 B1 在 Setting 2 下也是 0-shot 逻辑）。  
  - 输出 per-user 或 per-cycle 的 MAE/acc、shot、irregular，供 interaction 用。

建议：**主结果用 Setting 2（LLCO）**（与现有实现最近、每人一条测试、易解释）；**Setting 1（LOSO）** 作为严格 subject-wise 的稳健性检查（需新写一层 LOSO 循环）。

---

### 2.6 核心分析：Interaction（第六步）— 需显式做

- **当前**：只有整体「固定 vs personal」的对比，没有按 regular/irregular 分层，也没有 formal 的 **personalization gain × irregular** 的 interaction。
- **需要**：
  - **因变量**：每（用户或测试周期）的 **personalization gain**，例如  
    `gain_MAE = MAE(B1) - MAE(B2)` 或 `gain_acc = acc(B2) − acc(B1)`（正 = personalization 有利）。  
    若比较 B1 vs M3，同理定义 gain。
  - **自变量**：用户是否 **irregular**（二元，由 CV 或 std 切分），或连续 **CV(cycle_len)**。
  - **分析**：  
    - 简单版：按 regular / irregular 分层，报告两组的 mean(gain) 及检验（如 t-test 或 Mann–Whitney）。  
    - 正式版：线性或混合模型，如  
      `gain ~ irregular + (1|subject)` 或  
      `MAE ~ model * irregular + (1|subject)`，  
      看 **model × irregular** 的交互是否显著（H2：irregular 组 gain 更大）。
  - **H1**：Population model (B1) 的 MAE 或 error 是否在 irregular 组更大，可用 `MAE(B1) ~ irregular` 或分层表验证。

这样论文的主信息是：「personalization gain 在 irregular 用户上是否显著更大」，而不是「personalized 比 population 好 0.4 天」。

---

## 三、与当前代码/脚本的衔接清单

| 项目 | 当前状态 | 建议改动 |
|------|----------|----------|
| Irregular 定义 | 仅文档 | 在 `run_final_ov_menses` 或独立工具函数里：按 `id` 算 `cycle_len` 的 std/CV，写 `is_irregular`，并随 cycle_series/subj_order 一起往下传。 |
| B1 严格 Population-only | 日历用 `hist_cycle_len`（含用户历史） | 当「该用户历史周期数 = 0」或显式 B1 时，`avg_clen = 28`（或人群均值），`luteal = fixed`；B1 永远不用该用户 past_clens/past_luts。 |
| B2 按 k 个周期算 mean | 当前用**全部**过去周期 | 在算 `hist_cycle_len` 和 personal luteal 时，仅用 `past_clens[-k:]` / `past_luts[-k:]`（最近 k 个），k 由当前 shot 决定。 |
| Model 3 (bias) | `robust_eval` 有 LLCO+bias，但是日级 LGB | 在「排卵→月经」同一任务上：用 B1 或 B2 的 `pred` 作为 base，加上用该用户前 k 个周期残差估计的 `bias(uid)`，得到 M3；与 B1/B2 同口径评估（同 cycle、同 shot）。 |
| k-shot 记录 | 未显式 | 每个测试周期记录 `k = len(past_clens)`（或 LOSO 下该用户当前是第几个周期）；输出表含 `shot`, `is_irregular`, `model`, `MAE`, `acc_3d` 等。 |
| LOSO 月经评估 | 无 | 在 `run_detected_ov_experiment` 或新脚本中：对每个留出 subject，按时间顺序做 0/1/2/3-shot 月经预测，只在该 subject 的周期上评估，汇总到同一格式（shot, irregular, model, MAE/acc）。 |
| Interaction 分析 | 无 | 后处理脚本或 notebook：读 per-cycle/per-user 结果，算 gain，做 regular vs irregular 分层 + 回归/混合模型 interaction；输出表格与图（如 gain 的 regular vs irregular 箱线图）。 |

---

## 四、建议的实验与写作顺序

1. **先做 Setting 2（LLCO）**  
   - 在现有 `run_final_ov_menses` 上：加 irregular 定义、B1 严格 0-shot、B2 按 k 取 mean、M3 加 bias（同任务）、记录 shot 与 irregular。  
   - 输出：每测试周期一行的表（id, shot, is_irregular, MAE_B1, MAE_B2, MAE_M3, acc_B1, acc_B2, acc_M3）。

2. **再做 interaction 分析**  
   - 算 gain（如 B2−B1 的 acc 差或 MAE 差），做 H1（B1 的 MAE vs irregular）和 H2（gain vs irregular；分层 + 交互项）。  
   - 主结果写：「personalization gain 在 irregular 用户上是否显著更大」。

3. **可选：Setting 1（LOSO）**  
   - 实现 LOSO 下的月经预测与 k-shot 汇总，重复 interaction；在文中作为「strict subject-wise」的稳健性。

4. **3-shot 样本量**  
   - 先看有多少人满足「≥4 个周期」；若很少，主表用 0/1/2-shot，3-shot 放附录或敏感性。

---

## 五、简短总结

| Plan 步骤 | 结论 | 行动 |
|------------|------|------|
| 假设 H1/H2 | 合理，与 interaction 分析一致 | 保持；分析时显式检验 H1（B1 vs irregular）和 H2（gain vs irregular）。 |
| Irregular 定义 | 用 per-person cycle_len std/CV | 在代码里实现并输出 `is_irregular`。 |
| 预测任务 | 与现有一致 | 不变。 |
| B1 / B2 / M3 | 概念对应清晰，B1 需严格不用用户历史 | B1 强制人群均值+固定黄体期；B2 按 k 用前 k 个周期；M3 在同任务上加 user bias。 |
| k-shot | 0/1/2-shot 主报，3-shot 视样本量 | 每个测试周期记录 k；B2/M3 仅用前 k 个周期做 mean/bias。 |
| Protocol | Setting 2 = 当前 LLCO；Setting 1 = LOSO | 先做 Setting 2；LOSO 需在月经预测上加一层「留一 subject + 按该人周期顺序」评估。 |
| 核心分析 | Interaction：gain 在 irregular 上是否更大 | 算 per-cycle/per-user gain，做分层与 model×irregular 交互；主结论写 interaction，不写「谁最准」。 |

按上述调整后，plan 与当前实现可以完全对齐，且主故事落在「personalization 的异质性效果（在谁身上更有用）」上。
