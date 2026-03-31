# Personalization 研究计划 — 整理版与导师沟通要点  
# Personalization Research Plan — Summary and Supervisor Briefing

本文档把「分层与 Personalization」研究计划整理成可向导师汇报的结构，并给出**一句话总结**、**核心故事**和**可能的问答**，便于会议/邮件沟通。

*This document structures the stratification and personalization research plan for supervisor reporting, with a one-sentence summary, main narrative, and Q&A for meetings or email.*

---

## 一、研究计划总览（可直接给导师看）  
## I. Research Plan Overview (ready to share with supervisor)

### 1.1 研究问题 | Research question

在**可穿戴生理信号 → 排卵 → 月经日预测**这条 pipeline 上，我们想回答：

*Within the pipeline **wearable physiological signals → ovulation → menses date prediction**, we ask:*

- **人群模型（不用个人历史）在不规律用户上是否更差？**（H1）  
  *Does the population model (no individual history) perform worse in irregular users? (H1)*
- **个性化（用个人历史）的收益，是否在不规律用户身上更大？**（H2）  
  *Is the benefit of personalization (using individual history) larger in irregular users? (H2)*

即：**Personalization 的异质性效果——对谁更有用？**

*I.e. **Heterogeneous effect of personalization — who benefits more?***

### 1.2 核心假设（两句话）| Core hypotheses (two sentences)

| 假设 Hypothesis | 内容 Content |
|------|------|
| **H1** | 周期越不规律，population model（不用该用户历史）表现越差。 / The more irregular the cycle, the worse the population model (no user history) performs. |
| **H2** | Personalization 对不规律人群的提升 > 对规律人群的提升。 / Personalization gain for irregular users > for regular users. |

### 1.3 技术路线（六步）| Technical roadmap (six steps)

| 步骤 Step | 内容 (中文) | Content (English) |
|------|------|------|
| 1. 定义「不规律」 | 按**每人**周期长度的变异系数 CV = std(cycle_len)/mean(cycle_len)，例如 CV > 0.15 记为 irregular，可做敏感性分析。 | Define irregularity by **per-user** CV of cycle length, CV = std(cycle_len)/mean(cycle_len); e.g. irregular if CV > 0.15; sensitivity analysis on threshold. |
| 2. 预测任务 | 与现有一致：生理信号 → 排卵日 → 月经日（或周期长度）。 | Same as current: physiological signals → ovulation day → menses day (or cycle length). |
| 3. 三个模型/基线 | **B1** 纯人群（固定周期长 28d + 固定黄体期）；**B2** 个性化均值（用该用户前 k 个周期的均值）；**M3** 全局规则 + 用户级 bias 校正。 | **B1** population-only (fixed cycle 28d + fixed luteal); **B2** personalized mean (user’s prior k cycles); **M3** global rule + user-level bias correction. |
| 4. k-shot 设置 | 0-shot（无该用户历史）、1-shot、2-shot、3-shot（仅用前 1/2/3 个周期算 mean 或 bias）。 | 0-shot (no user history), 1/2/3-shot (use only the prior 1/2/3 cycles for mean or bias). |
| 5. 评估协议 | **主结果**：Setting 2 每人最后一个周期做测试、前面当历史（LLCO）；**稳健性**：Setting 1 LOSO（留一 subject，再按该人周期顺序做 0/1/2/3-shot）。 | **Main**: Setting 2 — last cycle per person as test, prior as history (LLCO). **Robustness**: Setting 1 LOSO (leave-one-subject-out, then 0/1/2/3-shot in chronological order). |
| 6. 核心分析 | **Interaction**：检验「personalization gain」在 irregular 组是否显著大于 regular 组（如分层比较 + 回归中的 model×irregular 交互），而不是只报「谁最准」。 | **Interaction**: test whether personalization gain is significantly larger in the irregular than in the regular group (stratified comparison + model×irregular in regression), not just “which model is best”. |

### 1.4 主结论预期（论文可写的一句话）| Expected main conclusion (one sentence for the paper)

> 「Personalization 的收益在周期不规律用户上显著更大」，即回答「个性化在谁身上更有用」。

> *“Personalization gain is significantly larger in users with irregular cycles,” i.e. answering “for whom is personalization more beneficial.”*

---

## 二、如何向导师解释（沟通要点）  
## II. How to explain to your supervisor (briefing points)

### 2.1 一句话总结（电梯演讲）| One-sentence summary (elevator pitch)

**中文：**  
我们做的是「**谁更需要个性化**」：用周期是否规律来分层，比较纯人群模型、个性化均值和全局+校准三种方式，看个性化带来的提升是否在不规律用户上更明显。

**English:**  
We study **who benefits more from personalization**: we stratify by cycle regularity, compare population-only, personalized mean, and global+calibration, and test whether personalization gain is larger in irregular cycles.

### 2.2 为什么要做这个（动机）| Why we do this (motivation)

- 现有结果多是「个性化 vs 不个性化」的整体对比，**没有回答对哪类用户更有用**。  
  *Existing results are mostly overall “personalized vs not”; they do not answer **for which user type** personalization helps more.*
- 临床/产品上更关心：**不规律用户**（更难预测）是否更需要个性化；若 H2 成立，可支持「对 irregular 用户优先做个性化」的策略。  
  *Clinically/product-wise we care whether **irregular users** (harder to predict) need personalization more; if H2 holds, it supports prioritizing personalization for irregular users.*
- 与当前 pipeline 完全兼容：任务不变（排卵→月经），只是在评估时加分群、分模型、分 k-shot，最后做 interaction 分析。  
  *Fully compatible with the current pipeline: same task (ovulation→menses), only adding stratification, model comparison, k-shot, and interaction analysis at evaluation.*

### 2.3 与当前实现的衔接（落地）| Link to current implementation (delivery)

- **预测任务**：不改，继续用现有排卵检测 + 黄体期/日历规则。  
  *Prediction task: unchanged; keep current ovulation detection + luteal/calendar rules.*
- **需补的**：  
  *To add:*  
  - 在代码里显式算每人 `cycle_len` 的 std/CV，打 `regular/irregular`；  
    *Compute per-user `cycle_len` std/CV and label `regular/irregular` in code.*  
  - B1 严格「不用该用户历史」（人群均值 + 固定黄体期）；  
    *B1 strictly uses no user history (population mean + fixed luteal).*  
  - B2/M3 按 k-shot 只用「前 k 个周期」；  
    *B2/M3 use only the prior k cycles per k-shot.*  
  - 输出每测试周期的 shot、is_irregular、各模型 MAE/acc，再做 gain 与 interaction 分析。  
    *Output per test cycle: shot, is_irregular, each model’s MAE/acc; then compute gain and interaction.*
- **建议顺序**：先做 Setting 2（LLCO，与现有实现最近），再做 interaction；LOSO 可作为稳健性。  
  *Suggested order: implement Setting 2 (LLCO, closest to current code) first, then interaction; LOSO as robustness check.*

### 2.4 预期产出 | Expected outputs

- **主表**：按 regular/irregular 分层，0/1/2-shot（及 3-shot 若样本量够）下 B1/B2/M3 的 MAE 或 ±2d/±3d 准确率。  
  *Main table: MAE or ±2d/±3d accuracy for B1/B2/M3 by regular/irregular and 0/1/2-shot (and 3-shot if sample size allows).*
- **核心结果**：Personalization gain（如 B2−B1）在 irregular 组是否显著大于 regular 组（分层 + 交互项）。  
  *Core result: whether personalization gain (e.g. B2−B1) is significantly larger in the irregular than in the regular group (stratification + interaction term).*
- **一句话结论**：Personalization gain 在 irregular 用户上是否显著更大。  
  *One-sentence conclusion: whether personalization gain is significantly larger in irregular users.*

---

## 三、导师可能问的问题（Q&A）  
## III. Questions your supervisor might ask (Q&A)

**Q1：什么叫「不规律」？**  
**Q1: What is “irregular”?**

按**每人**已完成周期的长度算变异系数 CV = std(cycle_len)/mean(cycle_len)，超过某阈值（如 0.15）定义为 irregular；阈值可做敏感性分析（如 0.12/0.15/0.18）。

*We compute the coefficient of variation CV = std(cycle_len)/mean(cycle_len) over each user’s completed cycles; above a threshold (e.g. 0.15) we label irregular. Threshold can be varied in sensitivity analysis (e.g. 0.12 / 0.15 / 0.18).*

**Q2：三个模型具体是什么？**  
**Q2: What exactly are the three models?**

- **B1**：完全不用该用户历史，用人群平均周期长（如 28 天）+ 固定黄体期（如 12–13 天）。  
  *B1: No user history; population mean cycle length (e.g. 28 days) + fixed luteal (e.g. 12–13 days).*
- **B2**：用该用户**前 k 个**周期的平均周期长和平均黄体期。  
  *B2: That user’s mean cycle length and mean luteal from their **prior k** cycles.*
- **M3**：在同一个「排卵→月经」规则上，加每用户的残差校正（bias），bias 只用该用户前 k 个周期估计。  
  *M3: Same ovulation→menses rule plus a per-user residual (bias) correction, with bias estimated from that user’s prior k cycles.*

**Q3：k-shot 是什么意思？**  
**Q3: What does k-shot mean?**

预测当前周期时，只允许用该用户**此前已完成**的 k 个周期（0 = 不用，1 = 用 1 个，2 = 用 2 个，3 = 用 3 个）来算个性化均值或 bias，用来模拟「新用户 vs 有 1/2/3 个历史周期的用户」。

*When predicting the current cycle, we only use that user’s **prior k completed** cycles (0 = none, 1/2/3 = one/two/three) to compute personalized mean or bias, mimicking “new user vs user with 1/2/3 historical cycles”.*

**Q4：为什么主结果用「最后周期当测试」而不是 LOSO？**  
**Q4: Why use “last cycle as test” for main results instead of LOSO?**

LLCO（每人最后一个周期当测试）与现有实现一致、每人一条测试、易解释；LOSO 更严格（训练时完全不含测试用户），计划作为稳健性检查，需要额外实现留一 subject 再按该人周期顺序做 k-shot。

*LLCO (last cycle per person as test) matches current implementation, gives one test point per person, and is easy to explain. LOSO is stricter (training excludes the test user entirely) and is planned as a robustness check, requiring extra implementation (leave one subject out, then k-shot in that subject’s chronological order).*

**Q5：样本量够吗？**  
**Q5: Is the sample size sufficient?**

中位数约 3 个周期/人，3-shot 需要至少 4 个周期，人数会变少；主表先报 0/1/2-shot，3-shot 视样本量放主表或附录/敏感性。

*Median is ~3 cycles per person; 3-shot needs ≥4 cycles so the effective n drops. Report 0/1/2-shot in the main table; put 3-shot in main or appendix/sensitivity depending on sample size.*

**Q6：和现有实验的关系？**  
**Q6: How does this relate to existing experiments?**

任务不变（同一套排卵→月经 pipeline），只是在评估时：  
(1) 显式区分 B1/B2/M3；  
(2) 按 k 限制使用的历史周期数；  
(3) 按 regular/irregular 分层并做 interaction。现有 `run_final_ov_menses` 类脚本是基础，需按文档《Stratification_Plan_vs_Current_Implementation.md》做上述补充。

*Same task (same ovulation→menses pipeline). We only change evaluation: (1) explicitly compare B1/B2/M3; (2) limit history to k cycles; (3) stratify by regular/irregular and run interaction. Current scripts like `run_final_ov_menses` are the base; we add the above as in Stratification_Plan_vs_Current_Implementation.md.*

---

## 四、文档与代码对应  
## IV. Document and code reference

| 内容 Content | 文档 Document | 代码/脚本 Code / script |
|------|------|-----------|
| 计划与实现对照、衔接清单 Plan vs implementation, checklist | [Stratification_Plan_vs_Current_Implementation.md](Stratification_Plan_vs_Current_Implementation.md) | `run_final_ov_menses`, `run_detected_ov_experiment` |
| 本沟通要点 This briefing | 本文档 This document | — |

---

**总结 Summary**

向导师解释时，突出「**谁更需要个性化**」和「**用 interaction 检验 gain 在 irregular 上是否更大**」，并说明与当前任务、代码的衔接和需补的几步即可。

*When explaining to your supervisor, stress “**who benefits more from personalization**” and “**using an interaction to test whether gain is larger in irregular users**”, and briefly state how this connects to the current task and code and what needs to be added.*
