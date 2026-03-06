# 组会汇报稿：经期预测系统 — 数据清洗、模型实现与问题发现

# Group Meeting Script: Menstrual Cycle Prediction — Data Cleaning, Implementation & Problems

---

## 一、项目背景 / Project Background

大家好，今天汇报我的毕业项目进展。我的项目目标是基于可穿戴设备数据预测用户下次经期的开始时间。

Hello everyone, today I'd like to present the progress on my final year project. The goal is to predict when a user's next menstruation will start, using wearable device data.

我使用的数据集叫做 mcPHASES，包含 42 名受试者佩戴 Fitbit Sense 手环收集的数据。数据种类非常丰富，包括分钟级的心率、腕温、心率变异性（HRV），以及睡眠质量、自报症状，还有实验室采集的 LH 和雌激素激素数据。

The dataset I'm using is called mcPHASES, which contains data from 42 participants wearing Fitbit Sense. It includes minute-level heart rate, wrist temperature, heart rate variability (HRV), as well as sleep quality, self-reported symptoms, and lab-collected LH and estrogen hormone data.

我的模型做的是一个日级回归任务：给定某一天的特征向量，预测距离下次经期还有多少天。这个预测值叫 `days_until_next_menses`。

My model performs a day-level regression task: given a feature vector on any given day, predict how many days until the next menstruation starts — a value called `days_until_next_menses`.

---

## 二、数据清洗 / Data Cleaning

### 整体流程 / Overall Pipeline

数据清洗是一个四阶段的管线。

Data cleaning consists of a four-stage pipeline.

**第一阶段是周期清洗。** 原始数据有 5659 行来自 42 个人。我需要先定义"周期"——以经期（Menstrual phase）首次出现为起点分割每个月经周期。然后对 LH 和雌激素的缺失值进行周期内插补，如果某个周期连续缺失超过 5 天，就整个丢弃。我还丢弃了不完整的 cycle0、天数少于 6 天的短周期，以及 day_in_study 有断天的周期。最后，我基于 LH surge 检测标注了每天的排卵概率。清洗后剩下 4825 行，共 173 个有效周期。

**Stage 1 is cycle cleaning.** The raw data has 5,659 rows from 42 people. I first need to define "cycles" — splitting at the first occurrence of the Menstrual phase. Then I interpolate missing LH and estrogen values within each cycle; if a cycle has more than 5 consecutive missing days, the entire cycle is discarded. I also remove incomplete cycle0, short cycles under 6 days, and cycles with gaps in day_in_study. Finally, I annotate daily ovulation probability based on LH surge detection. After cleaning, 4,825 rows remain across 173 valid cycles.

**第二阶段是穿戴数据过滤。** 原始穿戴数据量非常大——心率有 6300 万行，腕温有 690 万行。我以清洗后的周期作为锚点，用 inner join 只保留落在有效周期内的数据。过滤后心率剩 5400 万行，腕温剩 600 万行。

**Stage 2 is wearable data filtering.** The raw wearable data is massive — 63 million rows of heart rate, 6.9 million rows of wrist temperature. I use the cleaned cycles as anchors and inner-join to keep only data falling within valid cycles. After filtering, 54 million HR rows and 6 million WT rows remain.

**第三阶段是日级聚合。** 把分钟级数据聚合成每天一行。心率取日均值、标准差、最小值、最大值，过滤掉 bpm 不在 30-220 范围或 confidence 低于 0.5 的异常值。HRV 只保留覆盖率大于 0.6 的记录。腕温过滤掉偏差超过 ±5°C 的值。这一步输出 4825 行、54 列的日级特征表。

**Stage 3 is daily aggregation.** Minute-level data is aggregated to one row per day. Heart rate is summarized as daily mean, std, min, and max, filtering out values outside 30-220 bpm or with confidence below 0.5. HRV keeps only records with coverage above 0.6. Wrist temperature filters out deviations beyond ±5°C. This step outputs 4,825 rows with 54 columns.

**第四阶段是特征管线 v4，也是改进最大的阶段。** 这个阶段包含 10 个处理步骤，最终输出 3276 行、106 维的特征，模型实际使用其中 23 维。

**Stage 4 is the feature pipeline v4, which is where the biggest improvements happened.** It includes 10 processing steps and outputs 3,276 rows with 106 features, of which the model uses 23.

### v4 的四个关键修复 / Four Key Fixes in v4

这里我要重点讲一下从 v3 到 v4 的四个关键修复，因为这些修复带来了巨大的性能提升——MAE 从 4.30 降到了 3.34，降幅 22%；±3 天准确率从 46% 升到了 65%，提升了 19 个百分点。

I want to highlight the four key fixes from v3 to v4, because they brought a massive performance improvement — MAE dropped from 4.30 to 3.34, a 22% reduction; ±3-day accuracy rose from 46% to 65%, an improvement of 19 percentage points.

**第一个修复是 day_in_cycle_frac。** 这是特征重要性排名第二的特征。v3 的实现是 day_in_cycle 除以固定的 28 天。但问题是，如果一个人的周期是 35 天，那她在第 28 天的时候 frac 已经等于 1.0 了，但实际上她的周期才完成了 80%。我把它改成了 day 除以 hist_cycle_len_mean，就是基于这个人历史周期的平均长度来计算。

**The first fix is day_in_cycle_frac.** This is the #2 feature in importance ranking. In v3, it was computed as day_in_cycle divided by a fixed 28 days. The problem is, for someone with a 35-day cycle, the fraction would already reach 1.0 at day 28, even though the cycle is only 80% complete. I changed it to divide by hist_cycle_len_mean — the individual's historical average cycle length.

**第二个修复是静息心率的聚合方式。** 原来用的是 mean，我改成了 median。这和 mcPHASES 官方代码一致。Median 对异常值更鲁棒，比如偶尔记录到的异常高心率不会影响整体统计。

**The second fix is the resting heart rate aggregation.** It was using mean, and I changed it to median, consistent with the mcPHASES official code. Median is more robust to outliers — occasional abnormally high heart rate readings won't skew the statistic.

**第三个修复是移除边界周期。** 每个受试者在每个 study interval 的最后一个周期，可能被研究截止日期截断了。比如一个人的周期原本是 30 天，但研究在第 22 天就结束了，那我们的标签（还有多少天来经期）就是错的。我把这些边界周期全部移除了，减少了 62 个周期、1549 行数据，但换来了更干净的标签。

**The third fix is removing boundary cycles.** The last cycle of each subject in each study interval may have been truncated by the study end date. For example, if someone has a 30-day cycle but the study ended on day 22, our label (days until next menses) would be wrong. I removed all such boundary cycles — losing 62 cycles and 1,549 rows, but gaining much cleaner labels.

**第四个修复是新增夜间温度波动特征。** 数据集中有一个 baseline_relative_nightly_standard_deviation，反映的是每晚体温的波动程度。黄体期由于孕酮水平升高，体温波动会增大，所以这是一个有生理学意义的特征。

**The fourth fix is adding the nightly temperature standard deviation feature.** The dataset contains a baseline_relative_nightly_standard_deviation, which captures how much body temperature fluctuates each night. During the luteal phase, progesterone levels rise and temperature variability increases, so this is a physiologically meaningful feature.

---

## 三、特征工程与归一化 / Feature Engineering & Normalization

归一化策略是一个需要特别注意的地方。我把特征分成了三类，采用不同的处理方式。

The normalization strategy requires careful consideration. I divided features into three categories with different processing approaches.

**A 类是已经基线化的温度特征**，比如 wt_mean 这些来自 temperature_diff_from_baseline 的值。它们已经是相对于个人基线的差值了，所以我只做 per-cycle centering（减去周期前 5 天的均值），不除以标准差，避免双重归一化。

**Category A is already-baselined temperature features**, like wt_mean from temperature_diff_from_baseline. They are already differences from personal baselines, so I only do per-cycle centering (subtracting the first 5 days' mean) without dividing by std, to avoid double normalization.

**B 类是绝对值生理信号**，包括心率、HRV、夜间温度、静息心率、呼吸频率等。这些做 per-cycle-early z-normalization：用周期前 5 天的均值和标准差作为基线来标准化。这种方式的好处是每个周期有自己的基线，而且只用了周期早期的数据，保证因果性。96% 的周期成功使用了这种 per-cycle-early 基线，只有 4% 因为早期数据不足回退到了 per-subject 基线。

**Category B is absolute physiological signals**, including heart rate, HRV, nightly temperature, resting heart rate, and respiratory rate. These undergo per-cycle-early z-normalization: standardized using the mean and std of the first 5 days of each cycle as the baseline. The advantage is that each cycle has its own baseline, and only early-cycle data is used, ensuring causality. 96% of cycles successfully used per-cycle-early baselines; only 4% fell back to per-subject baselines due to insufficient early data.

**C 类是非生理特征**，包括自报症状和周期位置先验，这些不做归一化。

**Category C is non-physiological features**, including self-reported symptoms and cycle position priors, which are not normalized.

最终模型使用 23 维特征。其中 6 维是周期位置先验，比如 day_in_cycle、历史周期均长、预期剩余天数等；14 维是穿戴设备的 z-score，包括 HRV 四项、HR 四项、腕温四项、夜间温度和静息心率；还有 3 维是呼吸频率和温度波动。

The final model uses 23 features. Six are cycle position priors, such as day_in_cycle, historical cycle length mean, and expected remaining days. Fourteen are wearable z-scores covering HRV (4), HR (4), wrist temperature (4), nightly temperature, and resting HR. The remaining three are respiratory rate and temperature variability.

通过消融实验，我还移除了双相转折、变化率、睡眠架构和 PMS 症状这四组特征，因为它们对性能没有正向贡献。

Through ablation experiments, I also removed four feature groups — biphasic shift, deltas, sleep architecture, and PMS symptoms — because they had zero or negative contribution to performance.

---

## 四、模型实现 / Model Implementation

模型使用的是 LightGBM，一种梯度提升决策树。损失函数是 Huber loss，delta 约为 4.0。Huber loss 的特点是误差小的时候用 L2（平方损失，精确拟合），误差大的时候用 L1（绝对损失，抗极端值），这样对长 horizon 的预测不会过度惩罚。

The model uses LightGBM, a gradient boosted decision tree algorithm. The loss function is Huber loss with delta around 4.0. Huber loss uses L2 (squared loss, precise fitting) for small errors and L1 (absolute loss, resistant to outliers) for large errors, avoiding over-penalization of long-horizon predictions.

数据分割采用 subject-level split，15% 的人放入测试集，确保同一个人的数据不会同时出现在训练和测试集中，防止数据泄漏。因为我们只有 40 个人，大约 6 个人进入测试集，随机分到谁对结果影响很大，所以我采用 10-seed multi-seed evaluation，跑 10 次取均值，消除小样本方差。

Data splitting uses subject-level split with 15% of subjects in the test set, ensuring no one's data appears in both training and test sets, preventing data leakage. Since we only have 40 subjects — roughly 6 in each test set — who gets selected significantly affects results. So I use 10-seed multi-seed evaluation, running 10 times and averaging to reduce small-sample variance.

超参数调优使用了 Optuna，跑了 80 个 trials，每个 trial 用 3 个 seeds 评估。

Hyperparameter tuning was done with Optuna, running 80 trials with 3 seeds each.

### 当前性能 / Current Performance

来说一下结果。v3 的基线 MAE 是 4.30 天，±3 天准确率只有 46%。经过 v4 的四项修复，MAE 降到了 3.34 天，±3 天准确率升到了 65%。再加上 Optuna 调参，当前最优性能是 MAE 3.29 天，±3 天准确率 66.5%。

Let me share the results. The v3 baseline MAE was 4.30 days with only 46% ±3-day accuracy. After v4's four fixes, MAE dropped to 3.34 days with 65% ±3-day accuracy. With Optuna tuning, the current best is MAE 3.29 days and 66.5% ±3-day accuracy.

分层来看，模型在 6-10 天 horizon 表现最好（MAE 2.62），1-5 天和 11-15 天都在 3.0 左右，21 天以上最难（MAE 3.97）。

Looking at stratified performance, the model performs best at the 6-10 day horizon (MAE 2.62), around 3.0 for 1-5 and 11-15 days, and worst at 21+ days (MAE 3.97).

---

## 五、发现的问题 / Problems Discovered

接下来是最重要的部分——我在实验过程中发现的几个关键问题。

Now for the most important part — several key problems I discovered during experiments.

### 问题一：可穿戴特征贡献几乎为零 / Problem 1: Wearable Features Contribute Nearly Zero

这是最令人意外的发现。通过消融实验，我发现仅使用 6 个周期先验特征（day_in_cycle、历史周期均长等），就已经达到了 MAE 3.27、±3d 66.9% 的性能。而加上全部 17 个可穿戴特征后，MAE 反而略微升高到 3.29，±3d 降到 66.5%。

This is the most surprising finding. Through ablation experiments, I found that using only 6 cycle prior features (day_in_cycle, historical cycle length, etc.) already achieves MAE 3.27 and 66.9% ±3-day accuracy. Adding all 17 wearable features actually slightly worsens MAE to 3.29 and ±3d to 66.5%.

换句话说，心率、HRV、体温这些可穿戴信号对最终的经期预测几乎没有贡献。模型的预测力几乎完全来自 "历史平均周期长度减去当前在周期中的天数" 这个简单的日历法逻辑，LightGBM 只是在用更灵活的方式学这个关系。

In other words, heart rate, HRV, and temperature — all these wearable signals — contribute almost nothing to the final menstruation prediction. The model's predictive power comes almost entirely from "historical average cycle length minus current day in cycle" — a simple calendar method logic, just learned in a more flexible way by LightGBM.

### 问题二：可穿戴信号的信噪比太低 / Problem 2: Wearable Signal-to-Noise Ratio Too Low

为了理解为什么可穿戴特征没有帮助，我做了信号质量分析。文献中最有价值的方法是通过温度检测排卵，然后用黄体期长度做倒计时预测。所以我分析了排卵前后各种信号的变化。

To understand why wearable features don't help, I conducted a signal quality analysis. The most promising approach in the literature is detecting ovulation through temperature, then using luteal phase length for countdown prediction. So I analyzed signal changes before and after ovulation.

结果发现，腕温在排卵前后的平均偏移只有 0.21°C，但每天的噪声标准差高达 0.64°C，信噪比 SNR 只有 0.32。心率的情况类似，排卵后平均上升 0.75 个 z-score 单位，但噪声是 2.41，SNR 也只有 0.31。HRV 更差，SNR 是负的 0.19。所有信号的 SNR 都远低于可靠检测所需的约 1.0。

The results show that wrist temperature shifts only 0.21°C on average around ovulation, but daily noise has a standard deviation of 0.64°C, giving an SNR of just 0.32. Heart rate is similar — it rises 0.75 z-score units post-ovulation, but noise is 2.41, yielding an SNR of only 0.31. HRV is even worse with an SNR of negative 0.19. All signals have SNR far below the approximately 1.0 needed for reliable detection.

### 问题三：排卵检测算法全部失败 / Problem 3: All Ovulation Detection Algorithms Failed

我一共测试了 8 种排卵检测算法。

I tested a total of 8 ovulation detection algorithms.

最经典的是 3-over-6 coverline rule，就是看最近 3 天的均温是否比前 6 天高出一个阈值。我还试了 EMA 平滑后再做 coverline、CUSUM 变点检测、滑动均值比较、双相阶跃函数拟合、以及带稳定性约束的变点检测。此外，我还训练了一个 LightGBM 二分类器来判断每天是否处于黄体期——它在测试集上达到了 AUC 0.94、准确率 82%。最后，我还试了把原始温度特征（不做 z-norm）直接加入回归模型，以及用分类器概率做软融合。

The most classic is the 3-over-6 coverline rule — checking if the last 3 days' mean temperature exceeds the previous 6 days by a threshold. I also tried EMA-smoothed coverline, CUSUM change-point detection, running mean comparison, biphasic step-function fitting, and change-point detection with stability constraints. Additionally, I trained a LightGBM binary classifier to determine whether each day is in the luteal phase — it achieved AUC 0.94 and 82% accuracy on the test set. Finally, I tried adding raw temperature features (without z-normalization) directly to the regression model, and soft blending with the classifier probability.

结论是：没有任何一种方法能超越纯 LightGBM 基线。排卵检测的最好 ±3 天准确率只有 46%，而一旦检测不准确，混合模型的预测反而更差。

The conclusion: none of these approaches outperformed the pure LightGBM baseline. The best ovulation detection ±3-day accuracy was only 46%, and once detection is inaccurate, the hybrid model actually performs worse.

### 问题四：Oracle 上限证明方向正确，但数据不支持 / Problem 4: Oracle Shows the Direction is Right, but Data Can't Support It

不过我做了一个非常重要的 Oracle 实验。假设排卵检测完全准确（使用 LH 标注作为真值），然后对排卵后的日子使用"个人黄体期长度减去排卵后天数"做倒计时预测，排卵前继续用 LightGBM。

However, I conducted a very important Oracle experiment. Assuming perfect ovulation detection (using LH annotations as ground truth), I used "personal luteal phase length minus days since ovulation" as a countdown prediction for post-ovulation days, while continuing to use LightGBM for pre-ovulation days.

结果非常惊人：排卵后倒计时的 MAE 只有 1.14 天，±3 天准确率高达 93.9%。混合模型整体 MAE 从 3.55 降到了 3.02，±3 天准确率从 60.6% 升到了 68.7%，提升了 8 个百分点。

The results are striking: the post-ovulation countdown achieves MAE of only 1.14 days with 93.9% ±3-day accuracy. The hybrid model's overall MAE drops from 3.55 to 3.02, and ±3-day accuracy rises from 60.6% to 68.7% — an 8 percentage point improvement.

这证明了"排卵检测加黄体期倒计时"这个方向是完全正确的。瓶颈不在方法，而在于 Fitbit 腕温的 SNR 只有 0.32，不足以支撑可靠的排卵检测。

This proves that the "ovulation detection plus luteal phase countdown" direction is entirely correct. The bottleneck is not the method, but that Fitbit wrist temperature's SNR of 0.32 is insufficient to support reliable ovulation detection.

### 问题五：v5 到 v7 的实验均失败 / Problem 5: v5 Through v7 Experiments All Failed

除了排卵检测，我还尝试了多种特征扩展方案。v5 加入了上一个周期的长度和偏差，但因为和 hist_cycle_len_mean 信息冗余，没有提升。v6 加入了 3 个相位估计特征和 11 个转变检测特征，共 37 维，因为特征过多反而过拟合了。v7 做得最大，加入了 20 个从分钟级数据提取的子日特征（如夜间心率谷值、温度稳态平台等）和 5 个黄体期个人化特征，共 48 维，结果严重过拟合，性能反而变差了。

Beyond ovulation detection, I also tried multiple feature expansion approaches. V5 added previous cycle length and deviation, but showed no improvement due to redundancy with hist_cycle_len_mean. V6 added 3 phase estimation features and 11 transition detection features for 37 total, but overfitted due to too many features. V7 was the largest attempt, adding 20 sub-daily features extracted from minute-level data (like nocturnal HR nadir, temperature plateau, etc.) and 5 luteal personalization features for 48 total — it severely overfitted and performance actually worsened.

v7 还有一个特别值得注意的教训：它的精简版看似 MAE 降到了 3.25，但后来我发现存在数据泄漏。一个特征用了测试周期的数据来计算均值，另一个特征用了周期内的"未来"温度信息来做回溯标注。所以那个改善是不可信的。

V7 also has a particularly notable lesson: its slim version seemed to reduce MAE to 3.25, but I later discovered data leakage. One feature used test cycle data to compute averages, and another used "future" within-cycle temperature data for retrospective annotation. So that improvement was unreliable.

---

## 六、与文献的差距及根因 / Performance Gap with Literature & Root Causes

跟文献对比，我们的差距是明显的。Apple 在 2025 年的论文用 Apple Watch 数据做到了 MAE 1.65 天、±3 天准确率 89.4%。Li 等人 2022 年用贝叶斯方法在 5000 人数据上做到了 MAE 约 2.0 天。而我们是 MAE 3.29 天、±3 天准确率 66.5%。

Compared with the literature, our gap is significant. Apple's 2025 paper achieved MAE 1.65 days and 89.4% ±3-day accuracy using Apple Watch data. Li et al. 2022 achieved MAE around 2.0 days using Bayesian methods on 5,000 users. We are at MAE 3.29 days and 66.5% ±3-day accuracy.

但经过深入分析，我认为差距主要来自四个方面，而不是算法本身。

But after thorough analysis, I believe the gap comes mainly from four aspects, not the algorithm itself.

**第一是传感器精度。** Apple Watch 的温度传感器精度远高于 Fitbit 的腕温传感器。我们测到的温度 SNR 只有 0.32，Apple 那边很可能超过 1.0。

**First is sensor precision.** Apple Watch's temperature sensor has much higher precision than Fitbit's wrist temperature sensor. Our measured temperature SNR is only 0.32; Apple's is likely above 1.0.

**第二是样本量。** Apple 的研究有 260 人，Li 的研究有 5000 人，而我们只有 40 人，比 Apple 少了 6.5 倍。样本量小意味着很难做有效的个人化建模。

**Second is sample size.** Apple's study has 260 participants, Li's has 5,000, while we only have 40 — 6.5 times fewer than Apple. Small sample size means effective personalization modeling is very difficult.

**第三是预测范围不同。** Apple 论文的算法只在排卵检测成功之后才做预测，跳过了约 20% 检测失败的周期。而我们是全周期每天都预测，包括排卵前那些天然不确定性更高的日子。当我把我们的模型限制在黄体期（类似 Apple 的预测范围），MAE 从 3.29 降到了 2.87，差距缩小了，但仍然显著。

**Third is prediction scope.** Apple's algorithm only predicts after ovulation is successfully detected, skipping about 20% of cycles where detection fails. We predict every day of every cycle, including pre-ovulation days with inherently higher uncertainty. When I restrict our model to the luteal phase (similar to Apple's prediction scope), MAE drops from 3.29 to 2.87 — the gap narrows but remains significant.

**第四是每个用户的数据量。** 我们的受试者中位数只有 3 个完整周期，这对于学习个人模式来说太少了。

**Fourth is per-user data volume.** Our subjects have a median of only 3 complete cycles, which is too few for learning individual patterns.

---

## 七、总结与核心洞察 / Summary & Key Insights

总结一下，我想分享三个核心洞察。

To summarize, I'd like to share three core insights.

**第一，数据质量远比算法复杂度重要。** v4 的 19 个百分点的提升几乎全部来自数据修复——修正 frac 计算、改用 median 聚合、移除不完整周期、新增温度波动特征。这些都是很"朴素"的修复，但效果远超后来尝试的所有复杂算法。

**First, data quality matters far more than algorithmic complexity.** V4's 19 percentage point improvement came almost entirely from data fixes — correcting the frac calculation, switching to median aggregation, removing incomplete cycles, and adding temperature variability. These are all "humble" fixes, but their effect far exceeded all the complex algorithms I tried later.

**第二，可穿戴信号在当前数据条件下贡献有限。** 这不是说可穿戴信号没有生理学意义——排卵后确实有温度上升和心率变化——而是 Fitbit 腕温的信噪比（0.32）太低，加上样本量太小（40 人），使得这些信号无法被可靠地利用。模型的预测力几乎全部来自周期先验。

**Second, wearable signals contribute minimally under current data conditions.** This doesn't mean wearable signals lack physiological significance — there is indeed a post-ovulation temperature rise and heart rate change. But Fitbit wrist temperature's SNR (0.32) is too low, combined with our small sample size (40 subjects), making these signals unreliable to utilize. The model's prediction power comes almost entirely from cycle priors.

**第三，排卵检测加倒计时是正确的方向，但需要更好的数据。** Oracle 实验证明，完美排卵检测能把 ±3 天准确率从 61% 提到 69%。这恰好是文献中高性能方法的核心策略。但实现它需要更高精度的传感器或更大的样本量来学习个人模式。

**Third, ovulation detection plus countdown is the right direction, but needs better data.** The Oracle experiment proves that perfect ovulation detection can raise ±3-day accuracy from 61% to 69%. This is precisely the core strategy of high-performing methods in the literature. But achieving it requires higher-precision sensors or larger sample sizes to learn individual patterns.

我目前的最优配置是 LightGBM v4 加 Optuna 调参，MAE 3.29 天，±3 天准确率 66.5%。

My current best configuration is LightGBM v4 with Optuna tuning, achieving MAE 3.29 days and 66.5% ±3-day accuracy.

以上就是我的汇报，谢谢大家。

That concludes my presentation, thank you.

---

*2026-03-03*
