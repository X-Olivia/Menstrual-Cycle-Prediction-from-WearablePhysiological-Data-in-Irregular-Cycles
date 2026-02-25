# mcPHASES 数据集文档说明

## 数据集概述

**mcPHASES** 是一个综合性的生理健康追踪数据集，专门用于月经健康研究和可穿戴设备数据分析。该数据集整合了来自多种个人健康追踪设备和调查问卷的数据，为研究人员提供了丰富的生理、激素和自我报告的事件及症状数据。

### 基本信息
- **数据集版本**: 1.0.0
- **许可证**: PhysioNet Restricted Health Data License Version 1.5.0
- **数据收集时间**: 
  - 第一阶段：2022年1月-4月
  - 第二阶段：2024年7月-10月
- **总数据量**: 约1.1亿行记录
- **参与者数量**: 42名女性参与者

## 数据结构与组织

### 核心标识符
所有数据表都可以通过以下关键字段进行连接：
- `id`: 唯一参与者标识符
- `day_in_study`: 从每个参与者研究开始的标准化天数索引（从第1天开始）
- `study_interval`: 数据收集阶段标识（2022或2024）
- `is_weekend`: 布尔值，标识记录日期是否为周末

### 时间跨度测量
对于跨时间的测量（如睡眠），表格可能包含以下列：
- `sleep_start_day_in_study` / `sleep_end_day_in_study`
- `start_day_in_study` / `end_day_in_study`

## 数据文件详细说明

### 1. 参与者基本信息

#### subject-info.csv (42行)
包含参与者的人口统计学和背景调查信息：
- `birth_year`: 出生年份
- `gender`: 自我认同的性别
- `ethnicity`: 自我认同的种族/民族
- `education`: 最高教育水平
- `sexually_active`: 是否性活跃（是/否）
- `self_report_menstrual_health_literacy`: 参与者自评的月经健康知识水平
- `age_of_first_menarche`: 初潮年龄

#### height_and_weight.csv (43行)
参与者自报的身高体重数据：
- `height_2022/2024`: 身高（厘米）
- `weight_2022/2024`: 体重（公斤）

### 2. 激素与自我报告数据

#### hormones_and_selfreport.csv (5,660行)
结合了Mira生育设备的激素数据和每日自我报告症状调查：

**月经周期阶段标注**：
- `phase`: 月经周期阶段标签 - **100%覆盖率**（5,658/5,659条记录有标注）
  - **Follicular（卵泡期）**: 1,386条记录（24.5%）
    - 月经结束后到排卵前
    - 特征：LH基础水平（1-5 mIU/mL），E3G逐渐上升
  - **Fertility（排卵期/生育窗口期）**: 1,281条记录（22.6%）
    - 排卵前后的生育窗口期
    - 特征：LH峰值（可达40+ mIU/mL），E3G高水平（200-400 ng/mL）
  - **Luteal（黄体期）**: 1,912条记录（33.8%）
    - 排卵后到月经前
    - 特征：LH回落到基础水平，E3G下降，PDG上升
  - **Menstrual（月经期）**: 1,079条记录（19.1%）
    - 月经出血期间
    - 特征：有月经流量报告（flow_volume），激素水平较低

**阶段标注依据**：
根据README说明，阶段标签是"available or derived"（可用或推导的），结合数据分析，标注依据包括：
1. **LH峰值检测**：LH激增（>10 mIU/mL）标志排卵，用于确定Fertility期
2. **月经流量报告**：自我报告的月经流量用于确定Menstrual期
3. **激素模式**：E3G和LH的综合变化模式
4. **时间推算**：基于典型月经周期的时间规律

**激素指标**（通过尿液检测）：
- `lh`: **黄体生成素**水平（mIU/mL）- 5,339条有效记录
  - 正常范围：1-5 mIU/mL（基础水平），排卵前可达到峰值
  - 用于预测排卵时间
- `estrogen`: **雌酮-3-葡萄糖醛酸苷**（E3G）水平（ng/mL）- 5,338条有效记录
  - 雌激素的代谢产物
  - 反映体内雌激素水平
  - 在卵泡期逐渐上升，排卵前达到峰值
- `pdg`: **孕二醇葡萄糖醛酸苷**（PdG）水平（mcg/mL）- 1,864条有效记录
  - 孕酮的代谢产物
  - 主要在黄体期分泌
  - 用于确认排卵是否发生

**数据特点**：
- 数据来源：Mira Fertility Plus智能生育监测仪
- 检测方法：尿液免疫层析法
- 采样频率：**接近每日测量**
  - LH和E3G：94.3%的天数有数据（5,339/5,659天）
  - 大多数参与者覆盖率>90%，部分达到100%
  - 数据连续性高达99.8%（极少跳过天数）
- PDG数据较少（1,864条）是因为主要在黄体期测量

**自我报告症状**（0-5李克特量表，0="完全没有"，5="非常高"）：
- `flow_volume`: 月经流量
- `flow_color`: 月经颜色分类
- `appetite`: 食欲水平
- `exerciselevel`: 运动水平
- `headaches`: 头痛严重程度
- `cramps`: 痉挛严重程度
- `sorebreasts`: 乳房胀痛程度
- `fatigue`: 疲劳程度
- `sleepissue`: 睡眠问题
- `moodswing`: 情绪波动       
- `stress`: 压力水平
- `foodcravings`: 食物渴望
- `indigestion`: 消化不良
- `bloating`: 腹胀程度

**自我报告症状缺失率**（基于实际数据分析）：
- **总记录数**: 5,659行
- **平均缺失率**: **41.54%**
- **各字段缺失率**:
  - `flow_volume`: 43.65% (2,470/5,659)
  - `flow_color`: 43.56% (2,465/5,659)
  - 其他症状字段（appetite, exerciselevel, headaches, cramps, sorebreasts, fatigue, sleepissue, moodswing, stress, foodcravings, indigestion, bloating）: **约41.14-41.33%** (2,328-2,339/5,659)
- **说明**: 自我报告症状的缺失率较高，可能是因为参与者并非每天都填写症状调查问卷

### 3. 活动与运动数据

#### active_minutes.csv (5,553行)
按强度分类的身体活动持续时间：
- `sedentary`: 久坐运动分钟数
- `lightly`: 轻度运动分钟数
- `moderately`: 中度运动分钟数
- `very`: 高强度运动分钟数

#### active_zone_minutes.csv (154,483行)
不同心率区间的活动时间：
- `timestamp`: 记录时间
- `heart_zone_id`: 心率区间（燃脂、有氧、峰值）
- `total_minutes`: 在相应心率区间的总分钟数

#### exercise.csv (7,283行)
详细的运动会话记录：
- `activityname`: 活动名称（如步行、骑行）
- `activitytypeid`: Fitbit内部活动类型ID
- `averageheartrate`: 平均心率（bpm）
- `calories`: 消耗卡路里
- `duration`: 持续时间（毫秒）
- `steps`: 步数
- `elevationgain`: 海拔增益（米）
- `hasgps`: 是否有GPS数据

#### steps.csv (7,666,950行)
全天步数追踪：
- `timestamp`: 记录时间
- `steps`: 记录的步数

#### distance.csv (7,666,950行)
距离追踪数据：
- `timestamp`: 记录时间
- `distance`: 覆盖距离（米）

#### calories.csv (20,166,976行)
卡路里消耗记录：
- `timestamp`: 记录时间
- `calories`: 消耗的卡路里数

#### altitude.csv (90,879行)
相对海拔变化：
- `timestamp`: 记录时间
- `altitude`: 海拔增益（米，非GPS数据）

### 4. 心率相关数据

#### heart_rate.csv (63,100,277行) - **最大数据文件**
连续心率测量：
- `timestamp`: 记录时间
- `bpm`: 每分钟心跳数
- `confidence`: Fitbit生成的心率读数置信度

**连续心率数据缺失率**（基于采样分析）：
- **总记录数**: 63,100,277行（连续测量，分钟级数据）
- **缺失率**: 基于前10,000行采样分析，`bpm`和`confidence`字段缺失率约为 **0%**
- **说明**: 连续心率数据为高频时间序列，采样显示数据完整性良好

#### resting_heart_rate.csv (13,737行)
每日静息心率：
- `value`: 估计静息心率（bpm）
- `error`: 估计的误差范围

**静息心率数据缺失率**（基于实际数据分析）：
- **总记录数**: 13,737行
- **缺失率**: 所有字段缺失率为 **0%** (无缺失)
- **说明**: 每日静息心率数据完整，无缺失值

#### heart_rate_variability_details.csv (436,263行)
心率变异性详细数据（睡眠期间5分钟间隔）：
- `rmssd`: 连续心跳间隔差值的均方根（时域HRV指标）
- `coverage`: 数据点覆盖率（数据质量指标）
- `low_frequency`: **LF-HRV** - 低频心率变异性，测量长期心率变化，反映交感和副交感神经活动
- `high_frequency`: **HF-HRV** - 高频心率变异性，测量短期心率变化，反映副交感神经活动和呼吸窦性心律不齐

**注意**：HF-HRV数据主要在睡眠期间采集，采样间隔为5分钟，是研究自主神经系统活动的重要指标。

#### time_in_heart_rate_zones.csv (5,554行)
在不同心率区间的时间分布：
- `in_default_zone_3`: 峰值区间时间
- `in_default_zone_2`: 有氧区间时间
- `in_default_zone_1`: 燃脂区间时间
- `below_default_zone_1`: 燃脂区间以下时间

### 5. 睡眠数据

#### sleep.csv (14,765行)
睡眠会话详细记录：
- `duration`: 睡眠总时长（毫秒）
- `minutestofallasleep`: 入睡时间（分钟）
- `minutesasleep`: 实际睡眠时间（分钟）
- `minutesawake`: 醒着的时间（分钟）
- `efficiency`: 睡眠效率（在床时间中睡眠时间的百分比）
- `levels`: 睡眠阶段详情（JSON格式，包含深睡、浅睡、REM等）

**睡眠数据缺失率**（基于实际数据分析）：
- **sleep.csv**: 所有字段缺失率为 **0%** (14,765行记录完整)
- **说明**: 睡眠会话记录数据完整，无缺失值

#### sleep_score.csv (5,308行)
每日睡眠质量评分：
- `overall_score`: 总体睡眠评分（满分100）
- `composition_score`: 睡眠结构评分
- `revitalization_score`: 恢复性评分
- `duration_score`: 睡眠时长评分
- `deep_sleep_in_minutes`: 深睡眠分钟数
- `restlessness`: 基于运动的睡眠不安程度

**睡眠评分数据缺失率**（基于实际数据分析）：
- **总记录数**: 5,308行
- **缺失率**:
  - `overall_score`, `revitalization_score`, `restlessness`: **0%** (无缺失)
  - `composition_score`, `duration_score`: **29.97%** (1,591/5,308)
  - `deep_sleep_in_minutes`: **0.08%** (4/5,308)
- **说明**: 部分睡眠评分字段存在缺失，可能是由于某些睡眠会话无法计算完整的睡眠结构评分

#### respiratory_rate_summary.csv (6,302行)
呼吸频率汇总（每晚睡眠）：
- `full_sleep_breathing_rate`: 整夜平均呼吸频率
- `deep_sleep_breathing_rate`: 深睡眠期间呼吸频率
- `light_sleep_breathing_rate`: 浅睡眠期间呼吸频率
- `rem_sleep_breathing_rate`: REM睡眠期间呼吸频率

### 6. 生理指标

#### glucose.csv (837,131行)
连续血糖监测数据（Dexcom设备）：
- `timestamp`: 记录时间
- `glucose_value`: 血糖浓度（mmol/L）

#### wrist_temperature.csv (6,856,020行)
腕部皮肤温度变化：
- `timestamp`: 记录时间
- `temperature_diff_from_baseline`: 与个人基线的温度差异（摄氏度）

**原始温度数据缺失率**（基于采样分析）：
- **总记录数**: 6,856,020行（时间戳级别，分钟级数据）
- **缺失率**: 基于前10,000行采样分析，`temperature_diff_from_baseline`字段缺失率约为 **0%**
- **说明**: 原始温度数据为高频时间序列，采样显示数据完整性良好

#### computed_temperature.csv (5,575行)
计算的温度读数（主要在睡眠期间）：
- `nightly_temperature`: 计算的夜间平均皮肤温度
- `baseline_relative_sample_sum`: 与基线温度的偏差总和
- `baseline_relative_nightly_standard_deviation`: 夜间温度相对基线的标准差

**计算的温度数据缺失率**（基于实际数据分析）：
- **总记录数**: 5,575行
- **缺失率**:
  - `nightly_temperature`: **0%** (无缺失)
  - `baseline_relative_sample_sum`: **6.80%** (379/5,575)
  - `baseline_relative_nightly_standard_deviation`: **6.80%** (379/5,575)
- **说明**: 夜间平均温度数据完整，但部分记录的基线相对统计量存在缺失

#### estimated_oxygen_variation.csv (3,070,313行)
估计血氧变化（睡眠期间）：
- `infrared_to_red_signal_ratio`: 红外与红光吸收比率

### 7. 健康评估

#### demographic_vo2_max.csv (11,483行)
最大摄氧量估计：
- `demographic_vo2_max`: 基于人口统计学和心率数据的VO2 Max估计
- `demographic_vo2_max_error`: 估计误差
- `filtered_demographic_vo2_max`: 过滤后的VO2 Max值

#### stress_score.csv (7,933行)
压力管理评分：
- `stress_score`: 压力评分
- `sleep_points`: 睡眠组件得分
- `responsiveness_points`: 反应性组件得分
- `exertion_points`: 运动组件得分

## 数据质量与完整性

### 数据规模统计
- **最大文件**: heart_rate.csv (63,100,277行)
- **最小文件**: subject-info.csv (42行)
- **高频数据**: 心率、步数、距离、卡路里（分钟级记录）
- **中频数据**: 血糖（5分钟间隔）、温度变化
- **低频数据**: 睡眠、激素、自我报告（日级记录）

### 数据完整性
- 部分参与者可能缺少某些测量数据
- 身高体重数据存在较多缺失值
- 激素数据仅在特定时期收集
- 所有时间戳均为本地时间

## 研究应用场景

### 1. 月经周期研究
- 激素水平变化分析
- 症状模式识别
- 周期预测模型开发

### 2. 睡眠质量分析
- 睡眠阶段与月经周期关系
- 睡眠质量影响因素
- 个性化睡眠建议

### 3. 生理指标监测
- 心率变异性分析
- 血糖波动模式
- 体温变化趋势

### 4. 行为模式研究
- 运动习惯与月经周期
- 压力水平与生理指标
- 自我报告症状验证

### 5. 可穿戴设备数据挖掘
- 多模态数据融合
- 异常检测算法
- 个性化健康建议

## 数据使用注意事项

### 隐私保护
- 所有数据已去标识化
- 遵循PhysioNet限制性健康数据许可证
- 禁止尝试识别个人身份
- 不得与他人共享数据访问权限

### 技术要求
- 大文件处理能力（心率数据>6300万行）
- JSON数据解析（睡眠详情、运动数据）
- 时间序列数据处理
- 多表关联查询

### 数据质量考虑
- 设备精度限制
- 用户佩戴依从性
- 自我报告主观性
- 缺失数据处理

## 文件完整性验证

数据集包含SHA256校验和文件（SHA256SUMS.txt），可用于验证文件完整性：
```bash
sha256sum -c SHA256SUMS.txt
```