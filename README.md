# TrajectoryFeatureGeneration
generate feature of trajectory.

## 时间间隔分布统计特性

### geolife

[Gap sign] <=0 ratio: 0.1149  (zero: 0.1149, neg: 0.0000)
[Stats] #gaps = 27718
[Stats] raw gaps minutes: mean=444.077, median=16.000, p95=99.752, p99=13685.512
[Stats] log gaps: mean=0.944, std=5.525

[MLE Parameters]
Normal on log-gaps:
  nu(df)=0.990804, loc=2.979157, scale=0.903595
  shape(s)=1.590147, loc=0.000000, scale=17.469668

### moreuser

[Gap sign] <=0 ratio: 0.3180  (zero: 0.3180, neg: 0.0000)
[Stats] #gaps = 6801712
[Stats] raw gaps minutes: mean=22.208, median=4.617, p95=97.417, p99=201.483
[Stats] log gaps: mean=-2.879, std=7.691

[MLE Parameters]
Normal on log-gaps:
  mu=-2.878650, sigma=7.691448
Student-t on log-gaps:
  nu(df)=12074209537.129366, loc=-2.878629, scale=7.691454
LogNormal on raw gaps (minutes):
  shape(s)=2.225509, loc=0.000000, scale=9.221990

合并之后的数据：
[Gap sign] <=0 ratio: 0.0000  (zero: 0.0000, neg: 0.0000)
[Stats] #gaps = 2994935
[Stats] raw gaps minutes: mean=46.597, median=31.683, p95=147.733, p99=274.083
[Stats] log gaps: mean=3.063, std=1.662

[MLE Parameters]
Normal on log-gaps:
  mu=3.063127, sigma=1.662139
Student-t on log-gaps:
  nu(df)=2.413243, loc=3.446235, scale=0.949215
LogNormal on raw gaps (minutes):
  shape(s)=1.662142, loc=0.000000, scale=21.394329


python 6DistributionofUserTimeIntervals_2datasets.py \
  --geolife_csv <PATH_TO_GEOLIFE_CSV> \
  --moreuser_csv <PATH_TO_MOREUSER_CSV> \
  --out_pdf <OUTPUT_PDF_PATH>

python 6DistributionofUserTimeIntervals_2datasets.py \
  --geolife_csv ./Data/Output/all_users_context_combined.csv \
  --moreuser_csv ./Data/MoreUser/all.csv \
  --out_pdf ./Pictures/gap_2x2_geolife_moreuser.pdf


python 6DistributionofUserTimeIntervals_2datasets.py --geolife_csv ./Data/Output/all_users_context_combined_gapLE0_merged.csv --moreuser_csv ./Data/MoreUser/all_gapLE0_merged.csv --out_pdf ./Pictures/gap_2x2_geolife_moreuser.png

## 1. 类似GeoLife数据处理

### 主要步骤

1. 将每个用户的plt文件都合并为1个csv文件。
2. 生成停留点和移动点。
3. 删除异常值。
4. 生成指定数据格式的轨迹数据。
   1. 生成时序数据。
   2. 矩阵数据。
   3. ~~大模型微调的提示词。~~

### 数据中需要注意的地方

1. 需要删除区域之外的点。
2. 生成停留点的时候，需要去掉停留点为空的用户。
3. Geolife的数据是按天提供的，需要删除停留点超过24小时的点。
4. 使用最直接最原始的方式对栅格进行编号。也就是按顺序进行编号。

### 其他

1. 只要不用特征附着，那么可以不用多线程和polars。

## 2. 生成外部输入上下文

### 主要步骤

1. 输出的上下文格式为：{user id}会在{time/time period}去{grid}。也就是用户1将会在（一个精确时间）或者（一个模糊的时间段）访问（grid）。
   1. User {userID} will go to the area with grid number {grid number} at {time/time period}.
2. {time/time period}是随机对未来某个stay的进行说明。
3. 已经给出了上下文信息的停留点不会重复给出上下文。（没有处理改变意图的情况）
4. 对于出现频率最高的3个停留点不会给出上下文。这对应的解释是长期有规律的活动不需要上下文。
