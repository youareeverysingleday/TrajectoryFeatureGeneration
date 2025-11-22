# TrajectoryFeatureGeneration
generate feature of trajectory.

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
