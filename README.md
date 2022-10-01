[toc]

# 短视频推荐经验总结
总结一下目前工作中涉及到的短视频推荐领域相关优化经验，个别地方表述未必准确，欢迎大家批评指正~

## 时长建模

### WCE加权分类

- 优化目标

建模用户观看时长

- 方案借鉴

参考YouTube的[Deep Neural Networks for YouTube](Recommendationshttps://dl.acm.org/doi/pdf/10.1145/2959100.2959190)，是否点击作为正负样本，将正样本的观看时长作为其样本权重，用加权逻辑回归进行训练，线上infer用odds作为预估时长，具体推导过程参考：https://zhuanlan.zhihu.com/p/435912211

- 何为WCE？

普通二值交叉熵损失函数定义如下：
$$Loss=- \frac{1}{N} \sum_{k=1}^N [y_{i} \cdot log(p_{i}) + (1-y_{i}) \cdot log(1-p_{i})] $$
WCE(加权交叉熵损失函数)定义如下：
$$Loss=- \frac{1}{N} \sum_{k=1}^N [w_{i} \cdot y_{i} \cdot log(p_{i}) + (1-y_{i}) \cdot log(1-p_{i})]$$

这和普通二值交叉熵仅仅有一点变化，就是在**正样本**的判别上加了一个 $w_{i}$ 系数，而该系数的设定则有很多方法。

- 短视频领域优化方案

短视频推荐初期优化指标是ctr，而视频本身时长及播放时长的差异，导致正负样本的定义较难界定。建模时长采用的方案是WCE，即每个样本都是正样本，其观看时长作为权重，由于沉浸式业务场景没有负样本，所以会对每个样本复制一份作为负样本，采用加权Logloss训练，具体实现方案有两种：
1. 直接修改loss函数
$$Loss=- \frac{1}{N} \sum_{k=1}^N [w_{i} \cdot log(p_{i}) + log(1-p_{i})]$$
2. 样本复制一份作为负样本，权重为1，然后一起训练

两种方案除了工程性能的差异，最终的训练效果是否有差异？？？

- 短视频WCE具体实现方案

WCE建模时长的方案在国内某手app上得到充分验证，具体有以下两种方案：

|  | 方案1 | 方案2 |
| :----:| :----: | :----: |
| 观看时长$wt$ | $wt \in (0, +\infty]$ | $wt \in (0, +\infty]$ |
| 正样本权重$w$ | $w_{1} = \log_2 (wt+1)$ | $w_{2} = \log_2(wt+1)+1$ |
| 负样本权重 | 1 | 1 |
| 模型预测值<br>模型输出经过sigmoid变换 | $p_{1}$| $p_{2}$ |
| $odds$ | $odds = \frac{p_{1}}{1-p_{1}} = w_{1}$ <br> $p_{1} = \frac{w_{1}}{w_{1} +1}$ | $odds = \frac{p_{2}}{1-p_{2}} = w_{2}$ <br> $p_{2} = \frac{w_{2}}{w_{2} +1}$ |
| 模型预测值还原为时长 | $wt = 2^{\frac{p_{1}}{1-p_{1}}} -1 $ | $wt = 2^{(\frac{p_{2}}{1-p_{2}} -1)} -1$ |
| 变量边界条件 | $wt \in (0, +\infty]$ <br> $w_{1} \in (0, +\infty]$ <br> $p_{1} \in (0, 1]$ | $wt \in (0, +\infty]$ <br> $w_{2} \in (1, +\infty]$ <br> $p_{2} \in (\frac{1}{2}, 1]$ |
| 时长边界条件 | $当wt=1时，$ <br> $w_{1} = 1$ <br> $p_{1} = \frac{1}{2}$ | $当wt=1时，$ <br> $w_{2} = 2$ <br> $p_{2} = \frac{2}{3}$ |
| 模型预测值图示 | ![1.png](https://github.com/ShaoQiBNU/videoRecTips/blob/main/imgs/1.png) | ![2.png](https://github.com/ShaoQiBNU/videoRecTips/blob/main/imgs/2.png) |



### 多分类及衍化

## 用户冷启动优化
待补充
PPNet
POSO





