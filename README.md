[toc]

# 短视频推荐经验总结
总结一下目前工作中涉及到的短视频推荐领域相关优化经验，个别地方表述未必准确，欢迎大家批评指正~

## 时长建模

### WCE加权分类

- 优化目标
建模用户观看时长

- 方案借鉴
参考YouTube的[Deep Neural Networks for YouTube](Recommendationshttps://dl.acm.org/doi/pdf/10.1145/2959100.2959190)，是否点击作为正负样本，将正样本的观看时长作为其样本权重，用加权逻辑回归进行训练，线上infer用odds作为预估时长，具体推导过程参考：https://zhuanlan.zhihu.com/p/435912211

- 短视频领域优化方案
短视频推荐初期优化指标是ctr，而视频本身时长及播放时长的差异，导致正负样本的定义较难界定。建模时长采用的方案是WCE，即每个样本都是正样本，其观看时长作为权重，由于沉浸式业务场景没有负样本，所以会对每个样本复制一份作为负样本，采用加权Logloss训练，具体实现方案有两种：
1. 修改loss函数
2. 样本复制一份作为负样本，权重为1，然后一起训练

两种方案除了工程性能的差异，最终的训练效果是否有差异？？？

#### 何为WCE？

普通二值交叉熵损失函数定义如下：

$$Loss=- \frac{1}{N} \sum_{k=1}^N k^2 $$

![](https://latex.codecogs.com/svg.image?Loss&space;=&space;-\frac{1}{N}\sum_{i=1}^{n}[y_{i}\cdot&space;log(p_{i})&plus;(1-y_{i})\cdot&space;log(1-p_{i})])

WCE(加权交叉熵损失函数)定义如下：

![](https://latex.codecogs.com/svg.image?Loss&space;=&space;-\frac{1}{N}\sum_{i=1}^{n}[w_{i}\cdot&space;y_{i}\cdot&space;log(p_{i})&plus;(1-y_{i})\cdot&space;log(1-p_{i})])

这和二值交叉熵仅仅有一点变化，就是在**正样本**的判别上加了一个![](https://latex.codecogs.com/svg.image?w_{i})系数，而该系数的设定则有很多方式确定。

#### 短视频WCE具体实现方案

$$E=mc^2$$

行内的公式 $$E=mc^2$$ 行内的公式，行内的 $$E=mc^2$$ 公式。

- 正样本
用户点击并播放视频video，权重![](https://latex.codecogs.com/svg.image?w_{i})设置为：![](https://latex.codecogs.com/svg.image?w_{i}&space;=&space;log_{2}(wt&plus;1),&space;wt&space;=&space;watchTime)

![](https://latex.codecogs.com/svg.image?watchTime)为用户对视频的播放时长，单位为s；

- 负样本
以上正样本复制一份，权重设置为：1；

- 模型预测结果线上还原
模型假定DeepFM，则模型预测结果记作![](https://latex.codecogs.com/svg.image?\inline&space;p)，则：

![](https://latex.codecogs.com/svg.image?\inline&space;Odds&space;=&space;\frac{p}{1-p}&space;=e^{\theta^{T}&space;x}=w)

预测结果![](https://latex.codecogs.com/svg.image?\inline&space;p)为预测是正样本的概率，![](https://latex.codecogs.com/svg.image?w)为正样本权重，线上使用时，会根据权重公式还原为观看时长，具体如下：


- 边界条件推导
![](https://latex.codecogs.com/svg.image?\because&space;wt\in&space;[0,&plus;\infty]\:&space;\;&space;\:&space;\;&space;w\in&space;[0,&plus;\infty]\:&space;\;&space;\:&space;\;&space;p\in&space;[0,1])

![](https://latex.codecogs.com/svg.image?\therefore&space;\frac{p}{1-p}\in&space;[0,&plus;\infty]\:&space;\;&space;\:&space;\;&space;e^{\theta^{T}&space;x}&space;\in&space;[0,&plus;\infty]\:&space;\;&space;\:&space;\;&space;\theta^{T}&space;x&space;\in&space;[-\infty,&plus;\infty])

根据sigmoid函数曲线，当![](https://latex.codecogs.com/svg.image?p=\frac{1}{2})时，可推导得出![](https://latex.codecogs.com/svg.image?wt=1)，即当预测值![](https://latex.codecogs.com/svg.image?p>\frac{1}{2})时，预测用户播放视频时长![](https://latex.codecogs.com/svg.image?wt>1s&space;)；反之，预测用户播放视频时长![](https://latex.codecogs.com/svg.image?wt<1s&space;)。

此外，对于正样本权重![](https://latex.codecogs.com/svg.image?w)的设置，也有另一种方式，如下：

![](https://latex.codecogs.com/svg.image?w=log_{2}(wt&plus;1)&plus;1)

则边界条件相应变为：

![](https://latex.codecogs.com/svg.image?\because&space;wt\in&space;[0,&plus;\infty]\:&space;\;&space;\:&space;\;&space;w\in&space;[1,&plus;\infty]\:&space;\;&space;\:&space;\;&space;p\in&space;[\frac{1}{2},1])

![](https://latex.codecogs.com/svg.image?\therefore&space;\frac{p}{1-p}\in&space;[1,&plus;\infty]\:&space;\;&space;\:&space;\;&space;e^{\theta^{T}&space;x}&space;\in&space;[1,&plus;\infty]\:&space;\;&space;\:&space;\;&space;\theta^{T}&space;x&space;\in&space;[0,&plus;\infty])

### 多分类及衍化

## 用户冷启动优化
待补充
PPNet
POSO





