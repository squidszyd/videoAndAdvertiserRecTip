[toc]

# 短视频推荐优化经验总结

总结一下目前工作中涉及到的短视频推荐领域相关优化经验，个别地方表述未必准确，欢迎大家批评指正~



## 模型结构优化

待补充



## 样本及损失函数优化

### weighted cross entropy loss

短视频推荐初期优化指标是ctr，而视频本身时长及播放时长的差异，导致正负样本的定义较难界定，所以常采用的建模方式是weighted cross entropy loss，即WCE。

#### 何为WCE？

普通二值交叉熵损失函数定义如下：

https://latex.codecogs.com/svg.image?Loss&space;=&space;-\frac{1}{N}\sum_{i=1}^{n}[y_{i}\cdot&space;log(p_{i})&plus;(1-y_{i})\cdot&space;log(1-p_{i})]



WCE的样本构造方式如下：

正样本
