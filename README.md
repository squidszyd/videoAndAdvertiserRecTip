[toc]

# 短视频推荐经验总结
总结一下目前工作中涉及到的短视频推荐领域相关优化经验，个别地方表述未必准确，欢迎大家批评指正~

## 召回

## 粗排

## 精排

### 时长建模

#### mse建模
用mse损失函数对观看时长拟合，由于观看时长数值范围较大，直接对原始值拟合会导致 $logit$ 出现较大的range，导致模型陷入局部最优，引起梯度爆炸。

对观看时长做 $log$ 或 $e^{0.3}$ 变换后进行拟合，由于变换函数不是线性的，该方法对时长拟合是有偏估计，对于长观看时长的拟合会出现低估现象。

**mse隐含的假设是error（即label-pred）服从正态分布，然后最大化error的似然；然而很多情境下这个假设并不成立。当error不服从正太分布时，mse的效果就有可能受损。**

#### WCE加权分类

- 优化目标

建模用户观看短视频的时长

- 方案借鉴

参考YouTube的[Deep Neural Networks for YouTube](Recommendationshttps://dl.acm.org/doi/pdf/10.1145/2959100.2959190)，是否点击作为正负样本，将正样本的观看时长作为其样本权重，负样本权重为1，用加权逻辑回归进行训练，线上infer用odds作为预估时长，具体推导过程参考：https://zhuanlan.zhihu.com/p/435912211

- WCE加权分类

普通二值交叉熵损失函数定义如下：
$$Loss=- \frac{1}{N} \sum_{i=1}^N [y_{i} \cdot log(p_{i}) + (1-y_{i}) \cdot log(1-p_{i})] $$
WCE(加权交叉熵损失函数)定义如下：
$$Loss=- \frac{1}{N} \sum_{i=1}^N [w_{i} \cdot y_{i} \cdot log(p_{i}) + (1-y_{i}) \cdot log(1-p_{i})]$$

这和普通二值交叉熵仅仅有一点变化，就是在**正样本**的判别上加了一个 $w_{i}$ 系数，而该系数的设定则有很多方法。

- WCE具体方案

Youtube的页面是上下滑的列表，因此是否点击就可以作为正负样本的区分，但是短视频是沉浸式业务，每个视频都是点击的状态，都有播放时长，正负样本的定义较难界定，因此业界的做法是：用样本的观看时长作为正样本的权重，同时复制一份样本作为负样本，采用加权Logloss训练，因此 $odds$ 的表征会做如下调整：

$$ odds = \frac{p}{1-p} = \frac{\sum_{i=1}^k T_{i}}{N - k + k} = \frac{\sum_{i=1}^k T_{i}}{N} = E(T) $$

具体实现方案有两种：
1. 直接修改loss函数
$$Loss=- \frac{1}{N} \sum_{i=1}^N [w_{i} \cdot log(p_{i}) + log(1-p_{i})]$$

此时，正样本的梯度变为原来的 $w_{i}$ 倍，即 
$$w_{i}*(1-p_{i})$$

负样本的梯度保持不变，即 
$$1*(0-p_{i})$$

样本梯度推导见：https://lukebest.github.io/posts/d28f

2. 样本复制一份作为负样本，权重为1，然后一起训练

两种方案除了工程性能的差异，最终的训练效果是否有差异？？？

- 短视频WCE具体实现方案

WCE建模时长的方案在国内某手app上得到充分验证，对于权重 $w_{i}$ 有以下3种方案：

|  | 方案1 | 方案2 | 方案3 |
| :----:| :----: | :----: | :----: |
| 观看时长 $wt$ | $wt \in (0, +\infty]$ | $wt \in (0, +\infty]$ | $wt \in (0, +\infty]$ |
| 正样本权重 $w$ | $w_{1} = \log_2 (wt+1)$ | $w_{2} = \log_2(wt+1)+1$ | $w_{3} = wt$ |
| 负样本权重 | 1 | 1 | 1 |
| 模型预测值 <br> 模型输出经过sigmoid变换 | $p_{1} = \frac{1}{1+e^{-\theta_{1} x}}$| $p_{2} = \frac{1}{1+e^{-\theta_{2} x}}$ | $p_{3} = \frac{1}{1+e^{-\theta_{3} x}}$ | 
| $odds$ | $odds = \frac{p_{1}}{1-p_{1}} = e^{\theta_{1} x} = w_{1}$ <br> $p_{1} = \frac{w_{1}}{w_{1} +1}$ | $odds = \frac{p_{2}}{1-p_{2}} = e^{\theta_{2} x} = w_{2}$ <br> $p_{2} = \frac{w_{2}}{w_{2} +1}$ | $odds = \frac{p_{3}}{1-p_{3}} = e^{\theta_{3} x} = w_{3}$ <br> $p_{3} = \frac{w_{3}}{w_{3} +1}$ |
| 模型预测值还原为时长 | $wt = 2^{\frac{p_{1}}{1-p_{1}}} -1 $ | $wt = 2^{(\frac{p_{2}}{1-p_{2}} -1)} -1$ | $wt = \frac{p_{3}}{1-p_{3}}$ |
| 变量边界条件 | $wt \in (0, +\infty]$ <br> $w_{1} \in (0, +\infty]$ <br> $p_{1} \in (0, 1]$ | $wt \in (0, +\infty]$ <br> $w_{2} \in (1, +\infty]$ <br> $p_{2} \in (\frac{1}{2}, 1]$ | $wt \in (0, +\infty]$ <br> $w_{3} \in (0, +\infty]$ <br> $p_{3} \in (0, 1]$ |
| 时长边界条件 | $当wt=1时，$ <br> $w_{1} = 1$ <br> $p_{1} = \frac{1}{2}$ | $当wt=1时，$ <br> $w_{2} = 2$ <br> $p_{2} = \frac{2}{3}$ | $当wt=1时，$ <br> $w_{3} = 1$ <br> $p_{3} = \frac{1}{2}$ |
| 模型预测值图示 | ![1.png](https://github.com/ShaoQiBNU/videoRecTips/blob/main/imgs/1.png) | ![2.png](https://github.com/ShaoQiBNU/videoRecTips/blob/main/imgs/2.png) |

对于权重 $w$，业界有两种处理方法：

一种是对其做log变换，如方案1和2所示，快手采用该处理方法：观看时长 $wt \in (0, +\infty]$，该值可以无限大，但 $w$ 不能无限大，因此会对 $w = \log_2 (wt+1)$ 设置阈值上界，经验值设置为8，原因？

一种是直接用原值，如方案3所示：此方法与YouTube的论文和字节的处理方式一致

- WCE隐含的假设是y服从几何分布 $p(y) = p^y \cdot (1-p)$ 最大似然可得wce loss，即wce的预估值等于数学期望，是无偏预估。以方案3为例，具体推导如下：

$$Loss=- \frac{1}{N} \sum_{i=1}^N [w_{i} \cdot log(p_{i}) + log(1-p_{i})]， p_{i} = \frac{1}{1 + e^{-\theta x}}$$

求解该损失，得到无偏的预估时长 $\widehat{wt_{i}} = \frac{p_{i}}{1-p_{i}} = e^{\theta x_{i}}$ ，用预估值来表示损失函数可写为：

$$Loss=- \frac{1}{N} \sum_{i=1}^N [w_{i} \cdot log(\frac{\widehat{wt_{i}}}{1+\widehat{wt_{i}}}) + log(1-\frac{\widehat{wt_{i}}}{1+\widehat{wt_{i}}})]$$

$$ =- \frac{1}{N} \sum_{i=1}^N [w_{i} \cdot log(\widehat{wt_{i}}) - (1+w_{i}) \cdot log(1+\widehat{wt_{i}})]$$

$$ =- \frac{1}{N} \sum_{i=1}^N log[\widehat{wt_{i}}^{w_{i}} \cdot (1+\widehat{wt_{i}})^{-(1+w_{i})}] $$

$$ =- \frac{1}{N} \sum_{i=1}^N log[(\frac{\widehat{wt_{i}}}{\widehat{wt_{i}}+1})^{w_{i}} \cdot \frac{1}{\widehat{wt_{i}}+1}] $$

当 $w_{i} \in (0,1,2,..., +\infty]$，令 $p_{i} = \frac{1}{\widehat{wt_{i}}+1}$，有 $P(w_{i}=k|x_{i}) = (1-p_{i})^k \cdot p_{i}$，其满足几何分布，数学期望为 $\frac{1 - p_{i}}{p_{i}} = \widehat{wt_{i}}$ ，即WCE的预估值等于数学期望，是无偏预估。

几何分布的期望和方差如下：
![6.png](https://github.com/ShaoQiBNU/videoRecTips/blob/main/imgs/6.png)

几何分布：https://en.wikipedia.org/wiki/Geometric_distribution

wce的缺点：
- wce隐含的假设是y服从几何分布。虽然wce是无偏预估，预估值等于数学期望，但其隐含的假设是y服从几何分布，如果label的分布和几何分布较大，则WCE效果会变差；
- wce在低估和高估的时候梯度大小不同。对loss梯度做简单推导后可以发现wce在低估和高估的时候梯度大小不同，具体推导如下：

$$Loss=- \frac{1}{N} \sum_{i=1}^N [w_{i} \cdot log(\widehat{wt_{i}}) - (1+w_{i}) \cdot log(1+\widehat{wt_{i}})]$$

$$ \frac {\partial Loss} {\partial \widehat{wt_{i}}}=- \frac{1}{N} \sum_{i=1}^N [w_{i} \cdot \frac{1}{\widehat{wt_{i}}} - (1+w_{i}) \cdot \frac{1}{1+\widehat{wt_{i}}}]$$

$$ = \frac{1}{N} \sum_{i=1}^N \frac{\widehat{wt_{i}} - w_{i}}{\widehat{wt_{i}} \cdot (1+\widehat{wt_{i}})}$$

以 $w_{i}=20$ 为例，画出梯度图像如下所示：
![4.png](https://github.com/ShaoQiBNU/videoRecTips/blob/main/imgs/4.png)

可以看出，在低估 $\widehat{wt_{i}}<20$ 和高估 $\widehat{wt_{i}}>20$ 时，梯度具有严重的不对称性。低估时，梯度较大；高估时，梯度较小。在训练时，长视频样本回传梯度大，作用到全样本上，也贡献了短视频的预估，所以会出现长视频低估、短视频高估现象。

#### softmax多分类
将观看时长进行分桶离散化，进而将回归问题转为多分类问题，业界方案如下：

- 对观看时长 $wt$ 做等频划分，划为 $K$ 个桶：

$$ [wt_{1},wt_{2},wt_{3}...,wt_{K}] $$
$$ wt_{k}为桶的边界值，k=1,2,3..,K$$


- 根据样本的观看时长得到label转换后的类别：

当 $wt_{k-1} < wt_{i} < wt_{k}$时， 该样本转换后的类别为 $b_{i}$


- 模型优化由回归变为多分类问题，loss函数如下：

$$Loss=- \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K I(b_{i}) \cdot log(p_{i,k})$$

$$I(b_{i})只在对应类别为1，其余为0$$


- 线上serving时，得到样本的预测时长

$$ \widehat{wt_{i}} = \sum_{k=1}^K m_{k} \cdot p_{i,k} $$

$$ m_{k}是第k个桶的均值或中值，p_{i,k}表示样本i预测时长是第k类的概率 $$

#### distill softmax多分类

softmax多分类采用的是0-1的hard label，目标只能离散到一个桶里，这种硬标签忽略了负标签直接的差别。例如，某样本被分到第 $k$ 个桶， 损失函数只区分了预估为 $k$ 或者不为 $k$，当不为 $k$ 时，预测为 $k+1$ 或者 $k+n$ 的损失都为0，没有区别。但是作为回归问题，分桶值大小有顺序的含义，预测为 $k+n$ 的损失应该大于 $k+1$ 的损失.

distill softmax多分类借鉴了知识蒸馏中soft label的思路，为负标签增加信息，从而缓解上述问题。

假设时长分桶服从某个先验分布 $p(wt_{k})$，可以用KL散度来学习预测分布 $p_{i,k}$ 和 $p(wt_{k})$ 的相似性，即：

$$Loss=\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K p_{i}(wt_{k}) \cdot log\frac{p_{i}(wt_{k})}{p_{i,k}}$$

$$=\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K p_{i}(wt_{k}) \cdot [log(p_{i}(wt_{k})) - log(p_{i,k})]$$

$$=\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K p_{i}(wt_{k}) \cdot log(p_{i}(wt_{k})) - p_{i}(wt_{k}) \cdot log(p_{i,k})$$

$$=-\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K p_{i}(wt_{k}) \cdot log(p_{i,k}) + const$$


$$当 p_{i}(wt_{k}) \sim \mathcal{N}(wt_{i}, \sigma)时，p_{i}(wt_{k}) = \frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{1}{2}(\frac{wt_{k} - wt_{i}}{\sigma})^{2}}$$ 

$$Loss = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K e^{-\frac{1}{2}(\frac{wt_{k} - wt_{i}}{\sigma})^{2}} \cdot log(p_{i,k})$$

$$当 p_{i}(wt_{k}) \sim Laplace(wt_{i}, \sigma)时，p_{i}(wt_{k}) = e^{-\frac{|wt_{k} - wt_{i}|}{\sigma}}$$ 

$$Loss = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K e^{-\frac{|wt_{k} - wt_{i}|}{\sigma}} \cdot log(p_{i,k})$$

$$ wt_{i}是样本i的观看时长，为真实label值，wt_{k}是时长分桶第k个桶的桶内均值或者桶边界，此处以桶边界为例 $$

$$ \sigma 为超参数，可以设置为定值，也可以label-aware，如 \sigma = 1.5 \cdot \sqrt{wt}，越大的label其概率分布越平缓 $$

分桶策略有以下几种：

- uniform
设定bin_size，进行等距分桶

- segment
根据时长分布，手动分桶

- exp_bin
```python
tf.constant([np.exp(x/40.0) - 1 for x in range(bucket_size)])
```

- 线上serving时，得到样本的预测时长

$$ \widehat{wt_{i}} = \sum_{k=1}^K m_{k} \cdot p_{i,k} $$

$$ m_{k}是第k个桶的均值或中值，p_{i,k}表示样本i预测时长是第k类的概率 $$

![5.png](https://github.com/ShaoQiBNU/videoRecTips/blob/main/imgs/5.png)

#### 分桶LogLoss建模/ordinal regression建模

简单的多分类存在没有考虑非目标类的之间的序关系的问题，Ordinal Regression则是一种考虑类间序关系的回归方法，推导过程参考：https://zhuanlan.zhihu.com/p/573572151

![3.png](https://github.com/ShaoQiBNU/videoRecTips/blob/main/imgs/3.png)

具体做法如下：

- 观看时长 $wt$ 划为 $K$ 个桶：

$$ [wt_{1},wt_{2},wt_{3}...,wt_{K}] $$
$$ wt_{k}为桶的边界值，k=1,2,3..,K$$

- 根据样本的观看时长得到转换后的label：

当 $wt_{k-1} < wt_{i} < wt_{k}$时，即 $b_{i}=k$，该样本转换后的label为：

[1, 1, ..., 1, 0, ..., 0]

$wt_{i}$ 的样本在 $[wt_{k-1}, wt_{k}]$ 下类别为1，在 $[wt_{1}, wt_{2}]，[wt_{2}, wt_{3}]，...，[wt_{k-2}, wt_{k-1}]$ 下类别为1，在 $[wt_{k}, wt_{k+1}]，[wt_{k+1}, wt_{k+2}]，...，[wt_{K-1}, wt_{K}]$ 下类别为0


- 模型优化由回归问题转化为多个独立的二分类问题，loss函数如下：

$$Loss=- \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K I(k \le b_{i}) \cdot log(p_{i,k})$$

$$I(k \le b_{i})表示小于等于类别b_{i}时为1，其余为0$$

- 线上serving时，得到样本的预测时长

$$ \widehat{wt_{i}} = \sum_{k=1}^K I(k \le b_{i}) \cdot (m_{k} - m_{k-1}) \cdot p_{i,k} $$

$$ = 1 \cdot (m_{1} - m_{0}) \cdot p_{i,1} + 1 \cdot (m_{2} - m_{1}) \cdot p_{i,2} + ... + 1 \cdot (m_{b_{i}} - m_{b_{i}-1}) \cdot p_{i,b_{i}}  + 0 \cdot (m_{k+2} - m_{k+1}) \cdot p_{i,k+1} + ... + 0 \cdot (m_{K} - m_{K-1}) \cdot p_{i,K} $$

$$ = 1 \cdot (m_{1} - m_{0}) \cdot p_{i,1} + 1 \cdot (m_{2} - m_{1}) \cdot p_{i,2} + ... + 1 \cdot (m_{b_{i}} - m_{b_{i}-1}) \cdot p_{i,b_{i}} $$

$$ b_{i}=k表示样本i的时长lable是k，m_{k}是第k个桶的均值或中值，m_{0} = 0 $$

$$ p_{i,k}表示样本i预测时长是第k个桶的概率，I(k \le b_{i})表示小于等于类别b_{i}时为1，其余为0$$

## 用户冷启动优化
待补充
PPNet
POSO





