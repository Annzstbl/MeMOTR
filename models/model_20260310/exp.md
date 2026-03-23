
# 存在的问题:

1. 如果用avg_pool的话，越上层值越小
2. PIHead中，光谱分支中，dict数量为128，余弦距离也是128维度的；空间分支中特征为64维度+全局特征64维度。这两个分支其表达的一是含义不同，而是维度不对应，导致spec_proj前后的特征对目标处均不明显
3. PIHead的输出值，普遍都偏高，不知道是否和Spec分支有关，还是和门控增强有关
4. 多特征融合的时候，感觉spec有一些问题
VID: data30-10 data36-13 data39-1 data46-12 data48-1

5. GlobalToken的问题
 原来是怎么做的？权重系数是通过conv得到的。权重系数是在global token + 4*2个采样点上做softmax的。所有的token在经过deformable之后，再加上与global token交互的部分。global token通过这个系数吸收所有原来的token进行更新。

 现在是怎么做的？现在是系数通过原有attentnion得到，之后直接在global token维度softmax, 迫使所有的原token必须和全局token交互，且交互力度和self attention的力度一样大。同时全局token没有更新。

# 一些记录:

使用了gamma对特征图进行增强, 梯度是允许反传的