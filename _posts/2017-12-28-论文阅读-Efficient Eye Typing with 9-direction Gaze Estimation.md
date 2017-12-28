---
layout:     post
title:      论文阅读-Efficient Eye Typing with 9-direction Gaze Estimation
subtitle:   摘要
date:       2017-12-28
author:     顾剑成
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - Gaze Estimation
    - Deep Learning
    - Human Computer Interaction
---

# 完整交互过程

1. 位置校准
	- Screen mode
	> Screen size: 30x40 cm Distance: 25-40cm Cost Time: 10~20sec
	
	- Off-screen mode
	> Distance: 20~30cm

2. Gaze estimation
	*Convolutional Neural Network*
	
	- input: 32*128(two eyes) 32*64(one eye)
	
	- output: 10 (classes, 0-blink eye,1-9-9 directions)
	
	- model: 3 conv, 1fc
		- details:
			- Conv Block: Conv(3x3,p1,s1)尺度不变卷积->Bn->ReLu->MaxPool(2x2,s2).
	
	- training:
		- offline augmentation: rotate(-2.5~+2.5 degree rotation->crop eye region),scale(1.2~1.5倍),shift(8个方向 5 pixel)
		> 扩充到74101张图
		
		- online augmentation: random flip, HSV channel random adjustment
	
	- dataset:
		*162* video files, collected from *25* people
		时长*10-30*秒
		收集到了*832*个训练图像，*728*个测试图像


3. 文字录入系统
	- 噪声处理
	维护一个length为18的缓冲数据队列，出现最频繁的预测作为当前时刻的预测结果，等价于超过8帧的预测才被当作正式预测结果。
	由于不自觉的眨眼时常在0.1-0.17秒，这种方式解决了眨眼问题。

	- 交互输入：T9 method
# Reference
***Zhang C, Yao R, Cai J. Efficient Eye Typing with 9-direction Gaze Estimation[J]. 2017.***
