---
layout:     post
title:      MultiGPU-ComputeGradient-坑
subtitle:   Issue
date:       2018-1-10
author:     顾剑成
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - Tensorflow
    - Multi-GPU
    - Issue
    - Gradient
    - Optimizer
---
> Environment: Tensorflow-1.2

# MultiGPU-ComputeGradient-坑
## Issue


问题是这样的，在使用多GPU进行数据并行的训练时，我们需要针对每个GPU设备单独分配部分的数据，我通过以下代码解决。
	
	devices = [0,1,2]
	X = tf.placeholder(dtype=tf.float32,shape[None,224,224,3])
	Y = tf.placeholder(dtype=tf.int32,shape[None])
	input_tensors = tf.split(X,len(devices),0)
	label_tensors = tf.split(Y,len(devices),0)

接下来就到了出错的代码部分：

	grad_list = []
	opt = tf.train.MomentumOptimizer(learning_rate=0.001)
	for i in range(len(devices)):
		device = devices[i]
        with tf.device('/gpu:%d' % device):
            with tf.name_scope('%d' % device) as scope:
                logits=MyModel(input_tensors[i])
				...
                loss =	...
                grad = opt.compute_gradients(loss)
                grad_list.append(grad)
				...
                tf.get_variable_scope().reuse_variables()	
	...
	avg_grad = Average_gradients(grad_list)
	opt.apply_gradients(avg_grad)

Error:

	File "train.py", line 79, in train
    train_op = opt.apply_gradients(avggrads,global_step=global_step)
	File "/home/hui89.liu/anaconda2/lib/python2.7/site-packages/tensorflow/python/training/optimizer.py", line 446, in apply_gradients
	ValueError: Variable blockr01/conv/W/Momentum/ does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=None in VarScope?

同样的问题，在Github的[issue 6220](https://github.com/tensorflow/tensorflow/issues/6220) 已经得到解答。 （浏览详细解答过程，见[issue 6220](https://github.com/tensorflow/tensorflow/issues/6220))

由于这个issue提的日期较早，大多数用户tensorflow版本是0.12.0，我的版本是1.2.0，以下是官方解决方案，以及我最终的写法：

	with tf.variable_scope(tf.get_variable_scope(),reuse=False):
		grad_list = []
		opt = tf.train.MomentumOptimizer(learning_rate=0.001)
		for i in range(len(devices)):
			device = devices[i]
	        with tf.device('/gpu:%d' % device):
	            with tf.name_scope('%d' % device) as scope:
	                logits=MyModel(input_tensors[i])
					...
	                loss =	...
	                grad = opt.compute_gradients(loss)
	                grad_list.append(grad)
					...
	                tf.get_variable_scope().reuse_variables()	
		...
		avg_grad = Average_gradients(grad_list)
	with tf.variable_scope(tf.get_variable_scope(),reuse=False): # My personal written
		opt.apply_gradients(avg_grad)

## Conclusion
由于版本的不同，官方文档的欠缺，tensorflow的尝试需要我们踏破铁鞋，使用 ***tf.get_variable_scope().reuse_variables()*** 可以使得变量跨设备被重复利用。 在遇到问题和探索的过程中，我们需要确信***MomentumOptimizer*** 在被调用的过程中，***reuse=False*** ，也许0.12.0版本中，在 ***with tf.variable_scope(tf.get_variable_scope(),reuse=False):*** 大括号范围内，如没有特殊设定，可以保持***reuse=False***,但1.2.0版本中，设定具有历史记录，需要强制再次使用大括号进行***reuse***的转换。

