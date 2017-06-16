# *CNN图像风格迁移*
    根据前辈提供的资料，重新用keras写的图像风格迁移,一下是迁移10次的结果和迁移的图片,图片大小影响神经元节点数，即影响计算量，在内存为4G的CPU环境下当800x600图片写出一张图片需要平均2.4h,但是图片压缩为原来1/4。，写出一张图片只需要不超过8min。
因此，只在CPU运行环环境下图片不需要太大，可以按长宽比例进行压缩。
## 图片
### 迁移图像
![image text](https://raw.github.com/KANG99/Kang-keras-style-transfer/master/results/15.png)
### 内容图像 
![image text](https://raw.github.com/KANG99/Kang-keras-style-transfer/master/images/Taipei101.jpg)
### 风格图片
![image text](https://raw.github.com/KANG99/Kang-keras-style-transfer/master/images/guernica.jpg)
## 下面是代码，以及自身对于实现的理解，谢谢！
```python
#! /usr/bin/env python3
# -*-coding=utf-8-*-
import time
from scipy.misc import imsave
from scipy.optimize import fmin_l_bfgs_b
from keras import backend as K
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.applications.imagenet_utils import preprocess_input
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
#使用tensorflow环境编程
os=K.os
np=K.np
#定义目标图像长宽
img_rows=400
img_columns=300
#读入图片文件，以数组形式展开成三阶张量，后用numpy扩展为四阶张量
#最后使用对图片进行预处理：（1）去均值,（2）三基色RGB->BGR(3)调换维度 
def read_img(filename):
	img=load_img(filename,target_size=(img_columns,img_rows))
	img=img_to_array(img)
	img=np.expand_dims(img,axis=0)
	img=preprocess_input(img)
	return img
#写入/存储图片，将输出数组转换为三维张量，量化高度层BGR,并将BGR->RGB
#经灰度大小截断在（0,255）
def write_img(x,ordering):
	x=x.reshape((img_columns,img_rows,3))
	x[:,:,0]+=103.939
	x[:,:,1]+=116.779
	x[:,:,2]+=123.68
	x=x[:,:,::-1]
	x=np.clip(x,0,255).astype('uint8')
	result_file=('results/%s'%str(ordering).zfill(2))+'.png'
	if not os.path.exists('results'):
		os.mkdir('results')
	imsave(result_file,x)
	print(result_file)
#建立vgg19模型
def vgg19_model(input_tensor):
	img_input=Input(tensor=input_tensor,shape=(3,600,800,3))
	#Blocks 1
	x=Conv2D(64,(3,3),activation='relu',padding='same',name='block1_conv1')(img_input)
	x=Conv2D(64,(3,3),activation='relu',padding='same',name='block1_conv2')(x)
	x=MaxPooling2D((2,2),strides=(2,2),name='block1_pooling')(x)
	#Block 2
	x=Conv2D(128,(3,3),activation='relu',padding='same',name='block2_conv1')(x)
	x=Conv2D(128,(3,3),activation='relu',padding='same',name='block2_conv2')(x)
	x=MaxPooling2D((2,2),strides=(2,2),name='block2_pooling')(x)
	#Block3
	x=Conv2D(256,(3,3),activation='relu',padding='same',name='block3_conv1')(x)
	x=Conv2D(256,(3,3),activation='relu',padding='same',name='block3_conv2')(x)
	x=Conv2D(256,(3,3),activation='relu',padding='same',name='block3_conv3')(x)
	x=Conv2D(256,(3,3),activation='relu',padding='same',name='block3_conv4')(x)
	x=MaxPooling2D((2,2),strides=(2,2),name='block3_pooling')(x)
	#Block 4
	x=Conv2D(512,(3,3),activation='relu',padding='same',name='block4_conv1')(x)
	x=Conv2D(512,(3,3),activation='relu',padding='same',name='block4_conv2')(x)
	x=Conv2D(512,(3,3),activation='relu',padding='same',name='block4_conv3')(x)
	x=Conv2D(512,(3,3),activation='relu',padding='same',name='block4_conv4')(x)
	x=MaxPooling2D((2,2),strides=(2,2),name='block4_pooling')(x)
	#Block 5
	x=Conv2D(512,(3,3),activation='relu',padding='same',name='block5_conv1')(x)
	x=Conv2D(512,(3,3),activation='relu',padding='same',name='block5_conv2')(x)
	x=Conv2D(512,(3,3),activation='relu',padding='same',name='block5_conv3')(x)
	x=Conv2D(512,(3,3),activation='relu',padding='same',name='block5_conv4')(x)
	x=MaxPooling2D((2,2),strides=(2,2),name='block5_pooling')(x)
	x=GlobalAveragePooling2D()(x)
	inputs=get_source_inputs(input_tensor)
	model=Model(inputs,x,name='vgg19')
	weights_path='vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
	model.load_weights(weights_path)
	return model
#生成输入的张量,将内容，风格和迁移图像（中间量）一起输入到vgg模型中，返回三合一张量，和中间图张量
def create_tensor(content_path,style_path):
	content_tensor=K.variable(read_img(content_path))
	style_tensor=K.variable(read_img(style_path))
	transfer_tensor=K.placeholder((1,img_columns,img_rows,3))
	input_tensor=K.concatenate([content_tensor,style_tensor,transfer_tensor],axis=0)
	return input_tensor,transfer_tensor
#设置Gram_matrix矩阵的计算图，输入为某一层的representation
def gram_matrix(x):
	features=K.batch_flatten(K.permute_dimensions(x,(2,0,1)))
	gram=K.dot(features,K.transpose(features))
	return gram
#风格loss
def style_loss(style_img_feature,transfer_img_feature):
	style=style_img_feature
	transfer=transfer_img_feature
	A=gram_matrix(style)
	G=gram_matrix(transfer)
	channels=3
	size=img_rows*img_columns
	loss=K.sum(K.square(A-G))/(4.*(channels**2)*(size**2))
	return loss
#内容loss
def content_loss(content_img_feature,transfer_img_feature):
	content=content_img_feature
	transfer=transfer_img_feature
	loss=K.sum(K.square(transfer-content))
	return loss		 
#变量loss,一段迷一样的表达式×-×，施加全局差正则表达式，全局差正则用于使生成的图片更加平滑自然
def total_variation_loss(x):
	a=K.square(x[:,:img_columns-1,:img_rows-1,:]-x[:,1:,:img_rows-1,:])
	b=K.square(x[:,:img_columns-1,:img_rows-1,:]-x[:,:img_columns-1,1:,:])
	loss=K.sum(K.pow(a+b,1.25))
	return loss
#total loss
def total_loss(model,loss_weights,transfer_tensor):
	loss=K.variable(0.)
	layer_features_dict=dict([(layer.name,layer.output) for layer in model.layers])
	layer_features=layer_features_dict['block4_conv2']
	content_img_features=layer_features[0,:,:,:]
	transfer_img_features=layer_features[2,:,:,:]
	loss+=loss_weights['content']*content_loss(content_img_features,transfer_img_features)
	feature_layers=['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
	for layer_name in feature_layers:
		layer_features=layer_features_dict[layer_name]
		style_img_features=layer_features[1,:,:,:]
		transfer_img_features=layer_features[2,:,:,:]
		loss+=(loss_weights['style']/len(feature_layers))*(style_loss(style_img_features,transfer_img_features))
	loss+=loss_weights['total']*total_variation_loss(transfer_tensor)
	return loss
#通过K.gradient获取反向梯度，同时得到梯度和损失，
def create_outputs(total_loss,transfer_tensor):
	gradients=K.gradients(total_loss,transfer_tensor)
	outputs=[total_loss]
	if isinstance(gradients,(list,tuple)):
		print('list/tuple')
		outputs+=gradients
	else:
		outputs.append(gradients)
	return outputs
#计算输入图像的关于损失函数的倒数和对应损失值
def eval_loss_and_grads(x):
	x=x.reshape((1,img_columns,img_rows,3))
	outs=outputs_func([x])
	loss_value=outs[0]
	if len(outs[1:])==1:
		grads_value=outs[1].flatten().astype('float64')
	else:
		grads_value=np.array(outs[1:]).flatten().astype('float64')
	return loss_value,grads_value
#获取评价程序
class Evaluator(object):
	def __init__(self):
		self.loss_value=None
		self.grads_value=None
	def loss(self,x):
		loss_value,grads_value= eval_loss_and_grads(x)
		self.loss_value=loss_value
		self.grads_value=grads_value
		return self.loss_value
	def grads(self,x):
		grads_value=np.copy(self.grads_value)
		self.loss_value=None
		self.grads_value=None
		return grads_value
#main函数
if __name__=='__main__':
	print('')
	print('Welcom!')
	path={'content':'images/Macau.jpg','style':'images/StarryNight.jpg'}
	input_tensor,transfer_tensor=create_tensor(path['content'],path['style'])
	loss_weights={'style':1.0,'content':0.025,'total':1.0}
	model=vgg19_model(input_tensor)
	#生成总的反向特征缺失
	total_loss=total_loss(model,loss_weights,transfer_tensor)
	#生成正向输出
	outputs=create_outputs(total_loss,transfer_tensor)
	#获取计算图(反向输入图)
	outputs_func=K.function([transfer_tensor],outputs)
	#生成处理器
	evaluator=Evaluator()
	#生成噪声
	x=np.random.uniform(0,225,(1,img_columns,img_rows,3))-128
	#迭代训练15次
	for ordering in range(15):
		print('Start:',ordering)
		start_time=time.time()
		x,min_val,info=fmin_l_bfgs_b(evaluator.loss,x.flatten(),fprime=evaluator.grads,maxfun=20)
		print('Current_Loss:',min_val)
		img=np.copy(x)
		write_img(img,ordering)
		end_time=time.time()
		print('Used %ds'%(end_time-start_time))
```
## 使用VGG19模型，示意图如下：
![image text](https://raw.github.com/KANG99/Kang-keras-style-transfer/master/vgg19_img.png)
### 使用了前辈训练好的模型，模型下载地址如下：
[c-code](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5)
## 主要流程
![image text](https://raw.github.com/KANG99/Kang-keras-style-transfer/master/proc.png)
