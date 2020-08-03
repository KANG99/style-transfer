# *CNN图像风格迁移*

   
&nbsp;&nbsp;&nbsp;&nbsp;距离2017年用keras书写图像风格迁移的代码，已经过去三年了。2020年7月20号更新了一下keras的代码，并且利用pytorch重写了一下这段代码，用pytorch写的时候才发现原来里面很多原理都是很不清楚的。阔别三年重写这个项目别有一番感受。代码程序虽然进行了重写，但是参考了官方文档的大量代码,代码重写都是在业余时间完成的，代码中可能较多不足之处，欢迎提issue指正，不胜感激！
   
+ images文件夹里存放了素材图片（包括内容图片和风格图片）

+ results文件里面存放了最终效果图

+ [pytorch官方示例](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)

+ [keras官方示例](https://keras.io/examples/generative/neural_style_transfer/)

    
## 图片
### 内容图像 
![Alt text](https://github.com/KANG99/style-transfer/blob/master/images/Taipei101.jpg)
### keras实现效果
![Alt text](https://github.com/KANG99/style-transfer/blob/master/results/show_keras.png)
### pytorch实现效果
![Alt text](https://github.com/KANG99/style-transfer/blob/master/results/show_torch.png)

## 代码
### keras实现
```python
#! /usr/bin/env python3
# -*-coding=utf-8-*-
'''
（1）图像风格迁移：给定一张普通图片和一种艺术风格图片，生成一张呈现艺术风格
    和普通图片内容的迁移图片。
（2）此次实现中使用了VGG19的卷积神经网络模型，优化过程使用了scipy.optimizer
    基于L-BFGS算法的fmin_l_bfgs_b方法
（3）每次反向优化20次写出一张图片，在代码运行过程中发现超过10次loss减少量减少
    趋于平缓，所以只写出15张图片
（4）从images文件夹中选择普通图片和风格图片，并且不同风格和内容图片中间过程生成
    的图片都在results文件夹中
 (5)由于保存权值的.h5文件较大，这里给出下载地址
    https://github.com/fchollet/deep-learning-models/
    releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5
'''
#! /usr/bin/env python3
# -*-coding=utf-8-*-
import os
os.environ['KERAS_BACKEND']='tensorflow'
import time
from imageio import imwrite
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
#from keras.applications.imagenet_utils import _obtain_input_shape
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
	imwrite(result_file,x)
	print(result_file)
#建立vgg19模型
def vgg19_model(input_tensor):
	img_input=Input(tensor=input_tensor,shape=(img_columns,img_rows,3))
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
	path={'content':'images/Macau.png','style':'images/StarryNight.png'}
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
### pytorch实现
```python
import __future__
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import time
import os
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ContentLoss(torch.nn.Module):
    def __init__(self,content_feature,weight):
        super(ContentLoss,self).__init__()
        self.content_feature = content_feature.detach()
        self.criterion = torch.nn.MSELoss()
        self.weight = weight

    def forward(self,combination):
        self.loss = self.criterion(combination.clone()*self.weight,self.content_feature.clone()*self.weight)
        return combination

class GramMatrix(torch.nn.Module):
    def forward(self, input):
        b, n, h, w = input.size()  
        features = input.view(b * n, h * w) 
        G = torch.mm(features, features.t()) 
        return G.div(b * n * h * w)

class StyleLoss(torch.nn.Module):
    def __init__(self,style_feature,weight):
        super(StyleLoss,self).__init__()
        self.style_feature = style_feature.detach()
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = torch.nn.MSELoss()

    def forward(self,combination):
        #output = combination
        style_feature = self.gram(self.style_feature.clone()*self.weight)
        combination_features = self.gram(combination.clone()*self.weight)
        self.loss = self.criterion(combination_features,style_feature)
        return combination

class StyleTransfer:
    def __init__(self,content_image,style_image,style_weight=5,content_weight=0.025):
        # Weights of the different loss components
        self.vgg19 =  models.vgg19()
        self.vgg19.load_state_dict(torch.load('vgg19-dcbb9e9d.pth'))
        self.img_ncols = 400
        self.img_nrows = 300
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.content_tensor,self.content_name = self.process_img(content_image)
        self.style_tensor,self.style_name = self.process_img(style_image)
        self.conbination_tensor = self.content_tensor.clone()

    def process_img(self,img_path):
        img = Image.open(img_path)
        img_name  = img_path.split('/')[-1][:-4]
        loader = transforms.Compose([transforms.Resize((self.img_nrows,self.img_ncols)),
        transforms.ToTensor()])
        img_tensor = loader(img)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor.to(device, torch.float),img_name
    
    def deprocess_img(self,x,index):
        unloader = transforms.ToPILImage()
        x = x.cpu().clone()
        img_tensor = x.squeeze(0)
        img = unloader(img_tensor)
        result_folder = f'{self.content_name}_and_{self.style_name}'
        os.path.exists(result_folder) or os.mkdir(result_folder)
        filename = f'{result_folder}/rersult_{index}.png'
        img.save(filename)
        print(f'save {filename} successfully!')
        print()

    def get_loss_and_model(self,vgg_model,content_image,style_image):
        vgg_layers = vgg_model.features.to(device).eval()
        style_losses = []
        content_losses = []
        model = torch.nn.Sequential()
        style_layer_name_maping = {
                '0':"style_loss_1",
                '5':"style_loss_2",
                '10':"style_loss_3",
                '19':"style_loss_4",
                '28':"style_loss_5",
            }
        content_layer_name_maping = {'30':"content_loss"}
        for name,module in vgg_layers._modules.items():
            model.add_module(name,module)
            if name in content_layer_name_maping:
                content_feature = model(content_image).clone()
                content_loss = ContentLoss(content_feature,self.content_weight)
                model.add_module(f'{content_layer_name_maping[name]}',content_loss)
                content_losses.append(content_loss)
            if name in style_layer_name_maping:
                style_feature = model(style_image).clone()
                style_loss = StyleLoss(style_feature,self.style_weight)
                style_losses.append(style_loss)
                model.add_module(f'{style_layer_name_maping[name]}',style_loss)
        return content_losses,style_losses,model

    def get_input_param_optimizer(self,input_img):
        input_param = torch.nn.Parameter(input_img.data)
        optimizer = torch.optim.LBFGS([input_param])
        return input_param, optimizer

    def main_train(self,epoch=10):
        combination_param, optimizer = self.get_input_param_optimizer(self.conbination_tensor)
        content_losses,style_losses,model = self.get_loss_and_model(self.vgg19,self.content_tensor,self.style_tensor)
        cur,pre = 10,10
        for i in range(1,epoch+1):
            start = time.time()
            def closure():
                combination_param.data.clamp_(0,1)
                optimizer.zero_grad()
                model(combination_param)
                style_score = 0
                content_score = 0
                for cl in content_losses:
                    content_score += cl.loss
                for sl in style_losses:
                    style_score += sl.loss
                loss =  content_score+style_score
                loss.backward()
                return style_score+content_score
            loss = optimizer.step(closure)
            cur,pre = loss,cur
            end = time.time()
            print(f'|using:{int(end-start):2d}s |epoch:{i:2d} |loss:{loss.data}')
            if pre<=cur:
                print('Early stopping!')
                break
            combination_param.data.clamp_(0,1)
            if i%5 == 0:
                self.deprocess_img(self.conbination_tensor,i//5)
            
if __name__ == "__main__":
    pass
    print('welcome')
    content_file = 'images/Taipei101.jpg'
    style_file = 'images/StarryNight.jpg'
    st = StyleTransfer(content_file,style_file)
    epoch = 100
    st.main_train(epoch=epoch)
```

### VGG模型，结构图如下：
![Alt text](https://github.com/KANG99/style-transfer/blob/master/operation_principle/vgg_img.png)

### VGG19模型，示意图如下：
![Alt text](https://github.com/KANG99/style-transfer/blob/master/operation_principle/vgg19_img.png)

### VGG16模型，示意图如下：
![Alt text](https://github.com/KANG99/style-transfer/blob/master/operation_principle/vgg16_img.png)


### 主要流程
![Alt text](https://github.com/KANG99/style-transfer/blob/master/operation_principle/proc.png)

### vgg16流程
![Alt text](https://github.com/KANG99/style-transfer/blob/master/operation_principle/proc_vgg16.png)

