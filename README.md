参考：https://github.com/TowardsNorth/yolo_v1_tensorflow_guiyu
项目地址：

## pascal_voc.py

主要研究 `load_pascal_annotation(self, index)`函数：

其返回值为：`return label, len(objs) #返回label，以及该index文件中object的数量`

label数组的定义：

```python
# label数组维度 7*7*25， 一个cell只负责预测一个类别
# 7*7为网格的个数，25的0为标志位，用于是否检测到object
# 1-4为box的坐标信息，5-24位为是否存在20个相应类别的标志
label = np.zeros((self.cell_size, self.cell_size, 25))
```

  其中的位置信息根据xml文件得知，并且转换为448下的尺寸信息

## yolo_net.py

定义了Yolo_v1网络的结构：

网络结构，计算IOU，Loss函数，激活函数(leaky_relu)

### 网络结构  build_network

用slim构建网络

![网络结构](https://i.loli.net/2020/10/23/IU2XlfY3TirNqgQ.png)

### 计算IOU **calc_iou**

是一个**计算两个bbox之间的IOU**的函数

### Loss函数 loss_layer

![image-20201025160647234](https://i.loli.net/2020/10/25/ELlpFVHZTUozYCI.png)

![image-20201025161138225](https://i.loli.net/2020/10/25/Z1NmX2H4ehGpMc7.png)

这句话表明了在第i个cell 中的第j个bounding_box是否有物体的表示方式。

![image-20201025161154350](https://i.loli.net/2020/10/25/nr9BcwTsXPF1gkM.png)

增加了边界框坐标预测的损失，减少了不包含对象的边界框坐标预测的损失。我们使用了两个参数，在代码中的定义为`noobject_scale` `coord_scale`

![image-20201025214023920](C:/Users/crazyjack/AppData/Roaming/Typora/typora-user-images/image-20201025214023920.png)![image-20201025214033667](C:/Users/crazyjack/AppData/Roaming/Typora/typora-user-images/image-20201025214033667.png)

#### 网络的输出信息提取

主要提取了：

1. predict_classes 预测每个格子目标的类别： 形状[batch_size,7,7,20]

2. predict_scales 预测每个格子中两个边界框的置信度 形状[batch_size,7,7,2]

3. predict_boxes 预测每个格子中的两个边界框, (x,y)表示边界框**相对于格子边界框**的中心

    w,h的开根号**相对于整个图片** 形状[batch_size,7,7,2,4]

#### label 中的信息提取

主要提取了：

1. response 标签的置信度,表示这个地方是否有框 形状[batch_size,7,7,1]
2. boxes 标签的边界框 (x,y)表示边界框**相对于整个图片的中心**,通过**除以image_size归一化**， 形状[batch_size,7,7,1,4]，张量沿着axis=3重复两边，扩充后[batch_size,7,7,2,4]
3. classes 分类信息 （7\*7\*20）

#### 获取需要的 predict_boxes_tran

接下的代码用于获取predict_boxes_tran的信息，即网络输出的预测框的信息，它是由网络的输出信息提取出来的网络的输出**predict_boxes**变换而来，其形式需要和label 中的boxes信息格式一致，即：

1. **相对于格子边界框**的中心  ->>  **相对于整个图片的中心**,通过**除以image_size归一化**
2. 宽和高是**相对于图片归一化**后的大小的**开方**  ->> 不开方

具体分析如下：

predict_boxes的输出是网络**前向传播后预测的候选框**，predict_boxes中的**前两位**，是predict_boxes**中心坐标**离**所属格子（response）左上角**的**坐标**。而predict_boxes中的**后两位**，是predict_boxes的宽度高度**相对于图片归一化**后的大小的**开方**。

这就需要将predict_boxes的中心坐标转换为相对于整张图来说的（x，y）中心坐标。
这里引入了offset_tran，其构造过程由self.offset逐步产生。

```python
(predict_boxes[..., 0] + offset) / self.cell_size,
(predict_boxes[..., 1] + offset_tran) / self.cell_size,
```
就是先将上面图中的x和y变成（x+offset x，y+offset y），然后除以cell_size=7,相当于对中心坐标进行了归一化，
```python
tf.square(predict_boxes[..., 2]), 
tf.square(predict_boxes[..., 3])],
```
就是将原来的宽度（归一化）的开方和高度（归一化）的开方恢复成：（宽度（归一化），高度（归一化）），那么predict_bbox中的坐标信息，全部通过这段代码，恢复成了和labels中坐标相同格式的了

![img](https://i.loli.net/2020/10/25/uSE4foU6IHO3TNZ.jpg)


##### offset_tran 的产生
其构造过程由self.offset逐步产生，self.offset的定义：
```python
# 3.reshape之后再转置，变成7*7*2的三维数组
self.offset = np.transpose(
    # 2.创建完成后reshape为2*7*7的三维数组
    np.reshape(
        # 1.创建 14*7的二维数组
        np.array([np.arange(self.cell_size)] * self.cell_size *
                    self.boxes_per_cell),
        (self.boxes_per_cell, self.cell_size, self.cell_size)),
    (1, 2, 0))
```
<img src="https://i.loli.net/2020/10/25/1s2VOugyjAYqb6G.png" alt="image-20201025170401225" style="zoom:50%;" />

然后offset变量为self.offset增加第0维度:batch_size后所得 ：
先reshape，由[7\*7\*2]变为[1\*7\*7\*2]
再tile复制成[batch_size\*7\*7\*2]
最后的offset_tran的定义为
```python
offset_tran = tf.transpose(offset, (0, 2, 1, 3))
```
忽略axis=0,其值为下图b：
<img src="https://i.loli.net/2020/10/25/4XICamltvobZLpT.png" alt="image-20201025172554187" style="zoom:50%;" />

#### 计算每个格子预测边界框与真实边界框之间的IOU
```python
iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)
```

#### 求论文中的1ijobj参数 object_mask noobject_mask

![img](https://i.loli.net/2020/10/25/VySQkr3NxFbgTKc.png)

1ijobj：表示网格单元i的第j个编辑框预测器’负责‘该预测 ，[batch_size,7,7,2] 
当格子中的确有目标时，取值为[1,0],[0,1]
比如某一个格子的值为[1,0]，表示第一个边界框负责该格子目标的预测  [0,1]：表示第二个边界框负责该格子目标的预测
当格子没有目标时，取值为[0,0]

即满足**两个以下条件**的对象

（1） 该对象属于的框是response框，**负责**检测物体

（2） 该对象是所属框中的，**与实际物体IOU比例较大**的那个

相应的位置为1，其余为0即可求得 1ijobj ![img](https://i.loli.net/2020/10/25/VySQkr3NxFbgTKc.png)

noobject_mask就表示每个边界框不负责该目标的置信度

![img](https://pic4.zhimg.com/80/v2-06102ee6c189c4012fdd89b9a2c010ef_720w.png)

#### boxes_tran 将标签中的boxes参数调整为方便loss计算形式
1. 中心的(x,y)由相对整个图像->相对当前格子
2. 长和宽开方

#### 计算各部分loss

##### 1. class_loss

![image-20201025212917214](https://i.loli.net/2020/10/25/uMfJPbUdovlN8cT.png)

计算类别的损失,如果目标出现在网格中 response为1，否则response为0  原文代价函数公式第5项.该项表明当格子中有目标时，预测的类别越接近实际类别，代价值越小。

##### 2. object_loss

 ![image-20201025212900468](https://i.loli.net/2020/10/25/AC3hXZHkej6Ysby.png)


有目标物体存在的置信度预测损失   原文代价函数公式第3项，该项表明当格子中有目标时，负责该目标预测的边界框的置信度越越接近预测的边界框与实际边界框之间的IOU时，代价值越小，有目标的时候，置信度损失函数

##### 3. noobject_loss

![image-20201025212847051](https://i.loli.net/2020/10/25/Ef7KBWcX6rl2pM8.png)

没有目标物体存在的置信度的损失(此时iou_predict_truth为0)  原文代价函数公式第4项
该项表名当格子中没有目标时，预测的两个边界框的置信度越接近0，代价值越小，没有目标的时候，置信度的损失函数

##### 4. coord_loss
边界框坐标损失 shape 为 [batch_size, 7, 7, 2, 1]  原文代价函数公式1,2项

![image-20201025212647375](https://i.loli.net/2020/10/25/T2optAP4kD91r3F.png)

该项表明当格子中有目标时，预测的边界框越接近实际边界框，代价值越小，只计算有目标的cell中iou最大的那个框的损失，即用这个iou最大的框来负责预测这个框，其它不管，乘以0

### 激活函数 leaky_relu

![image-20201025214250826](https://i.loli.net/2020/10/25/TD7pXFZ9vljIS1w.png)

```python
def leaky_relu(alpha):  #leaky_relu激活函数
    def op(inputs):
        return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
    return op
```

# 参考
1. https://www.cnblogs.com/sddai/p/10288074.html
2. https://zhuanlan.zhihu.com/p/89143061

