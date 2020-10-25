import numpy as np
import tensorflow as tf
import yolo.config as cfg
import tf_slim as slim
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


#定义YoloNet网络
class YOLONet(object):
    def __init__(self, is_training=True):  #类的初始化
        self.classes = cfg.CLASSES  #PASCAL VOC数据集的20个数据类
        self.num_class = len(self.classes)  #20个类别
        self.image_size = cfg.IMAGE_SIZE  #图片的大小
        self.cell_size = cfg.CELL_SIZE  #整张输入图片划分为cell_size * cell_size的网格
        self.boxes_per_cell = cfg.BOXES_PER_CELL  #每个cell负责预测多少个(mayebe 2)bounding box
        self.output_size = (self.cell_size * self.cell_size) *\
            (self.num_class + self.boxes_per_cell * 5)    #最后输出的tensor大小，其为S*S*(C+5*B),具体可以看论文
        self.scale = 1.0 * self.image_size / self.cell_size  #每个cell像素的大小
        self.boundary1 = self.cell_size * self.cell_size * self.num_class  #类似于7*7*20
        self.boundary2 = self.boundary1 +\
            self.cell_size * self.cell_size * self.boxes_per_cell  #类似于 7*7*20 + 7*7*2

        self.object_scale = cfg.OBJECT_SCALE  #这些是论文中涉及的参数，具体可看论文(You Only Look Once: Unified, Real-Time Object Detection)
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE

        self.learning_rate = cfg.LEARNING_RATE  #学习率
        self.batch_size = cfg.BATCH_SIZE  #batch_size
        self.alpha = cfg.ALPHA  #

        # 3.reshape之后再转置，变成7*7*2的三维数组
        self.offset = np.transpose(
            # 2.创建完成后reshape为2*7*7的三维数组
            np.reshape(
                # 1.创建 14*7的二维数组
                np.array([np.arange(self.cell_size)] * self.cell_size *
                         self.boxes_per_cell),
                (self.boxes_per_cell, self.cell_size, self.cell_size)),
            (1, 2, 0))

        self.images = tf.compat.v1.placeholder(
            tf.float32, [None, self.image_size, self.image_size, 3],
            name='images'
        )  #定义输入的placeholder，需要喂饱的数据, batch_size * 448 * 448 *3 ,
        self.logits = self.build_network(
            self.images,
            num_outputs=self.output_size,
            alpha=self.alpha,
            is_training=is_training
        )  # 构建网络，预测值，在本程序中，其格式为 [batch_size , 7 * 7 * （20 + 2 * 5）]，其中的20表示PASCAL VOC数据集的20个类别

        if is_training:  #training为true时
            self.labels = tf.compat.v1.placeholder(
                tf.float32,
                [None, self.cell_size, self.cell_size, 5 + self.num_class
                 ])  # 需要喂饱的数据，在本程序中其格式为：[batch_size,7 ,7 ,25]
            self.loss_layer(self.logits, self.labels)  #预测值和真实值的比较，得到loss
            self.total_loss = tf.compat.v1.losses.get_total_loss()  #将所有的loss求和
            tf.compat.v1.summary.scalar('total_loss', self.total_loss)

    def build_network(
            self,  #用slim构建网络，简单高效
            images,
            num_outputs,
            alpha,
            keep_prob=0.5,
            is_training=True,
            scope='yolo'):
        with tf.compat.v1.variable_scope(scope):
            # with 限制作用域于此with,并且在进行with内部操作前后分别加上__entry__ and __exit__
            #定义变量命名空间
            with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],  #卷积层加上全连接层
                    activation_fn=leaky_relu(alpha),  #用的是leaky_relu激活函数
                    weights_regularizer=tf.keras.regularizers.l2(
                        0.5 * (0.0005)),  #L2正则化，防止过拟合
                    weights_initializer=tf.compat.v1.
                    truncated_normal_initializer(0.0, 0.01)  #权重初始化
            ):

                #这里先执行填充操作
                # t = [[2, 3, 4], [5, 6, 7]], paddings = [[1, 1], [2, 2]]，mode = "CONSTANT"
                #
                # 那么sess.run(tf.pad(t, paddings, "CONSTANT"))
                # 的输出结果为：
                #
                # array([[0, 0, 0, 0, 0, 0, 0],
                #        [0, 0, 2, 3, 4, 0, 0],
                #        [0, 0, 5, 6, 7, 0, 0],
                #        [0, 0, 0, 0, 0, 0, 0]], dtype=int32)
                #
                # 可以看到，上，下，左，右分别填充了1, 1, 2, 2
                # 行刚好和paddings = [[1, 1], [2, 2]]
                # 相等，零填充
                #因为这里有4维，batch和channel维没有填充，只填充了image_height,image_width这两个维度，0填充
                #pad_1 填充 454x454x3
                net = tf.pad(tensor=images,
                             paddings=np.array([[0, 0], [3, 3], [3, 3], [0,
                                                                         0]]),
                             name='pad_1')

                logging.info('Layer pad_1  {0}'.format(net.shape))
                #卷积层conv_2 s=2 对于padding 方式为 VALID，输出的形状计算为(n-f+1)/s向上取整 224x224x64
                net = slim.conv2d(net,
                                  64,
                                  7,
                                  2,
                                  padding='VALID',
                                  scope='conv_2')
                logging.info('Layer conv_2 {0}'.format(net.shape))
                #池化层pool_3 112x112x64
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                logging.info('Layer pool_3 {0}'.format(net.shape))

                #卷积层conv_4、3x3x192 s=1  对于padding 方式为 SAME，输出的形状计算如下： n/s向上取整 112x112x192
                net = slim.conv2d(net, 192, 3, scope='conv_4')
                logging.info('Layer conv_4 {0}'.format(net.shape))

                #池化层pool_5 56x56x192
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                logging.info('Layer pool_5 {0}'.format(net.shape))

                #卷积层conv_6、1x1x128 s=1  n/s向上取整  56x56x128
                net = slim.conv2d(net, 128, 1, scope='conv_6')
                logging.info('Layer conv_6 {0}'.format(net.shape))

                #卷积层conv_7、3x3x256 s=1  n/s向上取整 56x56x256
                net = slim.conv2d(net, 256, 3, scope='conv_7')
                logging.info('Layer conv_7 {0}'.format(net.shape))

                #卷积层conv_8、1x1x256 s=1  n/s向上取整 56x56x256
                net = slim.conv2d(net, 256, 1, scope='conv_8')
                logging.info('Layer conv_8 {0}'.format(net.shape))
                #卷积层conv_9、3x3x512 s=1  n/s向上取整 56x56x512
                net = slim.conv2d(net, 512, 3, scope='conv_9')
                logging.info('Layer conv_9 {0}'.format(net.shape))
                #池化层pool_10 28x28x512
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                logging.info('Layer pool_10 {0}'.format(net.shape))
                #卷积层conv_11、1x1x256 s=1  n/s向上取整 28x28x256
                net = slim.conv2d(net, 256, 1, scope='conv_11')
                logging.info('Layer conv_11 {0}'.format(net.shape))
                #卷积层conv_12、3x3x512 s=1  n/s向上取整 28x28x512
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                logging.info('Layer conv_12 {0}'.format(net.shape))
                #卷积层conv_13、1x1x256 s=1  n/s向上取整 28x28x256
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                logging.info('Layer conv_13 {0}'.format(net.shape))
                #卷积层conv_14、3x3x512 s=1  n/s向上取整 28x28x512
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                logging.info('Layer conv_14 {0}'.format(net.shape))
                #卷积层conv_15、1x1x256 s=1  n/s向上取整 28x28x256
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                logging.info('Layer conv_15 {0}'.format(net.shape))
                #卷积层conv_16、3x3x512 s=1  n/s向上取整 28x28x512
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                logging.info('Layer conv_16 {0}'.format(net.shape))
                #卷积层conv_17、1x1x256 s=1  n/s向上取整 28x28x256
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                logging.info('Layer conv_17 {0}'.format(net.shape))
                #卷积层conv_18、3x3x512 s=1  n/s向上取整 28x28x512
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                logging.info('Layer conv_18 {0}'.format(net.shape))
                #卷积层conv_19、1x1x512 s=1  n/s向上取整 28x28x512
                net = slim.conv2d(net, 512, 1, scope='conv_19')
                logging.info('Layer conv_19 {0}'.format(net.shape))
                #卷积层conv_20、3x3x1024 s=1  n/s向上取整 28x28x1024
                net = slim.conv2d(net, 1024, 3, scope='conv_20')
                logging.info('Layer conv_20 {0}'.format(net.shape))
                #池化层pool_21 14x14x1024
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                logging.info('Layer pool_21 {0}'.format(net.shape))
                #卷积层conv_22、1x1x512 s=1  n/s向上取整 14x14x512
                net = slim.conv2d(net, 512, 1, scope='conv_22')
                logging.info('Layer conv_22 {0}'.format(net.shape))
                #卷积层conv_23、3x3x1024 s=1  n/s向上取整 14x14x1024
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                logging.info('Layer conv_23 {0}'.format(net.shape))
                #卷积层conv_24、1x1x512 s=1  n/s向上取整 14x14x512
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                logging.info('Layer conv_24 {0}'.format(net.shape))
                #卷积层conv_25、3x3x1024 s=1  n/s向上取整 14x14x1024
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                logging.info('Layer conv_25 {0}'.format(net.shape))

                #卷积层conv_26、3x3x1024 s=1  n/s向上取整 14x14x1024
                net = slim.conv2d(net, 1024, 3, scope='conv_26')
                logging.info('Layer conv_26 {0}'.format(net.shape))

                #pad_27 填充 16x16x2014
                net = tf.pad(net,
                             np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),
                             name='pad_27')
                logging.info('Layer pad_27 {0}'.format(net.shape))

                #卷积层conv_28、3x3x1024 s=2  (n-f+1)/s向上取整 7x7x1024
                net = slim.conv2d(net,
                                  1024,
                                  3,
                                  2,
                                  padding='VALID',
                                  scope='conv_28')

                logging.info('Layer conv_28 {0}'.format(net.shape))
                #卷积层conv_29、3x3x1024 s=1  n/s向上取整 7x7x1024
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                logging.info('Layer conv_29 {0}'.format(net.shape))
                #卷积层conv_30、3x3x1024 s=1  n/s向上取整 7x7x1024
                net = slim.conv2d(net, 1024, 3, scope='conv_30')
                logging.info('Layer conv_30 {0}'.format(net.shape))
                #trans_31 转置[None,1024,7,7]
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                logging.info('Layer trans_31 {0}'.format(net.shape))
                #flat_32 展开 50176
                net = slim.flatten(net, scope='flat_32')
                logging.info('Layer flat_32 {0}'.format(net.shape))
                #全连接层fc_33  512
                net = slim.fully_connected(net, 512, scope='fc_33')
                logging.info('Layer fc_33 {0}'.format(net.shape))
                #全连接层fc_34  4096
                net = slim.fully_connected(net, 4096, scope='fc_34')
                logging.info('Layer fc_34 {0}'.format(net.shape))
                #弃权层dropout_35 4096
                net = slim.dropout(net,
                                   keep_prob=keep_prob,
                                   is_training=is_training,
                                   scope='dropout_35')
                logging.info('Layer dropout_35 {0}'.format(net.shape))
                #全连接层fc_36 1470
                net = slim.fully_connected(net,
                                           num_outputs,
                                           activation_fn=None,
                                           scope='fc_36')
                logging.info('Layer fc_36 {0}'.format(net.shape))
        return net

    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
        这个函数的主要作用是计算两个 bounding box 之间的 IoU。输入是两个 5 维的bounding box,输出的两个 bounding Box 的IoU
         
        Args:         
          boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
          注意这里的参数x_center, y_center, w, h都是归一到[0,1]之间的，分别表示预测边界框的中心相对整张图片的坐标，宽和高
        Return:
          iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):
            # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
            #把以前的中心点坐标和长和宽转换成了左上角和右下角的两个点的坐标
            boxes1_t = tf.stack(
                [
                    boxes1[..., 0] - boxes1[..., 2] / 2.0,  #左上角x
                    boxes1[..., 1] - boxes1[..., 3] / 2.0,  #左上角y
                    boxes1[..., 0] + boxes1[..., 2] / 2.0,  #右下角x
                    boxes1[..., 1] + boxes1[..., 3] / 2.0
                ],  #右下角y
                axis=-1)

            boxes2_t = tf.stack([
                boxes2[..., 0] - boxes2[..., 2] / 2.0, boxes2[..., 1] -
                boxes2[..., 3] / 2.0, boxes2[..., 0] + boxes2[..., 2] / 2.0,
                boxes2[..., 1] + boxes2[..., 3] / 2.0
            ],
                                axis=-1)

            # calculate the left up point & right down point
            #lu和rd就是分别求两个框相交的矩形的左上角的坐标和右下角的坐标，因为对于左上角，
            #选择的是x和y较大的，而右下角是选择较小的，可以想想两个矩形框相交是不是这中情况
            lu = tf.maximum(boxes1_t[..., :2],
                            boxes2_t[..., :2])  #两个框相交的矩形的左上角(x1,y1)
            rd = tf.minimum(boxes1_t[..., 2:],
                            boxes2_t[..., 2:])  #两个框相交的矩形的右下角(x2,y2)

            # intersection 这个就是求相交矩形的长和宽，所以有rd-ru，相当于x1-x2和y1-y2，
            #之所以外面还要加一个tf.maximum是因为删除那些不合理的框，比如两个框没交集，
            #就会出现左上角坐标比右下角还大。
            intersection = tf.maximum(0.0, rd - lu)
            #inter_square这个就是求面积了，就是长乘以宽。
            inter_square = intersection[..., 0] * intersection[..., 1]

            # calculate the boxs1 square and boxs2 square
            #square1和square2这个就是求面积了，因为之前是中心点坐标和长和宽，所以这里直接用长和宽
            square1 = boxes1[..., 2] * boxes1[..., 3]
            square2 = boxes2[..., 2] * boxes2[..., 3]

            #union_square就是就两个框的交面积，因为如果两个框的面积相加，那就会重复了相交的部分，
            #所以减去相交的部分，外面有个tf.maximum这个就是保证相交面积不为0,因为后面要做分母。
            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        #最后有一个tf.clip_by_value,这个是将如果你的交并比大于1,那么就让它等于1,如果小于0,那么就
        #让他变为0,因为交并比在0-1之间。
        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def loss_layer(self, predicts, labels, scope='loss_layer'):
        '''
            计算预测和标签之间的损失函数
            
            args：
                predicts：Yolo网络的输出 形状[None,1470] 
                        0：7*7*20：表示预测类别  
                        7*7*20:7*7*20 + 7*7*2:表示预测置信度，即预测的边界框与实际边界框之间的IOU
                        7*7*20 + 7*7*2：1470：预测边界框    目标中心是相对于当前格子的，宽度和高度的开根号是相对当前整张图像的(归一化的)
                labels：标签值 形状[None,7,7,25]
                        0:1：置信度，表示这个地方是否有目标
                        1:5：目标边界框  目标中心，宽度和高度(没有归一化)
                        5:25：目标的类别
            '''
        with tf.variable_scope(scope):
            '''
            网络的输出信息提取：将网络输出分离为类别和置信度以及边界框的大小
            网络的输出维度为7*7*20(类别) + 7*7*2(bounding box的置信度) + 7*7*2*4(bounding box 位置信息)=1470            
            '''
            #1. 预测每个格子目标的类别 形状[45,7,7,20],其中45为batch_size
            predict_classes = tf.reshape(predicts[:, :self.boundary1], [
                self.batch_size, self.cell_size, self.cell_size, self.num_class
            ])
            #2. 预测每个格子中两个边界框的置信度 形状[45,7,7,2]
            predict_scales = tf.reshape(
                predicts[:, self.boundary1:self.boundary2], [
                    self.batch_size, self.cell_size, self.cell_size,
                    self.boxes_per_cell
                ])
            #3. 预测每个格子中的两个边界框, (x,y)表示边界框相对于格子边界框的中心 w,h的开根号相对于整个图片  形状[45,7,7,2,4]
            predict_boxes = tf.reshape(predicts[:, self.boundary2:], [
                self.batch_size, self.cell_size, self.cell_size,
                self.boxes_per_cell, 4
            ])



            '''
            label 中的信息提取：标签的置信度信息，标签的边界框坐标,分类信息
            '''
            #标签的置信度,表示这个地方是否有框 形状[45,7,7,1]
            response = tf.reshape(
                labels[..., 0],
                [self.batch_size, self.cell_size, self.cell_size, 1])

            #标签的边界框 (x,y)表示边界框相对于整个图片的中心 形状[45,7,7,1,4]
            boxes = tf.reshape(
                labels[..., 1:5],
                [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            #标签的边界框 通过除以image_size归一化后 张量沿着axis=3重复两边，扩充后[45,7,7,2,4]
            boxes = tf.tile(
                boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size

            # 分类信息 （7*7*20）
            classes = labels[..., 5:]



            '''
                predict_boxes_tran：
                offset变量用于把预测边界框predict_boxes中的坐标中心(x,y)由相对当前格子转换为相对当前整个图片        
                offset，这个是构造的[1,7,7,2]矩阵，这里忽略axis=0，则每一行都是[7,2]的矩阵，值为[[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6]]
                这个变量是为了将每个cell的坐标对齐，后一个框比前一个框要多加1
                比如我们预测了cell_size的每个中心点坐标，那么我们这个中心点落在第几个cell_size
                就对应坐标要加几，这个用法比较巧妙，构造了这样一个数组，让他们对应位置相加
                '''
            #将self.offset [7*7*2]reshape成offset 其shape为[1,7,7,2] 如果忽略axis=0，则每一行都是  [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6]]
            offset = tf.reshape(
                tf.constant(self.offset, dtype=tf.float32),
                [1, self.cell_size, self.cell_size, self.boxes_per_cell])
            #再利用tile复制成shape为[45,7,7,2]
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
            #shape为[45,7,7,2]  如果忽略axis=0 第i行为[[i,i],[i,i],[i,i],[i,i],[i,i],[i,i],[i,i]]
            offset_tran = tf.transpose(offset, (0, 2, 1, 3))
            #shape为[45,7,7,2,4]  计算每个格子中的预测边界框坐标(x,y)相对于整个图片的位置  而不是相对当前格子
            #假设当前格子为(3,3)，当前格子的预测边界框为(x0,y0)，则计算坐标(x,y) = ((x0,y0)+(3,3))/7
            #offset_tran如下，只不过batch_size=1
            # [[[[0. 0.]
            # [0. 0.]
            # [0. 0.]
            # [0. 0.]
            # [0. 0.]
            # [0. 0.]
            # [0. 0.]]
            #
            # [[1. 1.]
            #  [1. 1.]
            # [1. 1.]
            # [1. 1.]
            # [1. 1.]
            # [1. 1.]
            # [1. 1.]]
            #
            # [[2. 2.]
            #  [2. 2.]
            # [2. 2.]
            # [2. 2.]
            # [2. 2.]
            # [2. 2.]
            # [2. 2.]]
            #
            # [[3. 3.]
            #  [3. 3.]
            # [3. 3.]
            # [3. 3.]
            # [3. 3.]
            # [3. 3.]
            # [3. 3.]]
            #
            # [[4. 4.]
            #  [4. 4.]
            # [4. 4.]
            # [4. 4.]
            # [4. 4.]
            # [4. 4.]
            # [4. 4.]]
            #
            # [[5. 5.]
            #  [5. 5.]
            # [5. 5.]
            # [5. 5.]
            # [5. 5.]
            # [5. 5.]
            # [5. 5.]]
            #
            # [[6. 6.]
            #  [6. 6.]
            # [6. 6.]
            # [6. 6.]
            # [6. 6.]
            # [6. 6.]
            # [6. 6.]]]]

            # 计算每个格子中的预测边界框坐标(x,y)相对于整个图片的位置  而不是相对当前格子
            # 假设当前格子为(3,3)，当前格子的预测边界框为(x0,y0)，则计算坐标(x,y) = ((x0,y0)+(3,3))/7
            predict_boxes_tran = tf.stack(  #相对于整张特征图来说，找到相对于特征图大小的中心点，和宽度以及高度的开方， 其格式为[batch_size, 7, 7, 2, 4]
                [
                    (predict_boxes[..., 0] + offset) / self.cell_size,  # x
                    (predict_boxes[..., 1] + offset_tran) / self.cell_size,  #y
                    tf.square(predict_boxes[..., 2]),  #宽度的平方，和论文中的开方对应，具体请看论文
                    tf.square(predict_boxes[..., 3])  #高度的平方
                ],
                axis=-1)  #高度的平方，

            #计算每个格子预测边界框与真实边界框之间的IOU,  其格式为： [batch_size, 7, 7, 2]
            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            #这个是求论文中的1ijobj参数，[45,7,7,2]     1ijobj：表示网格单元i的第j个编辑框预测器’负责‘该预测
            #先计算每个框交并比最大的那个，因为我们知道，YOLO每个格子预测两个边界框，一个类别。在训练时，每个目标只需要
            #一个预测器来负责，我们指定一个预测器"负责"，根据哪个预测器与真实值之间具有当前最高的IOU来预测目标。
            #所以object_mask就表示每个格子中的哪个边界框负责该格子中目标预测？哪个边界框取值为1，哪个边界框就负责目标预测
            #当格子中的确有目标时，取值为[1,1]，[1,0],[0,1]
            #比如某一个格子的值为[1,0]，表示第一个边界框负责该格子目标的预测  [0,1]：表示第二个边界框负责该格子目标的预测
            #当格子没有目标时，取值为[0,0]
            object_mask = tf.reduce_max(
                input_tensor=iou_predict_truth, axis=3, keepdims=True
            )  # Computes the maximum of elements across dimensions of a tensor, 在第四个维度上，维度从0开始算
            object_mask = tf.cast(
                (iou_predict_truth >= object_mask), tf.float32
            ) * response  #其维度为[batch_size, 7, 7, 2]  , 如果cell中真实有目标，那么该cell内iou最大的那个框的相应位置为1（就是负责预测该框），其余为0

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            # noobject_mask就表示每个边界框不负责该目标的置信度，
            # 使用tf.onr_like，使得全部为1,再减去有目标的，也就是有目标的对应坐标为1,这样一减，就变为没有的了。[45,7,7,2]
            noobject_mask = tf.ones_like(
                object_mask, dtype=tf.float32
            ) - object_mask  #其维度为[batch_size, 7 , 7, 2]， 真实没有目标的区域都为1，真实有目标的区域为0


            # boxes_tran 这个就是把之前的坐标换回来(相对整个图像->相对当前格子)，长和宽开方(原因在论文中有说明)
            # 后面求loss就方便。 shape为(4, 45, 7, 7, 2)
            boxes_tran = tf.stack(  #stack这是一个矩阵拼接的操作， 得到x_center, y_center相对于该cell左上角的偏移值， 宽度和高度是相对于整张图片的比例
                [
                    boxes[..., 0] * self.cell_size - offset,
                    boxes[..., 1] * self.cell_size - offset_tran,
                    tf.sqrt(boxes[..., 2]),  #宽度开方，和论文对应
                    tf.sqrt(boxes[..., 3])
                ],
                axis=-1)  #高度开方，和论文对应



            ################################# 开始计算loss ############################
            # class_loss, 计算类别的损失
            # 如果目标出现在网格中 response为1，否则response为0  原文代价函数公式第5项
            #该项表明当格子中有目标时，预测的类别越接近实际类别，代价值越小  原文代价函数公式第5项
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(  #平方差损失函数
                input_tensor=tf.reduce_sum(input_tensor=tf.square(class_delta),
                                           axis=[1, 2, 3]),
                name='class_loss'
            ) * self.class_scale  # self.class_scale为损失函数前面的系数


            # object_loss 有目标物体存在的置信度预测损失   原文代价函数公式第3项
            #该项表名当格子中有目标时，负责该目标预测的边界框的置信度越越接近预测的边界框与实际边界框之间的IOU时，代价值越小
            # 有目标的时候，置信度损失函数
            object_delta = object_mask * (
                predict_scales - iou_predict_truth
            )  #用iou_predict_truth替代真实的置信度，真的妙，佩服的5体投递
            object_loss = tf.reduce_mean(  #平方差损失函数
                input_tensor=tf.reduce_sum(
                    input_tensor=tf.square(object_delta), axis=[1, 2, 3]),
                name='object_loss') * self.object_scale


            #noobject_loss  没有目标物体存在的置信度的损失(此时iou_predict_truth为0)  原文代价函数公式第4项
            #该项表名当格子中没有目标时，预测的两个边界框的置信度越接近0，代价值越小
            # 没有目标的时候，置信度的损失函数
            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(  #平方差损失函数
                input_tensor=tf.reduce_sum(
                    input_tensor=tf.square(noobject_delta), axis=[1, 2, 3]),
                name='noobject_loss') * self.noobject_scale


            # coord_loss 边界框坐标损失 shape 为 [batch_size, 7, 7, 2, 1]  原文代价函数公式1,2项
            #该项表名当格子中有目标时，预测的边界框越接近实际边界框，代价值越小
            # 框坐标的损失，只计算有目标的cell中iou最大的那个框的损失，即用这个iou最大的框来负责预测这个框，其它不管，乘以0
            coord_mask = tf.expand_dims(
                object_mask, 4
            )  # object_mask其维度为：[batch_size, 7, 7, 2]， 扩展维度之后变成[batch_size, 7, 7, 2, 1]
            boxes_delta = coord_mask * (
                predict_boxes - boxes_tran
            )  #predict_boxes维度为： [batch_size, 7, 7, 2, 4]，这些框的坐标都是偏移值
            coord_loss = tf.reduce_mean(  #平方差损失函数
                input_tensor=tf.reduce_sum(input_tensor=tf.square(boxes_delta),
                                           axis=[1, 2, 3, 4]),
                name='coord_loss') * self.coord_scale


            #将各个损失总结起来
            tf.compat.v1.losses.add_loss(class_loss)
            tf.compat.v1.losses.add_loss(object_loss)
            tf.compat.v1.losses.add_loss(noobject_loss)
            tf.compat.v1.losses.add_loss(coord_loss)

            # 将每个损失添加到日志记录
            tf.compat.v1.summary.scalar('class_loss', class_loss)
            tf.compat.v1.summary.scalar('object_loss', object_loss)
            tf.compat.v1.summary.scalar('noobject_loss', noobject_loss)
            tf.compat.v1.summary.scalar('coord_loss', coord_loss)

            tf.compat.v1.summary.histogram('boxes_delta_x', boxes_delta[...,
                                                                        0])
            tf.compat.v1.summary.histogram('boxes_delta_y', boxes_delta[...,
                                                                        1])
            tf.compat.v1.summary.histogram('boxes_delta_w', boxes_delta[...,
                                                                        2])
            tf.compat.v1.summary.histogram('boxes_delta_h', boxes_delta[...,
                                                                        3])
            tf.compat.v1.summary.histogram('iou', iou_predict_truth)


def leaky_relu(alpha):  #leaky_relu激活函数
    def op(inputs):
        return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')

    return op
