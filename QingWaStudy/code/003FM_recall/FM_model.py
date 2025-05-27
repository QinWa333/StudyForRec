import keras
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorboard.plugins.scalar.summary import scalar
from tensorflow.python.keras.utils.data_utils import next_sample


# 定义FM模型
class FM_layer(tf.keras.layers.Layer):
    def __init__(self,dim_v,w_reg,v_reg):
        super(FM_layer,self).__init__()
        self.v = None
        self.w = None
        self.w0 = None
        self.k = dim_v
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self,input_shape):
        self.w0 = self.add_weight(name='w0',shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w',shape=(input_shape[-1],1),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.w_reg)
                                 )
        self.v = self.add_weight(name='v',shape=(input_shape[-1],self.k),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.v_reg)
                                 )

    def call(self, inputs, *args, **kwargs):
        # 线性部分
        linear_part = tf.matmul(inputs, self.w) + self.w0

        # 交叉部分使用优化完的式子（优化过程小青蛙学习笔记）
        inter_part1 = tf.pow(tf.matmul(inputs, self.v), 2)
        inter_part2 = tf.matmul(tf.pow(inputs,2),tf.pow(self.v,2))
        inter_part = 0.5*tf.reduce_sum(inter_part1-inter_part2,axis=-1,keepdims=True) #reduce_sum 按axis维度做操作后 求和

        out_put = linear_part + inter_part
        return tf.nn.sigmoid(out_put)


# 定义FM模型
class FM(tf.keras.Model):
    def __init__(self,dim_v,w_reg,v_reg):
        super(FM,self).__init__()
        self.fm = FM_layer(dim_v,w_reg,v_reg)

    def call(self, inputs, training=None, mask=None):
        output = self.fm(inputs)
        return output



if __name__=='__main__':
    np.random.seed(666)
    tf.random.set_seed(666)

    n_samples = 1000
    n_features = 5

    X = np.random.randn(n_samples,n_features)

    scalar = StandardScaler()
    X = scalar.fit_transform(X)

    w = np.random.randn(n_features,1)*0.5
    v = np.random.randn(n_features, 8) * 0.5  # 隐向量维度为8

    # 计算线性部分
    linear_part = np.dot(X, w)

    # 计算交互部分 (使用原始公式)
    interaction_part = np.zeros((n_samples, 1))
    for i in range(n_samples):
        for j in range(n_features):
            for k in range(j + 1, n_features):
                interaction_part[i] += X[i, j] * X[i, k] * np.dot(v[j], v[k])
    interaction_part *= 0.5

    # 计算logits (添加偏置)
    bias = 0.5
    logits = linear_part + interaction_part + bias

    # 转换为概率并生成二分类标签
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs > 0.5).astype(int).reshape(-1)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = FM(dim_v=8,w_reg=0.01,v_reg=0.01)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss = 'binary_crossentropy',
        metrics=['accuracy']
    )

    # 训练
    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    # 评估模型
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n测试集准确率: {test_acc:.4f}")
