# zzp 20250512
# FM用与召回时，只使用了二阶交叉部分
# 将用户特征隐向量相加作为user_embedding
# 将物品特征隐向量相加作为item_embedding
# 该数据集特征比较少，估计效果不好，当作学习尝试一下


import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Input, Embedding, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import numpy as np


class Similarity(Layer):
    """
    相似度计算层
    """
    def __init__(self, type='cos', **kwargs):
        self.type = type
        super(Similarity, self).__init__(**kwargs)

    def call(self, inputs):
        user_vec, item_vec = inputs
        # 确保向量是2D的
        user_vec = tf.expand_dims(user_vec, axis=1)  # [batch_size, 1, dim]
        item_vec = tf.expand_dims(item_vec, axis=1)  # [batch_size, 1, dim]

        if self.type == "cos":
            # 修改为 axis=2，因为现在是 [batch_size, 1, dim] 的形状
            user_vec = tf.nn.l2_normalize(user_vec, axis=2)
            item_vec = tf.nn.l2_normalize(item_vec, axis=2)
            # 计算点积并保持维度
            return tf.reduce_sum(user_vec * item_vec, axis=2, keepdims=True)  # [batch_size, 1, 1]
        else:
            return tf.reduce_sum(user_vec * item_vec, axis=2, keepdims=True)  # [batch_size, 1, 1]

    def compute_output_shape(self, input_shape):
        return (None, 1, 1)


class PredictionLayer(Layer):
    """
    预测层
    """
    def __init__(self, task='binary', use_bias=True, **kwargs):
        self.task = task
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.use_bias:
            self.global_bias = self.add_weight(
                shape=(1,), initializer=tf.keras.initializers.Zeros(),
                name="global_bias")
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x, self.global_bias)
        if self.task == "binary":
            x = tf.sigmoid(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class FMRecallModel(Model):
    def __init__(self, user_feature_configs, item_feature_configs, l2_reg_embedding=1e-6,
                 seed=1024, metric='cos', **kwargs):
        super(FMRecallModel, self).__init__(**kwargs)
        self.user_feature_configs = user_feature_configs
        self.item_feature_configs = item_feature_configs
        self.l2_reg_embedding = l2_reg_embedding
        self.seed = seed
        self.metric = metric

        # 创建embedding层
        self.embedding_layers = self._create_embedding_layers()

        # 构建输入层
        self.user_features = self._build_input_features(user_feature_configs)
        self.item_features = self._build_input_features(item_feature_configs)

        # 构建模型层
        self.similarity = Similarity(type=metric)
        self.prediction = PredictionLayer("binary", False)

    def _create_embedding_layers(self):
        """创建embedding层"""
        embedding_layers = {}
        for feat_config in self.user_feature_configs + self.item_feature_configs:
            embedding_layers[feat_config['name']] = Embedding(
                input_dim=feat_config['num_buckets'],
                output_dim=feat_config['dimension'],
                embeddings_regularizer=l2(self.l2_reg_embedding),
                mask_zero=True,
                name=f'embedding_{feat_config["name"]}'
            )
        return embedding_layers

    def _build_input_features(self, feature_configs):
        """构建输入特征"""
        input_features = {}
        for config in feature_configs:
            input_features[config['name']] = Input(shape=(1,), name=config['name'], dtype=tf.int32)
        return input_features

    def call(self, inputs):
        # 分离用户和物品输入
        user_inputs = {k: inputs[k] for k in self.user_features.keys()}
        item_inputs = {k: inputs[k] for k in self.item_features.keys()}

        # 处理用户特征
        user_embeddings = []
        for feat_config in self.user_feature_configs:
            # 确保输入是二维的
            input_tensor = tf.reshape(user_inputs[feat_config['name']], [-1, 1])
            embedding = self.embedding_layers[feat_config['name']](input_tensor)
            user_embeddings.append(embedding)
        user_dnn_input = tf.concat(user_embeddings, axis=1)
        user_vector = tf.reduce_sum(user_dnn_input, axis=1)  # [batch_size, dim]

        # 处理物品特征
        item_embeddings = []
        for feat_config in self.item_feature_configs:
            # 确保输入是二维的
            input_tensor = tf.reshape(item_inputs[feat_config['name']], [-1, 1])
            embedding = self.embedding_layers[feat_config['name']](input_tensor)
            item_embeddings.append(embedding)
        item_dnn_input = tf.concat(item_embeddings, axis=1)
        item_vector = tf.reduce_sum(item_dnn_input, axis=1)  # [batch_size, dim]

        # 计算相似度
        score = self.similarity([user_vector, item_vector])  # [batch_size, 1, 1]

        # 压缩维度
        score = tf.squeeze(score, axis=2)  # [batch_size, 1]

        # 输出层
        output = self.prediction(score)  # [batch_size, 1]

        return output

    def get_user_embedding(self, user_inputs):
        """获取用户embedding，用于预测时使用"""
        user_embeddings = []
        for feat_config in self.user_feature_configs:
            # 确保输入是二维的
            input_tensor = tf.reshape(user_inputs[feat_config['name']], [-1, 1])
            embedding = self.embedding_layers[feat_config['name']](input_tensor)
            user_embeddings.append(embedding)

        # 拼接所有特征的embedding
        user_dnn_input = tf.concat(user_embeddings, axis=1)
        print(f"Concatenated embeddings shape: {user_dnn_input.shape}")

        # 对每个特征的embedding分别求和，然后拼接
        user_vector = tf.reduce_sum(user_dnn_input, axis=1)
        print(f"User vector shape before reshape: {user_vector.shape}")

        # 确保返回的是二维数组，形状为 [batch_size, embedding_dim]
        return tf.reshape(user_vector, [-1, self.user_feature_configs[0]['dimension']])

    def get_item_embedding(self, item_inputs):
        """获取物品embedding，用于保存到faiss"""
        item_embeddings = []
        for feat_config in self.item_feature_configs:
            # 确保输入是二维的
            input_tensor = tf.reshape(item_inputs[feat_config['name']], [-1, 1])
            embedding = self.embedding_layers[feat_config['name']](input_tensor)
            item_embeddings.append(embedding)

        # 拼接所有特征的embedding
        item_dnn_input = tf.concat(item_embeddings, axis=1)

        # 对每个特征的embedding分别求和，然后拼接
        item_vector = tf.reduce_sum(item_dnn_input, axis=1)

        # 确保返回的是二维数组，形状为 [n_items, embedding_dim]
        return tf.reshape(item_vector, [-1, self.item_feature_configs[0]['dimension']])


def create_fm_recall_model():
    # 定义用户特征配置
    user_feature_configs = [
        {
            'name': 'user_id',
            'num_buckets': 10000,
            'dimension': 16
        },
        {
            'name': 'age',
            'num_buckets': 100,
            'dimension': 16
        },
        {
            'name': 'gender',
            'num_buckets': 2,
            'dimension': 16
        }
    ]

    # 定义物品特征配置
    item_feature_configs = [
        {
            'name': 'item_id',
            'num_buckets': 50000,
            'dimension': 16
        },
        {
            'name': 'category_id',
            'num_buckets': 1000,
            'dimension': 16
        },
        {
            'name': 'brand_id',
            'num_buckets': 500,
            'dimension': 16
        }
    ]

    # 创建模型
    model = FMRecallModel(
        user_feature_configs=user_feature_configs,
        item_feature_configs=item_feature_configs,
        l2_reg_embedding=1e-6,
        seed=1024,
        metric='cos'
    )

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model