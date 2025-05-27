import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from FM_forRecall import create_fm_recall_model
import faiss


# 生成测试数据
def generate_test_data(n_samples=1000):
    np.random.seed(42)

    # 生成用户特征
    user_ids = np.random.randint(0, 1000, n_samples)
    ages = np.random.randint(0, 100, n_samples)
    genders = np.random.randint(0, 2, n_samples)

    # 生成物品特征
    item_ids = np.random.randint(0, 5000, n_samples)
    category_ids = np.random.randint(0, 100, n_samples)
    brand_ids = np.random.randint(0, 50, n_samples)

    # 生成标签
    labels = np.zeros(n_samples)
    for i in range(n_samples):
        if (user_ids[i] % 10) == (item_ids[i] % 10):
            labels[i] = np.random.choice([0, 1], p=[0.3, 0.7])
        else:
            labels[i] = np.random.choice([0, 1], p=[0.7, 0.3])

    # 创建数据框
    data = pd.DataFrame({
        'user_id': user_ids,
        'age': ages,
        'gender': genders,
        'item_id': item_ids,
        'category_id': category_ids,
        'brand_id': brand_ids,
        'label': labels
    })

    # 划分训练集和测试集
    train_data, test_data, train_labels, test_labels = train_test_split(
        data.drop('label', axis=1),
        data['label'],
        test_size=0.2,
        random_state=42
    )

    return train_data, test_data, train_labels, test_labels


def create_faiss_index(item_embeddings, dimension=16):
    """创建faiss索引"""
    # 确保item_embeddings是numpy数组
    if isinstance(item_embeddings, tf.Tensor):
        item_embeddings = item_embeddings.numpy()

    # 确保item_embeddings是二维数组
    if len(item_embeddings.shape) == 1:
        item_embeddings = np.expand_dims(item_embeddings, axis=0)

    # 获取实际的embedding维度
    actual_dim = item_embeddings.shape[1]
    print(f"Creating FAISS index with dimension: {actual_dim}")
    print(f"Input embeddings shape: {item_embeddings.shape}")

    # 创建索引，使用实际的embedding维度
    index = faiss.IndexFlatIP(actual_dim)  # 使用内积作为相似度度量
    index.add(item_embeddings)  # 添加物品embedding
    return index


def main():
    # 生成测试数据
    train_data, test_data, train_labels, test_labels = generate_test_data()
    # 准备模型输入
    train_inputs = {
        'user_id': train_data['user_id'].values.reshape(-1, 1),  # 确保是二维的
        'age': train_data['age'].values.reshape(-1, 1),
        'gender': train_data['gender'].values.reshape(-1, 1),
        'item_id': train_data['item_id'].values.reshape(-1, 1),
        'category_id': train_data['category_id'].values.reshape(-1, 1),
        'brand_id': train_data['brand_id'].values.reshape(-1, 1)
    }

    test_inputs = {
        'user_id': test_data['user_id'].values.reshape(-1, 1),  # 确保是二维的
        'age': test_data['age'].values.reshape(-1, 1),
        'gender': test_data['gender'].values.reshape(-1, 1),
        'item_id': test_data['item_id'].values.reshape(-1, 1),
        'category_id': test_data['category_id'].values.reshape(-1, 1),
        'brand_id': test_data['brand_id'].values.reshape(-1, 1)
    }

    # 确保标签是二维的
    train_labels = train_labels.values.reshape(-1, 1)
    test_labels = test_labels.values.reshape(-1, 1)

    # 创建和训练模型
    model = create_fm_recall_model()

    history = model.fit(
        train_inputs,
        train_labels,
        batch_size=32,
        epochs=10,
        validation_data=(test_inputs, test_labels),
        verbose=1
    )

    # 评估模型
    test_loss, test_accuracy = model.evaluate(test_inputs, test_labels)
    print(f"\n测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")

    # 获取所有物品的embedding
    all_items = {
        'item_id': np.arange(50000),
        'category_id': np.random.randint(0, 1000, 50000),
        'brand_id': np.random.randint(0, 500, 50000)
    }
    item_embeddings = model.get_item_embedding(all_items)

    # 确保item_embeddings是numpy数组
    if isinstance(item_embeddings, tf.Tensor):
        item_embeddings = item_embeddings.numpy()

    # 打印embedding的形状，用于调试
    print(f"Item embeddings shape: {item_embeddings.shape}")

    # 创建faiss索引
    index = create_faiss_index(item_embeddings)

    # 测试召回
    test_user = {
        'user_id': np.array([[1]]),  # 注意这里改成二维数组
        'age': np.array([[25]]),
        'gender': np.array([[1]])
    }

    # 打印输入的形状，用于调试
    print("\n测试用户输入:")
    for key, value in test_user.items():
        print(f"{key}: {value.shape}")

    user_embedding = model.get_user_embedding(test_user)
    # 确保user_embedding是numpy数组
    if isinstance(user_embedding, tf.Tensor):
        user_embedding = user_embedding.numpy()

    # 打印user_embedding的形状，用于调试
    print(f"User embedding shape: {user_embedding.shape}")

    # 使用faiss进行召回
    k = 10  # 召回数量
    distances, indices = index.search(user_embedding, k)
    print("\n召回结果:")
    print(f"物品ID: {indices[0]}")
    print(f"相似度分数: {distances[0]}")


if __name__ == "__main__":
    main()