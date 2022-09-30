import os
import struct
import numpy as np

import matplotlib.pyplot as plt

# 当前文件目录

current_path = os.path.dirname(__file__)

# 数据集目录
dataset_path = current_path + "\\..\\dataset\\handwritten_digit\\"

# 训练图片文件名
train_images_path = "train-images-idx3-ubyte"
# 训练标签文件名
train_labels_path = "train-labels-idx1-ubyte"

# 测试图片文件名
test_images_path = "t10k-images-idx3-ubyte"

# 测试标签文件名
test_labels_path = "t10k-labels-idx1-ubyte"

def load_train_data():
    # 文件的比特形式
    train_images_bytes = open(dataset_path + train_images_path, "rb").read()

    # 偏移量
    offset = 0

    # 文件头格式
    fmt_header = ">iiii"

    # 解析文件
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, train_images_bytes, offset)

    # 图片尺寸
    image_size = num_rows * num_cols

    # 偏移文件头
    offset += struct.calcsize(fmt_header)

    # 图片文件格式
    fmt_image = ">" + str(image_size) + "B"

    # 训练数据集，矩阵大小：（图片数量，图片行数，图片列数）
    train_data = np.empty((num_images, num_rows, num_cols))

    for i in range(num_images):
        # 逐个读取图片
        train_data[i] = np.array(struct.unpack_from(fmt_image, train_images_bytes, offset)).reshape(
            (num_rows, num_cols))
        # 计算偏移
        offset += struct.calcsize(fmt_image)

    # 读取训练标签数据集
    fmt_header = ">ii"
    offset = 0
    train_labels_bytes = open(dataset_path + train_labels_path, "rb").read()

    # magic_number 魔数，文件头
    # num_labels，标签数量
    magic_number, num_labels = struct.unpack_from(fmt_header, train_labels_bytes, offset)

    train_labels = np.empty(num_labels, dtype="int")
    offset += struct.calcsize(fmt_header)
    fmt_label = ">" + str(1) + "B"
    for i in range(num_labels):
        train_labels[i] = np.array(struct.unpack_from(fmt_label, train_labels_bytes, offset), dtype="int")
        offset += struct.calcsize(fmt_label)

    return train_data, train_labels

def load_test_data():
    # 文件的比特形式
    test_images_bytes = open(dataset_path + test_images_path, "rb").read()

    # 偏移量
    offset = 0

    # 文件头格式
    fmt_header = ">iiii"

    # 解析文件
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, test_images_bytes, offset)

    # 图片尺寸
    image_size = num_rows * num_cols

    # 偏移文件头
    offset += struct.calcsize(fmt_header)

    # 图片文件格式
    fmt_image = ">" + str(image_size) + "B"

    # 测试数据集，矩阵大小：（图片数量，图片行数，图片列数）
    test_data = np.empty((num_images, num_rows, num_cols))

    for i in range(num_images):
        # 逐个读取图片
        test_data[i] = np.array(struct.unpack_from(fmt_image, test_images_bytes, offset)).reshape(
            (num_rows, num_cols))
        # 计算偏移
        offset += struct.calcsize(fmt_image)

    # 读取测试标签数据集
    fmt_header = ">ii"
    offset = 0
    test_labels_bytes = open(dataset_path + test_labels_path, "rb").read()

    # magic_number 魔数，文件头
    # num_labels，标签数量
    magic_number, num_labels = struct.unpack_from(fmt_header, test_labels_bytes, offset)

    test_labels = np.empty(num_labels, dtype="int")
    offset += struct.calcsize(fmt_header)
    fmt_label = ">" + str(1) + "B"
    for i in range(num_labels):
        test_labels[i] = np.array(struct.unpack_from(fmt_label, test_labels_bytes, offset), dtype="int")
        offset += struct.calcsize(fmt_label)

    return test_data, test_labels

train_data, train_label = load_train_data()

print(train_data[0])