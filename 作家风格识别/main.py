# 导入相关包
import copy
import os
import random
import numpy as np
import jieba as jb
import jieba.analyse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as f
from torchtext import data
from torchtext import datasets
from torchtext.data import Field
from torchtext.data import Dataset
from torchtext.data import Iterator
from torchtext.data import Example
from torchtext.data import BucketIterator


# ----------------------------- 请加载您最满意的模型 -------------------------------
# 加载模型(请加载你认为的最佳模型)
# 加载模型,加载请注意 model_path 是相对路径, 与当前文件同级。
# 如果你的模型是在 results 文件夹下的 temp.pth 模型，则 model_path = 'results/temp.pth'

import torch
import jieba as jb

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 加载模型(请加载你认为的最佳模型)
model_path = 'results/temp.pth'  # 模型路径
checkpoint = torch.load(model_path)

# 提取词典和作家信息
word_to_number = checkpoint['word2int']
number_to_author = checkpoint['int2author']
word_number = len(word_to_number)

# 创建模型实例
model = torch.nn.Sequential(
    torch.nn.Linear(word_number, 700),
    torch.nn.ReLU(),
    torch.nn.Linear(700, 6)
).to(device)

model.load_state_dict(checkpoint['model'])
model.eval()  # 设置为评估模式

# -------------------------请勿修改 predict 函数的输入和输出-------------------------
def predict(text):
    """
    :param text: 中文字符串
    :return: 字符串格式的作者名缩写
    """
    # ----------- 实现预测部分的代码，以下样例可代码自行删除，实现自己的处理方式 -----------
    feature = torch.zeros(word_number, dtype=torch.float).to(device)
    
    # 对输入文本进行分词
    for word in jb.lcut(text):  # 使用精确分词模式
        if word in word_to_number:
            feature[word_to_number[word]] += 1
    
    # 归一化处理，防止特征值过大影响模型表现
    if feature.sum() > 0:
        feature /= feature.sum()
    else:
        return "无法识别"  # 如果分词后没有匹配到任何词，返回无法识别
    
    # 模型预测
    with torch.no_grad():
        feature = feature.unsqueeze(0)  # 添加一个batch维度
        output = model(feature)  # 通过模型前向传播得到输出
        predicted_label = torch.argmax(output, dim=1).item()  # 获取预测的作家编号
    
    # 返回作家名缩写
    prediction = number_to_author[predicted_label]
    # --------------------------------------------------------------------------

    return prediction
