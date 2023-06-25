# -*- coding: utf-8 -*-
# built in python 3.9
# Author: Ziyi Guo
"""
src源代码，该文件夹包含以下模块：

preprocessing
    数据预处理

vectorize
    文本向量化

cls（已弃用）
    分类器的训练与评估

modules
    独立的功能模块

tfidf
    计算文档tfidf值的模块

symdict
    专业词典有关的模块，包括读取、拆分、划分从属关系等功能函数

similarity
    计算余弦相似度的模块

evaluate
    用于评估模型分类效果的模块

"""
import src.loadfiles as loadfiles

load_pickle = loadfiles.load_pickle
save_pickle = loadfiles.save_pickle


