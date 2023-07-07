"""
处理./data文件夹下的数据。
数据预处理，数据集划分，生成文件并保存

预处理完毕的文件保存至 ../out/dump 目录下，保存.xlsx和.pkl两个版本
划分的数据集保存至 ../out/datasets 目录下，保存.xlsx和.pkl两个版本，共四个文件（训练集+测试集各两个格式）
"""
import os
from src.loadfiles import *
from src.preprocessing import *
from src.modules import split_df_on_label


if __name__ == "__main__":
    EXP_NAME = "guanzhi_fix_bert_abbr_2941"
    df = load_data("../data/1-管制专业不安全事件匹配结果-fix.xlsx")  # 读取数据库

    # 乱码表
    loader = Loader()
    gibberish = loader.load_gibberish('../nlp/gibberish.txt')

    # 读取标签
    label_dict = loader.load_label_dict("../data/label-guanzhi.txt")

    # 读取缩写字典
    abbr_dict = loader.load_abbr("../data/abbr/管制.csv")

    # 预处理
    df = preprocess_bert_df(
        df, label_dict=label_dict, gibberish=gibberish, abbr_dict=abbr_dict)

    # 保存预处理文件
    if not os.path.exists(f'../out/dump/{EXP_NAME}/'):
        os.makedirs(f'../out/dump/{EXP_NAME}/')
    df.to_excel(f'../out/dump/{EXP_NAME}/{EXP_NAME}.xlsx', index=False)  # 仅供阅读
    save_pickle(df, f'../out/dump/{EXP_NAME}/{EXP_NAME}.pkl')

    # 数据集划分
    df_train, df_test = split_df_on_label(df)
    if not os.path.exists(f'../out/datasets/{EXP_NAME}/'):
        os.makedirs(f'../out/datasets/{EXP_NAME}/')
    df_train.to_excel(f"../out/datasets/{EXP_NAME}/{EXP_NAME}-train.xlsx", index=False)
    df_test.to_excel(f"../out/datasets/{EXP_NAME}/{EXP_NAME}-test.xlsx", index=False)
    save_pickle(df_train, f"../out/datasets/{EXP_NAME}/{EXP_NAME}-train.pkl")
    save_pickle(df_test, f"../out/datasets/{EXP_NAME}/{EXP_NAME}-test.pkl")

    pass
