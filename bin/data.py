"""
数据预处理，数据集划分，生成文件并保存

预处理完毕的文件保存至 ../out/dump 目录下，保存.xlsx和.pkl两个版本
划分的数据集保存至 ../out/datasets 目录下，保存.xlsx和.pkl两个版本，共四个文件（训练集+测试集各两个格式）
"""
from src.loadfiles import *
from src.preprocessing import *
from src.modules import split_df_on_label


if __name__ == "__main__":
    EXP_NAME = "guanzhi_bert_2941"
    df = load_data("../data/1-管制专业不安全事件匹配结果.xlsx")  # 读取数据库

    # 乱码表
    loader = Loader()
    gibberish = loader.load_gibberish('../nlp/gibberish.txt')

    # 读取标签
    label_dict = {}
    for l in [line.strip() for line in open("../data/label-guanzhi.txt", encoding='UTF-8').readlines()]:
        v, k = l.split('\t')  # 1	跑道侵入
        label_dict[k] = int(v)

    # 预处理
    df = preprocess_bert_df(df, label_dict=label_dict, gibberish=gibberish)

    # 保存预处理文件
    df.to_excel(f'../out/dump/{EXP_NAME}.xlsx', index=False)  # 仅供阅读
    save_pickle(df, f'../out/dump/{EXP_NAME}.pkl')

    # 数据集划分
    df_train, df_test = split_df_on_label(df)
    df_train.to_excel(f"../out/datasets/{EXP_NAME}-train.xlsx", index=False)
    df_test.to_excel(f"../out/datasets/{EXP_NAME}-test.xlsx", index=False)
    save_pickle(df_train, f"../out/datasets/{EXP_NAME}-train.pkl")
    save_pickle(df_test, f"../out/datasets/{EXP_NAME}-test.pkl")

    pass