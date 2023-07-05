"""
处理./out/resample文件夹下的数据。
无需预处理，直接划分数据集。
"""
from src.loadfiles import *
from src.modules import split_df_on_label


if __name__ == "__main__":
    EXP_NAME = "guanzhi_fix_bert_2941_del_labels_14_1845"
    df = load_pickle(f"../out/resample/{EXP_NAME}/{EXP_NAME}.pkl")  # 读取数据库

    # 数据集划分
    df_train, df_test = split_df_on_label(df)
    if not os.path.exists(f'../out/datasets/{EXP_NAME}/'):
        os.makedirs(f'../out/datasets/{EXP_NAME}/')
    df_train.to_excel(f"../out/datasets/{EXP_NAME}/{EXP_NAME}-train.xlsx", index=False)
    df_test.to_excel(f"../out/datasets/{EXP_NAME}/{EXP_NAME}-test.xlsx", index=False)
    save_pickle(df_train, f"../out/datasets/{EXP_NAME}/{EXP_NAME}-train.pkl")
    save_pickle(df_test, f"../out/datasets/{EXP_NAME}/{EXP_NAME}-test.pkl")

    pass
