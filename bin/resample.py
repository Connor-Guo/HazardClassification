from src.loadfiles import *

from src.sampling import Resampler

if __name__ == "__main__":
    EXP_NAME = "guanzhi_fix_bert"

    # 读取数据
    df = load_pickle(f'../out/dump/{EXP_NAME}/{EXP_NAME}.pkl')

    # 重采样
    resampler = Resampler(df,
                          labels_cols=['label', 'label2', 'label3', 'label4', 'label5'],
                          chinese_labels_cols=['不安全事件', '不安全事件2', '不安全事件3', '不安全事件4', '不安全事件5'],
                          )
    df_resample = resampler.del_labels([6, 7, 12, 15, 16, 18, 20, 22])
    print(df_resample)
    print("Rest n_samples: {}".format(len(df_resample)))

    # 保存结果
    SAVE_NAME = EXP_NAME + f"_del_labels_14_{len(df_resample)}"
    if not os.path.exists(f'../out/resample/{SAVE_NAME}/'):
        os.makedirs(f'../out/resample/{SAVE_NAME}/')
    df_resample.to_excel(f"../out/resample/{SAVE_NAME}/{SAVE_NAME}.xlsx", index=False)
    save_pickle(df_resample, f"../out/resample/{SAVE_NAME}/{SAVE_NAME}.pkl")
