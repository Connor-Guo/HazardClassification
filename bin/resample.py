from src.loadfiles import *

from src.sampling import Resampler

EXP_NAME = "guanzhi_bert_2941"
df = load_pickle(f'../out/dump/{EXP_NAME}.pkl')
resampler = Resampler(df,
                      labels_cols=['label', 'label2', 'label3', 'label4', 'label5'],
                      chinese_labels_cols=['不安全事件', '不安全事件2', '不安全事件3', '不安全事件4', '不安全事件5'],
                      )
df_1 = resampler.del_labels([9, 12, 14, 15, 17, 19])
print(df_1)
print("Rest n_samples: {}".format(len(df_1)))
df_1.to_excel("../out/resample/guanzhi_del_labels_13_{}.xlsx".format(len(df_1)), index=False)
