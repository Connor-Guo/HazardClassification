import os
import openai
import pandas as pd

from src.loadfiles import *
from src.sampling import *
"""
利用ChatGPT3.5生成数据。
仅针对训练集生成。

生成的数据按标签存放在`../out/ChatGPT/{EXP_NAME}/`文件夹下。
除了单个标签外，还包括所有生成数据的合并文件、所有（原+生成）数据的合并文件。
后者会被复制一份到`../out/dataset/{EXP_NAME}/`文件夹下。

"""

if __name__ == "__main__":
    # 参数设置
    EXP_NAME = "guanzhi_fix_bert_abbr_2941"
    SAVE_NAME = "guanzhi_fix_bert_abbr_2941_GPT"
    save_folder = f"../out/ChatGPT/{SAVE_NAME}/"
    dataset_folder = f"../out/dataset/{SAVE_NAME}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    openai.api_key = "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"
    prompt_str = """
    你是一名空中交通管理专家。

    你可以将这段事件描述用其他方式转述出来吗？请提供{}种转述。请确保转述中包含所有重要的细节，例如潜在的后果、系统的名称、以及其他相关的专业名词或缩写。请用中文转述。
    
    注意：事件描述可能由若干个子事件构成，这些子事件有时会由分号等符号分割。在转述时，请务必囊括所有子事件，而不是只包含其中的一个或数个。
    
    输出应当是JSON格式，包括以下关键字："Paraphrase"，并且"Paraphrase"下的元素应该是一个包含所有转述字符串组成的列表。

    事件描述：
    “{}”
    """
    r_dict = {
        1: 150,
        5: 150,
        8: 150,
        9: 150,
        10: 150,
        13: 200,
        14: 150,
        17: 150,
        19: 200,
        21: 200,
    }  # 标签对应的采样目标数量

    df = load_pickle(f'../out/datasets/{EXP_NAME}/{EXP_NAME}-train.pkl')
    resampler = Resampler(df,
                          id_col="危险源编号",
                          text_col="后果",
                          labels_cols=['label', 'label2', 'label3', 'label4', 'label5'],
                          chinese_labels_cols=['不安全事件', '不安全事件2', '不安全事件3', '不安全事件4',
                                               '不安全事件5'],
                          )

    # df_new = pd.DataFrame(columns=df.columns)
    lst_df_generated = []
    for label, n_paraphrase in r_dict.items():
        df_generated = resampler.generate_with_gpt(
            label=label,
            prompt_str=prompt_str,
            expected_n=n_paraphrase,
            save_folder=save_folder,
            lb_text_len=8,
        )
        dataset_path_1 = os.path.join(save_folder, f"{label}.xlsx")
        df_generated.to_excel(dataset_path_1, index=False)
        lst_df_generated.append(df_generated)

    # 保存合并后的文件
    df_generated_all = pd.concat(lst_df_generated)
    df_all = pd.concat((df, df_generated_all)).sort_values(by="label", ascending=True)
    df_generated_all.to_excel(os.path.join(save_folder, "generated_all.xlsx"), index=False)
    df_all.to_excel(os.path.join(save_folder, f"{SAVE_NAME}-train.xlsx"), index=False)
    save_pickle(df_all, os.path.join(dataset_folder, f"{SAVE_NAME}-train.xlsx"))
    df_all.to_excel(os.path.join(dataset_folder, f"{SAVE_NAME}-train.xlsx"), index=False)
