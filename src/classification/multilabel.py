from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
from src.loadfiles import load_pickle
from src.evaluate import MultiLabelEvaluator
import os


def train_rf_multi(X_train, X_test, y_train, y_test, **params):

    cls = RandomForestClassifier(**params)
    cls.fit(X_train, y_train)

    y_pred = cls.predict(X_test)
    print(
        metrics.f1_score(y_test, y_pred, average="macro"),
        metrics.f1_score(y_test, y_pred, average="micro"),
        metrics.f1_score(y_test, y_pred, average="weighted"),
        metrics.f1_score(y_test, y_pred, average="samples")
    )

    eva = MultiLabelEvaluator(y_test, y_pred)
    print(eva.evaluate())

    return cls, eva


def convert_to_one_hot(y: np.ndarray):
    """
    y是一个m行n列的array，其中的值是类别标签。
    假设共有p个标签，将y转换成m*p的独热矩阵z。
    Note:
        y（输入）的标签从1开始，y中的0值会被忽略。
        z（输出）的矩阵列索引从0开始。也就是说y中的
        标签1对应z中的第0列。
    """
    n_class = y.max()
    z = np.zeros((y.shape[0], n_class), dtype=int)

    for i in range(y.shape[0]):  # i是行索引
        for j in range(y.shape[1]):
            if y[i, j] > 0:
                z[i, y[i, j]-1] = 1
    return z


def convert_to_label(y: np.ndarray):
    """
    convet_to_one_hot()的反函数。
    z（输出）的标签从1开始，z中的0值表示无标签。
    """
    n_class = y.shape[1]
    z = np.zeros((y.shape[0], 5), dtype=int)

    for i in range(y.shape[0]):  # i是行索引
        if np.sum(y[i, :]):  # 如果这行有标签
            for j, lbl in enumerate(np.argwhere(y[i, :])):
                z[i, j] = lbl + 1
    return z


if __name__ == "__main__":
    name = "guanzhi_sep_vec_each_1350_select"
    df_train = load_pickle(fr'C:\Users\10507\OneDrive\桌面\202301 危险源挖掘论文\不安全事件分类'
                           fr'\code\out\datasets\{name}-train.pkl')
    df_test = load_pickle(fr'C:\Users\10507\OneDrive\桌面\202301 危险源挖掘论文\不安全事件分类'
                          fr'\code\out\datasets\{name}-test.pkl')

    # 转化数据集
    x_train = np.array(df_train['doc_vec'].apply(lambda x: np.mean(x, axis=0)).tolist())
    # 在y_train中，y是从1开始的
    # 将y_train转化为独热编码
    y_train = df_train[['label', 'label2', 'label3', 'label4', 'label5']].fillna(0).to_numpy().astype(int)
    y_train = convert_to_one_hot(y_train)
    x_test = np.array(df_test['doc_vec'].apply(lambda x: np.mean(x, axis=0)).tolist())
    y_test = df_test[['label', 'label2', 'label3', 'label4', 'label5']].fillna(0).to_numpy().astype(int)
    y_test = convert_to_one_hot(y_test)

    # 设置参数，训练模型，并获得评估结果
    params = {
        'n_estimators': 500,
        'max_depth': 8
    }
    clf, eva = train_rf_multi(x_train, x_test, y_train, y_test, **params)

    # 保存评估结果
    eva_path = r'C:\Users\10507\OneDrive\桌面\202301 危险源挖掘论文\不安全事件分类\code\evaluation\1350'
    eva.save(eva_path, modelname="RF-n_est500-dep8")

    # 保存预测结果，以供后期分析
    result = convert_to_label(clf.predict(x_test))
    for i in range(1, 6):
        df_test[f'pred_label{i}'] = result[:, i-1]
    df_test.to_excel(f"../out/result/result_RF_{name}-test.xlsx", index=False)
