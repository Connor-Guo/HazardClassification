"""
用于评估模型

Evaluator
对于单标签分类问题，记录以下指标：
1. accuracy
    from sklearn.metrics

2. balanced accuracy
    from sklearn.metrics

3. confusion metrics & heat map
    from sklearn.metrics

4. precision, recall, f1-score for imbalance datasets
    from sklearn.metrics and imblearn.metrics

5. AUROC & figure (暂未实现)
    from sklearn.metrics

6. Cohen's Kappa score
    from sklearn.metrics
    The kappa statistic, which is a number between -1 and 1. The maximum
    value means complete agreement; zero or lower means chance agreement.

7. Matthew's correlation coefficient
    from sklearn.metrics

Examples

>>> model_name = "RandomModel"
>>> y_true = np.array([0, 0, 1, 0, 2, 2, 1])
>>> y_pred = np.array([0, 0, 2, 0, 1, 2, 1])
>>> eva = Evaluator(y_true, y_pred)
>>> eva.evaluate()
>>> eva.save(modelname=model_name)


MultiLabelEvaluator
1. accuracy (exact match ratio)
    from sklearn.metrics

2. precision
    from sklearn.metrics

3. recall
    from sklearn.metrics

4. f1-score
    from sklearn.metrics

5. Hamming loss
    from sklearn.metrics
"""
import pandas as pd
import torch
from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        classification_report,
        roc_auc_score,
        roc_curve,
        auc,
        cohen_kappa_score,
        matthews_corrcoef,
        hamming_loss,
)
from imblearn.metrics import classification_report_imbalanced
from torcheval.metrics.functional.classification.accuracy import topk_multilabel_accuracy
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from collections import Iterable
from typing import Union, Sequence


class Evaluator:
    def __init__(
        self,
        y_true,
        y_pred,
        exp_name=None,
        **kwargs,
    ):
        if exp_name:
            self.name = "eval_" + str(exp_name)
        else:
            self.name = "eval_" + time.strftime('%Y-%m-%d %H%M%S', time.localtime())

        if type(y_true) is list:
            y_true = np.array(y_true)
        if type(y_pred) is list:
            y_pred = np.array(y_pred)
        assert y_true.shape == y_pred.shape

        self.y_true = y_true
        self.y_pred = y_pred
        self.kwargs = kwargs
        self.n_labels = y_true.max() - y_true.min()
        self.scores = _PerfDict()
        self._scores = {}
        self.figs = {}

    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)

    def balanced_accuracy(self):
        return balanced_accuracy_score(self.y_true, self.y_pred)

    def confusion_mat(self):
        return confusion_matrix(self.y_true, self.y_pred)

    def plot_confusion_mat(self):
        C = self._confusion_mat()
        fig, ax = plt.subplots()

        img = ax.matshow(C, cmap=plt.cm.Reds, vmax=40)  # 根据最下面的图按自己需求更改颜色
        # plt.colorbar()

        for i in range(len(C)):
            for j in range(len(C)):
                ax.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

        # plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。

        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        # plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20}) # 设置字体大小。
        # plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
        # plt.xticks(range(0,5), labels=['a','b','c','d','e']) # 将x轴或y轴坐标，刻度 替换为文字/字符
        # plt.yticks(range(0,5), labels=['a','b','c','d','e'])
        return fig

    def cls_report(self, imbalanced=False, output_dict=False):
        if imbalanced:
            return classification_report_imbalanced(self.y_true, self.y_pred, output_dict=output_dict)
        else:
            return classification_report(self.y_true, self.y_pred, output_dict=output_dict)

    # def _auroc(self):
    #     return roc_auc_score(self.y_true, self.y_pred)

    # def _plot_auroc(self):
    #     # 计算每一类的ROC
    #     fpr = dict()
    #     tpr = dict()
    #     roc_auc = dict()
    #     for i in range(n_classes):
    #         fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    #         roc_auc[i] = auc(fpr[i], tpr[i])
    #
    #     # Compute micro-average ROC curve and ROC area（方法二）
    #     fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    #     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    #
    #     # Compute macro-average ROC curve and ROC area（方法一）
    #     # First aggregate all false positive rates
    #     all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    #
    #     # Then interpolate all ROC curves at this points
    #     mean_tpr = np.zeros_like(all_fpr)
    #     for i in range(n_classes):
    #         mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    #
    #     # Finally average it and compute AUC
    #     mean_tpr /= n_classes
    #     fpr["macro"] = all_fpr
    #     tpr["macro"] = mean_tpr
    #     roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    #
    #     fig, ax = plt.subplots()
    #     lw = 2
    #     ax.plot(fpr["micro"], tpr["micro"],
    #              label='micro-average ROC curve (area = {0:0.2f})'
    #                    ''.format(roc_auc["micro"]),
    #              color='deeppink', linestyle=':', linewidth=4)
    #
    #     plt.plot(fpr["macro"], tpr["macro"],
    #              label='macro-average ROC curve (area = {0:0.2f})'
    #                    ''.format(roc_auc["macro"]),
    #              color='navy', linestyle=':', linewidth=4)
    #
    #     colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    #     for i, color in zip(range(n_classes), colors):
    #         plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #                  label='ROC curve of class {0} (area = {1:0.2f})'
    #                        ''.format(i, roc_auc[i]))
    #
    #     plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Some extension of Receiver operating characteristic to multi-class')
    #     plt.legend(loc="lower right")
    #     plt.show()

    def cohen_kappa_score(self):
        return cohen_kappa_score(self.y_true, self.y_pred)

    def matthews_corrcoef(self):
        return matthews_corrcoef(self.y_true, self.y_pred)

    def evaluate(self, show_fig=False):
        self.scores["accuracy"] = self.accuracy()
        self.scores["balanced accuracy"] = self.balanced_accuracy()
        self.scores["confusion matrix"] = self.confusion_mat()
        self.scores["classification report"] = self.cls_report()
        self._scores["classification report dict"] = self.cls_report(output_dict=True)
        # self.scores["roc auc score"] = self._auroc()
        self.scores["cohen kappa score"] = self.cohen_kappa_score()
        self.scores["matthews correlation coefficient"] = self.matthews_corrcoef()
        self.figs["confusion matrix"] = self.plot_confusion_mat()
        # self.figs["auc roc curves"] = self._plot_auroc()
        if show_fig:
            for fig in self.figs.values():
                fig.show()
        return self.scores

    def save(self, path='./evaluation', modelname: str = None, savefig=True):
        if not os.path.exists(path):
            os.makedirs(path)
        if modelname is None:
            modelname = "unknown-model-" + time.strftime('%Y-%m-%d %H%M%S', time.localtime())

        filename_ = os.path.join(path, modelname + '.txt')
        if os.path.exists(filename_):
            raise FileExistsError(f"File '{modelname}' already exists.")

        header, values = '', ''
        tables = ''
        for name, score in self.scores.items():
            if not isinstance(score, Iterable):
                header += name + ','
                values += f"{score:.4f},"
            else:
                tables += name + '\n'
                tables += str(score) + '\n'
        for name in ['avg_pre', 'avg_rec', 'avg_spe', 'avg_f1', 'avg_geo', 'avg_iba', 'total_support']:
            score = self._scores['classification report dict'][name]
            header += name + ','
            values += f"{score:.4f},"

        with open(filename_, 'w', encoding='utf-8') as f:
            f.writelines([header, "\n", values, "\n", tables])
        print(f"Evaluation results saved to '{filename_}'.")

        if savefig:
            fig_name = modelname
            self.savefig(path, fig_name)

    def savefig(self, path='./', modelname=None):
        if modelname is None:
            modelname = time.strftime('%Y-%m-%d %H%M%S', time.localtime())

        for fig_name, fig in self.figs.items():
            filename_ = modelname + "_" + fig_name + '.jpg'
            fig.savefig(os.path.join(path, filename_))
            print(f"Figure saved to '{filename_}'.")


class MultiLabelEvaluator(Evaluator):
    """还可以加入macro的PRF1"""
    def __init__(self, y_true, y_pred, exp_name=None, **kwargs):
        super(MultiLabelEvaluator, self).__init__(y_true, y_pred, exp_name, **kwargs)
        # 确保是multi label问题
        assert len(y_true.shape) == 2
        assert y_true.shape[1] > 1

    def balanced_accuracy(self):
        """Invalid method for multilabel."""
        pass

    def confusion_mat(self):
        """Invalid method for multilabel."""
        pass

    def plot_confusion_mat(self):
        """Invalid method for multilabel."""
        pass

    def cls_report(self, imbalanced=False, output_dict=False):
        """imblearn does not support multilabel"""
        return classification_report(self.y_true, self.y_pred, output_dict=output_dict)

    def hamming_loss(self):
        return hamming_loss(self.y_true, self.y_pred)

    def evaluate(self, show_fig=False):
        self.scores["accuracy"] = self.accuracy()
        self.scores["classification report"] = self.cls_report()
        self._scores["classification report dict"] = self.cls_report(imbalanced=False, output_dict=True)
        self.scores["hamming_loss"] = self.hamming_loss()
        return self.scores

    def save(self, path='./evaluation', modelname: str = None, savefig=True):
        if not os.path.exists(path):
            os.makedirs(path)
        if modelname is None:
            modelname = "unknown-model-" + time.strftime('%Y-%m-%d %H%M%S', time.localtime())

        filename_ = os.path.join(path, modelname + '.txt')
        if os.path.exists(filename_):
            raise FileExistsError(f"File '{modelname}' already exists.")

        header, values = '', ''
        tables = ''
        for name, score in self.scores.items():
            if not isinstance(score, Iterable):
                header += name + ','
                values += f"{score:.4f},"
            else:
                tables += name + '\n'
                tables += str(score) + '\n'
        for name in ['micro avg', 'macro avg', 'weighted avg', 'samples avg']:
            for attr in ['precision', 'recall', 'f1-score']:
                score = self._scores['classification report dict'][name][attr]
                header += name + ' ' + attr + ','
                values += f"{score:.4f},"

        with open(filename_, 'w', encoding='utf-8') as f:
            f.writelines([header, "\n", values, "\n", tables])
        print(f"Evaluation results saved to '{filename_}'.")

        if savefig:
            fig_name = modelname
            self.savefig(path, fig_name)


class MultiLabelProbEvaluator(Evaluator):
    def __init__(
        self,
        y_true: Union[torch.Tensor, np.ndarray],
        y_pred: Union[torch.Tensor, np.ndarray],  # 是prob，真正的pred是pred2
        exp_name: str = None,
        **kwargs,
    ) -> None:
        super(MultiLabelProbEvaluator, self).__init__(y_true, y_pred, exp_name, **kwargs)
        assert isinstance(y_true, (torch.Tensor, np.ndarray))
        assert isinstance(y_pred, (torch.Tensor, np.ndarray))
        if isinstance(y_true, torch.Tensor):
            self.y_true = np.array(y_true)
        if isinstance(y_pred, torch.Tensor):
            self.y_pred = np.array(y_pred)

        self.threshold = kwargs.get("threshold")
        if self.threshold:
            self.y_pred2 = self._prob_2_pred()  # 计算预测的标签值
            self.mleva = MultiLabelEvaluator(self.y_true, self.y_pred2, exp_name, **kwargs)

    def top_k_accuracy(self, k: int = 5):
        y_pred, y_true = torch.tensor(self.y_pred), torch.tensor(self.y_true)
        perf = {
            "contain": topk_multilabel_accuracy(y_pred, y_true, criteria='contain', k=k),
            "overlap": topk_multilabel_accuracy(y_pred, y_true, criteria='overlap', k=k),
            # "hamming": topk_multilabel_accuracy(self.y_pred, self.y_true, criteria='hamming', k=k),
        }
        return perf

    def accuracy(self) -> float:
        """Exact match accuracy. This is rather slow."""
        return accuracy_score(self.y_true, self.y_pred2)

    def hamming_loss(self):
        return self.mleva.hamming_loss()

    def cls_report(self, imbalanced=False, output_dict=False):
        return self.mleva.cls_report(imbalanced, output_dict)

    def evaluate(self, show_fig=False):
        self.scores["accuracy"] = self.accuracy()
        self.scores["top-k accuracy contain"] = self.top_k_accuracy(k=5)["contain"]
        self.scores["top-k accuracy overlap"] = self.top_k_accuracy(k=5)["overlap"]
        self.scores["classification report"] = self.cls_report()
        self._scores["classification report dict"] = self.cls_report(output_dict=True)
        self.scores["hamming_loss"] = self.hamming_loss()
        return self.scores

    def _prob_2_pred(self):
        # if self.threshold is None:
        #     raise AttributeError("Threshold for deciding labels unprovided.")
        pred2 = (self.y_pred > self.threshold).astype(int)
        return pred2

    def _top_k(self, k):
        y_pred = torch.tensor(self.y_pred)
        top_p, top_class = torch.topk(y_pred, k, dim=1)
        top_class = top_class + 1
        return top_p, top_class


class _PerfDict(dict):
    def __init__(self):
        super().__init__(self)

    def __repr__(self):
        s = "{"
        for k, v in self.items():
            if isinstance(v, Iterable):
                s += f"\'{k}\':\n{v}, \n"
            else:
                s += f"\'{k}\':{v}, \n"
        s = s[:-3] + "}"
        return s


if __name__ == "__main__":
    df = pd.read_excel(r"C:\Users\10507\OneDrive\桌面\202301 危险源挖掘论文\不安全事件分类\code\out\result\result_guanzhi_sep_vec_each_1000_select.xlsx")
    y_test = df['label'].to_numpy()
    # y_pred = df['pred_label'].to_numpy()
    # eva = Evaluator(y_test, y_pred)
    # eva.evaluate()
    # eva.save("../evaluation", "SimilarityOnSelect")

