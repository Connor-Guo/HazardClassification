import torch
from torch import nn
from sklearn.model_selection import train_test_split
from d2l import torch as d2l
from src.networks import SoftmaxClassifier


class SoftmaxClassifier(nn.Module):
    def __init__(self):
        super(SoftmaxClassifier, self).__init__()
        self.fc1 = nn.Linear(300, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 18)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    with open(f'./out/dump/{EXP_NAME}.pkl', 'rb') as tf:
        df = pickle.load(tf)
    with open('./out/dump/term_vec.pkl', 'rb') as tf:
        term_vec = pickle.load(tf)

    # -----------------------方法2--------------------------
    # 对输出的向量序列求平均，作为句子向量，用句子向量训练一个softmax分类器
    df['sep'] = df['doc_vec'].apply(lambda x: np.mean(x, axis=0))
    train_data, val_data, train_labels, val_labels = train_test_split(df['sep'], df['label'], test_size=0.2,
                                                                      random_state=42)
    # Convert the training and validation data into numpy arrays
    train_data = np.array(train_data.tolist())
    val_data = np.array(val_data.tolist())
    # Convert the training and validation labels into numpy arrays
    train_labels = np.array(train_labels.tolist()) - 1
    val_labels = np.array(val_labels.tolist()) - 1
    # Convert the training and validation data into PyTorch tensors
    train_data = torch.from_numpy(train_data).float()
    val_data = torch.from_numpy(val_data).float()
    # Convert the training and validation labels into PyTorch tensors
    train_labels = torch.from_numpy(train_labels).long()
    val_labels = torch.from_numpy(val_labels).long()

    import random


    def data_iter(batch_size, features, labels):
        num_examples = len(features)
        indices = list(range(num_examples))
        # 这些样本是随机读取的，没有特定的顺序
        random.shuffle(indices)
        for i in range(0, num_examples, batch_size):
            batch_indices = torch.tensor(
                indices[i: min(i + batch_size, num_examples)])
            yield features[batch_indices], labels[batch_indices]


    num_epochs = 10
    batch_size = 32
    train_iter = data_iter(batch_size, train_data, train_labels)
    val_iter = data_iter(batch_size, val_data, val_labels)

    # To train the softmax classifier, you need to define the loss function and optimizer first
    net = SoftmaxClassifier()
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)


    # Now we can train the classifier using a loop
    d2l.train_ch3(net, train_iter, val_iter, loss, num_epochs, trainer)
