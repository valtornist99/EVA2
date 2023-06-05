import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import cv2

sys.path.append('../input/timm-pytorch-image-models/pytorch-image-models-master')

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import timm
from warnings import filterwarnings
import math
filterwarnings("ignore")

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

import pandas as pd
import seaborn as sns
from scipy import stats


margin = 0.5


class PrepsDataset(Dataset):
    def __init__(self, df, mode, transform=None):

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        img = cv2.imread(row.file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']

        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)

        if self.mode == 'test':
            return torch.tensor(img).float()
        else:
            return torch.tensor(img).float(), torch.tensor(row.label_group).long()


class ArcModule(nn.Module):
    def __init__(self, in_features, out_features, s=10, m=margin):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = torch.tensor(math.cos(math.pi - m))
        self.mm = torch.tensor(math.sin(math.pi - m) * m)

    def forward(self, inputs, labels):
        cos_th = F.linear(inputs, F.normalize(self.weight))
        cos_th = cos_th.clamp(-1, 1)
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
        cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)

        cond_v = cos_th - self.th
        cond = cond_v <= 0
        cos_th_m[cond] = (cos_th - self.mm)[cond]

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        onehot = torch.zeros(cos_th.size())
        labels = labels.type(torch.LongTensor)
        onehot.scatter_(1, labels, 1.0)
        outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
        outputs = outputs * self.s
        return outputs


class PrepsResNet(nn.Module):

    def __init__(self, channel_size, out_feature, dropout=0.5, backbone='resnet34', pretrained=True):
        super(PrepsResNet, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        self.channel_size = channel_size

        self.out_feature = out_feature
        self.in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.margin = ArcModule(in_features=self.channel_size, out_features=self.out_feature)
        self.bn1 = nn.BatchNorm2d(self.in_features)
        self.dropout = nn.Dropout2d(dropout, inplace=True)
        self.fc1 = nn.Linear(self.in_features * 16 * 16, self.channel_size)
        self.bn2 = nn.BatchNorm1d(self.channel_size)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        #4 = batch_size = features.shape[0]
        features = features.view(features.shape[0], -1)
        # features = self.bn1(features)
        # features = self.dropout(features)

        # features = self.fc1(features)

        #imgs_number % batch_size = 0
        features = self.bn2(features)

        # features = F.normalize(features)
        if labels is not None:
            return self.margin(features, labels)
        return features


def train_model():
    init_lr = 3e-4
    batch_size = 4
    n_worker = 0

    os.chdir('saved_files/ml2_data/')
    df_train = pd.read_csv('df_train.csv')
    os.chdir('images/')
    dataset_train = PrepsDataset(df_train, 'train')

    #model = PrepsResNet(512, df_train.label_group.nunique())
    model = timm.create_model('resnet34', pretrained=True)
    model.fc = nn.Linear(512, 5)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_worker)

    for epoch in range(3):

        losses = []
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            #outputs = model(inputs, labels)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print('Epoch: ' + str(epoch))
        print('Train loss: ' + str(np.mean(losses)))
        valid_preps_model(model)

    print('Finished Training')

    torch.save(model.state_dict(), '../model')


def valid_preps_model(model):
    batch_size = 1
    n_worker = 0

    df_valid = pd.read_csv('../df_test.csv')
    dataset_valid = PrepsDataset(df_valid, 'valid')

    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=n_worker)

    criterion = nn.CrossEntropyLoss()

    model.eval()

    losses = []
    with torch.no_grad():
        for i, data in enumerate(valid_loader, 0):
            inputs, labels = data
            outputs = model(inputs)

            predicted_class = F.softmax(model(inputs), -1).detach().numpy()[0].argmax()
            print(predicted_class)
            print("Label = " + str(labels.detach().numpy()[0]))

            loss = criterion(outputs, labels)
            losses.append(loss.item())

    model.train()

    print('Valid loss: ' + str(np.mean(losses)))


def model_process():
    batch_size = 1
    n_worker = 0

    os.chdir('saved_files/ml2_data/')
    df_train = pd.read_csv('df_train.csv')
    df_test = pd.read_csv('df_test.csv')
    os.chdir('images/')
    dataset_train = PrepsDataset(df_train, 'process')
    dataset_test = PrepsDataset(df_test, 'process')

    #model = PrepsResNet(512, df_train.label_group.nunique())
    model = timm.create_model('resnet34', pretrained=True)
    model.fc = nn.Linear(512, 5)

    model.load_state_dict(torch.load('../model'))
    model.eval()

    process_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=n_worker)
    features512_train = get_features(process_loader, model)
    np.savetxt('../features512_train.csv', features512_train, delimiter=',')

    process_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=n_worker)
    features512_test = get_features(process_loader, model)
    np.savetxt('../features512_test.csv', features512_test, delimiter=',')


def get_features(process_loader, model):
    res = []
    for i, data in enumerate(process_loader, 0):
        inputs, labels = data
        outputs = model(inputs)

        outputs_val = list(outputs.detach().numpy()[0])
        label = labels.detach().numpy()[0]

        outputs_val.append(label)
        res.append(outputs_val)

    res = np.array(res)

    return res


def rf_test_var1():
    os.chdir('saved_files')
    df_train = pd.read_csv('np_miheev2.csv', header=None)
    df_test = pd.read_csv('np_miheev2.csv', header=None)

    # df_train = df_train.sample(frac=0.9)
    # df_test = df_test.drop(df_train.index)

    train_data = np.array(df_train)
    x_train, y_train = train_data[:, :df_train.shape[1] - 1], train_data[:, df_train.shape[1] - 1]
    test_data = np.array(df_test)
    x_test, y_test = test_data[:, :df_test.shape[1] - 1], test_data[:, df_test.shape[1] - 1]

    #rf_tuning(x_train, y_train)

    # n_estimators=700, min_samples_split=12, min_samples_leaf=2, max_features='sqrt', max_depth=13, bootstrap=False
    rfc = RandomForestClassifier(n_estimators=700, min_samples_split=12, min_samples_leaf=2, max_features='sqrt', max_depth=13, bootstrap=False)
    rfc.fit(x_train, y_train)
    print(rfc.score(x_train, y_train))
    print(rfc.score(x_test, y_test))

    predictions = rfc.predict(x_test)
    print(predictions)
    cm = confusion_matrix(y_test, predictions)
    print(cm)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Reds',
                xticklabels=['EO1', 'EC1', 'MED', 'EC2', 'EO2'],
                yticklabels=['EO1', 'EC1', 'MED', 'EC2', 'EO2'])
    plt.show()


def rf_test():
    os.chdir('saved_files/ml2_data/')
    df_train = pd.read_csv('features512_train.csv', header=None)
    df_test = pd.read_csv('features512_test.csv', header=None)

    train_data = np.array(df_train)
    x_train, y_train = train_data[:, :df_train.shape[1] - 1], train_data[:, df_train.shape[1] - 1]
    test_data = np.array(df_test)
    x_test, y_test = test_data[:, :df_test.shape[1] - 1], test_data[:, df_test.shape[1] - 1]

    pca = PCA(n_components=3)
    pca.fit(x_train)
    print(pca.explained_variance_ratio_)

    pc_train = pca.transform(x_train)
    pc_test = pca.transform(x_test)

    data_vis(x_train, y_train, pc_train)
    data_vis(x_test, y_test, pc_test)

    # rf_tuning(x_train, y_train)

    # {'n_estimators': 400, 'min_samples_split': 23, 'min_samples_leaf': 12, 'max_features': 'sqrt', 'max_depth': 11, 'bootstrap': True} C1

    rfc = RandomForestClassifier(n_estimators=400, min_samples_split=23, min_samples_leaf=12, max_features='sqrt', max_depth=11, bootstrap=True)
    rfc.fit(x_train, y_train)
    print(rfc.score(x_train, y_train))
    print(rfc.score(x_test, y_test))

    predictions = rfc.predict(x_test)
    print(predictions)
    cm = confusion_matrix(y_test, predictions)
    print(cm)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Reds',
                xticklabels=['EO1', 'EC1', 'MED', 'EC2', 'EO2'],
                yticklabels=['EO1', 'EC1', 'MED', 'EC2', 'EO2'])
    plt.show()


def rf_tuning(x_train, y_train):
    rfc_tuning = RandomForestClassifier()

    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
    max_features = ['log2', 'sqrt']
    max_depth = [int(x) for x in np.linspace(start=1, stop=15, num=15)]
    min_samples_split = [int(x) for x in np.linspace(start=2, stop=50, num=10)]
    min_samples_leaf = [int(x) for x in np.linspace(start=2, stop=50, num=10)]
    bootstrap = [True, False]
    param_dist = {'n_estimators': n_estimators,
                  'max_features': max_features,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf,
                  'bootstrap': bootstrap}
    rs = RandomizedSearchCV(rfc_tuning,
                            param_dist,
                            n_iter=100,
                            cv=3,
                            verbose=1,
                            n_jobs=-1,
                            random_state=0)
    rs.fit(x_train, y_train)
    print(rs.best_params_)


def data_vis(x, y, pc):
    plt.figure()
    df = pd.DataFrame(x)
    corr = df.corr()
    sns.heatmap(corr)
    plt.show()

    res = [[] for i in range(int(max(y)) + 1)]
    for i in range(len(y)):
        res[int(y[i])].append(pc[i])
    res = [np.array(res[i]) for i in range(len(res))]
    res = [pd.DataFrame(res[i]) for i in range(len(res))]
    res = [res[i][(np.abs(stats.zscore(res[i])) < 3).all(axis=1)] for i in range(len(res))]
    #res = [res[i].sample(n=100) for i in range(len(res))]
    res = [res[i].to_numpy() for i in range(len(res))]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(len(res)):
        ax.scatter(res[i][:, 0], res[i][:, 1], res[i][:, 2])
        print(res[i].shape)
    plt.show()
