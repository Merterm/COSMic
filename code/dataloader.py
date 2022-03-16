import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json

class MetricDataset(Dataset):
    """docstring for MetricDataset. Define a dataset for training
    that contains both the training data and the labels.
    __getitem__ function returns one frame of spectrogram data."""

    def __init__(self, c1, l1, c2, l2, v1, y_s):
        super().__init__()
        self._c1 = [torch.tensor(c['CLS_features']).float() for c in c1]
        self._c2 = [torch.tensor(c['CLS_features']).float() for c in c2]

        self._l1 = [torch.tensor(l).float() for l in l1]
        self._l2 = [torch.tensor(l).float() for l in l2]

        self._v1 = [torch.tensor(v).float() for v in v1]

        self._y = [torch.tensor(y).float() for y in y_s]

    def __len__(self):
        return len(self._c1)

    def __getitem__(self, index):
        c1 = self._c1[index]
        c2 = self._c2[index]

        l1 = self._l1[index]
        l2 = self._l2[index]

        v1 = self._v1[index]

        y = self._y[index]
        return c1, l1, c2, l2, v1, y


def load_data(batch_size):
    datadir = '/ihome/malikhani/mei13/projects/BLEURT_google/output/'

    # 1.Load the data
    print('Loading the dataset...')
    # train
    trainC1 = []
    with open((datadir + 'gen_feat.jsonl')) as f:
        for line in f.readlines():
            trainC1.append(json.loads(line))
    trainC2 = []
    with open((datadir + 'ref_feat.jsonl')) as f:
        for line in f.readlines():
            trainC2.append(json.loads(line))
    trainL1 = np.load((datadir + 'gen_labels.npy'),
                              encoding='bytes', allow_pickle=True)
    trainL2 = np.load((datadir + 'ref_labels.npy'),
                              encoding='bytes', allow_pickle=True)
    trainV1 = np.load((datadir + 'img_feats.npy'),
                              encoding='bytes', allow_pickle=True)
    trainY = np.load((datadir + 'ratings.npy'),
                              encoding='bytes', allow_pickle=True)

    # validation
    valC1 = []
    with open((datadir + 'gen_feat.jsonl')) as f:
        for line in f.readlines():
            valC1.append(json.loads(line))
    valC2 = []
    with open((datadir + 'ref_feat.jsonl')) as f:
        for line in f.readlines():
            valC2.append(json.loads(line))
    valL1 = np.load((datadir + 'gen_labels.npy'),
                              encoding='bytes', allow_pickle=True)
    valL2 = np.load((datadir + 'ref_labels.npy'),
                              encoding='bytes', allow_pickle=True)
    valV1 = np.load((datadir + 'img_feats.npy'),
                              encoding='bytes', allow_pickle=True)
    valY = np.load((datadir + 'ratings.npy'),
                              encoding='bytes', allow_pickle=True)

    # test
    testC1 = []
    with open((datadir + 'gen_feat.jsonl')) as f:
        for line in f.readlines():
            testC1.append(json.loads(line))
    testC2 = []
    with open((datadir + 'ref_feat.jsonl')) as f:
        for line in f.readlines():
            testC2.append(json.loads(line))
    testL1 = np.load((datadir + 'gen_labels.npy'),
                              encoding='bytes', allow_pickle=True)
    testL2 = np.load((datadir + 'ref_labels.npy'),
                              encoding='bytes', allow_pickle=True)
    testV1 = np.load((datadir + 'img_feats.npy'),
                              encoding='bytes', allow_pickle=True)
    testY = np.load((datadir + 'ratings.npy'),
                              encoding='bytes', allow_pickle=True)

    print('TrainC1',len(trainC1))
    print('TrainC2',len(trainC2))
    print('TrainL1',len(trainL1))
    print('TrainL2',len(trainL2))
    print('TrainV1',len(trainV1))
    print('TrainY',len(trainY))

    print('ValC1',len(valC1))
    print('ValC2',len(valC2))
    print('ValL1',len(valL1))
    print('ValL2',len(valL2))
    print('ValV1',len(valV1))
    print('ValY',len(valY))

    print('TestC1',len(testC1))
    print('TestC2',len(testC2))
    print('TestL1',len(testL1))
    print('TestL2',len(testL2))
    print('TestV1',len(testV1))
    print('TestY',len(testY))

    print(trainV1.shape)

    input_cptn_size = len(trainC1[0]['CLS_features'])
    input_label_size = trainL1.shape[1]
    input_visual_size = trainV1.shape[1]

    print('Caption Input Size', input_cptn_size)
    print('Caption Label Size',input_label_size)
    print('Caption Visual Size',input_visual_size)

    # 4. Turn the data into datasets
    train_dataset = MetricDataset(trainC1[:10], trainL1[:10], trainC2[:10], trainL2[:10],
                                    trainV1[:10], trainY[:10])
    val_dataset = MetricDataset(valC1[:10], valL1[:10], valC2[:10], valL2[:10],
                                    valV1[:10], valY[:10])
    test_dataset = MetricDataset(testC1[:10], testL1[:10], testC2[:10], testL2[:10],
                                    testV1[:10], testY[:10])

    # train_dataset = MetricDataset(trainC1, trainL1, trainC2, trainL2,
    #                                 trainV1, trainY)
    # val_dataset = MetricDataset(valC1, valL1, valC2, valL2,
    #                                 valV1, valY)
    # test_dataset = MetricDataset(testC1, testL1, testC2, testL2,
    #                                 testV1, testY)

    # 5. Put everything into a dataloader
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True,\
                                pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True,\
                                pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,\
                                pin_memory=True)

    return (train_dataloader, val_dataloader, test_dataloader, input_cptn_size, \
            input_label_size, input_visual_size)
