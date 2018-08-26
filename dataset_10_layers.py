from torch.utils.data import Dataset
import numpy as np
import math
import torch
import matplotlib.pyplot as plt


class AdsCFT(Dataset):
    print('数据生成中...')

    def __init__(self, data_type='train', alpha=4, size=2000, a_1=1.5, b_1=0.2, split=1):
        self.data_type = data_type
        self.size = size
        self.split = split
        self.alpha = alpha
        self.data = []
        self.labels = []

        self.data.append(np.transpose([np.random.rand(self.alpha * self.size) * a_1]))
        self.data.append(np.transpose([np.random.rand(self.alpha * self.size) * 2 * b_1 - b_1]))
        self.data = np.concatenate(self.data, axis=1)
        self.labels = self.getLabels(self.ThetaPaiFin(self.data))

        while not (len(self.labels) == self.size):
            a1 = []
            a1.append(np.transpose([np.random.rand(self.alpha * self.size) * a_1]))
            a1.append(np.transpose([np.random.rand(self.alpha * self.size) * 2 * b_1 - b_1]))
            a1 = np.concatenate(a1, axis=1)
            b1 = self.getLabels(self.ThetaPaiFin(a1))

            self.data = np.concatenate((self.data, a1), axis=0)
            self.labels = np.concatenate((self.labels, b1), axis=0)
            # self.data = self.data.append(self.a, axis=1)
            print(
                '\n合并', '\n',
                # self.data, '\n',
                '合并后长度:', len(self.data)
            )
            # print('data:', self.data)
            # print('self.data[1]:', self.data[1])
            # print('labels:', self.labels)
            # 删除多余数据
            print('\n筛选:')

            self.pi = []
            self.phi = []
            self.pi_n = []
            self.phi_n = []
            pos_count = 0
            neg_count = 0
            pos_flag = False
            neg_flag = False
            pos_remove_count = 0
            neg_remove_count = 0
            for i in range(len(self.labels)):
                index = i - pos_remove_count - neg_remove_count
                    #print(index)
                if self.labels[index] == 1:
                    if pos_flag:
                        self.data = np.delete(self.data, index, 0)
                        self.labels = np.delete(self.labels, index, 0)
                        pos_remove_count += 1
                        if pos_remove_count % 10000 == 0:
                            print('pos_remove_count:', pos_remove_count)
                    else:
                            self.phi.append(self.data[index][0])
                            self.pi.append(self.data[index][1])
                            pos_count += 1
                else:
                    if neg_flag:
                        self.data = np.delete(self.data, index, 0)
                        self.labels = np.delete(self.labels, index, 0)
                        neg_remove_count += 1
                    else:
                        self.phi_n.append(self.data[index][0])
                        self.pi_n.append(self.data[index][1])
                        neg_count += 1

                if pos_count == self.size / 2:
                    pos_flag = True
                if neg_count == self.size / 2:
                    neg_flag = True
            print(
                'pos_count:', pos_count, '\n'
                'neg_count:', neg_count, '\n'
                'pos_remove_count:',pos_remove_count, '\n'
                'neg_remove_count:',neg_remove_count, '\n',
                len(self.data))
        plt.plot(self.phi, self.pi, 'yo')
        plt.plot(self.phi_n, self.pi_n, 'ro')
        plt.show()
        print('1:', len(self.data))
        print('2:', len(self.labels))

    def __getitem__(self, index):
        if self.data_type == 'train':
            return self.data[index], self.labels[index]
        elif self.data_type == 'test':
            return self.data[int(index + (self.size * (1-self.split)))], self.labels[int(index + (self.size *(1-self.split)))]

    def __len__(self):
        if self.data_type == 'train':
            return int(self.size * self.split)
        else:
            return round(self.size * (1-self.split))

    def ThetaPai(self, TP, eta):
        TP = np.transpose(TP)
        oTP = [TP[0] - 0.1 * TP[1], TP[1] + 0.1 * (3 * TP[1] / math.tanh(3 * eta) + TP[0] - TP[0] * TP[0] * TP[0])]
        oTP = np.concatenate([oTP])
        oTP = np.transpose(oTP)

        return oTP

    def ThetaPaiFin(self, TP):
        for i in np.arange(1, 0, -0.1):
            TP = self.ThetaPai(TP, i)

        return TP

    def getLabels(self, TP):
        TP = np.transpose(TP)
        return np.transpose([abs(20 * TP[1] + TP[0] - TP[0] * TP[0] * TP[0]) > 0.1]).astype(float)

    def get(self):
        np.set_printoptions(suppress=True)
        np.set_printoptions(threshold=np.inf)
        print('self.data:', self.data)
        print(len(self.data))
        return self.data

    def save(self):
        sav_e = (torch.from_numpy(self.data), torch.from_numpy(self.labels))
        return sav_e
        

if __name__ == "__main__":
    a = AdsCFT()
    b = a.save()
    c = a.get()
    np.save("initial_data.npy", c)
    torch.save(b, 'train_data.pt')
    print('b:', b)
