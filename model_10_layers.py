
import torch
from torch import nn, optim
import argparse
import numpy as np
import torch.nn.functional as F
from jt_loader import JtLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import time
import os

since = time.time()
# print ('(1)')
parser = argparse.ArgumentParser(description='dataset.npy')
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
# print ('(2)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.set_default_tensor_type(torch.DoubleTensor)
# print ('len(trainset):',len(trainset))
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
    JtLoader('./train_data.pt', train=True),
    batch_size=args.batch_size, shuffle=True, **kwargs)
# print(len(trainset))
# print(len(train_loader))
# print ('(3)')
yita = [0, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

# print ('(4)')
class Neuralnetwork(nn.Module):
    def __init__(self):
        super(Neuralnetwork, self).__init__()
        self.layer1 = nn.Linear(1, 1, bias=False)
        self.layer2 = nn.Linear(1, 1, bias=False)
        self.layer3 = nn.Linear(1, 1, bias=False)
        self.layer4 = nn.Linear(1, 1, bias=False)
        self.layer5 = nn.Linear(1, 1, bias=False)
        self.layer6 = nn.Linear(1, 1, bias=False)
        self.layer7 = nn.Linear(1, 1, bias=False)
        self.layer8 = nn.Linear(1, 1, bias=False)
        self.layer9 = nn.Linear(1, 1, bias=False)
        self.layer10 = nn.Linear(1, 1, bias=False)
        self.layers=[self.layer1, self.layer2, self.layer3,
                     self.layer4, self.layer5, self.layer6,
                     self.layer7, self.layer8, self.layer9, self.layer10]

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('self.layer1') != -1:
            torch.nn.init.normal_(m.weight.data, 1, 1)
            # torch.nn.init.normal(m.bias.data)
        if classname.find('self.layer2') != -1:
            torch.nn.init.normal_(m.weight.data,1/yita[2],1)
            # torch.nn.init.normal(m.bias.data)
        if classname.find('self.layer3') != -1:
            torch.nn.init.normal_(m.weight.data,1/yita[3],1)
            # torch.nn.init.normal(m.bias.data)
        if classname.find('self.layer4') != -1:
            torch.nn.init.normal_(m.weight.data,1/yita[4],1)
            # torch.nn.init.normal(m.bias.data)
        if classname.find('self.layer5') != -1:
            torch.nn.init.normal_(m.weight.data,1/yita[5],1)
            # torch.nn.init.normal(m.bias.data)
        if classname.find('self.layer6') != -1:
            torch.nn.init.normal_(m.weight.data,1/yita[6],1)
            # torch.nn.init.normal(m.bias.data)
        if classname.find('self.layer7') != -1:
            torch.nn.init.normal_(m.weight.data,1/yita[7],1)
            # torch.nn.init.normal(m.bias.data)
        if classname.find('self.layer8') != -1:
            torch.nn.init.normal_(m.weight.data,1/yita[8],1)
            # torch.nn.init.normal(m.bias.data)
        if classname.find('self.layer9') != -1:
            torch.nn.init.normal_(m.weight.data,1/yita[9],1)
            # torch.nn.init.normal(m.bias.data)
        if classname.find('self.layer9') != -1:
            torch.nn.init.normal_(m.weight.data,1/yita[10],1)
            # torch.nn.init.normal(m.bias.data)

    def forward(self, x):
        # wights=torch.zeros([9])
        # i=0
        # for layer in self.layers:
        #     for ele in layer.parameters():
        #         wights[i]=ele
        #         i+=1
        # print(wights)
        # print('第一层权重',self.layer1.parameters())
        x = torch.cat(((x[:, 0] - 0.1 * x[:, 1]).view(x.shape[0], 1),
                       (0.1 * x[:, 0].view(x.shape[0], 1) + x[:, 1].view(x.shape[0], 1) -
                        0.1 * x[:, 0].view(x.shape[0], 1) ** 3 + 0.1 * self.layer1(x[:, 1].view(x.shape[0], 1))).view(
                           x.shape[0], 1)), dim=1)
        # print('1:',predict = model(Variable(self.layer1)))
        x = torch.cat(((x[:, 0] - 0.1 * x[:, 1]).view(x.shape[0], 1),
                       (0.1 * x[:, 0].view(x.shape[0], 1) + x[:, 1].view(x.shape[0], 1) -
                        0.1 * x[:, 0].view(x.shape[0], 1) ** 3 + 0.1 * self.layer2(x[:, 1].view(x.shape[0], 1))).view(
                           x.shape[0], 1)), dim=1)
        # print('2:',self.layer2)

        x = torch.cat(((x[:, 0] - 0.1 * x[:, 1]).view(x.shape[0], 1),
                       (0.1 * x[:, 0].view(x.shape[0], 1) + x[:, 1].view(x.shape[0], 1) -
                        0.1 * x[:, 0].view(x.shape[0], 1) ** 3 + 0.1 * self.layer3(x[:, 1].view(x.shape[0], 1))).view(
                           x.shape[0], 1)), dim=1)
        # print('3:',self.layer3)
        x = torch.cat(((x[:, 0] - 0.1 * x[:, 1]).view(x.shape[0], 1),
                       (0.1 * x[:, 0].view(x.shape[0], 1) + x[:, 1].view(x.shape[0], 1) -
                        0.1 * x[:, 0].view(x.shape[0], 1) ** 3 + 0.1 * self.layer4(x[:, 1].view(x.shape[0], 1))).view(
                           x.shape[0], 1)), dim=1)
        # print('4:',self.layer4)
        x = torch.cat(((x[:, 0] - 0.1 * x[:, 1]).view(x.shape[0], 1),
                       (0.1 * x[:, 0].view(x.shape[0], 1) + x[:, 1].view(x.shape[0], 1) -
                        0.1 * x[:, 0].view(x.shape[0], 1) ** 3 + 0.1 * self.layer5(x[:, 1].view(x.shape[0], 1))).view(
                           x.shape[0], 1)), dim=1)
        # print('5:',self.layer5)
        x = torch.cat(((x[:, 0] - 0.1 * x[:, 1]).view(x.shape[0], 1),
                       (0.1 * x[:, 0].view(x.shape[0], 1) + x[:, 1].view(x.shape[0], 1) -
                        0.1 * x[:, 0].view(x.shape[0], 1) ** 3 + 0.1 * self.layer6(x[:, 1].view(x.shape[0], 1))).view(
                           x.shape[0], 1)), dim=1)
        # print('6:',self.layer6)
        x = torch.cat(((x[:, 0] - 0.1 * x[:, 1]).view(x.shape[0], 1),
                       (0.1 * x[:, 0].view(x.shape[0], 1) + x[:, 1].view(x.shape[0], 1) -
                        0.1 * x[:, 0].view(x.shape[0], 1) ** 3 + 0.1 * self.layer7(x[:, 1].view(x.shape[0], 1))).view(
                           x.shape[0], 1)), dim=1)
        # print('7:',self.layer7)
        x = torch.cat(((x[:, 0] - 0.1 * x[:, 1]).view(x.shape[0], 1),
                       (0.1 * x[:, 0].view(x.shape[0], 1) + x[:, 1].view(x.shape[0], 1) -
                        0.1 * x[:, 0].view(x.shape[0], 1) ** 3 + 0.1 * self.layer8(x[:, 1].view(x.shape[0], 1))).view(
                           x.shape[0], 1)), dim=1)
        # print('8:',self.layer8)
        x = torch.cat(((x[:, 0] - 0.1 * x[:, 1]).view(x.shape[0], 1),
                       (0.1 * x[:, 0].view(x.shape[0], 1) + x[:, 1].view(x.shape[0], 1) -
                        0.1 * x[:, 0].view(x.shape[0], 1) ** 3 + 0.1 * self.layer9(x[:, 1].view(x.shape[0], 1))).view(
                           x.shape[0], 1)), dim=1)
        # print('9:',self.layer9)
        x = torch.cat(((x[:, 0] - 0.1 * x[:, 1]).view(x.shape[0], 1),
                       (0.1 * x[:, 0].view(x.shape[0], 1) + x[:, 1].view(x.shape[0], 1) -
                        0.1 * x[:, 0].view(x.shape[0], 1) ** 3 + 0.1 * self.layer10(x[:, 1].view(x.shape[0], 1))).view(
                           x.shape[0], 1)), dim=1)
        # print('10:',self.layer10)

        x = (torch.nn.functional.tanh(100 * (x[:, 1] - 0.1)) - torch.nn.functional.tanh(100 * (x[:, 1] + 0.1)) + 2) / 2
        return x.view(x.shape[0], 1)


model = Neuralnetwork()
model = model.to(device)

# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)



def train(epoch):
    model.train()
    criterion = torch.nn.L1Loss()
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(batch_idx)
        if args.cuda:
            data, target, criterion = data.to(device), target.to(device), criterion.to(device)
        optimizer.zero_grad()
        output = model(data)
        # wights_numpy=wights.detach().numpy()
        # print(output)
        # print(target)
        # print('mse:',criterion(output, target))
        paras = []
        for para in model.parameters():
            paras += list(para)
#        para.write(str(paras))
        # print('----------------------')
        sum=0
        for i in range(len(paras)-1):
            # print (i)
            sum += 0.03*(((10-i)/10)**4)*(paras[i+1]-paras[i])**2
        # print('sum:', sum)
        loss = criterion(output, target) + sum
        loss.backward()
        optimizer.step()
        # break
# 打印参数、把参数写入TXT文本。
        if batch_idx % args.log_interval == 0:
            print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                     epoch, batch_idx * len(data), len(train_loader.dataset),
                     100. * batch_idx / len(train_loader), loss.item()))
            para = model.state_dict()
            weights = []
            for i, j in para.items():
                weights.append(j)
            # print('weights:', weights)
            weights = str(torch.cat(weights, 1))
            # print('weights:', weights)
            weights = weights.replace('\n', '')
            weights = weights.replace('        ', '')
            paras = open('./paras.txt', 'r+')
            paras.seek(0, 2)
            paras.write(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f},    {}    {} \n'.format(
                 epoch, batch_idx * len(data), len(train_loader.dataset),
                 100. * batch_idx / len(train_loader), loss.item(), weights,
                 time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
            )
            paras.close()


if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        train(epoch)

# 画图
params = model.state_dict()
for k, v in params.items():
    print(k, v)  # 打印网络中的变量名
x_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
x_2 = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
y_2 = [3/np.tanh((3*1)), 3/np.tanh((3*0.9)), 3/np.tanh((3*0.8)), 3/np.tanh((3*0.7)), 3/np.tanh((3*0.6)), 3/np.tanh((3*0.5)),
       3/np.tanh((3*0.4)), 3/np.tanh((3*0.3)), 3/np.tanh((3*0.2)), 3/np.tanh((3*0.1))]
y = torch.cat([
    params['layer1.weight']
    , params['layer2.weight']
    , params['layer3.weight']
    , params['layer4.weight']
    , params['layer5.weight']
    , params['layer6.weight']
    , params['layer7.weight']
    , params['layer8.weight']
    , params['layer9.weight']
    , params['layer10.weight']
])
# print('y:', y)
y = y.cpu().numpy()
print('y:', y)

plt.plot(x_1, y_2, label='True metric')
plt.plot(x_1, y, label='Emergent metric')
# 保存图片并显示
plt.legend()
plt.savefig("C:\\Users\\jingtan\\Desktop\\bs_10_epoch_10k_lr_-4.png")
plt.show()

print('总时长:{:.2f} h'.format((time.time() - since)/3600))

# 20秒后关机
# os.system("shutdown -s -t 20")
