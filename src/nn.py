from torch import nn


class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, 50), nn.Tanh())
        self.layer2 = nn.Sequential(nn.Linear(50, 150), nn.Tanh(), nn.Dropout(0.33))
        self.layer3 = nn.Sequential(nn.Linear(150, 50), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(50, 35), nn.ReLU())
        self.layer5 = nn.Sequential(nn.Linear(35, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
