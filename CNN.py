import torch as T
from torch import nn 
import torchsummary as ts

class Net(T.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=0), #[1, 28, 28] -> [32, 24, 24]
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size = 3, stride = 2), #[32, 24, 24] -> [32, 12, 12] 
            nn.Dropout(0.25))

        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=0), #[32, 12, 12] -> [64, 8, 8]
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=3, stride=2) #[64, 8, 8] -> [64, 3, 3]
        )

        self.classifier = nn.Sequential(
            nn.Linear(144, 50),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(50, 10)
        )

    def forward(self, x):
       z = self.layer1(x)
       z = self.layer2(z)
       z = z.reshape(-1, 144)
       z = self.classifier(z)
       return z
    
def main():
   net = Net()
   ts.summary(net, (1,28,28))

if(__name__ == '__main__'):
   main()





