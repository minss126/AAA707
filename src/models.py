import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_channels=3, num_classes=10): # 기본값 3채널 고정
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        
        # CIFAR-10 (32x32) -> MaxPool 2번 -> 5x5 feature map
        self.fc_input_dim = 64 * 5 * 5
        
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x