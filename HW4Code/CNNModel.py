import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #Convolution Blocks -> Convolution - BatchNorm - ReLU - MaxPool
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5, stride = 1, padding = 2),
                        nn.BatchNorm2d(num_features = 16),          
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size = 2, stride = 2),
        )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 1, padding = 2),
                        nn.BatchNorm2d(num_features = 32),  
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.conv3 = nn.Sequential(
                        nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 1, padding = 2),
                        nn.BatchNorm2d(num_features = 64),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.conv4 = nn.Sequential(
                        nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 2),
                        nn.BatchNorm2d(num_features = 128),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size = 2, stride = 2)
        )                  
        #Fully Connected layer
        self.fc = nn.Linear(1152, 10)

    def forward(self, x):
        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)        
        h5 = h4.view(h4.size(0), -1) # flatten the output of conv4 to (batch_size, 1152)
        output = self.fc(h5)
        return output