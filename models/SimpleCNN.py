import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        """Initialization of the CNN.
        
        Args:
            - in_channels: Set it to 1 for B&W or to 3 for RGB.
            - num_classes: Number of predictions that have to be generated.
        
        Be careful:
            - This model assumes the images have size 28x28!
        """
        super(CNN, self).__init__()
        # kernel_size=(3,3), stride=(1,1) & padding=(1,1) will keep the same dimensions
        # size_output = floor((size_input + 2*padding_size - kernel_size) / stride_size) + 1 = size_input
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16,
                               kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=(3,3), stride=(1,1), padding=(1,1))
        # out_channels = 32
        # we're going to use 2 maxpooling, so we're gonna have the input 2 times
        # so 28 becomes 14 and 14 becomes 7
        self.fc1 = nn.Linear(32 * 7 * 7, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1) # flatten conserving the num of batches
        x = self.fc1(x)

        return x

