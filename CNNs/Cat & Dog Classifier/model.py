import torch 
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(3,16,3, padding=1) #64x64x3 --> Image is 64x64 with 3 colour channels so the dim is 64x646x3
        # (starting with 3 channles because 3 color channels (RGB), next layer has 16)
        
        
        self.conv2 = nn.Conv2d(16,32,3, padding=1) #32x32x16  64/2 (maxpool with size2) = 32 but this time 16 colour channels 
        self.conv3 = nn.Conv2d(32,64,3, padding=1) #16x16x32
        self.conv4 = nn.Conv2d(64,128,3, padding=1) #8*8*64
        self.conv5 = nn.Conv2d(128,256,3, padding=1) #4*4*128
        
        self.maxpool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(1024, 512) # output from the CNN layer would be in the form of (4x4x128)/2 = 1024 hence 1024 inputlayers  
        self.fc2 = nn.Linear(512, 264)
        self.fc3 = nn.Linear(264, 10)
        self.fc4 = nn.Linear(10,2)
        self.dropout = nn.Dropout(p=0.2) # each layer has a 20% chance of dropout to allow for better model training
  
    #Defining Feed Forward. Maxpooled(CNN) --> Fully Connected Linear Layer --> Log_Softmax Output For Dim =1 (probabilities)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.maxpool(F.relu(self.conv3(x)))
        x = self.maxpool(F.relu(self.conv4(x)))
        x = self.maxpool(F.relu(self.conv5(x)))
        x = x.view(x.shape[0],-1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.log_softmax(self.fc4(x),dim=1) 
        return x 

model = Network() #assigning the "model" variable to this network
