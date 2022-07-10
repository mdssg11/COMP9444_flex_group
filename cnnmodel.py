import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import torchvision.transforms as transforms
import os
import torchvision.datasets as datasets
from pickletools import optimize

class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,1)
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.fc1 = nn.Linear(246016, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def main():
    # Setup the trainining device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cnn = Network()
    net = cnn.to(device)
    print(f'device selected: {device}')

    #Import the dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    img_dir = 'imagenette2/'

    traindir = os.path.join(img_dir, 'train')
    valdir = os.path.join(img_dir, 'val')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([        
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,])
    )

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    )

    # Hyperparameter
    batch_size = 32
    learning_rate = 0.001

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)


    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)


    for epoch in range(10):  # loop over the dataset multiple times
        
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0
        test_acc = 0.0
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            
            
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # statistics
            train_loss += loss.item()
            pred = torch.max(outputs, 1)[1]
            train_correct = (pred == labels).sum()
            train_acc += train_correct.item()
            
        # To get the best learned model, we need to do some statisticcs.
        # After training, we pick the model with best validation accuracy.
        with torch.no_grad():
            net.eval()

            for inputs, labels in valloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                predicts = net(inputs)

                loss = criterion(predicts, labels)
                val_loss += loss.item()
                pred = torch.max(predicts, 1)[1]
                val_correct = (pred == labels).sum()
                val_acc += val_correct.item()

            net.train()
        print("Epoch %d" % epoch )

        print('Training Loss: {:.6f}, Training Acc: {:.6f}, Validation Acc: {:.6f}'.format(train_loss / (len(train_dataset))*32,train_acc / (len(train_dataset)), val_acc / (len(val_dataset))))        


    torch.save(net.state_dict(),'test.pth')
    print('Finished Training')

if __name__ == '__main__':
    main()