import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import torch
import torchvision.models as models
from torchvision.models import ResNet34_Weights
import torch.nn as nn

import torch.optim as optim
from PIL import ImageFile
from torch.autograd import Variable
ImageFile.LOAD_TRUNCATED_IMAGES = True

use_cuda = torch.cuda.is_available()
import numpy as np

class DogDataset(Dataset):
    def __init__(self, main_dir, transform=None, resize=None):
        self.main_dir = main_dir
        self.transform = transform
        self.resize = resize
        
        sample_classes = [f.path for f in os.scandir(main_dir) if f.is_dir()]
        
        self.data = []
        self.class_map = {}
        for idx, class_path in enumerate(sample_classes):
            class_name = class_path.split("/")[-1]
            self.class_map[class_name] = idx
            for image in os.scandir(class_path):
                self.data.append([image.path, class_name])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        else:
            transformer = transforms.Compose([transforms.Resize((self.resize, self.resize)),
                                              transforms.ToTensor()])
            img = transformer(img)
        
        class_id = self.class_map[class_name]
        class_id = torch.tensor(class_id)
        
        return img, class_id


train_val_transform = transforms.Compose([transforms.RandomResizedCrop((224, 224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
                                     ])

test_transform = transforms.Compose([transforms.RandomResizedCrop((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
                                     ])

train_dataset = DogDataset(main_dir='./data/dogImages/train/', transform=train_val_transform)
val_dataset = DogDataset(main_dir='./data/dogImages/valid/', transform=train_val_transform)
test_dataset = DogDataset(main_dir='./data/dogImages/test/', transform=test_transform)



train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=50)
test_loader = DataLoader(test_dataset, batch_size=50)

loaders_transfer = {'train': train_loader, 'valid': val_loader, 'test': test_loader}



## TODO: Specify model architecture 
model_transfer = models.resnet34(weights=ResNet34_Weights.DEFAULT)

for param in model_transfer.parameters():
    param.requires_grad = False

model_transfer.fc = nn.Linear(512, 133)

# for name, param in model_transfer.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

if use_cuda:
    model_transfer = model_transfer.cuda()

criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.Adam(model_transfer.fc.parameters(), lr=0.03)
n_epochs = 10


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            
            loss.backward()
            optimizer.step()
            
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            with torch.no_grad():
                loss = criterion(model(data), target)
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        if valid_loss < valid_loss_min:
            valid_loss_min = valid_loss
            torch.save(model.state_dict(), save_path)
    # return trained model
    return model

def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))
    

model_transfer = train(n_epochs, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, './models/model_ada.pt')
model_transfer.load_state_dict(torch.load('./models/model_ada.pt'))

test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)