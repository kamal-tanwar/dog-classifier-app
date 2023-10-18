import os
from PIL import Image

import torch
import torchvision.models as models
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn

model_transfer = models.resnet34()

for param in model_transfer.parameters():
    param.requires_grad = False

model_transfer.fc = nn.Linear(512, 133)

state_dict = torch.load('./models/model_transfer.pt')
# print(state_dict.keys())

model_transfer.load_state_dict(state_dict)
model_transfer.cuda()

class_names = [f[4:].replace("_", " ").title() for f in os.listdir('./data/dogImages/train')]

def predict_breed_transfer(img_path):
    # load the image and return the predicted breed
    loader = transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])])
    image = Image.open(img_path).convert('RGB')
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    image = image.cuda()
    out = model_transfer(image)
    _, pred = torch.max(out, 1)
    class_name = class_names[pred.cpu().numpy().item()]
    return class_name


prediction = predict_breed_transfer(r'.\data\dogImages\test\014.Basenji\Basenji_01009.jpg')
print(prediction)