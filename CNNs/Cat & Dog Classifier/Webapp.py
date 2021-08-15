from flask import Flask
from flask import render_template  
import requests
import torch
from torch._C import ParameterDict 
from model import model 
from PIL import Image 
from torchvision import transforms 
import numpy as np 


###Flask Part I ###


### Model Code ###
model.load_state_dict(torch.load('goodboy.pt', map_location=torch.device('cpu')))

image_test = Image.open("C:/Users/thaku/OneDrive/Desktop/sd.jpg")
image_test.show()
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(65),
    transforms.CenterCrop(64)
])

image_test = transforms(image_test)
output = model(image_test[None, ...])

ps = torch.exp(output)

top_p, top_class = ps.topk(1,dim=1)

top_class = top_class.numpy()

perdiction = top_class[0]

if perdiction == [0]:
    print('I see a Cat')
else:
    print('I see a Dog')