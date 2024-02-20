
import torch
import torchvision.models as models
import torchvision.transforms as transforms


import urllib.request
url = 'https://farm4.staticflickr.com/1301/4694470234_6f27a4f602_o.jpg'
filename = 'corgi.jpg'
urllib.request.urlretrieve(url, filename)

from PIL import Image
from matplotlib import pyplot as plt

img = Image.open('corgi.jpg')
plt.imshow(img)

model = torch.hub.load('pytorch/vision:v0.10.0','resnet18', pretrained=True)
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model

from torchvision import transforms

norm = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

inv_norm = transforms.Normalize(
    mean=[-0.485/0.229,-0.456/0.224,-0.406/0.225],
    std=[1/0.229,1/0.224,1/0.225]
)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    norm,
])

image_tensor = preprocess(img)
input_tensor = image_tensor.unsqueeze(0)
input_tensor = input_tensor.to(device)

with torch.no_grad():
    output = model(input_tensor)
    _,predicted_class = output.max(1)
print(f'Predicted class:{predicted_class.item()}')

