import torch
import torchvision
from PIL import  Image


# Define the image path
image_path = "frame3600.jpg"

print('starting')

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

classifiers = { 'ResNet50' : torchvision.models.resnet50(pretrained=True),
                'ResNet152' : torchvision.models.resnet152(pretrained=True),
                'DenseNet121' : torchvision.models.densenet121(pretrained=True),
                'DenseNet161' : torchvision.models.densenet161(pretrained=True),
                'EfficientNet B7' : torchvision.models.efficientnet_b7(pretrained=True),
                'MobileNet V2' : torchvision.models.mobilenet_v2(pretrained=True),
                'SSD' : torchvision.models.detection.ssd300_vgg16(pretrained=True),
                'Faster RCNN' : torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True),
                'Mask RCNN ' :  torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True),
                'RetinaNet' : torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
                }
import time


# Run each classifier on the image
for n in classifiers:
    t0 = time.time()
    model = classifiers[n]
    print('timing ', n)
    print(f"Number of parameters: {sum(torch.numel(param) for param in model.parameters())}")
    model.to('cuda:0') 
    for _ in range(2000): 
        image = Image.open(image_path)
        image = transform(image)
        image = image.to('cuda:0')
        with torch.no_grad():
            model.eval()
            output = model(image.unsqueeze(0))
#            probabilities = torch.nn.functional.softmax(output, dim=1)

    t1 = time.time()
    total = t1-t0
    print('total ', total)
    print()

