from torchvision.models.resnet import resnet50
from torchsummary import summary
model = resnet50()
summary(model, (3, 256, 256), 1)
