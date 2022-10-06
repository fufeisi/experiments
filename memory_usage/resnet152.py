from torchvision.models.resnet import resnet152
from torchsummary import summary
model = resnet152()
summary(model, (3, 256, 256), 1)
