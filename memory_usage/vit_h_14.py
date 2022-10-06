from torchvision.models import vit_h_14
from torchsummary import summary
model = vit_h_14()
summary(model, (3, 224, 224), 1)
