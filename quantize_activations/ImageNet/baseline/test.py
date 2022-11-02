from sklearn.utils import shuffle
import torchvision, torch

from load_data import load_data
import presets

data_path = "/datasets01/imagenet_full_size/061417"
train_path = "/datasets01/imagenet_full_size/061417/train"
val_path = "/datasets01/imagenet_full_size/061417/val"

# args = get_args_parser().parse_args([])
# dataset, dataset_test, train_sampler, test_sampler = load_data(traindir=train_path, valdir=val_path, args = args)
dataset_train, dataset_test = load_data(traindir=train_path, valdir=val_path)
# dataset = torchvision.datasets.ImageFolder(
#      data_path,
#      presets.ClassificationPresetTrain(
#      crop_size=224))
data_loader = torch.utils.data.DataLoader(
        dataset_train, shuffle=True,
        batch_size=32,
        num_workers=1,
        pin_memory=True
    )
for (image, target) in data_loader:
     print(target)
