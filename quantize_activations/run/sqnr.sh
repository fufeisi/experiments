python ../training.py --data_name CIFAR10 --model_name vgg16 --quant True --batch_size 512 --epochs 3 --sqnr True --early_stop 90 100 --data_path ../../data --times 2
python ../training.py --data_name CIFAR10 --model_name vgg16 --quant True --batch_size 1024 --epochs 3 --sqnr True --early_stop 90 100 --data_path ../../data --times 2

python ../training.py --data_name CIFAR100 --model_name vgg19_bn --quant True --batch_size 256 --epochs 3 --sqnr True --early_stop 90 100 --data_path ../../data --times 2
python ../training.py --data_name CIFAR100 --model_name vgg19_bn --quant True --batch_size 512 --epochs 3 --sqnr True --early_stop 90 100 --data_path ../../data --times 2
