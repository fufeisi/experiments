for size in 256 512
do
     python ../training.py --data_name CIFAR100 --model_name vgg19_bn --quant 1 --batch_size $size --epochs 200 --lr 0.05 --early_stop 70 90 --data_path ../../data --times 10
done
