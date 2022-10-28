for size in 256 512
do
     python ../training.py --data_name CIFAR100 --model_name vgg19_bn --quant False --batch_size $size --epochs 200 --lr 0.05 --early_stop 90 100 --data_path ../../data --times 10
done
