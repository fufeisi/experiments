for size in 256 512
do
     python ../training.py --data_name CIFAR10 --model_name vgg16 --quant 0 --batch_size $size --epochs 200 --early_stop 90 100 --data_path ../../data --times 20
done
