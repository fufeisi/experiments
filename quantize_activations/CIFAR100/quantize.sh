#!/bin/sh
for bs in 256 512 1024
do
     python quantized.py --batch_size $bs
done