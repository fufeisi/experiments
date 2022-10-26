#!/bin/sh
for bs in 512 1024 2048
do
     python quantized.py --batch-size $bs
done