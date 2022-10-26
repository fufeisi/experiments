#!/bin/sh
for bs in 512 1024 2048
do
     python self_no_quantize.py --batch-size $bs
done