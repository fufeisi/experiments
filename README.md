# This repository includes Feisi's code to explore applying quantization during neural network training stage. 

## Contents
- 'repeat_exp' directory is to repeat the experiment in '8-bit Optimizers via Block-wise Quantization' on GLUE dataset. 
- 'memory_usage' directory is for study the memory usage during neural network training stage. 
- 'ImageNet' directory is to quantize the activation maps for ResNet18 or ResNet50 on ImageNet dataset.

## Requirements
### repeat_exp
- fairseq
- bitsandbytes (bnb)
- Load roberta_large model and GLUE data from fairseq. 
- Needs to register 8 bit optimizer, bnb.optim.Adam8bit(), in fairseq/fairseq/optim/.
### ImageNet
- Download the ImageNet dataset and set --data to the path. 

## Use
- call repeat_exp/baseline/baseline_FT.sh to fine-tuning the roberta_large model with baseline Adam. 
- call repeat_exp/8bit_optimizer/baseline_FT.sh to fine-tuning the roberta_large model with 8bit Adam. 
- call ImageNet/{quan, quan_resnet50}.sh to train {resnet18, resnet50} with quantizing the activation maps. 
