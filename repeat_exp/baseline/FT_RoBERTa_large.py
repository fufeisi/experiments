import torch
from fairseq.models.roberta import RobertaModel

# load model: 

roberta = RobertaModel.from_pretrained('roberta.large', checkpoint_file='model.pt')
roberta.eval()

roberta_mnli = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta_mnli.eval()
# (classification_heads): ModuleDict(
#      (mnli): RobertaClassificationHead(
#      (dense): Linear(in_features=1024, out_features=1024, bias=True)
#      (dropout): Dropout(p=0.3, inplace=False)
#      (out_proj): Linear(in_features=1024, out_features=3, bias=True)
#      )

# create a new head:

# load data: 

# fine-tune: 
