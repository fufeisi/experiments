import torch
from fairseq.models.roberta import RobertaModel

if __name__ == '__main__':
     models = ['roberta.base', 'roberta.large']
     models = models[:1]
     for model_name in models:
          model = RobertaModel.from_pretrained(model_name, checkpoint_file='model.pt')
          model.eval()
          print('-'*50 + model_name + '-'*50)
          print(model)
