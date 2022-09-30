import torch, time
from fairseq.models.roberta import RobertaModel

test_tasks = ['rte', ]
data_dir = {'mnli': 'MNLI/dev_matched.tsv', 'rte': 'RTE/dev.tsv'}
label_map = {'mnli': {0: 'contradiction', 1: 'neutral', 2: 'entailment'}, 'rte': {0: 'entailment', 1: 'not_entailment'}}
position = {'mnli': [8, 9], 'rte': [1, 2]}

if __name__ == '__main__':
     for task in test_tasks:
          print('-'*50 + 'Working on ' + task + '-'*50)
          model = RobertaModel.from_pretrained('/fsx/users/feisi/repos/models/RoBERTa_large', checkpoint_file=task+'.pt')
          ncorrect, nsamples = 0, 0
          if torch.cuda.is_available():
               print('-'*50 + 'Using cuda' + '-'*50)
               model.cuda()
          model.eval()
          start_time = time.time()
          with open('/fsx/users/feisi/repos/data/GLUE-baselines/glue_data/'+data_dir[task]) as fin:
               fin.readline()
               for index, line in enumerate(fin):
                    tokens = line.strip().split('\t')
                    sent1, sent2, target = tokens[position[task][0]], tokens[position[task][1]], tokens[-1]
                    en_tokens = model.encode(sent1, sent2)
                    model_output = model.predict('sentence_classification_head', en_tokens)
                    # import pdb; pdb.set_trace()
                    prediction = model_output.argmax().item()
                    prediction_label = label_map[task][prediction]
                    ncorrect += int(prediction_label == target)
                    nsamples += 1
          print('| Accuracy: ', float(ncorrect)/float(nsamples))
          print('Time cost: ', round(time.time()-start_time))
