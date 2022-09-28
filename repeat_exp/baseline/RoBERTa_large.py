import torch, time
from fairseq.models.roberta import RobertaModel

roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.eval()

label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
ncorrect, nsamples = 0, 0
if torch.cuda.is_available():
    print('-'*20 + 'Using cuda' + '-'*20)
    roberta.cuda()
roberta.eval()
start_time = time.time()
with open('/fsx/users/feisi/repos/data/GLUE-baselines/glue_data/MNLI/dev_matched.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[8], tokens[9], tokens[-1]
        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('mnli', tokens).argmax().item()
        prediction_label = label_map[prediction]
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))
print('Time cost: ', round(time.time()-start_time))
# gpurun: 174s; cpurun: 788s;
# Expected output: 0.9060