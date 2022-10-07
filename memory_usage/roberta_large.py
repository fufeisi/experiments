import torch, time
from fairseq.models.roberta import RobertaModel

roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
if torch.cuda.is_available():
    print('-'*20 + 'Using cuda' + '-'*20)
    roberta.cuda()
roberta.train()
print(f'Memory Peak after loading model: {torch.cuda.max_memory_allocated(device=None)/(10**6)} MB.')

optimizer = torch.optim.Adam(roberta.parameters(), lr=0.0001)
words = ['hello', 'world']
for word in words:
     print('-'*50)
     optimizer.zero_grad()
     tokens = roberta.encode('Hello world!')
     output = roberta.predict('mnli', tokens)
     loss = output.mean()
     print(f'Memory Peak model+buffer: {torch.cuda.max_memory_allocated(device=None)/(10**6)} MB.')
     loss.backward()
     print(f'Memory Peak model+gradient: {torch.cuda.max_memory_allocated(device=None)/(10**6)} MB.')
     optimizer.step()
     print(f'Memory Peak model+gradient+optimizer: {torch.cuda.max_memory_allocated(device=None)/(10**6)} MB.')

