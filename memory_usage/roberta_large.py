import torch, time
from fairseq.models.roberta import RobertaModel

roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
if torch.cuda.is_available():
    print('-'*20 + 'Using cuda' + '-'*20)
    roberta.cuda()
roberta.train()
print(f'Memory after loading model: {torch.cuda.memory_allocated(device=None)/(10**6)} MB.')

optimizer = torch.optim.Adam(roberta.parameters(), lr=0.0001)
words = ['hello', 'world']
for word in words:
     print('-'*50)
     optimizer.zero_grad()
     tokens = roberta.encode('Hello world!')
     output = roberta.predict('mnli', tokens)
     loss = output.mean()
     print(f'Memory model+buffer: {torch.cuda.memory_allocated(device=None)/(10**6)} MB.')
     loss.backward()
     print(f'Memory model+gradient: {torch.cuda.memory_allocated(device=None)/(10**6)} MB.')
     optimizer.step()
     print(f'Memory model+gradient+optimizer: {torch.cuda.memory_allocated(device=None)/(10**6)} MB.')

