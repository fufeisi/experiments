import torch
import quantize_activations.quantization as qt
from torch.ao.ns.fx.utils import compute_sqnr

def grad_diff(model, criterion, x, target):
     temp = qt.quan
     res = [] # res[0] is nonquan, res[1] is quan. 
     for is_quan in [False, True]:
          qt.quan = is_quan
          res.append([])
          output = model(x)
          loss = criterion(output, target)
          loss.backward()
          for weight in model.parameters():
               res[-1].append(weight.grad.clone().detach())
               weight.grad.data.zero_()
     sqnr = [compute_sqnr(res[0][i], res[1][i]).item() for i in range(len(res[0]))]
     qt.quan = temp
     return sqnr


def grad_diff_roberta(model, inputs):
     temp = qt.quan
     res = [] # res[0] is nonquan, res[1] is quan. 
     for is_quan in [False, True]:
          qt.quan = is_quan
          res.append([])
          outputs = model(**inputs)
          loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
          loss = loss.mean()
          loss.backward()
          for weight in model.parameters():
               res[-1].append(weight.grad.clone().detach())
               weight.grad.data.zero_()
     sqnr = [compute_sqnr(res[0][i], res[1][i]).item() for i in range(len(res[0]))]
     qt.quan = temp
     return sqnr