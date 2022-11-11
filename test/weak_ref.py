import weakref, torch

d = weakref.WeakKeyDictionary()
for i in range(10):
     x = torch.rand([10, i+1])
     d[x] = x
for item in d:
     print(d[item])
