class A:
     def __init__(self):
          A.store = []
     def save(self, x):
          A.store.append(x)
     def get(self):
          return A.store
b = A()
