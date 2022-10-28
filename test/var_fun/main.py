from my_def import b
# def store(x=None):
#      if x:
#           if 'my_store' in globals():
#                my_store.append(x)
#           else:
#                my_store = [x]
#      return my_store

if __name__ == '__main__':
     b.save(10)
     b.save(20)
     print(b.get())
