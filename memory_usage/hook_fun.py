import torch

class Memory:
    def __init__(self):
        self.outputs = {'max_memory_allocated':[], 'memory_allocated': []}
        
    def __call__(self, module, module_in, module_out):
        self.outputs['max_memory_allocated'].append(
            [module._get_name(), f'{torch.cuda.max_memory_allocated()/(10**6)} MB'])
        self.outputs['memory_allocated'].append(
            [module._get_name(), f'{torch.cuda.memory_allocated()/(10**6)} MB'])
    def clear(self):
        self.outputs = {'max_memory_allocated':[], 'memory_allocated': []}


def register_max_memory(model):
     memory_usage = Memory()
     hook_handles = []
     for layer in model.modules():
          handle = layer.register_forward_hook(memory_usage)
          hook_handles.append(handle)
     return memory_usage
