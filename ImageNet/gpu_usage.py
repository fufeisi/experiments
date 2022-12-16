import subprocess as sp
import os
from threading import Thread , Timer
import sched, time

def get_gpu_info():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=utilization.gpu --format=csv"
    try:
        gpu_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    gpu_use_values = sum([int(item.split()[0]) for item in gpu_use_info])/len(gpu_use_info)
    return gpu_use_values

class PrintGPU:
    def __init__(self) -> None:
        self.is_print = True
        self.values = []
    def print_gpu(self, n):
        if self.is_print:
            Timer(n, lambda :self.print_gpu(n)).start()
        self.values.append(get_gpu_info())
    def cancel(self):
        self.is_print= False
    def report(self):
        if len(self.values) >= 1:
            print(f'Avg GPU usage: {sum(self.values)/len(self.values)}.')

if __name__ == '__main__':
    A = PrintGPU()
    A.print_gpu(5)
    time.sleep(20)
    A.cancel()
