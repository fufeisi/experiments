from fairseq_cli.hydra_train import cli_main 
import torch

if __name__ == "__main__":
     cli_main()
     print(f'Memory Peak: {torch.cuda.max_memory_allocated(device=None)/(10**6)} MB.')
