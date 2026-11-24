import os

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
x = torch.ones(1, device="cuda") * (rank + 1)
dist.all_reduce(x)
print(f"rank {rank}/{dist.get_world_size()} device {torch.cuda.current_device()} all_reduce={x.item()}", flush=True)
dist.destroy_process_group()
