import os
import tempfile

os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from dakd.train import rank_slice

B, D, K = 7, 5, 3


def _data():
    g = torch.Generator().manual_seed(0)
    return torch.randn(B, D, generator=g), torch.randint(0, K, (B,), generator=g)


def _worker(rank, world, init_file, out_path):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    torch.manual_seed(1)
    net = DDP(torch.nn.Linear(D, K))
    x, y = _data()
    a, b = rank_slice(B, rank, world)
    n_loc = b - a
    if n_loc == 0:
        a, b, fix = 0, 1, 0.0
    else:
        fix = 1.0
    loss = F.cross_entropy(net(x[a:b]), y[a:b])
    (loss * fix * world * (b - a) / B).backward()
    if rank == 0:
        torch.save(net.module.weight.grad.clone(), out_path)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 3, 8])
def test_grad_matches_single_process(world):
    torch.manual_seed(1)
    ref_net = torch.nn.Linear(D, K)
    x, y = _data()
    F.cross_entropy(ref_net(x), y).backward()
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "g.pt")
        mp.spawn(_worker, args=(world, os.path.join(td, "init"), out), nprocs=world, join=True)
        g = torch.load(out)
    assert torch.allclose(g, ref_net.weight.grad, atol=1e-6)


def test_rank_slice_partition():
    for n in (1, 7, 32):
        for w in (1, 4, 8):
            cover = [i for r in range(w) for i in range(*rank_slice(n, r, w))]
            assert cover == list(range(n))
