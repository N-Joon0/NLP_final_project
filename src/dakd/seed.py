import random

import numpy as np


def set_seed(seed: int) -> None:
    import torch
    from transformers import set_seed as hf_set_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)
