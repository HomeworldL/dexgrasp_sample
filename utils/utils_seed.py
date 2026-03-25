import random

import numpy as np
import torch

try:
    import open3d as o3d
except Exception:
    o3d = None


def set_seed(random_seed: int) -> None:
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if o3d is not None and hasattr(o3d.utility, "random") and hasattr(o3d.utility.random, "seed"):
        o3d.utility.random.seed(int(random_seed))
