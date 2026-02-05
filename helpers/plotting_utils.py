"""
Written and maintained by Andreas Mentzelopoulos
Copyright (c) 2025, Andreas Mentzelopoulos. All Rights Reserved.

This code is the exclusive property of Andreas Mentzelopoulos
All associated materials (data, models, scripts) are the
exclusive property of Andreas Mentzelopoulos and LOBSTgER.

This code may be used openly and freely for research and education purposes. 
No part of this code may be used, copied, distributed, or modified for commercial use, 
without the prior written consent of Andreas Mentzelopoulos.

For permission requests, contact: Andreas Mentzelopoulos, ament@mit.edu.
"""

import torch
from matplotlib import pyplot as plt

@torch.no_grad()
def plot_image(x: torch.Tensor, title=None):
    assert x.ndim == 4 or x.ndim == 3, f"Expected tensor with three dimensions (unbatched) or four dimensions (batched), received {x.ndim}"
    plt.imshow(x.squeeze().detach().cpu().permute(1, 2, 0), interpolation="LANCZOS")
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.show()

