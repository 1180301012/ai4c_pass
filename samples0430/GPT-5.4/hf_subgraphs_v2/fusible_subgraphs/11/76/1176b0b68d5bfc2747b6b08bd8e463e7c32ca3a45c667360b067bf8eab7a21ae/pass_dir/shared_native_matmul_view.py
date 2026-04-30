import torch


@torch.fx.wrap
def native_batched_matmul_view(in_0, in_1, out_shape):
    return (in_1 @ in_0).view(out_shape)