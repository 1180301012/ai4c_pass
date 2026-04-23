import torch

from pass_dir.linear_triton_common import triton_linear_448_1536_bias


def _route_impl(in_0, in_1, in_2, in_3, batch_size: int):
    linear = triton_linear_448_1536_bias(in_3, in_2, in_1)
    tmp_4 = linear.reshape(batch_size, 49, 8, -1)
    split = tmp_4.split([32, 32, 128], dim=3)
    tmp_6 = split[0]
    tmp_7 = split[1]
    tmp_8 = split[2]
    tmp_9 = tmp_6.permute(0, 2, 1, 3)
    tmp_10 = tmp_7.permute(0, 2, 1, 3)
    tmp_11 = tmp_8.permute(0, 2, 1, 3)
    tmp_12 = in_0.to(torch.device(type='cuda', index=0))
    tmp_13 = tmp_10.transpose(-2, -1)
    return (tmp_9, tmp_12, tmp_13, tmp_11)


@torch.fx.wrap
def dispatch_qkv_replacement(in_0, in_1, in_2, in_3, route: str):
    if route == 'b1':
        return _route_impl(in_0, in_1, in_2, in_3, 1)
    if route == 'b4':
        return _route_impl(in_0, in_1, in_2, in_3, 4)
    if route == 'b8':
        return _route_impl(in_0, in_1, in_2, in_3, 8)
    if route == 'b128':
        return _route_impl(in_0, in_1, in_2, in_3, 128)
    if route == 'b256':
        return _route_impl(in_0, in_1, in_2, in_3, 256)
    if route == 'b512':
        return _route_impl(in_0, in_1, in_2, in_3, 512)
    raise ValueError(f'Unknown route: {route}')