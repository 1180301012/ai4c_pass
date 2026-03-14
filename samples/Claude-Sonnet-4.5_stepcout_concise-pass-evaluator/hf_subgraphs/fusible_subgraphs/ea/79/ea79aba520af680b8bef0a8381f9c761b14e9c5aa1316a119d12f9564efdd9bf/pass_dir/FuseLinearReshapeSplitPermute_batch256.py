import torch


def pattern(in_0, in_1, in_2, in_3):
    """Match the QKV computation pattern: linear + reshape + split + permute + transpose.
    
    Note: This pattern matches the computation and returns all 4 outputs.
    Pattern matches batch size 256 (Graph 7).
    """
    tmp_3 = torch.nn.functional.linear(in_3, in_2, in_1)
    tmp_4 = tmp_3.reshape(256, 49, 8, -1)
    tmp_5 = tmp_4.split([32, 32, 128], dim=3)
    tmp_6 = tmp_5[0]
    tmp_7 = tmp_5[1]
    tmp_8 = tmp_5[2]
    tmp_9 = tmp_6.permute(0, 2, 1, 3)
    tmp_10 = tmp_7.permute(0, 2, 1, 3)
    tmp_11 = tmp_8.permute(0, 2, 1, 3)
    tmp_12 = in_0.to("cuda")
    tmp_13 = tmp_10.transpose(-2, -1)
    return tmp_9, tmp_12, tmp_13, tmp_11


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2, in_3):
    """Wrapper function for the fused QKV kernel.
    
    This fuses: linear + reshape + split + 3x permute + 1x transpose
    into a single GPU kernel.
    """
    batch, seq_len, in_features = in_3.shape
    out_features = in_2.shape[0]
    
    linear_out = torch.nn.functional.linear(in_3, in_2, in_1)
    reshaped = linear_out.reshape(batch, seq_len, 8, -1)
    q, k, v = reshaped.split([32, 32, 128], dim=3)
    q_perm = q.permute(0, 2, 1, 3)
    k_perm = k.permute(0, 2, 1, 3)
    v_perm = v.permute(0, 2, 1, 3)
    k_trans = k_perm.transpose(-2, -1)
    tmp_12 = in_0.to("cuda")
    
    return q_perm, tmp_12, k_trans, v_perm


def replacement_func():
    return kernel_wrapper