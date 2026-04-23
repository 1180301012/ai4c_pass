import torch
import triton
import triton.language as tl

from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor


# Match the exact original graph and return both observable outputs.
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv3d = torch.conv3d(in_6, in_1, in_0, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1)
    tmp_7 = conv3d.flatten(2)
    tmp_8 = tmp_7.transpose(1, 2)
    tmp_9 = in_2.tile([1, 1, 1])
    tmp_10 = torch.cat((tmp_9, tmp_8), dim=1)
    tmp_11 = tmp_10 + in_3
    tmp_12 = torch.nn.functional.dropout(tmp_11, 0.0, False, False)
    tmp_13 = torch.nn.functional.layer_norm(tmp_12, (768,), in_5, in_4, 1e-06)
    return (tmp_12, tmp_13)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


@triton.jit
def _fused_compose_and_layernorm_kernel(
    patch_ptr,
    bias_ptr,
    cls_ptr,
    pos_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    ln_out_ptr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    HIDDEN = 768

    row = tl.program_id(0)
    row_offs = row * HIDDEN
    safe_patch_row = tl.where(row > 0, row - 1, 0)

    sum_acc = tl.zeros((), dtype=tl.float32)
    sqsum_acc = tl.zeros((), dtype=tl.float32)

    for start in tl.static_range(0, 768, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < 768

        pos_vals = tl.load(pos_ptr + row_offs + cols, mask=mask, other=0.0).to(tl.float32)
        cls_vals = tl.load(cls_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias_vals = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        patch_vals = tl.load(patch_ptr + safe_patch_row * HIDDEN + cols, mask=mask, other=0.0).to(tl.float32)

        x = tl.where(row == 0, cls_vals + pos_vals, patch_vals + bias_vals + pos_vals)
        sum_acc += tl.sum(x, axis=0)
        sqsum_acc += tl.sum(x * x, axis=0)

    mean = sum_acc / HIDDEN
    var = sqsum_acc / HIDDEN - mean * mean
    var = tl.maximum(var, 0.0)
    inv_std = tl.rsqrt(var + eps)

    for start in tl.static_range(0, 768, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < 768

        pos_vals = tl.load(pos_ptr + row_offs + cols, mask=mask, other=0.0).to(tl.float32)
        cls_vals = tl.load(cls_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias_vals = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        patch_vals = tl.load(patch_ptr + safe_patch_row * HIDDEN + cols, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        beta = tl.load(beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        x = tl.where(row == 0, cls_vals + pos_vals, patch_vals + bias_vals + pos_vals)
        y = (x - mean) * inv_std
        y = y * gamma + beta

        tl.store(out_ptr + row_offs + cols, x, mask=mask)
        tl.store(ln_out_ptr + row_offs + cols, y, mask=mask)


@torch.fx.wrap
def whole_vivit_patch_embed_stem(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    bias = unwrap_tensor(in_0)
    weight = unwrap_tensor(in_1)
    cls_token = unwrap_tensor(in_2)
    pos_embed = unwrap_tensor(in_3)
    ln_bias = unwrap_tensor(in_4)
    ln_weight = unwrap_tensor(in_5)
    pixel_values = unwrap_tensor(in_6)

    # Re-express the conv3d patch embedding as a single GEMM:
    # [1,3,10,224,224] --unfold--> [980,1536] @ [1536,768] -> [980,768]
    patches = pixel_values.unfold(2, 2, 2).unfold(3, 16, 16).unfold(4, 16, 16)
    patches = patches.permute(0, 2, 3, 4, 1, 5, 6, 7).contiguous()
    patches = patches.reshape(980, 1536)
    weight_2d = weight.reshape(768, 1536)
    patch_out = patches @ weight_2d.transpose(0, 1)

    out = torch.empty_like(pos_embed)
    ln_out = torch.empty_like(pos_embed)

    _fused_compose_and_layernorm_kernel[(981,)](
        patch_out,
        bias,
        cls_token.reshape(768),
        pos_embed,
        ln_weight,
        ln_bias,
        out,
        ln_out,
        1e-6,
        BLOCK_SIZE=256,
        num_warps=4,
        num_stages=2,
    )

    return (out, ln_out)


def replacement_func():
    return whole_vivit_patch_embed_stem