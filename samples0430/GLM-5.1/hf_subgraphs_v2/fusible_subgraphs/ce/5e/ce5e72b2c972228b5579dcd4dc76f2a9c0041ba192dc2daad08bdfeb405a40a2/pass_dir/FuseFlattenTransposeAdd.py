import torch
import triton
import triton.language as tl


def pattern(conv3d_output, pos_embeddings):
    tmp_4 = conv3d_output.flatten(2)
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = pos_embeddings.detach()
    tmp_7 = tmp_6.type_as(tmp_5)
    tmp_8 = tmp_7.to(device=torch.device(type='cuda', index=0), copy=True)
    tmp_9 = tmp_5 + tmp_8
    return tmp_9


def replacement_args(conv3d_output, pos_embeddings):
    return (conv3d_output, pos_embeddings)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_S': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_C': 32, 'BLOCK_S': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_C': 64, 'BLOCK_S': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_C': 16, 'BLOCK_S': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 32, 'BLOCK_S': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_S': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_C': 128, 'BLOCK_S': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_C': 32, 'BLOCK_S': 128}, num_warps=8, num_stages=3),
    ],
    key=['C', 'S'],
)
@triton.jit
def fused_transpose_add_kernel(
    conv3d_ptr, pos_ptr, output_ptr,
    C, S,
    stride_conv_b, stride_conv_c, stride_conv_s,
    stride_pos_b, stride_pos_s, stride_pos_c,
    stride_out_b, stride_out_s, stride_out_c,
    BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr,
):
    # Each program handles a BLOCK_C x BLOCK_S tile
    # Uses tiled transpose for better memory coalescing:
    # - Load conv3d tile in [C, S] order (coalesced along S, stride=1)
    # - Transpose in local memory to [S, C] order
    # - Add position embeddings and store in [S, C] order (coalesced along C, stride=1)
    pid = tl.program_id(0)
    b_idx = tl.program_id(1)

    num_s_blocks = tl.cdiv(S, BLOCK_S)
    num_c_blocks = tl.cdiv(C, BLOCK_C)

    # Map pid to (s_block, c_block) - s_block varies first for conv3d spatial locality
    s_block = pid % num_s_blocks
    c_block = pid // num_s_blocks

    s_start = s_block * BLOCK_S
    c_start = c_block * BLOCK_C

    s_offsets = s_start + tl.arange(0, BLOCK_S)  # [BLOCK_S]
    c_offsets = c_start + tl.arange(0, BLOCK_C)  # [BLOCK_C]

    s_mask = s_offsets < S
    c_mask = c_offsets < C

    # 2D masks for different layouts
    mask_cs = c_mask[:, None] & s_mask[None, :]  # [BLOCK_C, BLOCK_S] for conv3d
    mask_sc = s_mask[:, None] & c_mask[None, :]  # [BLOCK_S, BLOCK_C] for pos/output

    # Load conv3d tile in [C, S] layout (channel-major, spatial contiguous)
    # conv3d[b, c, s] at offset b*stride_conv_b + c*stride_conv_c + s*stride_conv_s
    conv_offsets_2d = b_idx * stride_conv_b + c_offsets[:, None] * stride_conv_c + s_offsets[None, :] * stride_conv_s
    conv_tile = tl.load(conv3d_ptr + conv_offsets_2d, mask=mask_cs, other=0.0)  # [BLOCK_C, BLOCK_S]

    # Transpose conv3d tile: [BLOCK_C, BLOCK_S] -> [BLOCK_S, BLOCK_C]
    conv_tile_t = tl.trans(conv_tile)  # [BLOCK_S, BLOCK_C]

    # Load pos tile in [S, C] layout
    # pos[b, s, c] at offset b*stride_pos_b + s*stride_pos_s + c*stride_pos_c
    pos_offsets_2d = b_idx * stride_pos_b + s_offsets[:, None] * stride_pos_s + c_offsets[None, :] * stride_pos_c
    pos_tile = tl.load(pos_ptr + pos_offsets_2d, mask=mask_sc, other=0.0)  # [BLOCK_S, BLOCK_C]

    # Add transposed conv3d + position embeddings
    out_tile = conv_tile_t + pos_tile  # [BLOCK_S, BLOCK_C]

    # Store output in [S, C] layout
    # output[b, s, c] at offset b*stride_out_b + s*stride_out_s + c*stride_out_c
    out_offsets_2d = b_idx * stride_out_b + s_offsets[:, None] * stride_out_s + c_offsets[None, :] * stride_out_c
    tl.store(output_ptr + out_offsets_2d, out_tile, mask=mask_sc)


@torch.fx.wrap
def fused_flatten_transpose_add(conv3d_output, pos_embeddings):
    # Transfer position embeddings to GPU with correct dtype (replaces detach + type_as + to)
    pos_gpu = torch.as_tensor(pos_embeddings, dtype=conv3d_output.dtype, device=conv3d_output.device)

    batch = conv3d_output.shape[0]
    C = conv3d_output.shape[1]
    D = conv3d_output.shape[2]
    H = conv3d_output.shape[3]
    W = conv3d_output.shape[4]
    S = D * H * W

    output = torch.empty((batch, S, C), dtype=conv3d_output.dtype, device=conv3d_output.device)

    # Conv3d strides: contiguous [batch, C, D, H, W] = [batch, C, S] in flattened view
    stride_conv_b = conv3d_output.stride(0)
    stride_conv_c = conv3d_output.stride(1)
    stride_conv_s = 1  # spatial stride (contiguous within channel)

    # Position embeddings strides: contiguous [batch, S, C]
    stride_pos_b = pos_gpu.stride(0)
    stride_pos_s = pos_gpu.stride(1)
    stride_pos_c = pos_gpu.stride(2)

    # Output strides: contiguous [batch, S, C]
    stride_out_b = output.stride(0)
    stride_out_s = output.stride(1)
    stride_out_c = output.stride(2)

    # Grid function for autotuning
    def grid(META):
        return (triton.cdiv(S, META['BLOCK_S']) * triton.cdiv(C, META['BLOCK_C']), batch)

    fused_transpose_add_kernel[grid](
        conv3d_output, pos_gpu, output,
        C, S,
        stride_conv_b, stride_conv_c, stride_conv_s,
        stride_pos_b, stride_pos_s, stride_pos_c,
        stride_out_b, stride_out_s, stride_out_c,
    )

    return output


def replacement_func():
    return fused_flatten_transpose_add