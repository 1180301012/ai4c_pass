"""
Shared dispatch wrapper for all optimization passes.
All pass files return this same function from replacement_func(),
ensuring only one unique replacement function exists (satisfying replacement_func_limit).

Routes:
  "inv_freq"       : cast in_1 [freq] → [1, freq, 1] float32
  "position_ids"   : cast in_3 [b, seq] → [b, 1, seq] float32
  "attn_mask_2"    : attention mask for N=2
  "attn_mask_3"    : attention mask for N=3
  "attn_mask_64"   : attention mask for N=64
  "attn_mask_128"  : attention mask for N=128
  "attn_mask_256"  : attention mask for N=256
  "attn_mask_512"  : attention mask for N=512

Signature: dispatch_wrapper(arg0, arg1, route)
  - Single-input patterns pass arg0=in_X, arg1=in_X (dummy, ignored)
  - Two-input patterns pass arg0=in_0, arg1=in_2
"""

import torch
import triton
import triton.language as tl
from pass_dir.shared_kernels import attn_mask_kernel, cast_to_float32_kernel


@torch.fx.wrap
def dispatch_wrapper(arg0, arg1, route):
    # ------------------------------------------------------------------ #
    #  Route: inv_freq                                                     #
    #  arg0 = in_1 [freq_size] float16/bf16/fp32                          #
    #  output: [1, freq_size, 1] float32                                  #
    # ------------------------------------------------------------------ #
    if route == "inv_freq":
        in_1 = arg0
        freq_size = in_1.shape[0]
        out = torch.empty((1, freq_size, 1), dtype=torch.float32, device=in_1.device)
        BLOCK_SIZE = 64
        grid = ((freq_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        cast_to_float32_kernel[grid](in_1, out, freq_size, BLOCK_SIZE=BLOCK_SIZE)
        return out

    # ------------------------------------------------------------------ #
    #  Route: position_ids                                                 #
    #  arg0 = in_3 [batch, seq] int64                                     #
    #  output: [batch, 1, seq] float32                                    #
    # ------------------------------------------------------------------ #
    elif route == "position_ids":
        in_3 = arg0
        batch = in_3.shape[0]
        seq_len = in_3.shape[1]
        out = torch.empty((batch, 1, seq_len), dtype=torch.float32, device=in_3.device)
        N = batch * seq_len
        BLOCK_SIZE = 256
        grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        cast_to_float32_kernel[grid](in_3, out, N, BLOCK_SIZE=BLOCK_SIZE)
        return out

    # ------------------------------------------------------------------ #
    #  Route: attn_mask_2  (N=2)                                          #
    #  arg0 = in_0 [batch, 2] int64                                       #
    #  arg1 = in_2 [2] int64                                              #
    #  output: [batch, 1, 2, 2] bool                                      #
    # ------------------------------------------------------------------ #
    elif route == "attn_mask_2":
        in_0, in_2 = arg0, arg1
        batch = in_0.shape[0]
        N = 2
        out = torch.empty((batch, 1, N, N), dtype=torch.bool, device=in_0.device)
        total = batch * N * N
        BLOCK_SIZE = 16
        grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        attn_mask_kernel[grid](in_0, in_2, out, batch, N=2, BLOCK_SIZE=16)
        return out

    # ------------------------------------------------------------------ #
    #  Route: attn_mask_3  (N=3)                                          #
    # ------------------------------------------------------------------ #
    elif route == "attn_mask_3":
        in_0, in_2 = arg0, arg1
        batch = in_0.shape[0]
        N = 3
        out = torch.empty((batch, 1, N, N), dtype=torch.bool, device=in_0.device)
        total = batch * N * N
        BLOCK_SIZE = 16
        grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        attn_mask_kernel[grid](in_0, in_2, out, batch, N=3, BLOCK_SIZE=16)
        return out

    # ------------------------------------------------------------------ #
    #  Route: attn_mask_64  (N=64)                                        #
    # ------------------------------------------------------------------ #
    elif route == "attn_mask_64":
        in_0, in_2 = arg0, arg1
        batch = in_0.shape[0]
        N = 64
        out = torch.empty((batch, 1, N, N), dtype=torch.bool, device=in_0.device)
        total = batch * N * N
        BLOCK_SIZE = 1024
        grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        attn_mask_kernel[grid](in_0, in_2, out, batch, N=64, BLOCK_SIZE=1024)
        return out

    # ------------------------------------------------------------------ #
    #  Route: attn_mask_128  (N=128)                                      #
    # ------------------------------------------------------------------ #
    elif route == "attn_mask_128":
        in_0, in_2 = arg0, arg1
        batch = in_0.shape[0]
        N = 128
        out = torch.empty((batch, 1, N, N), dtype=torch.bool, device=in_0.device)
        total = batch * N * N
        BLOCK_SIZE = 1024
        grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        attn_mask_kernel[grid](in_0, in_2, out, batch, N=128, BLOCK_SIZE=1024)
        return out

    # ------------------------------------------------------------------ #
    #  Route: attn_mask_256  (N=256)                                      #
    # ------------------------------------------------------------------ #
    elif route == "attn_mask_256":
        in_0, in_2 = arg0, arg1
        batch = in_0.shape[0]
        N = 256
        out = torch.empty((batch, 1, N, N), dtype=torch.bool, device=in_0.device)
        total = batch * N * N
        BLOCK_SIZE = 1024
        grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        attn_mask_kernel[grid](in_0, in_2, out, batch, N=256, BLOCK_SIZE=1024)
        return out

    # ------------------------------------------------------------------ #
    #  Route: attn_mask_512  (N=512)                                      #
    # ------------------------------------------------------------------ #
    elif route == "attn_mask_512":
        in_0, in_2 = arg0, arg1
        batch = in_0.shape[0]
        N = 512
        out = torch.empty((batch, 1, N, N), dtype=torch.bool, device=in_0.device)
        total = batch * N * N
        BLOCK_SIZE = 1024
        grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        attn_mask_kernel[grid](in_0, in_2, out, batch, N=512, BLOCK_SIZE=1024)
        return out

    else:
        # Should never reach here
        return arg0