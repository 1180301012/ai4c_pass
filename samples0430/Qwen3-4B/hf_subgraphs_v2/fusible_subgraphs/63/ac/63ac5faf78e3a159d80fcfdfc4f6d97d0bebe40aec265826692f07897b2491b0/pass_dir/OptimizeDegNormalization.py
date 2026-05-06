import torch
import triton
import triton.language as tl

def pattern(deg, row_indices, col_indices, edge_weights):
    tmp_2 = deg.pow(-0.5)
    tmp_3 = tmp_2 == torch.inf
    tmp_4 = tmp_2.masked_fill(tmp_3, 0)
    tmp_5 = tmp_4[row_indices]
    tmp_6 = tmp_5 * edge_weights
    tmp_7 = tmp_4[col_indices]
    tmp_8 = tmp_6 * tmp_7
    return tmp_8
def replacement_args(deg, row_indices, col_indices, edge_weights):
    return (deg, row_indices, col_indices, edge_weights)

defunc = triton.jit
@triton.jit
def optimized_kernel(\n    deg_ptr,
    row_indices_ptr,
    col_indices_ptr,
    edge_weights_ptr,
    out_ptr,
    n_edges: tl.int32,
    n_nodes: tl.int32,
    BLOCK_SIZE: tl.constexpr
):
    # Each thread handles a block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (offsets < n_edges)

    # Load row indices and their values
    row_indices = tl.load(row_indices_ptr + block_start + offsets, mask=mask, other=-1)
    col_indices = tl.load(col_indices_ptr + block_start + offsets, mask=mask, other=-1)
    edge_weights = tl.load(edge_weights_ptr + block_start + offsets, mask=mask, other=0.0)

    # Check if indices are valid
    valid_row = (row_indices >= 0) & (row_indices < n_nodes)
    valid_col = (col_indices >= 0) & (col_indices < n_nodes)

    # Load deg values from preprocessed deg tensor
    deg_row = tl.load(deg_ptr + row_indices, mask=valid_row, other=0.0)
    deg_col = tl.load(deg_ptr + col_indices, mask=valid_col, other=0.0)

    # Compute the weighted product
    row_weighted = deg_row * edge_weights
    out = row_weighted * deg_col

    # Store the result
    tl.store(out_ptr + block_start + offsets, out, mask=mask)

defunc = torch.fx.wrap
@torch.fx.wrap
def kernel_wrapper(deg, row_indices, col_indices, edge_weights):
    n_edges = row_indices.numel()
    n_nodes = deg.shape[0]
    BLOCK_SIZE = 256
    num_blocks = (n_edges + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.zeros(n_edges, dtype=deg.dtype, device=deg.device)

    optimized_kernel[(num_blocks,)](
        deg_ptr=deg,
        row_indices_ptr=row_indices,
        col_indices_ptr=col_indices,
        edge_weights_ptr=edge_weights,
        out_ptr=out,
        n_edges=n_edges,
        n_nodes=n_nodes,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
def replacement_func():
    return kernel_wrapper