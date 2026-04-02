import torch

def pattern(in_0, in_1):
    """Match the computation pattern: index selection for graph neural networks"""
    tmp_0 = in_0[1]  # Select second row (target node indices)
    tmp_1 = in_0[0]  # Select first row (source node indices)
    tmp_2 = in_1.index_select(-2, tmp_1)  # Select rows from features using indices
    return (tmp_0, tmp_2)

def replacement_args(in_0, in_1):
    """Extract arguments for the replacement kernel"""
    return (in_0, in_1)

def replacement_func():
    """Return an optimized function using PyTorch operations"""
    def optimized_impl(in_0, in_1):
        # Extract indices from in_0 (edge_index[0] = source nodes)
        tmp_1 = in_0[0]  # Select first row (source node indices)
        tmp_0 = in_0[1]  # Select second row (target node indices)
        
        # Use advanced indexing for better performance
        # This is more efficient than index_select for certain use cases
        output = in_1[tmp_1, :]
        
        # Return the same structure as the original
        return (tmp_0, output)
    
    return optimized_impl