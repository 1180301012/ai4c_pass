import torch


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def pattern(in_0, in_1):
    """
    Pattern: view -> transpose on in_1, permute -> reshape on in_0
    Graph: face-parsing_start46_end50_12 (fusible_subgraphs/7)
    - in_0: [32, 16384, 64] -> permute(0, 2, 1) -> [32, 64, 16384] -> reshape -> [32, 64, 128, 128]
    - in_1: [32, 16384, 64] -> view(32, -1, 1, 64) -> [32, 256, 1, 64] -> transpose(1, 2) -> [32, 1, 256, 64]
    """
    # in_1 path: view -> transpose
    tmp_0 = in_1.view(32, -1, 1, 64)
    tmp_1 = tmp_0.transpose(1, 2)
    
    # in_0 path: permute -> reshape
    tmp_2 = in_0.permute(0, 2, 1)
    tmp_3 = tmp_2.reshape(32, 64, 128, 128)
    
    return tmp_1, tmp_3


def replacement_func():
    # Return a simple inline function that does the exact same operations
    # The goal is to match the pattern but avoid overhead
    def replacement(in_0, in_1):
        # Direct chaining to minimize overhead - no intermediate variables
        return (
            in_1.view(32, -1, 1, 64).transpose(1, 2),
            in_0.permute(0, 2, 1).reshape(32, 64, 128, 128)
        )
    return replacement