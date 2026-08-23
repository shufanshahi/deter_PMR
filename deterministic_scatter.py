import torch


def deterministic_scatter_add(src, index, dim=0, dim_size=None):
    """CUDA-deterministic replacement for torch_scatter.scatter_add and PyG's
    default MessagePassing aggregation (both use atomic adds on CUDA, whose
    result depends on thread-completion order and is not fixed by any seed
    or by torch.backends.cudnn.deterministic).

    Built on Tensor.index_add_, which has a deterministic CUDA
    implementation when torch.use_deterministic_algorithms(True) is set
    (see train.py's seed_everything). index_add_'s contract already matches
    scatter_add's for the 1D-index case used throughout this codebase: index
    is 1D with length src.size(dim), giving each row of src a target row in
    the output along dim.
    """
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = src.new_zeros(out_shape)
    out.index_add_(dim, index, src)
    return out
