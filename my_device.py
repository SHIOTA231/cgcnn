import torch


def get_device(requested='auto'):
    """Resolve the requested execution device and reject unavailable CUDA."""
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA was requested, but PyTorch cannot access a CUDA device. '
            'Check the server GPU allocation and CUDA installation.'
        )
    return torch.device(requested)
