import torch
import torch.fft


def fftshift(x, axes=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    assert torch.is_tensor(x) is True
    if axes is None:
        axes = tuple(range(x.ndim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, int):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[axis] // 2 for axis in axes]
    return torch.roll(x, shift, axes)


def ifftshift(x, axes=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    assert torch.is_tensor(x) is True
    if axes is None:
        axes = tuple(range(x.ndim()))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[axis] // 2) for axis in axes]
    return torch.roll(x, shift, axes)


def fft2(data):
    assert data.shape[-1] == 2
    data = torch.view_as_complex(data)
    data = torch.fft.fftn(data, dim=(-2, -1), norm='ortho')
    data = torch.view_as_real(data)
    data = fftshift(data, axes=(-3, -2))
    return data


def ifft2(data):
    assert data.shape[-1] == 2
    data = ifftshift(data, axes=(-3, -2))
    data = torch.view_as_complex(data)
    data = torch.fft.ifftn(data, dim=(-2, -1), norm='ortho')
    data = torch.view_as_real(data)
    return data


def rfft2(data):
    assert data.shape[-1] == 1
    data = torch.cat([data, torch.zeros_like(data)], dim=-1)
    data = fft2(data)
    return data


def rifft2(data):
    assert data.shape[-1] == 2
    data = ifft2(data)
    data = data[..., 0].unsqueeze(-1)
    return data


def rA(data, mask):
    assert data.shape[-1] == 1
    data = torch.cat([data, torch.zeros_like(data)], dim=-1)
    data = fft2(data) * mask
    return data


def rAt(data, mask):
    assert data.shape[-1] == 2
    data = ifft2(data * mask)
    data = data[..., 0].unsqueeze(-1)
    return data


def rAtA(data, mask):
    assert data.shape[-1] == 1
    data = torch.cat([data, torch.zeros_like(data)], dim=-1)
    data = fft2(data) * mask
    data = ifft2(data)
    data = data[..., 0].unsqueeze(-1)
    return data
