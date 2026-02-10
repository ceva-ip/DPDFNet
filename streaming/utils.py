from torch import Tensor


def get_mag(x: Tensor) -> Tensor:
    assert x.shape[-1] == 2
    return (x ** 2).sum(dim=-1).sqrt()


def get_pow(x: Tensor) -> Tensor:
    assert x.shape[-1] == 2
    return (x ** 2).sum(dim=-1)







