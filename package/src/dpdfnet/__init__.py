from typing import TYPE_CHECKING

__all__ = [
    "enhance",
    "enhance_file",
    "available_models",
    "download",
]

if TYPE_CHECKING:
    from .api import available_models, download, enhance, enhance_file


def __getattr__(name: str):
    if name in {"enhance", "enhance_file", "available_models", "download"}:
        from . import api

        return getattr(api, name)
    raise AttributeError(f"module 'dpdfnet' has no attribute '{name}'")
