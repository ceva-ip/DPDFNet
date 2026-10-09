"""Small NumPy/ctypes adapter for the public C API; no ONNX dependency."""
import ctypes as ct
import hashlib
from pathlib import Path
import numpy as np

FP = ct.POINTER(ct.c_float)


class Info(ct.Structure):
    _fields_ = [("abi_version", ct.c_uint32), ("name", ct.c_char_p),
                ("weights_sha256", ct.c_char_p), ("sample_rate", ct.c_uint32),
                ("hop_samples", ct.c_uint32), ("spectrum_floats", ct.c_size_t),
                ("state_floats", ct.c_size_t), ("weight_floats", ct.c_size_t)]


def pointer(array):
    return array.ctypes.data_as(FP)


class Model:
    """One context and one stream. process() returns a borrowed output buffer."""
    def __init__(self, library, weights, size=8):
        self.handle = None
        self.size = size
        self.lib = lib = ct.CDLL(str(Path(library).resolve()))
        lib.dpdfnet_get_model_info.argtypes = [ct.c_int]
        lib.dpdfnet_get_model_info.restype = ct.POINTER(Info)
        lib.dpdfnet_create.argtypes = [ct.c_int, FP, ct.c_size_t]
        lib.dpdfnet_create.restype = ct.c_void_p
        lib.dpdfnet_process.argtypes = [ct.c_void_p, FP, FP, FP, FP]
        lib.dpdfnet_process.restype = ct.c_int
        lib.dpdfnet_init_state.argtypes = [ct.c_int, FP]
        lib.dpdfnet_init_state.restype = ct.c_int
        lib.dpdfnet_destroy.argtypes = [ct.c_void_p]
        lib.dpdfnet_destroy.restype = None
        lib.dpdfnet_owned_bytes.argtypes = [ct.c_void_p]
        lib.dpdfnet_owned_bytes.restype = ct.c_size_t
        info = lib.dpdfnet_get_model_info(size)
        if not info or info.contents.abi_version != 1:
            raise ValueError("Unsupported model or ABI")
        self.info = info.contents
        data = Path(weights).read_bytes()
        if hashlib.sha256(data).hexdigest() != self.info.weights_sha256.decode():
            raise ValueError("Weight SHA-256 does not match the model")
        weights_array = np.frombuffer(data, dtype="<f4")
        self.handle = lib.dpdfnet_create(size, pointer(weights_array), weights_array.size)
        if not self.handle:
            raise RuntimeError("Creation failed: check CPU support and memory availability")
        try:
            self.state = np.empty(self.info.state_floats, dtype=np.float32)
            self.output = np.empty((481, 2), dtype=np.float32)
            self.reset()
        except BaseException:
            self.close()
            raise

    def reset(self):
        if not self.handle:
            raise RuntimeError("Model is closed")
        if self.lib.dpdfnet_init_state(self.size, pointer(self.state)):
            raise RuntimeError("State reset failed")

    def process(self, spectrum):
        if not self.handle:
            raise RuntimeError("Model is closed")
        if (spectrum.shape != (481, 2) or spectrum.dtype != np.float32 or
                not spectrum.flags.c_contiguous or not np.isfinite(spectrum).all()):
            raise ValueError("Expected finite contiguous float32 spectrum [481,2]")
        if self.lib.dpdfnet_process(self.handle, pointer(spectrum), pointer(self.state),
                                   pointer(self.output), pointer(self.state)):
            raise RuntimeError("Inference failed")
        return self.output

    def close(self):
        if self.handle:
            self.lib.dpdfnet_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
