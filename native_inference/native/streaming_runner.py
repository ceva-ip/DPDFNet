"""Preallocated spectral-model runner for one stream and one calling thread.

The returned output and exposed state are borrowed buffers, overwritten by the
next call. Copy them explicitly when retaining history. Own the model separately;
do not close it or call it concurrently while this runner is in use.
"""
import numpy as np
from probe import initial_state, ptr


class StreamingRunner:
    def __init__(self, model):
        if not model.handle:
            raise ValueError('Model is closed')
        self.model = model
        self.input = np.empty((1, 1, 481, 2), dtype=np.float32)
        self.output = np.empty_like(self.input)
        self.state = initial_state(model)
        self._input_ptr = ptr(self.input)
        self._output_ptr = ptr(self.output)
        self._state_ptr = ptr(self.state)
        self._handle = model.handle
        self._process = model.lib.dpdf_model_process

    def _check_open(self):
        if self.model.handle != self._handle:
            raise RuntimeError('Model was closed or replaced')

    def reset(self):
        self._check_open()
        if self.model.lib.dpdf_model_init_state(self._state_ptr):
            raise RuntimeError('State initialization failed')

    def process(self, frame):
        self._check_open()
        if frame.shape != self.input.shape or frame.dtype != np.float32:
            raise ValueError('Expected float32 spectrum with shape (1, 1, 481, 2)')
        np.copyto(self.input, frame, casting='no')
        rc = self._process(self._handle, self._input_ptr, self._state_ptr,
                           self._output_ptr, self._state_ptr)
        if rc:
            raise RuntimeError(f'C model process failed: {rc}')
        return self.output
