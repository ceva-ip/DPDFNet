# Optional XDNA2 WAV preview

This Linux example runs the 48 kHz mono PCM path with a separately
installed XDNA2 provider. The Python package and CPU examples have no
additional runtime dependencies. The preview uses the locally verified
R851 DPDFNet 2 and R877 DPDFNet 8 profiles.

Requirements are NumPy, an AMD XDNA2 device with a working XRT runtime,
and a trusted version-1 PCM provider catalog. Use the
[qualified local integration checkpoint](https://github.com/varyagnord/DPDFNet/blob/f09e16db5971f9a77f00e17e283996ea56eb74b4/optimized_inference/docs/XDNA2_LOCAL_PCM_CHECKPOINT.md)
to stage an available provider bundle. The example accepts mono 16-bit
PCM WAV at 48 kHz.

```sh
python examples/xdna2/enhance_wav.py input.wav output.wav \
    --providers /path/to/providers.json --model 8
```

Each `pcm_profiles` entry identifies the provider library, model and
provider manifest by path and SHA-256. Paths can be absolute or relative
to the catalog. The provider must support the PCM ABI version 1 and quiet
close via `r103_destroy_quiet_v1`. This example uses a bounded worker
process, sends the 480-sample frames in order, and compensates the verified
2,400-sample provider delay. The output keeps the input sample count.

The provider bundle contains the complete device graph and its assets.
A reproducible public build and release of those bundles is follow-up work.
Set up the driver and runtime on the host and use an idle accelerator for
the offline preview. CPU inference remains available through the regular
DPDFNet tools.
