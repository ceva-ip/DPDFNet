# Opt-in fitted W7A8 kernels

These two source files are byte-identical copies of the measured
`scratch/w7_followup/pack_fit5` kernels, distributed here so consumers need
neither Python nor ignored research scratch files. Other operators, headers
and runtime dispatch are shared with the normal native implementation.

Select them through the parent CMake target with
`-DDPDF_EXPERIMENTAL_W7A8=ON`; do not replace production kernel files manually.
Both generated model sizes use these kernels. The C ABI exposes this profile
as `DPDF_PRESET_W7A8_FITTED` and rejects the ordinary W8A8 preset.

Source SHA-256:

```text
4b3e4c01cd0d09125decafedca140416bea4f3d66c553d98a218f8ac5c729e88  int8.c
ea78768b1239a672b704dd28a3f2a0989f83ae88d8a6bacedfcec4bd61b0224c  avx2.c
```

The parity check in `../verify_research_parity.py` verifies the resulting C
API against the preserved research libraries for both models. Run it again
if changing these files or their compiler settings; existing quality and
latency results should not be assumed to describe changed kernels.
