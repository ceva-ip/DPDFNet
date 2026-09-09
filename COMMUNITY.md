# Built with DPDFNet

A community directory of applications, plugins, libraries and experiments built with [DPDFNet](https://github.com/ceva-ip/DPDFNet) models. Integrations through sherpa-onnx and converted ONNX, Core ML or NNEF models are included.

**Last reviewed: 9 September 2026.** This snapshot lists **60 distinct public projects or integrations**, including **20 experimental or research entries**. Related components and duplicate repositories are grouped.

Entries are supported by public implementation evidence or authoritative runtime documentation. “Implemented” describes the inspected source; it does not certify release readiness or audio quality. Optional and experimental uses are marked. This is a growing directory, not a guarantee that every public or private adopter has been found.

## Microphones and audio plugins

| Project | How it uses DPDFNet | Status | Evidence |
| --- | --- | --- | --- |
| [HushMic](https://github.com/Fovty/HushMic) | System-wide virtual microphone for Linux/PipeWire. DPDFNet 48 kHz; Rust/ONNX Runtime. | Implemented | [Source](https://github.com/Fovty/HushMic/blob/c70e0c9a6ffec2154e2bcfae7e2a8b767678f115/crates/hushmic-denoiser/src/denoiser.rs) |
| [obs-dpdfnet](https://github.com/orienw/obs-dpdfnet) | Native speech-enhancement filter for OBS Studio. DPDFNet2/8 48 kHz HR; ONNX Runtime; Windows is the primary tested platform. | Implemented | [Source](https://github.com/orienw/obs-dpdfnet/blob/c8805aace268ff17805a0439cc73cc58925935c7/src/dpdfnet-processor.cpp) |
| [DPDFNet VST3 — Outsidepro Arts](https://github.com/outsidepro-arts/DPDFNet) | Windows VST3 speech-denoising plugin with embedded weights. DPDFNet8 48 kHz HR; C++/ONNX Runtime. | Implemented | [Source](https://github.com/outsidepro-arts/DPDFNet/blob/e413b61be1b07c8834dc3d225bdbf52b4e16e840/src/dsp/DpdfDenoiser.cpp) |
| [DPDFNet LADSPA — BigLinux](https://github.com/biglinux/dpdfnet-ladspa) | LADSPA noise-suppression plugins for Linux audio routing. Rust/OpenVINO; model-specific builds. | Implemented | [Source](https://github.com/biglinux/dpdfnet-ladspa/blob/fec73b3fd99802bde78ba97ea211890d86adb9a3/src/lib.rs) |
| [JenyaDereverb2](https://github.com/nepoimannykh/dereverb-vst3) · [alternate repo](https://github.com/jenyanepoimannykh-it/dereverb-vst3) | Speech-cleanup plugin for macOS, in VST3 and Audio Unit formats. Embedded DPDFNet ONNX model. | Implemented | [Source](https://github.com/nepoimannykh/dereverb-vst3/blob/9514611ce044915ae796da415c43881825983119/Source/NeuralEnhancer.cpp) |
| [ODE](https://github.com/jcanizalez/ode) | macOS virtual microphone and output-audio cleanup. DPDFNet2 48 kHz HR through sherpa-onnx. | Implemented | [Source](https://github.com/jcanizalez/ode/blob/c810eb47728adad7cc567973d8c54c384912583a/Sources/ODEKit/Denoiser.swift) |
| [VoiceFocus](https://github.com/maddylaneeee/VoiceFocus-macOS) | macOS system/app-output speech cleanup. DPDFNet2 48 kHz HR converted to Core ML. | Implemented | [Source](https://github.com/maddylaneeee/VoiceFocus-macOS/blob/2ade17e7638cde5bd8e6a1b8b93cf9c839140746/Sources/VoiceFocusCore/CoreMLSpeechEnhancer.swift) |
| [VoxMic](https://github.com/melody0709/vox_mic) | Uses an Android phone as a Windows microphone, with optional denoising. DPDFNet2 48 kHz HR through sherpa-onnx on the Windows host. | Optional | [Source](https://github.com/melody0709/vox_mic/blob/921c140f6d76181b2ed2242f6ce81d4988489d34/src/dsp/dpdfnet_processor.cpp) |
| [Soundboard](https://github.com/subenoeva/soundboard) | Virtual-microphone soundboard with a neural noise-suppression effect. DPDFNet2 48 kHz HR; custom Python streaming adapter. | Implemented | [Source](https://github.com/subenoeva/soundboard/blob/HEAD/src/soundboard/effects/neural.py) |

## Audio, transcription and speech applications

| Project | How it uses DPDFNet | Status | Evidence |
| --- | --- | --- | --- |
| [Open-Lyrics / openlrc](https://github.com/zh-plus/openlrc) | Noise suppression before transcription, translation and subtitle creation. Calls the dpdfnet Python package. | Optional | [Source](https://github.com/zh-plus/openlrc/blob/7b7f6186c34c1ba4c5ce7d61c5c3896f8896d2ad/openlrc/preprocess.py) |
| [denoise_gui](https://github.com/ErwinLiYH/denoise_gui) | GUI for cleaning audio tracks in video files. Multiple DPDFNet models through the Python package. | Optional | [Source](https://github.com/ErwinLiYH/denoise_gui/blob/6585efe89683a6629ce5fe4b1e69ed68b77290a2/denoiser.py) |
| [KIGTTS](https://github.com/LHT02/KIGTTS) | Android speech/communication app with selectable enhancement modes. DPDFNet2/4 through sherpa-onnx. | Optional | [Source](https://github.com/LHT02/KIGTTS/blob/b411a4fb69694acc1eba7f37df93ad796299bf5f/android-app/app/src/main/java/com/kgtts/app/audio/Engines.kt) |
| [PySAVR](https://github.com/sandergs92/PySAVR) | Synchronized audio/video recording with post-capture noise reduction. DPDFNet through sherpa-onnx. | Optional | [Source](https://github.com/sandergs92/PySAVR/blob/a74291847b8ef3ce378665d5118b46bf71d95f39/av_recorder.py) |
| [WhisperProject](https://github.com/fmam0126/WhisperProject) | Desktop transcription with a speech-enhancement preprocessing option. DPDFNet8 through a .NET/sherpa-onnx integration. | Optional | [Source](https://github.com/fmam0126/WhisperProject/blob/2f6ebe3c11b55c4d9bab61ab4ad17905ede7731d/WhisperProject.Core/class/VoiceEmphasisFilter.cs) |
| [Melina Audio](https://github.com/MichaelxBelmonte/melina-audio) | Audio enhancement with selectable neural speech-cleanup profiles. Multiple DPDFNet variants through sherpa-onnx; Android implementation verified. | Optional | [Source](https://github.com/MichaelxBelmonte/melina-audio/blob/40c0d393fb3224a17562e4fa6f55eea6e1210e62/app/src/main/java/it/michelina/focus/audio/NeuralSpeechEnhancer.kt) |
| [whisper-for-subs](https://github.com/hungshinlee/whisper-for-subs) | Subtitle/transcription workflow with speech enhancement. DPDFNet TFLite inference; weights fetched from Ceva-IP/DPDFNet. | Optional | [Source](https://github.com/hungshinlee/whisper-for-subs/blob/169cb57aea96f3f1e77732e9d9eefc9d920a55ed/speech_enhancer.py) |
| [Subflow](https://github.com/mochizuki0323/subflow) | Live subtitles and translation with an optional denoising stage. DPDFNet through sherpa-onnx. | Optional | [Source](https://github.com/mochizuki0323/subflow/blob/9d91565d205307150b3fcd95468ea6e4955d28d5/src/backend/audio/denoiser.cpp) |
| [voice-enh](https://github.com/nepoimannykh/voice-enhancer-vst) | Speech-recording enhancement CLI, including a DaVinci Resolve external-process launcher. DPDFNet8 48 kHz HR through the Python package. | Implemented | [Source](https://github.com/nepoimannykh/voice-enhancer-vst/blob/a2eff5f1fe96a3d0a8d9af9984d845b49863e4d1/voice_enh/dpdf_runner.py) |
| [Poeditcal](https://github.com/poetapps/Poeditcal) | macOS audio editing, cleanup and transcription. DPDFNet2 48 kHz HR through sherpa-onnx. | Optional | [Source](https://github.com/poetapps/Poeditcal/blob/bf1d689889cb5e791e6e73cbf3579a81e47c7d11/Sources/PoetAudio/AIDenoiser.swift) |
| [Capiau Talho](https://github.com/fernangcortes/capiau-talho-v03) | Local video-editing workflow with dialogue/audio cleanup. DPDFNet2 48 kHz HR through sherpa-onnx. | Optional | [Source](https://github.com/fernangcortes/capiau-talho-v03/blob/eeb268c3bc6c1d3db4ae3eb2e9f16f20798a7ea0/src/media/audio_denoise.py) |
| [qwen-audio-toolkits](https://github.com/QwenAudio/qwen-audio-toolkits) | Local audio-AI workspace with a denoising operation. DPDFNet2 48 kHz HR through a Rust/sherpa-onnx backend. | Optional | [Source](https://github.com/QwenAudio/qwen-audio-toolkits/blob/HEAD/src-tauri/src/audio_processing.rs) |
| [Cadence](https://github.com/bykcyc/Cadence) | Dictation and meeting-transcription preprocessing. DPDFNet4 through the Python package. | Optional | [Source](https://github.com/bykcyc/Cadence/blob/HEAD/ml/denoise.py) |
| [Anki Audio Quick Editor](https://github.com/ganqqwerty/anki-audio-tools) | Anki add-on for cleaning audio attached to learning cards. Invokes a bundled DPDFNet executable. | Optional | [Source](https://github.com/ganqqwerty/anki-audio-tools/blob/HEAD/addon/anki_audio_quick_editor/audio_noise_reduction_bundled.py) |
| [Voxera](https://github.com/aintoniodev/voxera) | Audio/video speech-enhancement CLI with multiple backends. DPDFNet2 adapter using the dpdfnet Python package. | Optional | [Source](https://github.com/aintoniodev/voxera/blob/1103f5c4eabbee5ebb143f2ed696fa9b10c4744c/src/voxera/backends/dpdfnet.py) |
| [Model Chain — Voice Pipeline](https://github.com/RJSprod/SD-Neo-ModelSwitchRefiner) | Cleans generated speech before a bandwidth-restoration stage. DPDFNet StreamEnhancer in a dedicated Python worker. | Optional | [Source](https://github.com/RJSprod/SD-Neo-ModelSwitchRefiner/blob/d63e1a6a32a8944062dac463881111160d1afe6a/pipeline_worker/worker.py) |
| [Music Assassin](https://github.com/mAbEngineers/Music-Assassin) | Optional speech cleanup after audio/music separation. DPDFNet through sherpa-onnx in the maintained v0.4.4 script. | Optional | [Source](https://github.com/mAbEngineers/Music-Assassin/blob/e9327e3a43fceba32ae2541acd53d0cc67cbbfd4/MusicAssassinGUI_v0.4.4.py) |
| [Nexus Anima / Enhanced Voice System](https://github.com/kekw2077/enhanced-voice-system) | Voice-assistant speech-input preprocessing. DPDFNet baseline through sherpa-onnx. | Optional | [Source](https://github.com/kekw2077/enhanced-voice-system/blob/fe00fabc408fb4094ca094422e218c80aaf8f02d/test1/sidecar/stt_engine.py) |
| [audio-filter](https://github.com/artur-matkowski/audio-filter) | Visual audio-filtering tool with an optional live speech denoiser. DPDFNet2 48 kHz HR through sherpa-onnx. | Optional | [Source](https://github.com/artur-matkowski/audio-filter/blob/HEAD/backend/src/audiofilter/ml/denoiser.py) |
| [DPDFNet WebAssembly](https://github.com/jenyanepoimannykh-it/dpdfnet-webassembly) | Browser-based audio/video speech cleanup. DPDFNet8 48 kHz HR; ONNX Runtime Web/WASM. | Implemented | [Source](https://github.com/jenyanepoimannykh-it/dpdfnet-webassembly/blob/3e63a51d7fc7c72bf3f2d120c105a6d9a37c6c89/src/models.ts) |

## Libraries, model ports and runtime examples

| Project | How it uses DPDFNet | Status | Evidence |
| --- | --- | --- | --- |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | Cross-platform speech-enhancement runtime and language bindings. Offline and online DPDFNet support, with pretrained model distribution. | Implemented | [Source](https://k2-fsa.github.io/sherpa/onnx/speech-enhancement/dpdfnet.html) |
| [DeepFilterNet-mlx](https://github.com/kylehowells/DeepFilterNet-mlx) | Swift/MLX/Core ML speech-enhancement implementation for Apple Silicon. DPDFNet4 16 kHz and DPDFNet8 48 kHz HR; batch and stateful processing. | Implemented | [Source](https://github.com/kylehowells/DeepFilterNet-mlx/blob/c6c4b9dc98c62bfa90ed66c5fd4f5d54e88c7b4f/Sources/DeepFilterNetCoreML/DPDFNetCoreMLStreamingEngine.swift) |
| [Audio-Denoiser-ONNX](https://github.com/DakeQQ/Audio-Denoiser-ONNX) | DPDFNet export, optimization and inference examples. End-to-end audio ONNX graph and ONNX Runtime inference. | Implemented | [Source](https://github.com/DakeQQ/Audio-Denoiser-ONNX/blob/8ab880c312b9022435a7bc1aa054a2e672d3897a/DPDFNet/Inference_DPDFNet_ONNX.py) |
| [react-native-sherpa-onnx](https://github.com/XDcobra/react-native-sherpa-onnx) | React Native speech-enhancement SDK. DPDFNet through sherpa-onnx; native iOS/Android wrappers. | Implemented | [Source](https://github.com/XDcobra/react-native-sherpa-onnx/blob/67ca2e5122ecb3972bbb58f4bd1fdc510496e692/ios/enhancement/sherpa-onnx-enhancement-wrapper.mm) |
| [mica-voice](https://github.com/lets-mica/mica-voice) | Java speech-services toolkit with a denoising service. DPDFNet through sherpa-onnx. | Implemented | [Source](https://github.com/lets-mica/mica-voice/blob/71efe92eac4dcda29231cb7ef949b480f5a51e4b/mica-voice-core/src/main/java/net/dreamlu/mica/voice/denoise/OfflineSpeechDenoiserService.java) |
| [audiosronnx](https://github.com/TigreGotico/audiosronnx) | Python/ONNX audio enhancement toolkit with a DPDFNet engine. Multiple 8/16/48 kHz model variants. | Implemented | [Source](https://github.com/TigreGotico/audiosronnx/blob/e31e2207dbaf452f9696d056d3931b64cfe36a13/audiosronnx/engines/dpdfnet.py) |
| [caitunai/audio](https://github.com/caitunai/audio) | Go audio-processing library with a DPDFNet denoiser. Custom ONNX Runtime integration. | Implemented | [Source](https://github.com/caitunai/audio/blob/bb2a123ff0ec13ef133659407054fae791dbb031/audio_denoise_dpdfnet.go) |
| [stt_flutter_package](https://github.com/sebbom/stt_flutter_package) | Flutter speech-recognition package with optional preprocessing. Offline DPDFNet through sherpa-onnx. | Optional | [Source](https://github.com/sebbom/stt_flutter_package/blob/51f2b01ce48c574983f3aeb249d721fcbc7e159a/lib/src/audio/audio_processor.dart) |
| [CaseHub Blocks — speech-sherpa](https://github.com/casehubio/blocks) | Reusable Java speech-denoising component. DPDFNet through sherpa-onnx native/FFM bindings. | Implemented | [Source](https://github.com/casehubio/blocks/blob/HEAD/speech-sherpa/src/main/java/io/casehub/blocks/speech/sherpa/SherpaOnnxSpeechDenoiser.java) |
| [vad-filter-onnx](https://github.com/lxp3/vad-filter-onnx) | C++ speech-denoising runtime and ONNX export tooling. Dedicated DPDFNet model adapter with stateful waveform processing. | Implemented | [Source](https://github.com/lxp3/vad-filter-onnx/blob/dae77fa8cb62ffd4e6f9699a19c603d0234c645b/vad-filter-onnx/denoise/dpdfnet-denoise-model.cc) |
| [torch-to-nnef — DPDFNet example](https://github.com/sonos/torch-to-nnef) | Exports DPDFNet to NNEF and runs it in Rust WAV-cleaner examples. NNEF/tract; streaming and pulse-mode examples. | Example | [Source](https://github.com/sonos/torch-to-nnef/blob/2793f1ca073d1f3e30feb938b9e9e016e523fc50/examples/speech_enhancement/dpdfnet/README.md) |

## Experimental applications and research

These projects contain DPDFNet-specific implementations, but their use is experimental, diagnostic, under development, or part of research tooling. They should not be read as a list of production deployments.

| Project | How it uses DPDFNet | Status | Evidence |
| --- | --- | --- | --- |
| [HushMic-Ahuenno](https://github.com/qwertyuiopftft/HushMic-Ahuenno) | HushMic derivative with additional noise-suppression and speaker-gating experiments. DPDFNet-based fork with project-specific model/voice experiments. | Experimental | [Source](https://github.com/qwertyuiopftft/HushMic-Ahuenno/blob/f64139314fbff9f24befdd994c23a9eed3476ad0/crates/hushmic-denoiser/src/model.rs) |
| [ASR Multi-Speaker Streaming](https://github.com/3002tad/ASR_Multi_Speaker_Streaming) | Meeting-transcription pipeline with an optional enhancement branch. Stateful DPDFNet through sherpa-onnx, with raw/enhanced blending. | Experimental | [Source](https://github.com/3002tad/ASR_Multi_Speaker_Streaming/blob/6cf6371302e8ce1ccbc1c5385bce6667f5e19605/backend/audio_pipeline.py) |
| [Realtime_Denoise](https://github.com/gulliman9000/Realtime_Denoise) | Real-time denoising tool for audio/radio listening. DPDFNet StreamEnhancer Python backend. | Prototype | [Source](https://github.com/gulliman9000/Realtime_Denoise/blob/780c4e0cb6c7e4a37c3c1f45aa5e59d75400ca7b/realtime_denoise.py) |
| [Music Assassin Live](https://github.com/mAbEngineers/music-assassin-live) | Linux/PipeWire live-audio processing application. Dedicated DPDFNet streaming processor. | Experimental | [Source](https://github.com/mAbEngineers/music-assassin-live/blob/06f0ac8eb65cbffe26f93961347289b7aae9582d/assassin_live/processors/dpdfnet.py) |
| [Drone Audio Feed Noise Reduction via AI](https://github.com/SlipStream90/Drone-Audio-Feed-Noise-Reduction-via-AI) | Drone-audio cleanup combining DSP and a neural post-filter. DPDFNet post-filter and associated fine-tuning code. | Research prototype | [Source](https://github.com/SlipStream90/Drone-Audio-Feed-Noise-Reduction-via-AI/blob/50be9c81a633b18930c116b37d58cee3153fe03a/ai_postfilter.py) |
| [Project Vaani](https://github.com/Subho4531/Project-Vaani) | ESP32 audio capture feeding a server-side enhancement pipeline. DPDFNet8 48 kHz HR in the Python server. | Prototype | [Source](https://github.com/Subho4531/Project-Vaani/blob/24cbcdf021559767a5533ee8867755ffa3e4dbe3/server/core/engine.py) |
| [Pownin](https://github.com/sanju1v4/Pownin-V2.M1.01) | Phone-recording workflow comparing raw and enhanced transcription. DPDFNet4 16 kHz in a laptop/server Python backend. | Prototype | [Source](https://github.com/sanju1v4/Pownin-V2.M1.01/blob/f5e376fec379f4d6c3f1340575f2ff2c53452d67/enhance.py) |
| [de-denoiser](https://github.com/finaea/de-denoiser) | Speech-denoising comparison and debugging harness. Stateful DPDFNet ONNX processors. | Research tooling | [Source](https://github.com/finaea/de-denoiser/blob/97b30dea9a47037f4947e6342012ab2ed5c2df38/nc_bench/processors/spec_onnx.py) |
| [denoize](https://github.com/penguin425/denoize) | Rust denoising backend with an experimental CLAP plugin path. Dedicated DPDFNet ONNX backend. | Experimental | [Source](https://github.com/penguin425/denoize/blob/cbb0fa14e5a7638029f2fdcf481f525753d24335/src/backend/dpdfnet.rs) |
| [GPTIRL](https://github.com/d8dzmf5mfn/GPTIRL) | Wake-word/voice-control experimentation with a DPDFNet front end. Streaming DPDFNet adapter through sherpa-onnx. | Diagnostic only | [Source](https://github.com/d8dzmf5mfn/GPTIRL/blob/7c9ac49c932b1aea21eab1209ee68afaf02cc564/Sources/GPTIRLSherpa/StreamingDPDFNetDenoiser.swift) |
| [Caper](https://github.com/joswayski/caper) | Browser chat/voice application with microphone noise suppression. DPDFNet8 audio worklet and microphone-pipeline integration. | In development | [Source](https://github.com/joswayski/caper/blob/HEAD/apps/web/src/media/microphone.ts) |
| [Noican](https://github.com/lightsound/noican) | macOS virtual-microphone project with switchable denoisers. Rust/ONNX Runtime DPDFNet model stage. | In development | [Source](https://github.com/lightsound/noican/blob/HEAD/crates/noican-models/src/stages/dpdfnet.rs) |
| [Clowd](https://github.com/clowd/Clowd) | Screen-recording application with audio cleanup in its rewrite. Embedded DPDFNet2 48 kHz HR in a Rust/ONNX Runtime helper. | Development rewrite | [Source](https://github.com/clowd/Clowd/blob/HEAD/clowd_ai/src/denoise.rs) |
| [speech-to-speech-mobile](https://github.com/loyality7/speech-to-speech-mobile) | Optional denoising/restoration pass for synthesized mobile speech. DPDFNet 48 kHz in the Android engine. | Experimental | [Source](https://github.com/loyality7/speech-to-speech-mobile/blob/HEAD/bindings/android/src/main/java/com/s2s/mobile/S2SEngine.kt) |
| [demo-v3 voice-agent pipeline](https://github.com/nlpkiddo-2001/demo-v3) | Voice-agent input cleanup before VAD/barge-in processing. DPDFNet4 through a server-side sherpa-onnx adapter. | Prototype | [Source](https://github.com/nlpkiddo-2001/demo-v3/blob/HEAD/server/speech/denoise.py) |
| [NextEngine — speech-timeline tools](https://github.com/kaifaty/NextEngine) | Optional ASR preprocessing in speech-timeline tooling. DPDFNet StreamEnhancer adapter. | Experimental | [Source](https://github.com/kaifaty/NextEngine/blob/HEAD/tools/speech-timeline/src/nextengine_speech_timeline/adapters/dpdfnet.py) |
| [WatchDog](https://github.com/shooter119/WatchDog) | Speech-input preprocessing in a Flutter assistant project. DPDFNet2 through sherpa-onnx before local ASR. | Prototype | [Source](https://github.com/shooter119/WatchDog/blob/HEAD/app/lib/services/local_asr_service.dart) |
| [Lyre](https://github.com/5aaee9/Lyre) | Voice-room application with a DPDFNet noise-cancellation provider. Rust/ONNX Runtime backend plus WASM DSP; multiple DPDFNet variants. | In development | [Source](https://github.com/5aaee9/Lyre/blob/90486d8b0e73d5f74fd21a4a6590764d82e1feb5/crates/lyre-noise-cancelling/src/dpdfnet.rs) |
| [nslab](https://github.com/Kinescope-Igor/nslab) | Browser laboratory comparing real-time denoisers. TypeScript DPDFNet2 implementation; 16 kHz and 48 kHz HR weights. | Research tooling | [Source](https://github.com/Kinescope-Igor/nslab/blob/19a0262400d01b5088af094b5ebc24c516e77ff5/src/audio/dpdfnet.ts) |
| [speech-core — laptop audio tools](https://github.com/atacolak/speech-core) | Laptop-microphone denoising proxy forwarding audio to a speech daemon. DPDFNet2 StreamEnhancer; Python and WebSocket. | Experimental | [Source](https://github.com/atacolak/speech-core/blob/ab2f994a7fee4b11ee4d4c1d6e18f7833629167f/laptop-audio/mic-denoise.py) |

## Related components, model assets and demos

These resources are additional ways to reuse or try DPDFNet. They are not added to the 60-project count.

| Resource | Relationship |
| --- | --- |
| [hushmic-denoiser](https://github.com/Fovty/HushMic/tree/HEAD/crates/hushmic-denoiser) and [HushMic's LADSPA plugin](https://github.com/Fovty/HushMic/tree/HEAD/crates/dpdfnet-ladspa) | Reusable components of HushMic; included in its project entry. |
| [hushmic-nix](https://github.com/Fovty/hushmic-nix) | Nix packaging for HushMic, its models and runtime; grouped with HushMic. |
| [DPDFNet4-CoreML](https://huggingface.co/iky1e/DPDFNet4-CoreML) and [DPDFNet8-48kHz-HR-CoreML](https://huggingface.co/iky1e/DPDFNet8-48kHz-HR-CoreML) | Converted model assets used by DeepFilterNet-mlx. |
| [audiosronnx-dpdfnet](https://huggingface.co/TigreGotico/audiosronnx-dpdfnet) | DPDFNet model assets for audiosronnx. |
| [bitsydarel/dpdfnet-onnx](https://huggingface.co/bitsydarel/dpdfnet-onnx) | Additional ONNX model distribution; no independent application verified. |
| [zliyk/DPDFNetDemo](https://huggingface.co/spaces/zliyk/DPDFNetDemo) | Additional hosted Gradio demo with DPDFNet model-loading code; not counted as a distinct application project. |

## Historical integrations and evaluations

These projects are useful parts of the ecosystem history. Their evidence supports earlier use, evaluation, or runtime work rather than an additional current application.

| Project | What the evidence supports |
| --- | --- |
| [MicYou](https://github.com/LanRhyme/MicYou) | Earlier DPDFNet integration was removed in the [PureVox change](https://github.com/LanRhyme/MicYou/commit/8ce2888d613a01156ca1c678ae889f86ddd67d10). A [fork retains the older implementation](https://github.com/realSalman/MicYou/blob/HEAD/tauri-app/crates/micyou-audio/src/dsp.rs); this is not counted as another current project. |
| [Kurn](https://github.com/carlosmazzei/Kurn) | [Historical DPDFNet Core ML tooling](https://github.com/carlosmazzei/Kurn/commit/822d997875e20698fbab21504bf5213f8c6df18f); the [current speech enhancer](https://github.com/carlosmazzei/Kurn/blob/848cec1a694ef80fe1d162afce561dcef37fb739/Kurn/Services/Enhancement/SpeechEnhancer.swift) uses GTCRN. |
| [cleanfeed](https://github.com/vedantggwp/cleanfeed) / [phonepod](https://github.com/vedantggwp/phonepod) | [Benchmark scripts](https://github.com/vedantggwp/cleanfeed/blob/HEAD/benchmark_denoisers.py) evaluate DPDFNet; the application selected DeepFilterNet3. Grouped as one evaluation. |
| [tract](https://github.com/sonos/tract) | [Runtime optimization work](https://github.com/sonos/tract/commit/6381da8225e1ae250be1d4320bed3c105c9d31e5) references DPDFNet workloads. The runnable DPDFNet integration is listed above under torch-to-nnef. |

## Add your project

Built something with DPDFNet? [Open an issue](https://github.com/ceva-ip/DPDFNet/issues/new) or submit a pull request with:

- Project name and public repository or product URL.
- A short description of how DPDFNet is used.
- Model variant and runtime, if known.
- A source-code, documentation or demo link that demonstrates the integration.
- Whether the integration is released, optional, experimental or historical.

Please identify renamed repositories and forks of already-listed projects so related work can be grouped accurately.
