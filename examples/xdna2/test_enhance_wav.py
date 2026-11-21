"""Provider catalog and WAV contracts that do not require an NPU."""

from hashlib import sha256
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import wave

from enhance_wav import provider_assets, wav_info


class PreviewContracts(unittest.TestCase):
    def test_catalog_resolves_relative_artifacts_and_checks_digests(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            entry = {}
            for key in ("library", "manifest", "model"):
                data = key.encode()
                (root / key).write_bytes(data)
                entry[key] = key
                entry[key + "_sha256"] = sha256(data).hexdigest()
            catalog = root / "providers.json"
            catalog.write_text(json.dumps({"version": 1, "pcm_profiles": {"8": entry}}))
            self.assertEqual(provider_assets(catalog, 8), [root / k for k in ("library", "manifest", "model")])
            with self.assertRaisesRegex(ValueError, "Model 2"):
                provider_assets(catalog, 2)
            (root / "manifest").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                provider_assets(catalog, 8)

    def test_only_nonempty_mono_48k_pcm_is_accepted(self):
        for channels, rate, samples, expected in ((1, 48000, 481, (481, 8)),
                                                  (2, 48000, 481, None),
                                                  (1, 44100, 481, None),
                                                  (1, 48000, 0, None)):
            with self.subTest(channels=channels, rate=rate, samples=samples):
                data = io.BytesIO()
                with wave.open(data, "wb") as wav:
                    wav.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
                    wav.writeframes(bytes(samples * 2 * channels))
                data.seek(0)
                with wave.open(data, "rb") as wav:
                    if expected:
                        self.assertEqual(wav_info(wav), expected)
                    else:
                        with self.assertRaises(ValueError):
                            wav_info(wav)


if __name__ == "__main__":
    unittest.main()
