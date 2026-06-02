"""Multi-mode beat_this inference: single-channel, multichannel stems, or ensemble.

Three input modes, all decoded with the beat-only ``hmm`` postprocessor by default
(beat-only DBN at min_bpm=80 + the model's original downbeats snapped to the grid):

  - ``single``    : a single-channel model on the full mix.
  - ``multi``     : a multichannel model on demixed stems (channels = demucs
                    drums/bass/other/vocals + full mix).
  - ``ensemble``  : several models whose framewise logits are averaged before decoding.

The demucs stems and the ``mix`` channel are computed to match how the multichannel
training spectrograms are built (htdemucs 44.1k stereo, split, overlap 0.1; per-stem mono
-> 22050 Hz -> LogMelSpect; mix = original audio mono -> 22050 Hz -> LogMelSpect).

Quick start::

    from beat_this.multichannel import build_predictor
    pred = build_predictor("ensemble", multi_ckpt=".../5stem.ckpt",
                           multi4_ckpt=".../4stem.ckpt", single_ckpt=".../fullmix.ckpt",
                           device="cuda")
    beats, downbeats = pred("song.mp3")          # numpy arrays of times in seconds
"""
from __future__ import annotations

import numpy as np
import soxr
import torch
import torchaudio

from beat_this.inference import load_model
from beat_this.model.postprocessor import Postprocessor
from beat_this.preprocessing import LogMelSpect

DEMUCS_STEMS = ("drums", "bass", "other", "vocals")
CHANNELS_5 = ["drums", "bass", "other", "vocals", "mix"]
CHANNELS_4 = ["drums", "bass", "other", "vocals"]
CHANNELS_MIX = ["mix"]


class BeatThisPredictor:
    """Flexible single / multichannel / ensemble beat_this inference.

    Args:
        members: list of ``(model, channels)`` pairs. ``model`` is a checkpoint path/name
            (loaded via :func:`beat_this.inference.load_model`) or an already-loaded
            ``BeatThis`` model. ``channels`` is a list drawn from ``{"drums","bass","other",
            "vocals","mix"}`` in the order the model's stem conv expects. Multiple members =>
            their framewise logits are averaged (ensemble).
        device: torch device.
        postprocessor: ``"hmm"`` (default), ``"minimal"`` or ``"dbn"``.
        chunk: frames per forward chunk (1500 = 30 s at 50 fps).
        float16: run the model in half precision.
    """

    def __init__(self, members, device="cpu", postprocessor="hmm",
                 chunk=1500, float16=False):
        self.device = torch.device(device)
        self.chunk = chunk
        self.float16 = float16
        self.members = [
            (load_model(m, self.device) if isinstance(m, str) else m, list(ch))
            for m, ch in members
        ]
        self.spect = LogMelSpect(device=self.device)
        self.post = Postprocessor(type=postprocessor)
        self.need_stems = any(
            any(c in DEMUCS_STEMS for c in ch) for _, ch in self.members
        )
        self._demucs = None

    def _demucs_model(self):
        if self._demucs is None:
            from demucs.pretrained import get_model

            self._demucs = get_model("htdemucs").to(self.device).eval()
        return self._demucs

    def _logmel(self, mono_22k: np.ndarray) -> torch.Tensor:
        x = torch.tensor(np.ascontiguousarray(mono_22k), dtype=torch.float32,
                         device=self.device)
        return self.spect(x)

    def _channels(self, wav: torch.Tensor, sr: int) -> dict:
        """wav: (channels, samples) tensor. Returns {name: (T,128) tensor on device}."""
        out = {}
        mono = wav.mean(0).cpu().numpy()
        out["mix"] = self._logmel(soxr.resample(mono, sr, 22050) if sr != 22050 else mono)
        if self.need_stems:
            from demucs.apply import apply_model
            from demucs.audio import convert_audio

            dm = self._demucs_model()
            w = convert_audio(wav, sr, dm.samplerate, dm.audio_channels)
            with torch.no_grad():
                stems = apply_model(dm, w[None].to(self.device), device=self.device,
                                    split=True, overlap=0.1)[0]
            for si, name in enumerate(dm.sources):
                s_mono = stems[si].mean(0).cpu().numpy()
                out[name] = self._logmel(soxr.resample(s_mono, dm.samplerate, 22050))
        return out

    def _forward(self, model, spect: torch.Tensor):
        """spect: (T,F) single-channel or (T,C,F) multichannel."""
        spect = spect.to(self.device)
        T = spect.shape[0]
        beat = torch.zeros(T)
        down = torch.zeros(T)
        for s in range(0, T, self.chunk):
            e = min(s + self.chunk, T)
            with torch.inference_mode():
                with torch.autocast(enabled=self.float16, device_type=self.device.type):
                    pr = model(spect[s:e].unsqueeze(0))
            beat[s:e] = pr["beat"][0].float().cpu()
            down[s:e] = pr["downbeat"][0].float().cpu()
        return beat, down

    def frames(self, audio, sr=None):
        """Return averaged framewise (beat_logits, downbeat_logits) for the audio."""
        if isinstance(audio, str):
            wav, sr = torchaudio.load(audio)            # (channels, samples)
        else:
            wav = audio if torch.is_tensor(audio) else torch.tensor(audio)
            if wav.ndim == 1:
                wav = wav[None, :]
            elif wav.shape[0] > wav.shape[1]:           # (samples, channels) -> (channels, samples)
                wav = wav.T
            if sr is None:
                raise ValueError("sr is required when passing a raw signal")
        ch = self._channels(wav, sr)
        bls, dls = [], []
        for model, channels in self.members:
            arrs = [ch[c] for c in channels]
            if len(channels) == 1:
                spect = arrs[0]
            else:
                n = min(a.shape[0] for a in arrs)
                spect = torch.stack([a[:n] for a in arrs], dim=1)   # (T,C,F)
            b, d = self._forward(model, spect)
            bls.append(b)
            dls.append(d)
        n = min(len(x) for x in bls)
        beat = torch.stack([x[:n] for x in bls]).mean(0)
        down = torch.stack([x[:n] for x in dls]).mean(0)
        return beat, down

    def __call__(self, audio, sr=None):
        beat, down = self.frames(audio, sr)
        return self.post(beat, down)


def build_predictor(mode="ensemble", *, multi_ckpt=None, multi4_ckpt=None,
                    single_ckpt=None, device="cpu", postprocessor="hmm", float16=False):
    """Build a :class:`BeatThisPredictor` for one of the presets.

    Args:
        mode: ``"single"`` (full mix, needs ``single_ckpt``), ``"multi"`` (5-channel stems,
            needs ``multi_ckpt``), or ``"ensemble"`` (5-stem + 4-stem + full-mix, needs all).
        multi_ckpt:  5-channel model checkpoint (drums/bass/other/vocals + mix).
        multi4_ckpt: 4-channel model checkpoint (drums/bass/other/vocals).
        single_ckpt: single-channel (full-mix) model checkpoint.
    """
    if mode == "single":
        if not single_ckpt:
            raise ValueError("mode='single' needs single_ckpt=<checkpoint>")
        members = [(single_ckpt, CHANNELS_MIX)]
    elif mode == "multi":
        if not multi_ckpt:
            raise ValueError("mode='multi' needs multi_ckpt=<checkpoint>")
        members = [(multi_ckpt, CHANNELS_5)]
    elif mode == "ensemble":
        if not (multi_ckpt and multi4_ckpt and single_ckpt):
            raise ValueError("mode='ensemble' needs multi_ckpt, multi4_ckpt and single_ckpt")
        members = [(multi_ckpt, CHANNELS_5), (multi4_ckpt, CHANNELS_4),
                   (single_ckpt, CHANNELS_MIX)]
    else:
        raise ValueError(f"unknown mode {mode!r} (single|multi|ensemble)")
    return BeatThisPredictor(members, device=device, postprocessor=postprocessor,
                             float16=float16)
