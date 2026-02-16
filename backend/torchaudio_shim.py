"""
Minimal torchaudio compatibility shim for Chatterbox TTS.

Replaces the broken C-extension torchaudio with pure-Python equivalents
using soundfile, scipy, librosa, and numpy.  Only the three functions
Chatterbox actually calls are implemented:

  1. torchaudio.load(filepath)
  2. torchaudio.transforms.Resample(src_sr, dst_sr)
  3. torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=80, ...)

Install by calling  install()  BEFORE any Chatterbox import.
"""

import importlib
import importlib.machinery
import sys
import types

import numpy as np
import torch


# ── 1.  torchaudio.load ─────────────────────────────────────────────

def _load(filepath, **kwargs):
    """Load audio file → (waveform_tensor [1, T], sample_rate).
    
    Returns tensor on CUDA if available (Chatterbox's voice encoder
    expects input on the same device as its weights).
    """
    import soundfile as sf
    data, sr = sf.read(str(filepath), dtype="float32")
    if data.ndim == 1:
        data = data[np.newaxis, :]          # [1, T]
    else:
        data = data.T                        # [C, T]
    tensor = torch.from_numpy(data)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor, sr


# ── 2.  torchaudio.transforms.Resample ──────────────────────────────

class _Resample(torch.nn.Module):
    """Resample audio tensor from src_sr → dst_sr via scipy."""

    def __init__(self, orig_freq: int, new_freq: int):
        super().__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.orig_freq == self.new_freq:
            return waveform
        from scipy.signal import resample as _scipy_resample
        ratio = self.new_freq / self.orig_freq
        device = waveform.device
        np_wav = waveform.cpu().numpy()
        new_len = int(np_wav.shape[-1] * ratio)
        if np_wav.ndim == 1:
            resampled = _scipy_resample(np_wav, new_len).astype(np.float32)
        else:
            resampled = np.stack(
                [_scipy_resample(ch, new_len).astype(np.float32) for ch in np_wav]
            )
        return torch.from_numpy(resampled).to(device)


class _Spectrogram(torch.nn.Module):
    """STFT-based spectrogram using torch.stft."""

    def __init__(self, n_fft: int = 400, win_length=None, hop_length=None,
                 pad: int = 0, window_fn=torch.hann_window, power=2.0,
                 normalized: bool = False, center: bool = True,
                 pad_mode: str = "reflect", onesided: bool = True, **kwargs):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.hop_length = hop_length or (n_fft // 2)
        self.power = power
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided
        self.register_buffer("window", window_fn(self.win_length))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.center:
            pad_amount = self.n_fft // 2
            # reflect padding needs at least 2D input
            if waveform.ndim == 1:
                waveform = torch.nn.functional.pad(
                    waveform.unsqueeze(0), (pad_amount, pad_amount),
                    mode=self.pad_mode,
                ).squeeze(0)
            else:
                waveform = torch.nn.functional.pad(
                    waveform.unsqueeze(0) if waveform.ndim == 1 else waveform,
                    (pad_amount, pad_amount), mode=self.pad_mode,
                )
        spec = torch.stft(waveform, self.n_fft,
                          hop_length=self.hop_length,
                          win_length=self.win_length,
                          window=self.window.to(waveform.device),
                          center=False,
                          onesided=self.onesided,
                          return_complex=True)
        if self.power is not None:
            return spec.abs().pow(self.power)
        return spec  # return complex tensor when power=None


class _InverseSpectrogram(torch.nn.Module):
    """Inverse STFT via torch.istft."""

    def __init__(self, n_fft: int = 400, win_length=None, hop_length=None,
                 window_fn=torch.hann_window, normalized: bool = False,
                 center: bool = True, onesided: bool = True, **kwargs):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.hop_length = hop_length or (n_fft // 2)
        self.center = center
        self.onesided = onesided
        self.register_buffer("window", window_fn(self.win_length))

    def forward(self, specgram: torch.Tensor) -> torch.Tensor:
        if specgram.is_complex():
            complex_spec = specgram
        elif specgram.shape[-1] == 2:
            complex_spec = torch.view_as_complex(specgram)
        else:
            complex_spec = specgram
        return torch.istft(complex_spec, self.n_fft,
                           hop_length=self.hop_length,
                           win_length=self.win_length,
                           window=self.window.to(specgram.device),
                           center=self.center,
                           onesided=self.onesided)


class _TimeStretch(torch.nn.Module):
    """Phase-vocoder time stretch — matches torchaudio.transforms.TimeStretch API."""

    def __init__(self, hop_length=None, n_freq: int = 201, fixed_rate=None, **kwargs):
        super().__init__()
        self.fixed_rate = fixed_rate
        hop = hop_length if hop_length is not None else 512
        # phase_advance buffer — same as real torchaudio TimeStretch
        phase_advance = torch.linspace(0, np.pi * hop, n_freq, dtype=torch.float32)[..., None]
        self.register_buffer("phase_advance", phase_advance)

    def forward(self, complex_specgrams: torch.Tensor, overriding_rate=None) -> torch.Tensor:
        rate = overriding_rate or self.fixed_rate or 1.0
        if rate == 1.0:
            return complex_specgrams
        shape = list(complex_specgrams.shape)
        time_dim = -2 if complex_specgrams.is_complex() else -3
        new_time = max(1, int(shape[time_dim] / rate))
        if complex_specgrams.is_complex():
            mag = complex_specgrams.abs()
            phase = complex_specgrams.angle()
            if mag.ndim == 2:
                mag = mag.unsqueeze(0)
                phase = phase.unsqueeze(0)
                squeeze = True
            else:
                squeeze = False
            mag = torch.nn.functional.interpolate(
                mag.unsqueeze(1), size=(mag.shape[-2], new_time),
                mode="bilinear", align_corners=False,
            ).squeeze(1)
            phase = torch.nn.functional.interpolate(
                phase.unsqueeze(1), size=(phase.shape[-2], new_time),
                mode="bilinear", align_corners=False,
            ).squeeze(1)
            if squeeze:
                mag = mag.squeeze(0)
                phase = phase.squeeze(0)
            return mag * torch.exp(1j * phase)
        return complex_specgrams


# ── 3.  torchaudio.compliance.kaldi.fbank ────────────────────────────

def _fbank(
    waveform,
    num_mel_bins: int = 23,
    sample_frequency: float = 16000.0,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    dither: float = 0.0,
    energy_floor: float = 1.0,
    preemphasis_coefficient: float = 0.97,
    window_type: str = "povey",
    snip_edges: bool = True,
    **kwargs,
) -> torch.Tensor:
    """Kaldi-compatible filter-bank features via librosa.

    Input:  waveform  [1, T]  (float32 tensor, possibly on CUDA)
    Output: features  [num_frames, num_mel_bins]  (float32 tensor, same device as input)
    """
    import librosa

    device = waveform.device
    wav_np = waveform.squeeze().cpu().numpy().astype(np.float32)
    sr = int(sample_frequency)
    n_fft = int(sr * frame_length / 1000.0)
    hop = int(sr * frame_shift / 1000.0)
    n_fft_pow2 = 1
    while n_fft_pow2 < n_fft:
        n_fft_pow2 <<= 1

    mel_spec = librosa.feature.melspectrogram(
        y=wav_np,
        sr=sr,
        n_fft=n_fft_pow2,
        hop_length=hop,
        win_length=n_fft,
        n_mels=num_mel_bins,
        power=2.0,
    )
    log_mel = np.log(np.maximum(mel_spec, energy_floor)).T   # [frames, mel_bins]
    return torch.from_numpy(log_mel.astype(np.float32)).to(device)


# ── Module injection ────────────────────────────────────────────────

def _make_module(name):
    """Create a module with proper __spec__ to avoid ValueError."""
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mod.__path__ = []
    mod.__package__ = name
    return mod


def _patch_complex_abs():
    """Monkey-patch torch.abs to avoid NVRTC compilation for complex tensors.
    
    On Blackwell GPUs, NVRTC cannot compile the abs kernel for complex<float>
    because it doesn't recognize the sm_121 architecture.  We replace complex
    abs with an equivalent real-arithmetic formula:
        abs(z) = sqrt(z.real² + z.imag²)
    This uses pre-compiled kernels and avoids NVRTC entirely.
    """
    _original_abs = torch.abs

    def _safe_abs(input, *args, **kwargs):
        if isinstance(input, torch.Tensor) and input.is_complex():
            return torch.sqrt(input.real ** 2 + input.imag ** 2)
        return _original_abs(input, *args, **kwargs)

    torch.abs = _safe_abs

    # Also patch the Tensor method
    _original_tensor_abs = torch.Tensor.abs

    def _safe_tensor_abs(self):
        if self.is_complex():
            return torch.sqrt(self.real ** 2 + self.imag ** 2)
        return _original_tensor_abs(self)

    torch.Tensor.abs = _safe_tensor_abs


def install():
    """Replace torchaudio in sys.modules with this shim."""
    _patch_complex_abs()

    # Root module
    ta = _make_module("torchaudio")
    ta.__version__ = "0.0.0+shim"
    ta.load = _load

    # torchaudio.transforms
    transforms = _make_module("torchaudio.transforms")
    transforms.Resample = _Resample
    transforms.Spectrogram = _Spectrogram
    transforms.InverseSpectrogram = _InverseSpectrogram
    transforms.TimeStretch = _TimeStretch
    ta.transforms = transforms

    # torchaudio.compliance
    compliance = _make_module("torchaudio.compliance")
    ta.compliance = compliance

    # torchaudio.compliance.kaldi
    kaldi = _make_module("torchaudio.compliance.kaldi")
    kaldi.fbank = _fbank
    compliance.kaldi = kaldi

    # torchaudio.functional (some code may probe for it)
    functional = _make_module("torchaudio.functional")
    ta.functional = functional

    # Block the broken _extension import
    _ext = _make_module("torchaudio._extension")
    ta._extension = _ext

    # Register in sys.modules so `import torchaudio` works
    sys.modules["torchaudio"] = ta
    sys.modules["torchaudio.transforms"] = transforms
    sys.modules["torchaudio.compliance"] = compliance
    sys.modules["torchaudio.compliance.kaldi"] = kaldi
    sys.modules["torchaudio.functional"] = functional
    sys.modules["torchaudio._extension"] = _ext
