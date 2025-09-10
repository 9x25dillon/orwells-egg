#!/usr/bin/env python3
# dual_llm_wavecaster_enhanced.py
# SPDX-License-Identifier: MIT
"""
Enhanced Dual LLM WaveCaster
---------------------------
Two-LLM orchestration (local final inference + remote resource-only summaries) → framed bits
→ modulated waveform (BFSK/BPSK/QPSK/16QAM/AFSK/OFDM) → WAV/IQ files (+ optional audio out)
with visualization, simple FEC, encryption, watermarking, and metadata.

Deps (minimum):
  pip install numpy scipy requests

Optional:
  pip install matplotlib sounddevice pycryptodome

Quick start:
  python dual_llm_wavecaster_enhanced.py modulate --text "hello airwaves" --scheme qpsk --wav --iq
  python dual_llm_wavecaster_enhanced.py cast --prompt "2-paragraph plan" \
      --resource-file notes.txt --local-url http://127.0.0.1:8080 --local-mode llama-cpp \
      --remote-url https://api.openai.com --remote-key $OPENAI_API_KEY --scheme bfsk --wav
"""

from __future__ import annotations
import argparse, base64, binascii, hashlib, json, logging, math, os, struct, sys, time, warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Callable
from enum import Enum, auto
from datetime import datetime

# ---------- Hard requirements ----------
try:
    import numpy as np
    from scipy import signal as sp_signal
    from scipy.fft import rfft, rfftfreq
except Exception as e:
    raise SystemExit("numpy and scipy are required: pip install numpy scipy") from e

# ---------- Optional dependencies ----------
try:
    import requests
except Exception:
    requests = None  # HTTP backends disabled if missing

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import sounddevice as sd
    HAS_AUDIO = True
except Exception:
    HAS_AUDIO = False

try:
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    from Crypto.Protocol.KDF import PBKDF2
    HAS_CRYPTO = True
except Exception:
    HAS_CRYPTO = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("wavecaster")

# =========================================================
# Enums / Config
# =========================================================

class ModulationScheme(Enum):
    BFSK = auto()
    BPSK = auto()
    QPSK = auto()
    QAM16 = auto()
    AFSK = auto()
    OFDM = auto()
    DSSS_BPSK = auto()

class FEC(Enum):
    NONE = auto()
    HAMMING74 = auto()
    REED_SOLOMON = auto()   # stub
    LDPC = auto()           # stub
    TURBO = auto()          # stub

@dataclass
class HTTPConfig:
    base_url: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    timeout: int = 60
    mode: str = "openai-chat"  # ["openai-chat","openai-completions","llama-cpp","textgen-webui"]
    verify_ssl: bool = True
    max_retries: int = 2
    retry_delay: float = 0.8

@dataclass
class OrchestratorSettings:
    temperature: float = 0.7
    max_tokens: int = 512
    style: str = "concise"
    max_context_chars: int = 8000

@dataclass
class ModConfig:
    sample_rate: int = 48000
    symbol_rate: int = 1200
    amplitude: float = 0.7
    f0: float = 1200.0     # BFSK 0
    f1: float = 2200.0     # BFSK 1
    fc: float = 1800.0     # PSK/QAM audio carrier (for WAV)
    clip: bool = True
    # OFDM (toy)
    ofdm_subc: int = 64
    cp_len: int = 16
    # DSSS
    dsss_chip_rate: int = 4800

@dataclass
class FrameConfig:
    use_crc32: bool = True
    use_crc16: bool = False
    preamble: bytes = b"\x55" * 8  # 01010101 * 8
    version: int = 1

# =========================================================
# Utilities
# =========================================================

def now_ms() -> int:
    return int(time.time() * 1000)

def crc32_bytes(data: bytes) -> bytes:
    return binascii.crc32(data).to_bytes(4, "big")

def crc16_ccitt(data: bytes) -> bytes:
    poly, crc = 0x1021, 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if (crc & 0x8000) else ((crc << 1) & 0xFFFF)
    return crc.to_bytes(2, "big")

def to_bits(data: bytes) -> List[int]:
    return [(byte >> i) & 1 for byte in data for i in range(7, -1, -1)]

def from_bits(bits: Sequence[int]) -> bytes:
    if len(bits) % 8 != 0:
        bits = list(bits) + [0] * (8 - len(bits) % 8)
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i:i+8]:
            byte = (byte << 1) | (1 if b else 0)
        out.append(byte)
    return bytes(out)

def chunk_bits(bits: Sequence[int], n: int) -> List[List[int]]:
    return [list(bits[i:i+n]) for i in range(0, len(bits), n)]

def safe_json(obj: Any) -> str:
    def enc(x):
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
        if isinstance(x, complex):
            return {"real": float(x.real), "imag": float(x.imag)}
        if isinstance(x, datetime):
            return x.isoformat()
        return str(x)
    return json.dumps(obj, ensure_ascii=False, indent=2, default=enc)

def write_wav_mono(path: Path, signal: np.ndarray, sample_rate: int):
    import wave
    sig = np.clip(signal, -1.0, 1.0)
    pcm = (sig * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())

def write_iq_f32(path: Path, iq: np.ndarray):
    if iq.ndim != 1 or not np.iscomplexobj(iq):
        raise ValueError("iq must be 1-D complex array")
    interleaved = np.empty(iq.size * 2, dtype=np.float32)
    interleaved[0::2] = iq.real.astype(np.float32)
    interleaved[1::2] = iq.imag.astype(np.float32)
    path.write_bytes(interleaved.tobytes())

def plot_wave_and_spectrum(path_png: Path, x: np.ndarray, sr: int, title: str):
    if not HAS_MPL: 
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,5))
    t = np.arange(len(x))/sr
    ax1.plot(t[:min(len(t), 0.05*sr)], x[:min(len(x), int(0.05*sr))])
    ax1.set_title(f"{title} (first 50ms)")
    ax1.set_xlabel("s"); ax1.set_ylabel("amplitude")
    spec = np.abs(rfft(x)) + 1e-12
    freqs = rfftfreq(len(x), 1.0/sr)
    ax2.semilogy(freqs, spec/spec.max())
    ax2.set_xlim(0, min(8000, sr//2)); ax2.set_xlabel("Hz"); ax2.set_ylabel("norm |X(f)|")
    plt.tight_layout(); fig.savefig(path_png); plt.close(fig)

def play_audio(x: np.ndarray, sr: int):
    if not HAS_AUDIO:
        log.warning("sounddevice not installed; cannot play audio")
        return
    sd.play(x, sr); sd.wait()

# =========================================================
# FEC (simple Hamming 7,4; heavy codes are stubs)
# =========================================================

def hamming74_encode(data_bits: List[int]) -> List[int]:
    if len(data_bits) % 4 != 0:
        data_bits = data_bits + [0] * (4 - len(data_bits) % 4)
    out = []
    for i in range(0, len(data_bits), 4):
        d0, d1, d2, d3 = data_bits[i:i+4]
        p1 = d0 ^ d1 ^ d3
        p2 = d0 ^ d2 ^ d3
        p3 = d1 ^ d2 ^ d3
        out += [p1, p2, d0, p3, d1, d2, d3]
    return out

def fec_encode(bits: List[int], scheme: FEC) -> List[int]:
    if scheme == FEC.NONE:
        return list(bits)
    if scheme == FEC.HAMMING74:
        return hamming74_encode(bits)
    if scheme in (FEC.REED_SOLOMON, FEC.LDPC, FEC.TURBO):
        raise NotImplementedError(f"{scheme.name} encoding not implemented in this minimal build")
    raise ValueError("Unknown FEC")

# =========================================================
# Framing / Security / Watermark
# =========================================================

@dataclass
class SecurityConfig:
    password: Optional[str] = None           # AES-GCM if provided
    watermark: Optional[str] = None          # prepended SHA256[0:8]
    hmac_key: Optional[str] = None           # HMAC-SHA256 appended

def aes_gcm_encrypt(plaintext: bytes, password: str) -> bytes:
    if not HAS_CRYPTO:
        raise RuntimeError("pycryptodome required for encryption")
    salt = get_random_bytes(16)
    key = PBKDF2(password, salt, dkLen=32, count=200_000)
    nonce = get_random_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ct, tag = cipher.encrypt_and_digest(plaintext)
    return b"AGCM" + salt + nonce + tag + ct

def apply_hmac(data: bytes, hkey: str) -> bytes:
    import hmac
    key = hashlib.sha256(hkey.encode("utf-8")).digest()
    mac = hmac.new(key, data, hashlib.sha256).digest()
    return data + b"HMAC" + mac

def add_watermark(data: bytes, wm: str) -> bytes:
    return hashlib.sha256(wm.encode("utf-8")).digest()[:8] + data

def frame_payload(payload: bytes, fcfg: FrameConfig) -> bytes:
    header = struct.pack(">BBI", 0xA5, fcfg.version, now_ms() & 0xFFFFFFFF)
    core = header + payload
    tail = b""
    if fcfg.use_crc32:
        tail += crc32_bytes(core)
    if fcfg.use_crc16:
        tail += crc16_ccitt(core)
    return fcfg.preamble + core + tail

def encode_text(
    text: str,
    fcfg: FrameConfig,
    sec: SecurityConfig,
    fec_scheme: FEC,
) -> List[int]:
    data = text.encode("utf-8")
    if sec.watermark:
        data = add_watermark(data, sec.watermark)
    if sec.password:
        data = aes_gcm_encrypt(data, sec.password)
    framed = frame_payload(data, fcfg)
    if sec.hmac_key:
        framed = apply_hmac(framed, sec.hmac_key)
    bits = to_bits(framed)
    bits = fec_encode(bits, fec_scheme)
    return bits

# =========================================================
# Modulators (audio & IQ)
# =========================================================

class Modulators:
    @staticmethod
    def bfsK(bits: Sequence[int], cfg: ModConfig) -> np.ndarray:
        sr, rb = cfg.sample_rate, cfg.symbol_rate
        spb = int(sr / rb)
        t = np.arange(spb) / sr
        s = []
        a = cfg.amplitude
        for b in bits:
            f = cfg.f1 if b else cfg.f0
            s.append(a * np.sin(2*np.pi*f*t))
        y = np.concatenate(s) if s else np.zeros(0, dtype=np.float64)
        return np.clip(y, -1, 1).astype(np.float32) if cfg.clip else y.astype(np.float32)

    @staticmethod
    def bpsK(bits: Sequence[int], cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        sr, rb, fc = cfg.sample_rate, cfg.symbol_rate, cfg.fc
        spb = int(sr / rb)
        t = np.arange(spb) / sr
        a = cfg.amplitude
        audio_blocks, iq_blocks = [], []
        for b in bits:
            phase = 0.0 if b else np.pi
            audio_blocks.append(a * np.sin(2*np.pi*fc*t + phase))
            iq_blocks.append(a * (np.cos(phase) + 1j*np.sin(phase)) * np.ones_like(t, dtype=np.complex64))
        audio = np.concatenate(audio_blocks) if audio_blocks else np.zeros(0, dtype=np.float64)
        iq = np.concatenate(iq_blocks) if iq_blocks else np.zeros(0, dtype=np.complex64)
        if cfg.clip: audio = np.clip(audio, -1, 1)
        return audio.astype(np.float32), iq

    @staticmethod
    def qpsK(bits: Sequence[int], cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        # Gray map: 00->(1+1j), 01->(-1+1j), 11->(-1-1j), 10->(1-1j)
        pairs = chunk_bits(bits, 2)
        syms = []
        for p in pairs:
            b0, b1 = (p + [0,0])[:2]
            if (b0, b1) == (0,0): s = 1 + 1j
            elif (b0, b1) == (0,1): s = -1 + 1j
            elif (b0, b1) == (1,1): s = -1 - 1j
            else: s = 1 - 1j
            syms.append(s / math.sqrt(2))  # unit energy
        return Modulators._psk_qam_to_audio_iq(np.array(syms, dtype=np.complex64), cfg)

    @staticmethod
    def qam16(bits: Sequence[int], cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        quads = chunk_bits(bits, 4)
        def map2(b0,b1):
            # Gray 2-bit to {-3,-1,1,3}
            val = (b0<<1) | b1
            return [-3,-1,1,3][val]
        syms = []
        for q in quads:
            b0,b1,b2,b3 = (q+[0,0,0,0])[:4]
            I = map2(b0,b1); Q = map2(b2,b3)
            syms.append((I + 1j*Q)/math.sqrt(10)) # unit average power
        return Modulators._psk_qam_to_audio_iq(np.array(syms, dtype=np.complex64), cfg)

    @staticmethod
    def _psk_qam_to_audio_iq(syms: np.ndarray, cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        sr, rb, fc = cfg.sample_rate, cfg.symbol_rate, cfg.fc
        spb = int(sr / rb)
        a = cfg.amplitude
        # Upsample each symbol to 'spb' samples (rectangular pulse)
        i = np.repeat(syms.real.astype(np.float32), spb)
        q = np.repeat(syms.imag.astype(np.float32), spb)
        t = np.arange(len(i)) / sr
        audio = a * (i*np.cos(2*np.pi*fc*t) - q*np.sin(2*np.pi*fc*t))
        iq = (a * i) + 1j*(a * q)
        if cfg.clip: audio = np.clip(audio, -1, 1)
        return audio.astype(np.float32), iq.astype(np.complex64)

    @staticmethod
    def afsK(bits: Sequence[int], cfg: ModConfig) -> np.ndarray:
        return Modulators.bfsK(bits, cfg)

    @staticmethod
    def dsss_bpsK(bits: Sequence[int], cfg: ModConfig) -> np.ndarray:
        # Very simple DSSS: chip with PN sequence at cfg.dsss_chip_rate
        pn = np.array([1, -1, 1, 1, -1, 1, -1, -1], dtype=np.float32)  # toy PN8
        sr = cfg.sample_rate
        chips_per_symbol = max(1, int(cfg.dsss_chip_rate / cfg.symbol_rate))
        spb = int(sr / (cfg.dsss_chip_rate))
        base = []
        for b in bits:
            bit_val = 1.0 if b else -1.0
            ch = bit_val * pn
            ch = np.repeat(ch, spb)
            base.append(ch)
        baseband = np.concatenate(base) if base else np.zeros(0, dtype=np.float32)
        # Upconvert to audio carrier
        t = np.arange(len(baseband))/sr
        audio = cfg.amplitude * baseband * np.sin(2*np.pi*cfg.fc*t)
        if cfg.clip: audio = np.clip(audio, -1, 1)
        return audio.astype(np.float32)

    @staticmethod
    def ofdm(bits: Sequence[int], cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        # Toy OFDM: QPSK mapping across N subcarriers, IFFT, add cyclic prefix
        N = cfg.ofdm_subc
        spb_sym = int(cfg.sample_rate / cfg.symbol_rate)  # samples per OFDM symbol (approx shaping)
        chunks = chunk_bits(bits, 2*N)
        a = cfg.amplitude
        wave = []
        iq = []
        for ch in chunks:
            # map 2 bits -> QPSK symbol
            qsyms = []
            pairs = chunk_bits(ch, 2)
            for p in pairs:
                b0,b1 = (p+[0,0])[:2]
                if (b0,b1)==(0,0): s = 1+1j
                elif (b0,b1)==(0,1): s = -1+1j
                elif (b0,b1)==(1,1): s = -1-1j
                else: s = 1-1j
                qsyms.append(s/math.sqrt(2))
            # pad to N
            if len(qsyms) < N:
                qsyms += [0j]*(N-len(qsyms))
            Xk = np.array(qsyms, dtype=np.complex64)
            xt = np.fft.ifft(Xk)  # time domain symbol (complex)
            # cyclic prefix
            cp = xt[-cfg.cp_len:]
            sym = np.concatenate([cp, xt])
            # stretch to samples-per-symbol for audio mixing
            reps = max(1, int(spb_sym/len(sym)))
            sym_up = np.repeat(sym, reps)
            # audio upconvert
            t = np.arange(len(sym_up))/cfg.sample_rate
            audio = a*(sym_up.real*np.cos(2*np.pi*cfg.fc*t) - sym_up.imag*np.sin(2*np.pi*cfg.fc*t))
            wave.append(audio.astype(np.float32))
            iq.append((a*sym_up).astype(np.complex64))
        audio = np.concatenate(wave) if wave else np.zeros(0, dtype=np.float32)
        iqc = np.concatenate(iq) if iq else np.zeros(0, dtype=np.complex64)
        if cfg.clip: audio = np.clip(audio, -1, 1)
        return audio, iqc

# =========================================================
# LLM backends (Local final inference; Remote resource-only)
# =========================================================

class BaseLLM:
    def generate(self, prompt: str, **kwargs) -> str: raise NotImplementedError

class LocalLLM(BaseLLM):
    def __init__(self, configs: List[HTTPConfig]):
        if requests is None:
            raise RuntimeError("LocalLLM requires 'requests' (pip install requests)")
        self.configs = configs
        self.idx = 0

    def generate(self, prompt: str, **kwargs) -> str:
        last = None
        for _ in range(len(self.configs)):
            cfg = self.configs[self.idx]
            try:
                out = self._call(cfg, prompt, **kwargs)
                return out
            except Exception as e:
                last = e
                self.idx = (self.idx + 1) % len(self.configs)
        raise last or RuntimeError("All local LLM configs failed")

    def _post(self, cfg: HTTPConfig, url: str, headers: dict, body: dict) -> dict:
        s = requests.Session()
        for attempt in range(cfg.max_retries):
            try:
                r = s.post(url, headers=headers, json=body, timeout=cfg.timeout, verify=cfg.verify_ssl)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt < cfg.max_retries-1:
                    time.sleep(cfg.retry_delay*(2**attempt))
                else:
                    raise

    def _call(self, cfg: HTTPConfig, prompt: str, **kwargs) -> str:
        mode = cfg.mode
        if mode == "openai-chat":
            url = f"{cfg.base_url.rstrip('/')}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            if cfg.api_key: headers["Authorization"] = f"Bearer {cfg.api_key}"
            body = {
                "model": cfg.model or "gpt-4o-mini",
                "messages": [{"role":"user","content":prompt}],
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 512),
            }
            data = self._post(cfg, url, headers, body)
            return data["choices"][0]["message"]["content"]
        if mode == "openai-completions":
            url = f"{cfg.base_url.rstrip('/')}/v1/completions"
            headers = {"Content-Type": "application/json"}
            if cfg.api_key: headers["Authorization"] = f"Bearer {cfg.api_key}"
            body = {
                "model": cfg.model or "gpt-3.5-turbo-instruct",
                "prompt": prompt,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 512),
            }
            data = self._post(cfg, url, headers, body)
            return data["choices"][0]["text"]
        if mode == "llama-cpp":
            url = f"{cfg.base_url.rstrip('/')}/completion"
            body = {"prompt": prompt, "temperature": kwargs.get("temperature",0.7), "n_predict": kwargs.get("max_tokens",512)}
            data = self._post(cfg, url, {}, body)
            if "content" in data: return data["content"]
            if "choices" in data and data["choices"]: return data["choices"][0].get("text","")
            return data.get("text","")
        if mode == "textgen-webui":
            url = f"{cfg.base_url.rstrip('/')}/api/v1/generate"
            body = {"prompt": prompt, "max_new_tokens": kwargs.get("max_tokens",512), "temperature": kwargs.get("temperature",0.7)}
            data = self._post(cfg, url, {}, body)
            return data.get("results",[{}])[0].get("text","")
        raise ValueError(f"Unsupported mode: {mode}")

class ResourceLLM(BaseLLM):
    def __init__(self, cfg: Optional[HTTPConfig] = None):
        self.cfg = cfg

    def generate(self, prompt: str, **kwargs) -> str:
        # Constrained to resources-only summarization
        if self.cfg is None or requests is None:
            return LocalSummarizer().summarize(prompt)
        url = f"{self.cfg.base_url.rstrip('/')}/v1/chat/completions"
        headers = {"Content-Type":"application/json"}
        if self.cfg.api_key: headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        system = ("You are a constrained assistant. ONLY summarize/structure the provided INPUT RESOURCES. "
                  "Do not add external knowledge.")
        body = {
            "model": self.cfg.model or "gpt-4o-mini",
            "messages":[{"role":"system","content":system},{"role":"user","content":prompt}],
            "temperature": kwargs.get("temperature",0.2),
            "max_tokens": kwargs.get("max_tokens",512),
        }
        s = requests.Session()
        r = s.post(url, headers=headers, json=body, timeout=self.cfg.timeout, verify=self.cfg.verify_ssl)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

class LocalSummarizer:
    def __init__(self):
        self.stop = {
            "the","a","an","and","or","but","in","on","at","to","for","of","with","by","is","are",
            "was","were","be","been","being","have","has","had","do","does","did","will","would",
            "could","should","from","that","this","it","as"
        }
    def summarize(self, text: str) -> str:
        txt = " ".join(text.split())
        if not txt: return "No content to summarize."
        sents = [s.strip() for s in txt.replace("?",".").replace("!",".").split(".") if s.strip()]
        if not sents: return txt[:300] + ("..." if len(txt)>300 else "")
        # score sentences by length + term frequency (very light heuristic)
        words = [w.lower().strip(",;:()[]") for w in txt.split()]
        freq: Dict[str,int] = {}
        for w in words:
            if w and w not in self.stop: freq[w] = freq.get(w,0)+1
        scored = []
        for s in sents:
            sw = [w.lower().strip(",;:()[]") for w in s.split()]
            score = len(s) * 0.1 + sum(freq.get(w,0) for w in sw)
            scored.append((s, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        keep = [s for s,_ in scored[: min(6,len(scored))]]
        keep.sort(key=lambda k: sents.index(k))
        out = " ".join(keep)
        return out[:800] + ("..." if len(out)>800 else "")

# =========================================================
# Orchestrator
# =========================================================

class DualLLMOrchestrator:
    def __init__(self, local: LocalLLM, resource: ResourceLLM, settings: OrchestratorSettings):
        self.local, self.resource, self.set = local, resource, settings

    def _load_resources(self, paths: List[str], inline: List[str]) -> str:
        parts = []
        for p in paths:
            pa = Path(p)
            if pa.exists() and pa.is_file():
                try:
                    parts.append(pa.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    parts.append(f"[[UNREADABLE_FILE:{pa.name}]]")
            else:
                parts.append(f"[[MISSING_FILE:{pa}]]")
        parts += [str(x) for x in inline]
        blob = "\n\n".join(parts)
        return blob[: self.set.max_context_chars]

    def compose(self, user_prompt: str, resource_paths: List[str], inline_resources: List[str]) -> Tuple[str,str]:
        res_text = self._load_resources(resource_paths, inline_resources)
        res_summary = self.resource.generate(
            f"INPUT RESOURCES:\n{res_text}\n\nTASK: Summarize/structure ONLY the content above.",
            temperature=0.2, max_tokens=self.set.max_tokens
        )
        final_prompt = (
            "You are a LOCAL expert system. Use ONLY the structured summary below; do not invent facts.\n\n"
            f"=== STRUCTURED SUMMARY ===\n{res_summary}\n\n"
            f"=== USER PROMPT ===\n{user_prompt}\n\n"
            f"STYLE: {self.set.style}. Be clear and directly actionable."
        )
        return final_prompt, res_summary

    def run(self, user_prompt: str, resource_paths: List[str], inline_resources: List[str]) -> Dict[str,str]:
        fp, summary = self.compose(user_prompt, resource_paths, inline_resources)
        ans = self.local.generate(fp, temperature=self.set.temperature, max_tokens=self.set.max_tokens)
        return {"summary": summary, "final": ans, "prompt": fp}

# =========================================================
# End-to-end casting
# =========================================================

@dataclass
class OutputPaths:
    wav: Optional[Path] = None
    iq: Optional[Path] = None
    meta: Optional[Path] = None
    png: Optional[Path] = None

def bits_to_signals(bits: List[int], scheme: ModulationScheme, mcfg: ModConfig) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if scheme == ModulationScheme.BFSK:
        return Modulators.bfsK(bits, mcfg), None
    if scheme == ModulationScheme.AFSK:
        return Modulators.afsK(bits, mcfg), None
    if scheme == ModulationScheme.BPSK:
        return Modulators.bpsK(bits, mcfg)
    if scheme == ModulationScheme.QPSK:
        return Modulators.qpsK(bits, mcfg)
    if scheme == ModulationScheme.QAM16:
        return Modulators.qam16(bits, mcfg)
    if scheme == ModulationScheme.OFDM:
        return Modulators.ofdm(bits, mcfg)
    if scheme == ModulationScheme.DSSS_BPSK:
        return Modulators.dsss_bpsK(bits, mcfg), None
    raise ValueError("Unknown modulation scheme")

def cast_to_files(
    text: str,
    outdir: Path,
    scheme: ModulationScheme,
    mcfg: ModConfig,
    want_wav: bool,
    want_iq: bool,
    title: str = "WaveCaster"
) -> OutputPaths:
    outdir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    base = outdir / f"cast_{scheme.name.lower()}_{ts}"
    # Minimal frame (no FEC/security here; caller handles)
    fcfg = FrameConfig()
    bits = to_bits(frame_payload(text.encode("utf-8"), fcfg))
    audio, iq = bits_to_signals(bits, scheme, mcfg)
    paths = OutputPaths()
    if want_wav and audio is not None and len(audio)>0:
        paths.wav = base.with_suffix(".wav"); write_wav_mono(paths.wav, audio, mcfg.sample_rate)
    if want_iq:
        if iq is None and audio is not None:
            # make a naive hilbert to IQ for convenience
            try:
                q = np.imag(sp_signal.hilbert(audio))
                iq = audio.astype(np.float32) + 1j*q.astype(np.float32)
            except Exception:
                iq = (audio.astype(np.float32) + 1j*np.zeros_like(audio, dtype=np.float32))
        if iq is not None:
            paths.iq = base.with_suffix(".iqf32"); write_iq_f32(paths.iq, iq)
    # Visualization
    if audio is not None and len(audio)>0 and HAS_MPL:
        paths.png = base.with_suffix(".png"); plot_wave_and_spectrum(paths.png, audio, mcfg.sample_rate, title)
    # Meta
    meta = {
        "timestamp": ts, "scheme": scheme.name, "sample_rate": mcfg.sample_rate,
        "symbol_rate": mcfg.symbol_rate, "framesec": len(audio)/mcfg.sample_rate if audio is not None else 0,
    }
    paths.meta = base.with_suffix(".json")
    paths.meta.write_text(safe_json(meta), encoding="utf-8")
    return paths

def full_cast_and_save(
    text: str,
    outdir: Path,
    scheme: ModulationScheme,
    mcfg: ModConfig,
    fcfg: FrameConfig,
    sec: SecurityConfig,
    fec_scheme: FEC,
    want_wav: bool,
    want_iq: bool,
    title: str = "WaveCaster"
) -> OutputPaths:
    outdir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    base = outdir / f"cast_{scheme.name.lower()}_{ts}"
    bits = encode_text(text, fcfg, sec, fec_scheme)
    audio, iq = bits_to_signals(bits, scheme, mcfg)
    paths = OutputPaths()
    if want_wav and audio is not None and len(audio)>0:
        paths.wav = base.with_suffix(".wav"); write_wav_mono(paths.wav, audio, mcfg.sample_rate)
    if want_iq:
        if iq is None and audio is not None:
            try:
                q = np.imag(sp_signal.hilbert(audio))
                iq = audio.astype(np.float32) + 1j*q.astype(np.float32)
            except Exception:
                iq = (audio.astype(np.float32) + 1j*np.zeros_like(audio, dtype=np.float32))
        if iq is not None:
            paths.iq = base.with_suffix(".iqf32"); write_iq_f32(paths.iq, iq)
    if audio is not None and len(audio)>0 and HAS_MPL:
        paths.png = base.with_suffix(".png"); plot_wave_and_spectrum(paths.png, audio, mcfg.sample_rate, title)
    meta = {
        "timestamp": ts, "scheme": scheme.name, "sample_rate": mcfg.sample_rate,
        "symbol_rate": mcfg.symbol_rate, "framesec": len(audio)/mcfg.sample_rate if audio is not None else 0,
        "fec": fec_scheme.name, "encrypted": bool(sec.password), "watermark": bool(sec.watermark),
        "hmac": bool(sec.hmac_key),
    }
    paths.meta = base.with_suffix(".json"); paths.meta.write_text(safe_json(meta), encoding="utf-8")
    return paths

# =========================================================
# CLI
# =========================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dual_llm_wavecaster_enhanced", description="Two-LLM orchestration → modulated waveform")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_mod_args(sp):
        sp.add_argument("--scheme", choices=[s.name.lower() for s in ModulationScheme], default="bfsk")
        sp.add_argument("--sample-rate", type=int, default=48000)
        sp.add_argument("--symbol-rate", type=int, default=1200)
        sp.add_argument("--amplitude", type=float, default=0.7)
        sp.add_argument("--f0", type=float, default=1200.0)
        sp.add_argument("--f1", type=float, default=2200.0)
        sp.add_argument("--fc", type=float, default=1800.0)
        sp.add_argument("--no-clip", action="store_true")
        sp.add_argument("--outdir", type=str, default="casts")
        sp.add_argument("--wav", action="store_true")
        sp.add_argument("--iq", action="store_true")
        sp.add_argument("--play", action="store_true", help="Play audio to soundcard (if available)")

        # OFDM / DSSS
        sp.add_argument("--ofdm-subc", type=int, default=64)
        sp.add_argument("--cp-len", type=int, default=16)
        sp.add_argument("--dsss-chip-rate", type=int, default=4800)

    # cast: 2-LLM orchestration then modulate
    sp_cast = sub.add_parser("cast", help="Compose via dual LLMs then modulate")
    sp_cast.add_argument("--prompt", type=str, required=True)
    sp_cast.add_argument("--resource-file", nargs="*", default=[])
    sp_cast.add_argument("--resource-text", nargs="*", default=[])
    # Local LLM
    sp_cast.add_argument("--local-url", type=str, default="http://127.0.0.1:8080")
    sp_cast.add_argument("--local-mode", choices=["openai-chat","openai-completions","llama-cpp","textgen-webui"], default="llama-cpp")
    sp_cast.add_argument("--local-model", type=str, default="local-gguf")
    sp_cast.add_argument("--local-key", type=str, default=None)
    # Remote Resource LLM
    sp_cast.add_argument("--remote-url", type=str, default=None)
    sp_cast.add_argument("--remote-model", type=str, default="gpt-4o-mini")
    sp_cast.add_argument("--remote-key", type=str, default=None)
    # Orchestration params
    sp_cast.add_argument("--style", type=str, default="concise")
    sp_cast.add_argument("--max-tokens", type=int, default=512)
    sp_cast.add_argument("--temperature", type=float, default=0.7)
    # Security / FEC
    sp_cast.add_argument("--password", type=str, default=None)
    sp_cast.add_argument("--watermark", type=str, default=None)
    sp_cast.add_argument("--hmac-key", type=str, default=None)
    sp_cast.add_argument("--fec", choices=[f.name.lower() for f in FEC], default="hamming74")
    add_mod_args(sp_cast)

    # modulate: direct text to waveform
    sp_mod = sub.add_parser("modulate", help="Modulate provided text directly")
    sp_mod.add_argument("--text", type=str, required=True)
    sp_mod.add_argument("--password", type=str, default=None)
    sp_mod.add_argument("--watermark", type=str, default=None)
    sp_mod.add_argument("--hmac-key", type=str, default=None)
    sp_mod.add_argument("--fec", choices=[f.name.lower() for f in FEC], default="none")
    add_mod_args(sp_mod)

    # visualize existing WAV
    sp_vis = sub.add_parser("visualize", help="Plot waveform + spectrum from WAV")
    sp_vis.add_argument("--wav", type=str, required=True)
    sp_vis.add_argument("--out", type=str, default=None)

    # analyze: print basic metrics
    sp_an = sub.add_parser("analyze", help="Basic audio metrics of WAV")
    sp_an.add_argument("--wav", type=str, required=True)

    return p

def make_modcfg(args: argparse.Namespace) -> ModConfig:
    return ModConfig(
        sample_rate=args.sample_rate, symbol_rate=args.symbol_rate, amplitude=args.amplitude,
        f0=args.f0, f1=args.f1, fc=args.fc, clip=not args.no_clip,
        ofdm_subc=getattr(args, "ofdm_subc", 64), cp_len=getattr(args,"cp_len",16),
        dsss_chip_rate=getattr(args,"dsss_chip_rate",4800),
    )

def parse_scheme(s: str) -> ModulationScheme:
    return ModulationScheme[s.upper()]

def parse_fec(s: str) -> FEC:
    return FEC[s.upper()]

def cmd_cast(args: argparse.Namespace) -> int:
    # Build LLMs
    local = LocalLLM([HTTPConfig(
        base_url=args.local_url, model=args.local_model, mode=args.local_mode, api_key=args.local_key
    )])
    rcfg = HTTPConfig(base_url=args.remote_url, model=args.remote_model, api_key=args.remote_key) if args.remote_url else None
    resource = ResourceLLM(rcfg)
    orch = DualLLMOrchestrator(local, resource, OrchestratorSettings(
        temperature=args.temperature, max_tokens=args.max_tokens, style=args.style
    ))
    result = orch.run(args.prompt, args.resource_file, args.resource_text)
    # Build pipeline
    mcfg = make_modcfg(args)
    fcfg = FrameConfig()
    sec = SecurityConfig(password=args.password, watermark=args.watermark, hmac_key=args.hmac_key)
    scheme = parse_scheme(args.scheme)
    fec_s = parse_fec(args.fec)
    paths = full_cast_and_save(
        text=result["final"], outdir=Path(args.outdir), scheme=scheme, mcfg=mcfg, fcfg=fcfg,
        sec=sec, fec_scheme=fec_s, want_wav=args.wav or (not args.iq), want_iq=args.iq,
        title=f"{scheme.name} | DualLLM Wave"
    )
    if args.play and paths.wav and HAS_AUDIO:
        import soundfile as sf
        try:
            data, sr = sf.read(str(paths.wav), dtype="float32")
            play_audio(data, sr)
        except Exception:
            # Fallback: play from generated buffer (we already have audio only inside full_cast if we changed it to return)
            log.warning("Install soundfile for playback of saved WAV, or use --play with 'modulate'")
    print(safe_json({
        "files": {"wav": str(paths.wav) if paths.wav else None,
                  "iq": str(paths.iq) if paths.iq else None,
                  "meta": str(paths.meta) if paths.meta else None,
                  "png": str(paths.png) if paths.png else None},
        "preview": result["final"][:400],
        "summary": result["summary"][:400],
    }))
    return 0

def cmd_modulate(args: argparse.Namespace) -> int:
    mcfg = make_modcfg(args)
    fcfg = FrameConfig()
    sec = SecurityConfig(password=args.password, watermark=args.watermark, hmac_key=args.hmac_key)
    scheme = parse_scheme(args.scheme)
    fec_s = parse_fec(args.fec)
    paths = full_cast_and_save(
        text=args.text, outdir=Path(args.outdir), scheme=scheme, mcfg=mcfg, fcfg=fcfg,
        sec=sec, fec_scheme=fec_s, want_wav=args.wav or (not args.iq), want_iq=args.iq,
        title=f"{scheme.name} | Direct Mod"
    )
    if args.play and paths.wav:
        try:
            import soundfile as sf
            data, sr = sf.read(str(paths.wav), dtype="float32"); play_audio(data, sr)
        except Exception:
            log.warning("Install soundfile for playback of saved WAV")
    print(safe_json({"files": {"wav": str(paths.wav) if paths.wav else None,
                               "iq": str(paths.iq) if paths.iq else None,
                               "meta": str(paths.meta) if paths.meta else None,
                               "png": str(paths.png) if paths.png else None}}))
    return 0

def cmd_visualize(args: argparse.Namespace) -> int:
    if not HAS_MPL:
        print("matplotlib is not installed.")
        return 1
    import wave
    with wave.open(args.wav, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        s = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)/32767.0
    out = Path(args.out or (Path(args.wav).with_suffix(".png")))
    plot_wave_and_spectrum(out, s, sr, f"Visualize: {Path(args.wav).name}")
    print(safe_json({"png": str(out), "sample_rate": sr, "seconds": len(s)/sr}))
    return 0

def cmd_analyze(args: argparse.Namespace) -> int:
    import wave
    with wave.open(args.wav, "rb") as w:
        sr = w.getframerate(); n = w.getnframes()
        s = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)/32767.0
    dur = len(s)/sr
    rms = float(np.sqrt(np.mean(s**2)))
    peak = float(np.max(np.abs(s)))
    spec = np.abs(rfft(s)); spec /= (spec.max()+1e-12)
    # simple SNR estimate
    snr = 10*np.log10(np.mean(s**2) / (np.var(s - np.mean(s)) + 1e-12))
    print(safe_json({"sample_rate": sr, "seconds": dur, "rms": rms, "peak": peak, "snr_db": float(snr)}))
    return 0

def main(argv: Optional[List[str]] = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if args.cmd == "cast": return cmd_cast(args)
    if args.cmd == "modulate": return cmd_modulate(args)
    if args.cmd == "visualize": return cmd_visualize(args)
    if args.cmd == "analyze": return cmd_analyze(args)
    p.print_help(); return 2

if __name__ == "__main__":
    raise SystemExit(main())
