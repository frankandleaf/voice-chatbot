"""Audio utility functions: resampling, format conversion, buffering."""

import numpy as np


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float32 audio (-1.0 to 1.0) to int16."""
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)


def int16_to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert int16 audio to float32 (-1.0 to 1.0)."""
    return audio.astype(np.float32) / 32768.0


def bytes_to_float32(data: bytes) -> np.ndarray:
    """Convert raw int16 bytes to float32 numpy array."""
    arr = np.frombuffer(data, dtype=np.int16)
    return int16_to_float32(arr)


def float32_to_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 numpy array to raw int16 bytes."""
    return float32_to_int16(audio).tobytes()


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Simple linear resampling using numpy interpolation.

    For production use, consider scipy.signal.resample or librosa.resample,
    but this avoids extra dependencies for basic use.
    """
    if orig_sr == target_sr:
        return audio

    duration = len(audio) / orig_sr
    target_len = int(duration * target_sr)
    indices = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def wav_bytes_to_float32(wav_bytes: bytes, expected_sr: int = 24000) -> tuple[np.ndarray, int]:
    """Parse WAV bytes and return (float32_samples, sample_rate).

    Handles standard PCM16 WAV format. Returns audio as float32 array.
    """
    import struct

    if len(wav_bytes) < 44:
        raise ValueError("WAV data too short")

    # Parse RIFF header
    riff, _, wave = struct.unpack_from("<4sI4s", wav_bytes, 0)
    if riff != b"RIFF" or wave != b"WAVE":
        raise ValueError("Not a valid WAV file")

    # Find data chunk
    offset = 12
    sample_rate = expected_sr
    data = None

    while offset < len(wav_bytes) - 8:
        chunk_id, chunk_size = struct.unpack_from("<4sI", wav_bytes, offset)
        offset += 8

        if chunk_id == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", wav_bytes, offset)
            _, num_channels, sample_rate, _, _, bits_per_sample = fmt
            offset += chunk_size
        elif chunk_id == b"data":
            data = wav_bytes[offset : offset + chunk_size]
            offset += chunk_size
        else:
            offset += chunk_size

    if data is None:
        raise ValueError("No data chunk found in WAV")

    if bits_per_sample == 16:
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    elif bits_per_sample == 32:
        samples = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif bits_per_sample == 8:
        samples = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        raise ValueError(f"Unsupported bit depth: {bits_per_sample}")

    if num_channels > 1:
        samples = samples.reshape(-1, num_channels).mean(axis=1)

    return samples.astype(np.float32), sample_rate


def calculate_rms(audio: np.ndarray) -> float:
    """Calculate RMS energy of audio signal."""
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def is_silence(audio: np.ndarray, threshold: float = 0.005) -> bool:
    """Check if audio is below the silence threshold."""
    return calculate_rms(audio) < threshold
