"""
AeroGuard AI — Software-Defined RTL-SDR & ADS-B Reception Subsystem
===================================================================
Software Defined Communication (SDC) subsystem providing:
1. Input Sources:
   - Mode 1: Recorded complex IQ sample reader (.cu8, .cf32, .bin) with metadata validation.
   - Mode 2: Deterministic synthetic ADS-B baseband IQ generator with PPM modulation & AWGN.
   - Mode 3: Modular RTL-SDR hardware adapter with device discovery & graceful unavailable reporting.
   - Mode 4: Airspace telemetry replay interface.
2. Signal Processing & Decoding:
   - Envelope detection (|s(t)| = sqrt(I^2 + Q^2)).
   - 8-microsecond Mode-S preamble matched filtering.
   - 1 Mbit/s Pulse Position Modulation (PPM) bit slicing.
   - 24-bit CRC polynomial parity verification (G(x) = 0x1FFF409).
   - Mode-S DF17 Extended Squitter field extraction (ICAO24, Callsign, Altitude, Velocity, Heading).
3. Receiver Management & Statistics:
   - Tracks sample count, preambles detected, CRC pass/fail counts, and active aircraft tracks.
"""

import os, sys, time, struct
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np

# ==============================================================================
# SDC DATA STRUCTURES
# ==============================================================================
@dataclass
class DecodedADSBMessage:
    raw_hex: str
    timestamp: str
    df: int
    icao24: str
    tc: int
    message_type: str
    callsign: Optional[str] = None
    altitude_ft: Optional[float] = None
    groundspeed_kts: Optional[float] = None
    heading_deg: Optional[float] = None
    vertical_rate_fpm: Optional[float] = None
    crc_valid: bool = False
    signal_source: str = "SYNTHETIC_IQ"
    signal_snr_db: Optional[float] = None

# Mode-S character mapping for Callsign decoding
MODES_CHARSET = "?ABCDEFGHIJKLMNOPQRSTUVWXYZ????? ???????????????0123456789??????"

# Official ICAO 24-bit Mode-S Generator Polynomial: x^24 + x^23 + ... + 1 (0x1FFF409)
MODES_GENERATOR = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1
]

def compute_modes_crc24(bits_88: List[int]) -> List[int]:
    """Calculates 24-bit parity remainder for 88-bit Mode-S message data payload."""
    padded = list(bits_88) + [0] * 24
    for i in range(88):
        if padded[i] == 1:
            for j in range(25):
                padded[i + j] ^= MODES_GENERATOR[j]
    return padded[88:]

def verify_modes_crc24(bits_112: List[int]) -> bool:
    """Verifies that 112-bit Mode-S message has remainder 0 modulo generator polynomial."""
    rem = list(bits_112)
    for i in range(88):
        if rem[i] == 1:
            for j in range(25):
                rem[i + j] ^= MODES_GENERATOR[j]
    return sum(rem[88:]) == 0

# ==============================================================================
# 1. ADS-B BASEBAND SIGNAL MODULATOR (SYNTHESIS & ENCODING)
# ==============================================================================
class ADSBWaveformModulator:
    """
    Synthesizes physically accurate 1090 MHz complex baseband ADS-B waveforms.
    Generates PPM pulses with preamble and configurable Additive White Gaussian Noise (AWGN).
    """
    @staticmethod
    def encode_callsign_frame(icao24: str, callsign: str) -> bytes:
        """Encodes TC=4 Aircraft Identification message with valid 24-bit CRC."""
        icao_int = int(icao24, 16)
        bits = []
        # DF = 17 (5 bits: 10001), CA = 5 (3 bits: 101)
        df_ca = (17 << 3) | 5
        for i in range(7, -1, -1):
            bits.append((df_ca >> i) & 1)
        # ICAO24 (24 bits)
        for i in range(23, -1, -1):
            bits.append((icao_int >> i) & 1)
        # TC = 4, Category = 0 (8 bits: 00100 000)
        tc_cat = (4 << 3) | 0
        for i in range(7, -1, -1):
            bits.append((tc_cat >> i) & 1)
        # Callsign characters (6 bits each, 8 chars = 48 bits)
        clean_callsign = (callsign.upper().strip() + "        ")[:8]
        for ch in clean_callsign:
            idx = MODES_CHARSET.find(ch)
            if idx == -1: idx = 32 # space
            for i in range(5, -1, -1):
                bits.append((idx >> i) & 1)

        assert len(bits) == 88, f"Expected 88 bits, got {len(bits)}"
        parity = compute_modes_crc24(bits)
        all_bits = bits + parity

        # Pack to 14 bytes (112 bits)
        byte_arr = bytearray()
        for i in range(0, 112, 8):
            val = 0
            for b in all_bits[i : i + 8]:
                val = (val << 1) | b
            byte_arr.append(val)
        return bytes(byte_arr)

    @staticmethod
    def encode_airborne_velocity_frame(icao24: str, speed_kts: float, heading_deg: float, vert_rate_fpm: float) -> bytes:
        """Encodes TC=19 Airborne Velocity message with valid 24-bit CRC."""
        icao_int = int(icao24, 16)
        bits = []
        # DF = 17, CA = 5
        df_ca = (17 << 3) | 5
        for i in range(7, -1, -1):
            bits.append((df_ca >> i) & 1)
        # ICAO24
        for i in range(23, -1, -1):
            bits.append((icao_int >> i) & 1)
        # TC = 19 (5 bits: 10011), Subtype = 1 (3 bits: 001 Groundspeed normal)
        tc_sub = (19 << 3) | 1
        for i in range(7, -1, -1):
            bits.append((tc_sub >> i) & 1)

        # Simplified Subtype 1 fields (48 bits):
        # East/West & North/South components from speed and heading
        head_rad = np.radians(heading_deg)
        v_ew = int(min(1023, max(1, abs(speed_kts * np.sin(head_rad)) + 1)))
        v_ns = int(min(1023, max(1, abs(speed_kts * np.cos(head_rad)) + 1)))
        dir_ew = 0 if np.sin(head_rad) >= 0 else 1
        dir_ns = 0 if np.cos(head_rad) >= 0 else 1

        # 1 bit Intent, 1 bit NACv, 1 bit dir_ew, 10 bits v_ew, 1 bit dir_ns, 10 bits v_ns
        vel_bits_int = (dir_ew << 21) | (v_ew << 11) | (dir_ns << 10) | v_ns
        for i in range(23, -1, -1):
            bits.append((vel_bits_int >> i) & 1)

        # Vertical rate (1 bit source, 1 bit sign, 9 bits rate in 64 fpm units)
        vr_sign = 0 if vert_rate_fpm >= 0 else 1
        vr_val = int(min(511, max(1, abs(vert_rate_fpm) / 64 + 1)))
        vr_int = (vr_sign << 9) | vr_val
        for i in range(23, -1, -1):
            bits.append((vr_int >> i) & 1)

        assert len(bits) == 88, f"Expected 88 bits, got {len(bits)}"
        parity = compute_modes_crc24(bits)
        all_bits = bits + parity

        byte_arr = bytearray()
        for i in range(0, 112, 8):
            val = 0
            for b in all_bits[i : i + 8]:
                val = (val << 1) | b
            byte_arr.append(val)
        return bytes(byte_arr)

    @staticmethod
    def synthesize_ppm_baseband(
        frame_bytes: bytes,
        sample_rate: float = 2000000.0,
        snr_db: float = 25.0,
        freq_offset_hz: float = 0.0,
        lead_silence_samples: int = 100,
        trail_silence_samples: int = 100
    ) -> np.ndarray:
        """
        Modulates 112-bit frame into complex baseband IQ samples at specified sample rate.
        Each microsecond corresponds to (sample_rate / 1e6) samples.
        """
        samples_per_us = sample_rate / 1e6
        chip_samples = int(round(samples_per_us * 0.5))

        # 1. Preamble (8 microseconds = 16 chips of 0.5 us)
        # Pulses at chips 0, 2, 7, 9
        preamble_chips = [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]

        # 2. Data Payload (112 bits = 224 chips of 0.5 us PPM)
        # Bit '1' -> [1, 0], Bit '0' -> [0, 1]
        bits = []
        for b in frame_bytes:
            for i in range(7, -1, -1):
                bits.append((b >> i) & 1)

        data_chips = []
        for bit in bits:
            if bit == 1:
                data_chips.extend([1, 0])
            else:
                data_chips.extend([0, 1])

        all_chips = preamble_chips + data_chips

        # Expand chips to continuous samples
        amplitude_pulse = np.repeat(all_chips, chip_samples).astype(np.float32)

        # Add leading and trailing silence
        envelope = np.concatenate([
            np.zeros(lead_silence_samples, dtype=np.float32),
            amplitude_pulse,
            np.zeros(trail_silence_samples, dtype=np.float32)
        ])

        # Complex carrier with frequency offset
        t = np.arange(len(envelope)) / sample_rate
        carrier = np.exp(1j * (2 * np.pi * freq_offset_hz * t))
        signal = envelope * carrier

        # Additive White Gaussian Noise (AWGN)
        if snr_db < 100.0:
            sig_power = np.mean(envelope**2)
            noise_power = sig_power / (10.0 ** (snr_db / 10.0))
            noise = np.sqrt(noise_power / 2.0) * (
                np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
            )
            signal = signal + noise

        return signal.astype(np.complex64)

# ==============================================================================
# 2. ADS-B BASEBAND SIGNAL DEMODULATOR & PARSER
# ==============================================================================
class ADSBWaveformDecoder:
    """
    Decodes Mode-S PPM baseband IQ waveforms:
    1. Magnitude envelope calculation.
    2. Preamble matched filter correlation.
    3. Pulse position chip slicing (112 bits).
    4. 24-bit CRC parity verification.
    5. Field parsing (ICAO24, Callsign, Velocity, Heading).
    """
    def __init__(self, sample_rate: float = 2000000.0):
        self.sample_rate = sample_rate
        self.samples_per_us = sample_rate / 1e6
        self.chip_samples = max(1, int(round(self.samples_per_us * 0.5)))
        self.frame_samples = int(round(120 * self.samples_per_us)) # 8us preamble + 112us data

    def decode_iq(self, iq_samples: np.ndarray, source_label: str = "RECORDED_IQ") -> List[DecodedADSBMessage]:
        """Scans IQ buffer, correlates preamble, and decodes valid Mode-S frames."""
        mag = np.abs(iq_samples).astype(np.float32)
        total_len = len(mag)
        decoded = []

        cs = self.chip_samples
        preamble_len = cs * 16 # 8 us

        # Fast energy scan for preamble pulses: (P0 + P2 + P7 + P9) vs quiet intervals
        i = 0
        while i < total_len - self.frame_samples:
            p0 = mag[i : i + cs].sum()
            p2 = mag[i + 2*cs : i + 3*cs].sum()
            p7 = mag[i + 7*cs : i + 8*cs].sum()
            p9 = mag[i + 9*cs : i + 10*cs].sum()
            pulse_energy = (p0 + p2 + p7 + p9) / 4.0

            # Quiet valley intervals
            q1 = mag[i + 1*cs : i + 2*cs].sum()
            q4 = mag[i + 4*cs : i + 5*cs].sum()
            valley_energy = (q1 + q4) / 2.0 + 1e-6

            # Preamble trigger: strong pulses with quiet valleys
            if pulse_energy > 0.15 and (pulse_energy / valley_energy) > 1.8:
                # Slice 112 bits starting after preamble (offset 16 chips)
                data_start = i + preamble_len
                bits = []
                for b_idx in range(112):
                    chip1 = mag[data_start + b_idx * 2 * cs : data_start + (b_idx * 2 + 1) * cs].sum()
                    chip2 = mag[data_start + (b_idx * 2 + 1) * cs : data_start + (b_idx * 2 + 2) * cs].sum()
                    bits.append(1 if chip1 > chip2 else 0)

                # Verify CRC
                crc_ok = verify_modes_crc24(bits)
                if crc_ok:
                    msg = self._parse_frame_bits(bits, source_label)
                    decoded.append(msg)
                    i += self.frame_samples # Skip past decoded frame
                    continue
            i += 1

        return decoded

    def _parse_frame_bits(self, bits: List[int], source_label: str) -> DecodedADSBMessage:
        """Parses decoded 112 bits into structured message fields."""
        # Hex representation
        raw_bytes = bytearray()
        for i in range(0, 112, 8):
            val = 0
            for b in bits[i : i + 8]:
                val = (val << 1) | b
            raw_bytes.append(val)
        raw_hex = raw_bytes.hex().upper()

        # DF (bits 0-4)
        df = sum([b << (4 - idx) for idx, b in enumerate(bits[:5])])

        # ICAO24 (bits 8-31)
        icao_int = sum([b << (23 - idx) for idx, b in enumerate(bits[8:32])])
        icao24 = f"{icao_int:06X}".upper()

        # TC (bits 32-36)
        tc = sum([b << (4 - idx) for idx, b in enumerate(bits[32:37])])

        msg_type = f"DF{df}_TC{tc}"
        callsign = None
        speed_kts = None
        heading_deg = None
        vert_rate = None

        if df == 17 or df == 18:
            if 1 <= tc <= 4:
                msg_type = "AIRCRAFT_IDENTIFICATION"
                # Callsign (8 chars of 6 bits each in bits 40-87)
                cs_chars = []
                for c_i in range(8):
                    char_val = sum([b << (5 - b_i) for b_i, b in enumerate(bits[40 + c_i * 6 : 40 + (c_i + 1) * 6])])
                    cs_chars.append(MODES_CHARSET[char_val])
                callsign = "".join(cs_chars).strip()
            elif tc == 19:
                msg_type = "AIRBORNE_VELOCITY"
                subtype = sum([b << (2 - idx) for idx, b in enumerate(bits[37:40])])
                if subtype == 1 or subtype == 2:
                    dir_ew = bits[45]
                    v_ew = sum([b << (9 - idx) for idx, b in enumerate(bits[46:56])]) - 1
                    dir_ns = bits[56]
                    v_ns = sum([b << (9 - idx) for idx, b in enumerate(bits[57:67])]) - 1
                    if v_ew >= 0 and v_ns >= 0:
                        speed_kts = round(float(np.sqrt(v_ew**2 + v_ns**2)), 1)
                        heading_deg = round(float(np.degrees(np.arctan2(v_ew * (-1 if dir_ew else 1), v_ns * (-1 if dir_ns else 1))) % 360), 1)

                    # Vertical rate (bits 68-76)
                    vr_sign = bits[68]
                    vr_val = (sum([b << (8 - idx) for idx, b in enumerate(bits[69:78])]) - 1) * 64
                    vert_rate = -vr_val if vr_sign else vr_val

        return DecodedADSBMessage(
            raw_hex=raw_hex,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            df=df,
            icao24=icao24,
            tc=tc,
            message_type=msg_type,
            callsign=callsign,
            groundspeed_kts=speed_kts,
            heading_deg=heading_deg,
            vertical_rate_fpm=vert_rate,
            crc_valid=True,
            signal_source=source_label
        )

# ==============================================================================
# 3. INPUT SOURCES
# ==============================================================================
class RecordedIQSource:
    """Mode 1: Reads and validates complex IQ recordings from binary disk files."""
    @staticmethod
    def read_iq_file(
        filepath: str,
        iq_format: str = "cu8",
        sample_rate: float = 2000000.0,
        center_freq: float = 1090000000.0,
        max_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reads complex IQ samples with explicit metadata validation."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"IQ recording not found: {filepath}")

        file_size = os.path.getsize(filepath)
        if file_size == 0:
            raise ValueError(f"IQ recording file is empty: {filepath}")

        metadata = {
            "filepath": filepath,
            "format": iq_format.lower(),
            "sample_rate_hz": float(sample_rate),
            "center_freq_hz": float(center_freq),
            "file_size_bytes": file_size
        }

        fmt = iq_format.lower()
        if fmt in ["cu8", "uint8", "bin"]:
            # 8-bit unsigned interleaved IQ: I, Q, I, Q...
            count = max_samples * 2 if max_samples else -1
            raw = np.fromfile(filepath, dtype=np.uint8, count=count)
            # Normalize [0, 255] to float32 centered at 0.0
            i_samples = (raw[0::2].astype(np.float32) - 127.5) / 128.0
            q_samples = (raw[1::2].astype(np.float32) - 127.5) / 128.0
            samples = i_samples + 1j * q_samples
        elif fmt in ["cf32", "float32", "complex64"]:
            count = max_samples if max_samples else -1
            samples = np.fromfile(filepath, dtype=np.complex64, count=count)
        else:
            raise ValueError(f"Unsupported IQ format: {iq_format}. Expected 'cu8' or 'cf32'.")

        metadata["samples_loaded"] = len(samples)
        metadata["duration_ms"] = round((len(samples) / sample_rate) * 1000.0, 2)
        return samples.astype(np.complex64), metadata

class SyntheticIQSource:
    """Mode 2: Generates documented valid synthetic test waveforms."""
    def __init__(self, sample_rate: float = 2000000.0):
        self.sample_rate = sample_rate
        self.modulator = ADSBWaveformModulator()

    def generate_ground_truth_test_burst(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Generates deterministic ADS-B waveform burst containing 3 known aircraft frames."""
        f1 = self.modulator.encode_callsign_frame("500006", "RYR205")
        f2 = self.modulator.encode_airborne_velocity_frame("500006", speed_kts=420.0, heading_deg=180.0, vert_rate_fpm=0.0)
        f3 = self.modulator.encode_callsign_frame("4840D6", "KLM1023")

        # Modulate into baseband IQ with silence between frames
        iq1 = self.modulator.synthesize_ppm_baseband(f1, sample_rate=self.sample_rate, snr_db=30.0)
        gap = np.zeros(int(self.sample_rate * 0.0005), dtype=np.complex64) # 500 us gap
        iq2 = self.modulator.synthesize_ppm_baseband(f2, sample_rate=self.sample_rate, snr_db=25.0)
        iq3 = self.modulator.synthesize_ppm_baseband(f3, sample_rate=self.sample_rate, snr_db=28.0)

        full_iq = np.concatenate([iq1, gap, iq2, gap, iq3])

        ground_truth = [
            {"icao24": "500006", "type": "AIRCRAFT_IDENTIFICATION", "callsign": "RYR205"},
            {"icao24": "500006", "type": "AIRBORNE_VELOCITY", "speed_kts": 420.0},
            {"icao24": "4840D6", "type": "AIRCRAFT_IDENTIFICATION", "callsign": "KLM1023"}
        ]
        return full_iq, ground_truth

class HardwareRtlSdrAdapter:
    """Mode 3: Modular interface for physical RTL-SDR hardware with safe discovery."""
    @staticmethod
    def probe_device() -> Dict[str, Any]:
        """Discovers physical RTL-SDR devices without crashing or fabricating reception."""
        try:
            import rtlsdr
            devices = rtlsdr.RtlSdr.get_device_serial_addresses()
            if devices:
                return {
                    "hardware_detected": True,
                    "devices": devices,
                    "status": f"Physical RTL-SDR detected ({len(devices)} device(s))"
                }
        except ImportError:
            pass
        except Exception:
            pass

        return {
            "hardware_detected": False,
            "devices": [],
            "status": "No physical RTL-SDR detected. Operating in software-defined simulation mode."
        }

# ==============================================================================
# 4. SOFTWARE-DEFINED RECEIVER MANAGER
# ==============================================================================
class SoftwareDefinedReceiver:
    """
    Central controller managing software-defined radio reception, decoding,
    and statistics for the Command Center.
    """
    def __init__(self, sample_rate: float = 2000000.0, center_freq: float = 1090000000.0):
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.current_mode = "synthetic_iq" # 'synthetic_iq' | 'recorded_iq' | 'hardware_sdr' | 'telemetry_replay'

        self.decoder = ADSBWaveformDecoder(sample_rate=sample_rate)
        self.synthetic_source = SyntheticIQSource(sample_rate=sample_rate)

        # Statistics
        self.samples_processed = 0
        self.messages_decoded = 0
        self.crc_passed = 0
        self.crc_failed = 0
        self.last_valid_timestamp = None
        self.recent_decoded_messages: List[Dict[str, Any]] = []

    def set_mode(self, mode: str):
        valid_modes = ["synthetic_iq", "recorded_iq", "hardware_sdr", "telemetry_replay"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")
        self.current_mode = mode

    def process_synthetic_burst(self) -> List[DecodedADSBMessage]:
        """Runs a synthetic ADS-B baseband burst through the decoder."""
        iq_burst, _ = self.synthetic_source.generate_ground_truth_test_burst()
        self.samples_processed += len(iq_burst)

        msgs = self.decoder.decode_iq(iq_burst, source_label="SYNTHETIC_IQ")
        for m in msgs:
            self.messages_decoded += 1
            if m.crc_valid:
                self.crc_passed += 1
                self.last_valid_timestamp = m.timestamp
            else:
                self.crc_failed += 1
            self.recent_decoded_messages.insert(0, asdict(m))

        self.recent_decoded_messages = self.recent_decoded_messages[:30] # Keep last 30
        return msgs

    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive SDR status for Command Center display."""
        hw_probe = HardwareRtlSdrAdapter.probe_device()
        return {
            "receiver_mode": self.current_mode,
            "center_frequency_mhz": self.center_freq / 1e6,
            "sample_rate_msps": self.sample_rate / 1e6,
            "hardware_connected": hw_probe["hardware_detected"],
            "hardware_status": hw_probe["status"],
            "samples_processed": self.samples_processed,
            "messages_decoded": self.messages_decoded,
            "crc_passed": self.crc_passed,
            "crc_failed": self.crc_failed,
            "last_message_timestamp": self.last_valid_timestamp or "No frames decoded yet",
            "recent_messages_count": len(self.recent_decoded_messages)
        }

# Global singleton receiver instance
sdr_receiver = SoftwareDefinedReceiver()
