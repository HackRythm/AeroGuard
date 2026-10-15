"""
Unit Tests for AeroGuard Software-Defined ADS-B Receiver Subsystem
==================================================================
Tests:
1. Mode-S 24-bit CRC polynomial calculation and verification.
2. Callsign & Velocity frame encoding.
3. Baseband PPM modulation & complex IQ synthesis.
4. End-to-end IQ waveform decoding (envelope detection, preamble correlation, bit slicing).
5. Recorded IQ sample loading & metadata validation.
6. Hardware adapter graceful probing (no hardware required).
7. SoftwareDefinedReceiver manager state & statistics.
"""

import os, sys, tempfile
sys.path.insert(0, os.path.abspath("."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import unittest
import numpy as np

from scripts.rtl_sdr_adapter import (
    ADSBWaveformModulator, ADSBWaveformDecoder, RecordedIQSource,
    SyntheticIQSource, HardwareRtlSdrAdapter, SoftwareDefinedReceiver,
    compute_modes_crc24, verify_modes_crc24
)

class TestADSBReceiver(unittest.TestCase):
    def test_crc24_parity(self):
        """Verifies 24-bit ICAO generator polynomial parity."""
        # Standard known test frame: KLM1023
        raw = bytes.fromhex("8D4840D6202CC371C32CE0576098")
        bits = []
        for b in raw:
            for i in range(7, -1, -1):
                bits.append((b >> i) & 1)
        self.assertEqual(len(bits), 112)
        self.assertTrue(verify_modes_crc24(bits))

    def test_callsign_encoding_and_crc(self):
        """Verifies callsign frame encoding produces valid 112-bit message with valid CRC."""
        frame = ADSBWaveformModulator.encode_callsign_frame("500006", "RYR205")
        self.assertEqual(len(frame), 14) # 14 bytes = 112 bits
        bits = []
        for b in frame:
            for i in range(7, -1, -1):
                bits.append((b >> i) & 1)
        self.assertTrue(verify_modes_crc24(bits))

    def test_airborne_velocity_encoding_and_crc(self):
        """Verifies velocity frame encoding produces valid 112-bit message with valid CRC."""
        frame = ADSBWaveformModulator.encode_airborne_velocity_frame("500006", speed_kts=420.0, heading_deg=180.0, vert_rate_fpm=0.0)
        self.assertEqual(len(frame), 14)
        bits = []
        for b in frame:
            for i in range(7, -1, -1):
                bits.append((b >> i) & 1)
        self.assertTrue(verify_modes_crc24(bits))

    def test_ppm_modulation_and_synthesis(self):
        """Verifies baseband PPM modulation outputs complex IQ with preamble pulses."""
        frame = ADSBWaveformModulator.encode_callsign_frame("500006", "RYR205")
        iq = ADSBWaveformModulator.synthesize_ppm_baseband(frame, sample_rate=2000000.0, snr_db=30.0)
        self.assertEqual(iq.dtype, np.complex64)
        self.assertTrue(len(iq) > 240)
        # Verify finite values
        self.assertFalse(np.isnan(iq).any())

    def test_end_to_end_iq_decoding(self):
        """Verifies end-to-end demodulation & decoding of synthetic baseband waveforms."""
        source = SyntheticIQSource(sample_rate=2000000.0)
        iq_burst, ground_truth = source.generate_ground_truth_test_burst()

        decoder = ADSBWaveformDecoder(sample_rate=2000000.0)
        decoded_msgs = decoder.decode_iq(iq_burst, source_label="SYNTHETIC_IQ")

        self.assertTrue(len(decoded_msgs) >= 2)
        # Check that RYR205 and KLM1023 were decoded
        callsigns = [m.callsign for m in decoded_msgs if m.callsign]
        self.assertIn("RYR205", callsigns)
        self.assertIn("KLM1023", callsigns)
        for m in decoded_msgs:
            self.assertTrue(m.crc_valid)
            self.assertEqual(m.df, 17)

    def test_recorded_iq_source(self):
        """Verifies binary IQ file loading and metadata validation."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            tmp_path = tmp.name
            # Write 1000 dummy complex samples as interleaved uint8
            raw_data = np.random.randint(0, 256, size=2000, dtype=np.uint8)
            raw_data.tofile(tmp)

        try:
            samples, meta = RecordedIQSource.read_iq_file(tmp_path, iq_format="cu8", sample_rate=2000000.0)
            self.assertEqual(len(samples), 1000)
            self.assertEqual(samples.dtype, np.complex64)
            self.assertEqual(meta["format"], "cu8")
            self.assertEqual(meta["sample_rate_hz"], 2000000.0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_hardware_adapter_probing(self):
        """Verifies hardware adapter safely reports device status without crashing."""
        probe = HardwareRtlSdrAdapter.probe_device()
        self.assertIn("hardware_detected", probe)
        self.assertIn("status", probe)

    def test_receiver_manager(self):
        """Verifies SoftwareDefinedReceiver state, statistics, and synthetic burst processing."""
        rx = SoftwareDefinedReceiver(sample_rate=2000000.0)
        self.assertEqual(rx.current_mode, "synthetic_iq")

        msgs = rx.process_synthetic_burst()
        self.assertTrue(len(msgs) >= 2)
        self.assertTrue(rx.crc_passed >= 2)
        self.assertEqual(rx.crc_failed, 0)

        status = rx.get_status()
        self.assertEqual(status["receiver_mode"], "synthetic_iq")
        self.assertTrue(status["messages_decoded"] >= 2)

if __name__ == "__main__":
    unittest.main()
