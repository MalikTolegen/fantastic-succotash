# -*- coding: utf-8 -*-
import os
import sys
import json
import glob
import math
import numpy as np
import time
from scipy import signal
import traceback
from collections import deque

try:
    from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
    from PyQt5.QtCore import QThread, pyqtSlot, pyqtSignal, QObject, QTimer, Qt
except ImportError:
    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
    from PySide6.QtCore import QThread, Slot as pyqtSlot, Signal as pyqtSignal, QObject, QTimer, Qt

# 파일명 대소문자 주의 (보통 소문자로 저장됨)
try:
    from mainwindow import MainWindow
except ImportError:
    from MainWindow import MainWindow

FS = 1000000.0  # 1 MHz


# --- BPF Logic ---
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # 안전 장치: Nyquist 주파수 넘지 않게
    if high >= 1.0: high = 0.99
    if low <= 0.0: low = 0.001

    sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.sosfilt(sos, data)
    return y


# --- Distance Calculator ---
class DistanceCalculator:
    """Calculate distance using ToF detection, TX centroid, and LSE model."""
    
    # LSE Model parameters
    BETA_0 = -14.057  # should be calibrated before experiment
    BETA_1 = 0.5
    
    def __init__(self, sampling_rate=1e6):
        """
        Initialize distance calculator.
        
        Args:
            sampling_rate: ADC sampling rate in Hz (default: 1 MHz = 1 sample/μs)
        """
        self.fs = sampling_rate
        self.nyquist = sampling_rate / 2.0
        
        # Filter parameters (defaults, can be overridden with UI settings)
        self.bandpass_low = 20e3   # 20 kHz lower cutoff
        self.bandpass_high = 60e3  # 60 kHz upper cutoff
        self.envelope_cutoff = 5e3  # 5 kHz lowpass for envelope
        self.filter_order = 4
        
        # ToF detection parameters
        self.crosstalk_skip = 2500  # Skip first 2500 samples (crosstalk zone)
        self.min_detection_distance = 200  # Minimum 200 samples AFTER crosstalk skip
        self.noise_estimation_start = 2500  # Start noise estimation AFTER crosstalk
        self.noise_estimation_length = 1000  # Use 1000 samples (2500-3500) for noise estimation
        self.snr_threshold = 12  # Signal-to-noise ratio threshold (dB)
        self.min_absolute_amplitude = 50  # Minimum absolute signal amplitude
        # Tilt correction (degrees) for final distance adjustment
        self.tilt_angle_deg = 8.05
        # Cache last valid distance to handle empty/invalid frames gracefully
        self.last_valid_distance = None
        # Debug toggle: log which ToF detection method succeeds
        self.debug_tof = True
    
    def load_adc_data(self, file_path):
        """Load raw ADC data from file. Returns None for empty/invalid files."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Skip empty files
        if os.path.getsize(file_path) == 0:
            return None

        # Read line by line (consistent with existing code)
        data = []
        with open(file_path, 'r', errors='replace') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(float(line))
                    except ValueError:
                        pass
        if len(data) == 0:
            return None
        return np.array(data)
    
    def bandpass_filter(self, signal_data, lowcut=None, highcut=None):
        """
        Apply bandpass filter to reduce out-of-band noise.
        
        Args:
            signal_data: Raw ADC signal
            lowcut: Low cutoff frequency (uses default if None)
            highcut: High cutoff frequency (uses default if None)
            
        Returns:
            Bandpass filtered signal
        """
        low = lowcut if lowcut is not None else self.bandpass_low
        high = highcut if highcut is not None else self.bandpass_high
        
        # Remove DC offset
        dc_offset = np.median(signal_data)
        signal_dc_removed = signal_data - dc_offset
        
        # Design Butterworth bandpass filter
        low_norm = low / self.nyquist
        high_norm = high / self.nyquist
        
        # Clamp to valid range (0, 1) for Nyquist normalization
        low_norm = np.clip(low_norm, 0.001, 0.999)
        high_norm = np.clip(high_norm, low_norm + 0.001, 0.999)
        
        b, a = signal.butter(self.filter_order, [low_norm, high_norm], btype='band')
        
        # Apply zero-phase filtering
        filtered = signal.filtfilt(b, a, signal_dc_removed)
        
        return filtered
    
    def envelope_detection(self, signal_data):
        """
        Envelope detection using Hilbert transform and lowpass filter.
        
        Args:
            signal_data: Filtered signal
            
        Returns:
            Envelope signal
        """
        # Rectify signal
        rectified = np.abs(signal_data)
        
        # Design lowpass filter for envelope extraction
        cutoff_norm = self.envelope_cutoff / self.nyquist
        cutoff_norm = np.clip(cutoff_norm, 0.001, 0.999)
        
        b, a = signal.butter(self.filter_order, cutoff_norm, btype='low')
        
        # Apply lowpass filter to rectified signal
        envelope = signal.filtfilt(b, a, rectified)
        
        return envelope
    
    def detect_tof(self, envelope, tx_signal=None):
        """
        Detect Time-of-Flight from envelope signal using adaptive threshold method.
        Optionally leverages TX post-burst noise for a stable baseline.
        
        Args:
            envelope: Envelope signal
            
        Returns:
            ToF sample index, or "Not Detected" if not found
        """
        tx_start_std = None
        tx_end_std = None
        use_basic_first = False

        if tx_signal is not None and len(tx_signal) >= 1000:
            tx_start_std = np.std(tx_signal[:1000])
            tx_end_std = np.std(tx_signal[-1000:])
            if tx_start_std < 10 * tx_end_std:
                use_basic_first = True

        if use_basic_first:
            self.crosstalk_skip = 3000
        else:
            self.crosstalk_skip = 2500

        if len(envelope) <= self.crosstalk_skip + self.min_detection_distance:
            return "Not Detected"

        # Search region: after crosstalk zone
        search_signal = envelope[self.crosstalk_skip:]

        def basic_criteria_detection(signal_data):
            rise_len = 100
            rise_delta_min = 10
            rise_end_min = 50
            slope_threshold = np.tan(np.deg2rad(22.5))

            if len(signal_data) > rise_len + 1:
                for start_idx in range(0, len(signal_data) - rise_len - 1):
                    end_idx = start_idx + rise_len
                    rising = True
                    for j in range(start_idx, end_idx):
                        if signal_data[j + 1] <= signal_data[j]:
                            rising = False
                            break
                    if not rising:
                        continue

                    if (signal_data[end_idx] - signal_data[start_idx]) < rise_delta_min:
                        continue
                    if signal_data[end_idx] < rise_end_min:
                        continue

                    onset_idx = None
                    for j in range(start_idx, end_idx):
                        if (signal_data[j + 1] - signal_data[j]) > slope_threshold:
                            onset_idx = j
                            break

                    if onset_idx is not None:
                        return onset_idx + self.crosstalk_skip

            return None

        if use_basic_first:
            tof_idx_basic = basic_criteria_detection(search_signal)
            if tof_idx_basic is not None:
                return int(tof_idx_basic)

        # Prefer TX-derived noise (post-burst) for stability; fallback to RX 1500-2000 region
        if tx_signal is not None and len(tx_signal) >= 2000:
            tx_centered = tx_signal - np.median(tx_signal)
            tx_env = np.abs(signal.hilbert(tx_centered))
            noise_region = tx_env[1500:2000]
        else:
            if len(envelope) < 2000:
                return "Not Detected"
            tx_env = None
            noise_region = envelope[1500:2000]

        noise_std = np.std(noise_region)
        noise_median = np.median(noise_region)
        noise_max = np.max(noise_region)
        
        # Calculate adaptive threshold using multiple strategies
        threshold_median = noise_median + 5 * noise_std
        threshold_percentile = np.percentile(noise_region, 90) + 4 * noise_std
        threshold_max = noise_max + 3 * noise_std
        
        # Use the median of thresholds
        adaptive_threshold = np.median([threshold_median, threshold_percentile, threshold_max])
        # Safety floor to avoid triggering on too-quiet TX
        MIN_THRESHOLD = 30
        adaptive_threshold = max(adaptive_threshold, MIN_THRESHOLD)
        
        # Find where signal crosses threshold
        threshold_crossings = np.where(search_signal > adaptive_threshold)[0]
        
        if len(threshold_crossings) > 0:
            first_crossing = threshold_crossings[0]
            
            # Reject if too close to crosstalk boundary
            if first_crossing < self.min_detection_distance:
                valid_crossings = threshold_crossings[threshold_crossings >= self.min_detection_distance]
                if len(valid_crossings) == 0:
                    first_crossing = None
                else:
                    first_crossing = valid_crossings[0]
            
            if first_crossing is not None:
                # Validate: signal must be SUSTAINED above threshold
                sustained = True
                for i in range(first_crossing, min(first_crossing + 5, len(search_signal))):
                    if search_signal[i] <= adaptive_threshold * 0.8:
                        sustained = False
                        break
                
                if not sustained:
                    first_crossing = None
            
            if first_crossing is not None:
                # Walk backward conservatively (reduced backtrack) to find rise start
                start_idx = first_crossing
                lookback_limit = max(self.min_detection_distance, first_crossing - 100)
                
                # Walk backward: stop when we find a point lower than the next few points
                # (i.e., a local minimum where the rise begins)
                for i in range(first_crossing - 1, lookback_limit, -1):
                    # Check if current point is lower than the next 3-5 points
                    # This means we found the bottom of the valley (start of rise)
                    if i + 5 < len(search_signal):
                        next_points = search_signal[i+1:i+6]
                        if search_signal[i] < np.min(next_points):
                            start_idx = i
                            break
                
                # Final validation: check absolute amplitude
                tof_idx = start_idx + self.crosstalk_skip
                if envelope[tof_idx] > self.min_absolute_amplitude:
                    return int(tof_idx)

        if not use_basic_first:
            # Method 1.5: Adaptive Threshold using RX noise region (2500-3000)
            if len(envelope) < 3000:
                return "Not Detected"

            noise_region_rx = envelope[2500:3000]
            noise_std_rx = np.std(noise_region_rx)
            noise_median_rx = np.median(noise_region_rx)
            noise_max_rx = np.max(noise_region_rx)

            threshold_median_rx = noise_median_rx + 5 * noise_std_rx
            threshold_percentile_rx = np.percentile(noise_region_rx, 90) + 4 * noise_std_rx
            threshold_max_rx = noise_max_rx + 3 * noise_std_rx

            adaptive_threshold_rx = np.median(
                [threshold_median_rx, threshold_percentile_rx, threshold_max_rx]
            )
            adaptive_threshold_rx = max(adaptive_threshold_rx, MIN_THRESHOLD)

            threshold_crossings_rx = np.where(search_signal > adaptive_threshold_rx)[0]

            if len(threshold_crossings_rx) > 0:
                first_crossing_rx = threshold_crossings_rx[0]

                if first_crossing_rx < self.min_detection_distance:
                    valid_crossings_rx = threshold_crossings_rx[
                        threshold_crossings_rx >= self.min_detection_distance
                    ]
                    if len(valid_crossings_rx) == 0:
                        first_crossing_rx = None
                    else:
                        first_crossing_rx = valid_crossings_rx[0]

                if first_crossing_rx is not None:
                    sustained = True
                    for i in range(
                        first_crossing_rx,
                        min(first_crossing_rx + 5, len(search_signal))
                    ):
                        if search_signal[i] <= adaptive_threshold_rx * 0.8:
                            sustained = False
                            break

                    if sustained:
                        tof_idx_rx = first_crossing_rx + self.crosstalk_skip
                        if envelope[tof_idx_rx] > self.min_absolute_amplitude:
                            return int(tof_idx_rx)

            tof_idx_basic = basic_criteria_detection(search_signal)
            if tof_idx_basic is not None:
                return int(tof_idx_basic)
        
        # Method 2: Energy-Based Detection (Fallback)
        window_size = 100  # 100 μs window
        hop_size = 20  # 20 μs hop
        energy_snr_threshold = 6
        energy_onset_k = 1.8
        energy_sustain_len = 5
        energy_lookback_factor = 4

        # Use TX-only noise region for this fallback
        energy_noise_region = None
        if tx_env is not None and len(tx_env) >= 3000:
            energy_noise_region = tx_env[2500:3000]
        
        max_energy = 0
        max_energy_idx = None
        
        for i in range(self.crosstalk_skip, len(envelope) - window_size, hop_size):
            window = envelope[i:i+window_size]
            energy = np.sum(window ** 2)
            
            if energy > max_energy:
                max_energy = energy
                max_energy_idx = i
        
        if energy_noise_region is None or len(energy_noise_region) == 0:
            return "Not Detected"

        noise_energy = np.sum(energy_noise_region ** 2) / len(energy_noise_region) * window_size
        energy_ratio_db = 10 * np.log10(max_energy / (noise_energy + 1e-10))
        
        if energy_ratio_db > energy_snr_threshold and max_energy_idx is not None:
            if max_energy_idx - self.crosstalk_skip >= self.min_detection_distance:
                if envelope[max_energy_idx] > self.min_absolute_amplitude:
                    # Refine to onset: find first sustained rise above a local threshold
                    energy_search_start = max(
                        self.crosstalk_skip,
                        max_energy_idx - (window_size * energy_lookback_factor)
                    )
                    energy_segment = envelope[energy_search_start:max_energy_idx + 1]
                    if len(energy_segment) > 0:
                        seg_median = np.median(energy_segment)
                        seg_std = np.std(energy_segment)
                        local_threshold = seg_median + energy_onset_k * seg_std

                        onset_idx = None
                        for i in range(len(energy_segment)):
                            if energy_segment[i] <= local_threshold:
                                continue
                            sustained = True
                            end_i = min(i + energy_sustain_len, len(energy_segment))
                            for j in range(i, end_i):
                                if energy_segment[j] <= local_threshold * 0.8:
                                    sustained = False
                                    break
                            if sustained:
                                onset_idx = energy_search_start + i
                                break

                        if onset_idx is not None and envelope[onset_idx] > self.min_absolute_amplitude:
                            if self.debug_tof:
                                print("[ToF] Method=EnergyFallback", flush=True)
                            return int(onset_idx)
                    if self.debug_tof:
                        print("[ToF] Method=EnergyFallback", flush=True)
                    return int(max_energy_idx)
        
        # Method 3: First Significant Rise Detection (Last Resort)
        derivative = np.diff(search_signal)
        significant_rise = derivative > (noise_std * 2.5)
        
        if np.any(significant_rise):
            rise_indices = np.where(significant_rise)[0]
            valid_rises = rise_indices[rise_indices >= self.min_detection_distance]
            
            if len(valid_rises) > 0:
                tof_idx = valid_rises[0] + self.crosstalk_skip
                if envelope[tof_idx] > adaptive_threshold * 0.7 and envelope[tof_idx] > self.min_absolute_amplitude:
                    return int(tof_idx)
        
        return "Not Detected"
    
    def detect_tx_start(self, tx_signal):
        """
        Detect TX burst start time (similar to RX ToF detection).
        
        Args:
            tx_signal: Raw TX signal
            
        Returns:
            TX start time in samples, or "Not Detected" if no burst detected
        """
        # Remove DC offset
        dc_offset = np.median(tx_signal)
        tx_centered = tx_signal - dc_offset
        
        # Extract envelope using Hilbert transform
        analytic_signal = signal.hilbert(tx_centered)
        tx_env = np.abs(analytic_signal)
        
        if len(tx_env) < 100:
            return "Not Detected"
        
        # Estimate noise floor from first 50 samples (before burst)
        noise_region = tx_env[:50]
        noise_std = np.std(noise_region)
        noise_max = np.max(noise_region)

        # Stricter threshold to avoid noise edges
        threshold = noise_max + 6 * noise_std
        
        # Find first threshold crossing
        threshold_crossings = np.where(tx_env > threshold)[0]
        
        if len(threshold_crossings) == 0:
            return "Not Detected"
        
        first_crossing = threshold_crossings[0]
        
        # Minimal backtracking: TX rise is sharp; rewind only a few samples if clearly below threshold
        start_idx = first_crossing
        lookback_limit = max(0, first_crossing - 5)

        for i in range(first_crossing - 1, lookback_limit, -1):
            if tx_env[i] < threshold * 0.5:
                start_idx = i + 1
                break
        
        return float(start_idx)
    
    def calculate_speed_of_sound(self, temp_C, humi_percent):
        """
        Calculate speed of sound based on temperature and humidity.
        
        Args:
            temp_C: Temperature in Celsius
            humi_percent: Humidity percentage
            
        Returns:
            Speed of sound in m/s
        """
        if temp_C is None or humi_percent is None:
            return None
        
        temp_factor = np.sqrt(1 + temp_C / 273.15)
        humidity_factor = 1 + 0.0036 * (humi_percent / 100)
        c = 331.3 * temp_factor * humidity_factor
        return c
    
    def predict_distance(self, tof_us, tx_start_us, temp_C, humi_percent):
        """
        Predict distance using the LSE model.
        
        Args:
            tof_us: Time-of-flight in microseconds (or "Not Detected")
            tx_start_us: TX start time in microseconds (or "Not Detected")
            temp_C: Temperature in Celsius
            humi_percent: Humidity percentage
            
        Returns:
            Distance in mm, or "Not Detected" if calculation not possible
        """
        if tof_us == "Not Detected" or tx_start_us == "Not Detected":
            if self.last_valid_distance is not None:
                return self.last_valid_distance
            return "Not Detected"

        if temp_C is None or humi_percent is None:
            if self.last_valid_distance is not None:
                return self.last_valid_distance
            return "Not Detected"
        
        # Calculate speed of sound
        c = self.calculate_speed_of_sound(temp_C, humi_percent)
        if c is None:
            if self.last_valid_distance is not None:
                return self.last_valid_distance
            return "Not Detected"
        
        # Calculate delta_t in microseconds
        delta_t = tof_us - tx_start_us
        
        # Predict distance in mm
        # distance = β₀ + β₁ × c × Δt / 1000
        # where c is in m/s and Δt is in μs
        distance_mm = self.BETA_0 + self.BETA_1 * c * delta_t / 1000

        # Apply tilt correction to final distance
        distance_mm = distance_mm * math.cos(math.radians(self.tilt_angle_deg))

        # Cache valid distance for future fallbacks
        self.last_valid_distance = distance_mm

        return distance_mm


# --- Distance Smoothing Filters ---
def hampel_filter(data, window=5, n_sigma=3.0):
    """
    Robust outlier detection and replacement using Hampel filter.
    
    Uses median and MAD (Median Absolute Deviation) for outlier detection.
    Outliers are replaced with local median values.
    
    Args:
        data: 1D array of measurements
        window: Window size for local statistics (must be odd)
        n_sigma: Threshold multiplier for outlier detection (typical: 3.0)
    
    Returns:
        Filtered data with outliers replaced by local median
    """
    if len(data) < window:
        return data.copy()
    
    filtered = data.copy()
    k = 1.4826  # Scale factor to make MAD consistent with standard deviation
    
    half_window = window // 2
    
    for i in range(len(data)):
        # Extract local window
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        window_data = data[start:end]
        
        # Compute robust statistics
        median_val = np.median(window_data)
        mad = np.median(np.abs(window_data - median_val))
        
        # Outlier detection threshold
        threshold = n_sigma * k * mad
        
        # Replace outlier with median
        if np.abs(data[i] - median_val) > threshold:
            filtered[i] = median_val
    
    return filtered


class DistanceKalmanFilter:
    """
    Kalman filter for distance tracking with velocity estimation.
    
    State vector: [distance, velocity]
    Measurement: distance only
    
    Model:
        x_k = A * x_{k-1} + w_k    (process model: constant velocity)
        z_k = H * x_k + v_k         (measurement model: observe distance)
    
    where:
        A = [[1, dt], [0, 1]]  (state transition matrix)
        H = [1, 0]              (measurement matrix)
        w_k ~ N(0, Q)           (process noise)
        v_k ~ N(0, R)           (measurement noise)
    """
    
    def __init__(self, process_noise_distance=0.5, process_noise_velocity=0.1, measurement_noise=15.0):
        """
        Initialize Kalman filter.
        
        Args:
            process_noise_distance: Process noise for distance (mm²)
            process_noise_velocity: Process noise for velocity (mm²/s²)
            measurement_noise: Measurement noise variance (mm²)
        """
        # State: [distance (mm), velocity (mm/s)]
        self.x = np.array([0.0, 0.0])
        
        # Initial state covariance (high uncertainty)
        self.P = np.eye(2) * 1000
        
        # Process noise covariance matrix
        self.Q = np.array([[process_noise_distance, 0],
                           [0, process_noise_velocity]])
        
        # Measurement noise variance
        self.R = measurement_noise
        
        # Measurement matrix (observe distance only)
        self.H = np.array([1.0, 0.0])
    
    def predict(self, dt):
        """
        Prediction step: propagate state forward in time.
        
        Args:
            dt: Time step (seconds)
        """
        # State transition matrix (constant velocity model)
        A = np.array([[1.0, dt],
                      [0.0, 1.0]])
        
        # Predict state
        self.x = A @ self.x
        
        # Predict covariance
        self.P = A @ self.P @ A.T + self.Q
    
    def update(self, measurement):
        """
        Update step: correct prediction using measurement.
        
        Args:
            measurement: Observed distance (mm)
        
        Returns:
            Updated distance estimate (mm)
        """
        # Innovation (measurement residual)
        y = measurement - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T / S
        
        # Update state
        self.x = self.x + K * y
        
        # Update covariance
        self.P = (np.eye(2) - np.outer(K, self.H)) @ self.P
        
        return self.x[0]  # Return distance estimate
    
    def get_velocity(self):
        """Get current velocity estimate (mm/s)."""
        return self.x[1]
    
    def reset(self):
        
        """Reset filter state (for frame jumps)."""
        self.x = np.array([0.0, 0.0])
        self.P = np.eye(2) * 1000


class ViewModel:
    def __init__(self): pass


# --- Background Worker (Trend Analysis) ---
class TrendScanner(QThread):
    progress = pyqtSignal(int, int)
    result_ready = pyqtSignal(object)

    def __init__(self, sensor_paths, total_frames, pfft_range_ms, bpf_config=None):
        super().__init__()
        self.sensor_paths = sensor_paths
        self.total_frames = total_frames
        self.pfft_range_ms = pfft_range_ms
        self.bpf_config = bpf_config  # {'apply': bool, 'low': float, 'high': float}
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        results = {}
        s_keys = list(self.sensor_paths.keys())
        print(f"[Trend] Thread Started. Total Frames: {self.total_frames}", flush=True)
        if self.bpf_config and self.bpf_config['apply']:
            print(f"[Trend] BPF Enabled: {self.bpf_config['low']}Hz ~ {self.bpf_config['high']}Hz", flush=True)

        # 결과 컨테이너 초기화
        for sid in self.sensor_paths.keys():
            results[sid] = {
                'speed': np.full(self.total_frames, np.nan),
                'fft_peak': np.full(self.total_frames, np.nan),
                'pfft_peak': np.full(self.total_frames, np.nan),
                'pfft_mag': np.full(self.total_frames, np.nan)
            }

        start_ms, end_ms = self.pfft_range_ms
        p_start_target = int(start_ms * FS / 1000.0)
        p_end_target = int(end_ms * FS / 1000.0)

        valid_frames_count = 0
        first_success_logged = False

        for idx in range(self.total_frames):
            if not self.is_running: break

            frame_has_valid_data = False

            for sid, paths in self.sensor_paths.items():
                if idx >= len(paths): continue
                folder_path = paths[idx]

                try:
                    # 1. Speed (info.json)
                    json_path = os.path.join(folder_path, "info.json")
                    if os.path.exists(json_path):
                        with open(json_path, 'r', encoding='utf-8') as f:
                            info = json.load(f)
                            gps_spd = info.get('gps', {}).get('speed', 0.0)
                            if gps_spd is not None:
                                results[sid]['speed'][idx] = gps_spd

                    # 2. Signal Analysis (rx_plain.dat)
                    rx_path = os.path.join(folder_path, "rx_plain.dat")
                    if os.path.exists(rx_path):
                        data = []
                        # errors='replace'로 인코딩 에러 무시하고 읽기 시도
                        with open(rx_path, 'r', errors='replace') as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    try:
                                        data.append(float(line))
                                    except ValueError:
                                        pass
                        data = np.array(data)

                        if len(data) > 0:
                            frame_has_valid_data = True
                            vol = data - np.mean(data)

                            # [BPF 적용]
                            if self.bpf_config and self.bpf_config['apply']:
                                try:
                                    vol = butter_bandpass_filter(vol, self.bpf_config['low'], self.bpf_config['high'],
                                                                 FS)
                                except Exception as e:
                                    # 필터 에러나면 원본 사용 (너무 짧은 데이터 등)
                                    pass

                            # --- [Full FFT] ---
                            n = len(vol)
                            n_padded = max(n * 4, 1024)
                            yf = np.fft.fft(vol, n=n_padded)
                            xf = np.fft.fftfreq(n_padded, 1 / FS)

                            # 양수 주파수만 (DC 제외)
                            half = n_padded // 2
                            mag = np.abs(yf[:half])
                            freqs = xf[:half]

                            if len(mag) > 1:
                                target_mag = mag[1:]
                                target_freqs = freqs[1:]
                                if len(target_mag) > 0:
                                    peak_i = np.argmax(target_mag)
                                    results[sid]['fft_peak'][idx] = target_freqs[peak_i]

                            # --- [Partial FFT] ---
                            p_start = min(p_start_target, len(vol) - 1)
                            p_end = min(p_end_target, len(vol))

                            if p_end > p_start + 10:
                                segment = vol[p_start:p_end]
                                n_seg = len(segment)
                                n_seg_padded = max(n_seg * 4, 512)
                                syf = np.fft.fft(segment, n=n_seg_padded)
                                sxf = np.fft.fftfreq(n_seg_padded, 1 / FS)

                                shalf = n_seg_padded // 2
                                smag = np.abs(syf[:shalf])
                                sfreqs = sxf[:shalf]

                                if len(smag) > 1:
                                    starget_mag = smag[1:]
                                    starget_freqs = sfreqs[1:]
                                    if len(starget_mag) > 0:
                                        speak_i = np.argmax(starget_mag)
                                        results[sid]['pfft_peak'][idx] = starget_freqs[speak_i]
                                        results[sid]['pfft_mag'][idx] = np.max(starget_mag)
                    else:
                        pass

                except Exception as e:
                    pass

            if frame_has_valid_data:
                valid_frames_count += 1

            # Progress Update
            if idx % 10 == 0 or idx == self.total_frames - 1:
                self.progress.emit(idx + 1, self.total_frames)

        print(
            f"[Trend] Scan Finished. Processed {self.total_frames} frames. Valid Data Found in {valid_frames_count} frames.",
            flush=True)
        self.result_ready.emit(results)


# --- Main Controller ---
class PlayerController(QObject):
    def __init__(self):
        super().__init__()
        self.view_model = ViewModel()
        self.window = MainWindow(self.view_model)

        self.sensor_dirs = {}
        self.timeline_folders = {}
        self.total_frames = 0
        self.current_index = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.scanner = None
        self.distance_calculator = DistanceCalculator(FS)
        
        # Smoothing state management
        self.distance_history = {}  # {sensor_id: deque(maxlen=20)} storing raw distances
        self.last_frame_index = -1  # Track previous frame index to detect jumps vs sequential navigation
        
        # Initialize smoothing state for all sensors
        for sid in range(6):
            self.distance_history[sid] = deque(maxlen=20)

        # High DPI settings check
        try:
            self.window.setAttribute(Qt.WA_AlwaysShowToolTips)
        except:
            pass

        self._connect_signals()
        self.window.show()

    def _connect_signals(self):
        for i, btn in enumerate(self.window.sensor_select_buttons):
            btn.clicked.connect(lambda checked, idx=i: self.browse_sensor_folder(idx))

        self.window.slider_seek.sliderMoved.connect(self.seek_frame)
        self.window.slider_seek.sliderPressed.connect(self.pause_playback)
        self.window.btn_prev.clicked.connect(self.prev_frame)
        self.window.btn_next.clicked.connect(self.next_frame)
        self.window.btn_play.clicked.connect(self.toggle_play)
        self.window.combo_speed.currentIndexChanged.connect(self.change_speed)
        self.window.right_main_tabs.currentChanged.connect(lambda idx: self.load_frame(self.current_index))
        
        # Left tabs (Control/Metadata) - trigger distance calculation when metadata tab is selected
        if hasattr(self.window, 'left_tabs'):
            self.window.left_tabs.currentChanged.connect(lambda idx: self.load_frame(self.current_index))

        # Trend View Signals
        if hasattr(self.window, 'trend_view'):
            self.window.trend_view.plot_clicked.connect(self.on_trend_clicked)
            self.window.trend_view.btn_go_detail.clicked.connect(self.switch_to_detail)

        self.window.btn_analyze.clicked.connect(self.start_trend_analysis)

        for i, btn in enumerate(self.window.meta_sensor_buttons):
            btn.clicked.connect(lambda checked, idx=i: self.switch_meta_view(idx))

        # STFT & BPF Parameter Changes -> Refresh current frame
        if hasattr(self.window, 'combo_stft_window'):
            self.window.combo_stft_window.currentTextChanged.connect(lambda: self.load_frame(self.current_index))
        if hasattr(self.window, 'spin_stft_overlap'):
            self.window.spin_stft_overlap.valueChanged.connect(lambda: self.load_frame(self.current_index))
        if hasattr(self.window, 'combo_stft_padding'):
            self.window.combo_stft_padding.currentTextChanged.connect(lambda: self.load_frame(self.current_index))

        # BPF Refresh
        self.window.chk_bpf_apply.toggled.connect(lambda: self.load_frame(self.current_index))
        self.window.spin_bpf_low.valueChanged.connect(lambda: self.load_frame(self.current_index))
        self.window.spin_bpf_high.valueChanged.connect(lambda: self.load_frame(self.current_index))
        
        # Reset Button
        self.window.btn_reset.clicked.connect(self.reset_all)

    def _calculate_distance(self, sensor_id, folder_path, temp_C, humi_percent):
        """
        Calculate distance for a sensor frame with smoothing.
        
        Args:
            sensor_id: Sensor ID (0-5)
            folder_path: Path to frame folder
            temp_C: Temperature in Celsius
            humi_percent: Humidity percentage
        Returns:
            Dictionary with keys: 'distance_mm' (smoothed), 'distance_raw', 'distance_filtered', 
            'distance_smooth', 'tof_us', 'tx_centroid_us', 'processing_time_ms', 'status'
        """
        start_time = time.time()
        result = {
            'distance_mm': "Not Detected",  # Final smoothed distance (for display)
            'distance_raw': "Not Detected",
            'distance_filtered': "Not Detected",
            'distance_smooth': "Not Detected",
            'tof_us': "Not Detected",
            'tx_start_us': "Not Detected",
            'processing_time_ms': 0.0,
            'status': 'Error'
        }
        
        # Convert temp_C and humi_percent to float/None (handle strings, None, '-', etc.)
        temp_C_float = None
        humi_percent_float = None
        try:
            if temp_C is not None and temp_C != '-' and temp_C != '':
                temp_C_float = float(temp_C)
        except (ValueError, TypeError):
            temp_C_float = None
        
        try:
            if humi_percent is not None and humi_percent != '-' and humi_percent != '':
                humi_percent_float = float(humi_percent)
        except (ValueError, TypeError):
            humi_percent_float = None
        
        try:
            rx_path = os.path.join(folder_path, "rx_plain.dat")
            tx_path = os.path.join(folder_path, "tx_plain.dat")

            # Get BPF settings from UI if available (use UI settings if BPF is enabled)
            bpf_low = None
            bpf_high = None
            if hasattr(self.window, 'chk_bpf_apply') and self.window.chk_bpf_apply.isChecked():
                bpf_low = self.window.spin_bpf_low.value()
                bpf_high = self.window.spin_bpf_high.value()

            # Load TX once for both baseline and start detection
            tx_signal = None
            if os.path.exists(tx_path):
                try:
                    tx_signal = self.distance_calculator.load_adc_data(tx_path)
                except Exception as e:
                    print(f"[Error loading TX] {e}")

            # Process RX for ToF detection
            if os.path.exists(rx_path):
                try:
                    rx_signal = self.distance_calculator.load_adc_data(rx_path)
                    if rx_signal is not None and len(rx_signal) > 0:
                        filtered = self.distance_calculator.bandpass_filter(rx_signal, bpf_low, bpf_high)
                        envelope = self.distance_calculator.envelope_detection(filtered)
                        tof_sample = self.distance_calculator.detect_tof(envelope, tx_signal)
                        # Convert samples to microseconds (FS = 1MHz, so 1 sample = 1 μs)
                        if tof_sample != "Not Detected":
                            result['tof_us'] = float(tof_sample)  # Already in μs since FS = 1MHz
                        else:
                            result['tof_us'] = "Not Detected"
                    else:
                        result['tof_us'] = "Not Detected"
                except Exception as e:
                    print(f"[Error processing RX for ToF] {e}")
                    result['tof_us'] = "Not Detected"
            
            # Process TX for start time detection (reuse tx_signal if loaded)
            if tx_signal is not None and len(tx_signal) > 0:
                try:
                    tx_start = self.distance_calculator.detect_tx_start(tx_signal)
                    if tx_start != "Not Detected":
                        result['tx_start_us'] = float(tx_start)  # Already in μs since FS = 1MHz
                    else:
                        result['tx_start_us'] = "Not Detected"
                except Exception as e:
                    print(f"[Error processing TX for start time] {e}")
                    result['tx_start_us'] = "Not Detected"
            else:
                result['tx_start_us'] = "Not Detected"
            
            # Predict raw distance
            if result['tof_us'] != "Not Detected" and result['tx_start_us'] != "Not Detected":
                distance_raw = self.distance_calculator.predict_distance(
                    result['tof_us'], 
                    result['tx_start_us'], 
                    temp_C_float, 
                    humi_percent_float
                )
                if distance_raw != "Not Detected":
                    distance_raw = float(distance_raw)
                    result['distance_raw'] = distance_raw
                    result['status'] = 'OK'
                    
                    # ========== SMOOTHING PIPELINE ==========
                    # Stage 1: Store in history
                    self.distance_history[sensor_id].append(distance_raw)
                    
                    # Stage 2: Hampel Filter (outlier removal)
                    distance_filtered = distance_raw
                    history_array = np.array(list(self.distance_history[sensor_id]))
                    if len(history_array) >= 5:  # Window size = 5
                        filtered_array = hampel_filter(history_array, window=5, n_sigma=3.0)
                        distance_filtered = float(filtered_array[-1])  # Use last (current) value
                    result['distance_filtered'] = distance_filtered
                    
                    # Stage 3: No Kalman smoothing (use Hampel output only)
                    result['distance_smooth'] = distance_filtered
                    
                    # Final distance for display (smoothed)
                    result['distance_mm'] = result['distance_smooth']
                else:
                    result['status'] = 'Error'
            else:
                result['status'] = 'Error'
                
        except Exception as e:
            print(f"[Error in distance calculation] {e}")
            traceback.print_exc()
            result['status'] = 'Error'
        
        finally:
            result['processing_time_ms'] = (time.time() - start_time) * 1000.0
        
        return result
    
    def switch_meta_view(self, idx):
        for i, btn in enumerate(self.window.meta_sensor_buttons):
            btn.setChecked(i == idx)
        self.window.stacked_meta_info.setCurrentIndex(idx)
        # Calculate distance for the newly selected sensor
        if 0 <= idx < 6 and idx in self.timeline_folders and self.current_index < len(self.timeline_folders[idx]):
            folder_path = self.timeline_folders[idx][self.current_index]
            json_path = os.path.join(folder_path, "info.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                    widget = self.window.stacked_meta_info.widget(idx)
                    if widget:
                        distance_result = self._calculate_distance(
                            idx, folder_path, info.get('temp'), info.get('humi')
                        )
                        if distance_result['status'] == 'OK':
                            widget.lbl_distance.setText(f"{distance_result['distance_mm']:.3f} mm")
                            widget.lbl_tof.setText(f"{distance_result['tof_us']:.1f} μs")
                            widget.lbl_processing_time.setText(f"{distance_result['processing_time_ms']:.2f} ms")
                        else:
                            widget.lbl_distance.setText("Not Detected")
                            widget.lbl_tof.setText("Not Detected" if distance_result['tof_us'] == "Not Detected" else f"{distance_result['tof_us']:.1f} μs")
                            widget.lbl_processing_time.setText(f"{distance_result['processing_time_ms']:.2f} ms")
                except Exception as e:
                    print(f"[Error loading distance on sensor switch] {e}")

    def browse_sensor_folder(self, sensor_id):
        path = QFileDialog.getExistingDirectory(self.window, f"Select Data Folder for Sensor {sensor_id + 1}")
        if not path: return
        self.sensor_dirs[sensor_id] = path
        self.window.set_sensor_ready(sensor_id, True)
        self.load_structure()

    def load_structure(self):
        self.timeline_folders = {}
        for sid, s_path in self.sensor_dirs.items():
            # 폴더만 가져오기
            subfolders = sorted(glob.glob(os.path.join(s_path, "*")))
            subfolders = [f for f in subfolders if os.path.isdir(f)]
            if subfolders: self.timeline_folders[sid] = subfolders

        if not self.timeline_folders: return

        max_len = 0
        for sid in self.timeline_folders:
            max_len = max(max_len, len(self.timeline_folders[sid]))
        self.total_frames = max_len
        self.window.slider_seek.setRange(0, self.total_frames - 1)
        self.window.lbl_index.setText(f"{self.current_index} / {self.total_frames}")

        if self.current_index >= self.total_frames: self.current_index = 0
        self.load_frame(self.current_index)

    def start_trend_analysis(self):
        if self.total_frames == 0:
            QMessageBox.warning(self.window, "Warning", "No data loaded!")
            return

        selected_opts = []
        if self.window.chk_trend_speed.isChecked(): selected_opts.append('speed')
        if self.window.chk_trend_fft.isChecked(): selected_opts.append('fft_peak')
        if self.window.chk_trend_pfft_peak.isChecked(): selected_opts.append('pfft_peak')
        if self.window.chk_trend_pfft_mag.isChecked(): selected_opts.append('pfft_mag')

        if not selected_opts:
            QMessageBox.warning(self.window, "Check", "Please select at least one trend option.")
            return

        # BPF Config 가져오기
        bpf_config = {
            'apply': self.window.chk_bpf_apply.isChecked(),
            'low': self.window.spin_bpf_low.value(),
            'high': self.window.spin_bpf_high.value()
        }

        print(f"[Main] Trend Analysis Requested. BPF: {bpf_config}", flush=True)

        self.window.trend_view.setup_plots(selected_opts)
        # X축 범위 설정
        for opt, plot_data in self.window.trend_view.active_plots.items():
            plot_data['widget'].setXRange(0, self.total_frames)

        p_start_ms = self.window.pfft_start_spin.value()
        p_end_ms = self.window.pfft_end_spin.value()

        if self.scanner and self.scanner.isRunning():
            self.scanner.stop()
            self.scanner.wait()

        self.scanner = TrendScanner(self.timeline_folders, self.total_frames, (p_start_ms, p_end_ms), bpf_config)
        self.scanner.progress.connect(self.update_scan_progress)
        self.scanner.result_ready.connect(self.finalize_trend)
        self.scanner.start()

    def update_scan_progress(self, current, total):
        self.window.setWindowTitle(f"Scanning Trend... {int(current / total * 100)}% ({current}/{total})")

    def finalize_trend(self, results):
        self.window.setWindowTitle("Multi-Sensor Data Player - Ready")
        print("[Main] Analysis Complete. Updating Graphs...", flush=True)
        self.window.trend_view.update_data(results)
        self.window.right_main_tabs.setCurrentIndex(1)  # 트렌드 탭으로 이동

    def on_trend_clicked(self, idx):
        if 0 <= idx < self.total_frames:
            self.seek_frame(int(idx))

    def switch_to_detail(self):
        self.window.right_main_tabs.setCurrentIndex(0)

    def toggle_play(self):
        if self.window.btn_play.isChecked():
            self.is_playing = True
            self.window.btn_play.setText("⏸ Pause")
            self.timer.start(100)
        else:
            self.pause_playback()

    def pause_playback(self):
        self.is_playing = False
        self.window.btn_play.setChecked(False)
        self.window.btn_play.setText("▶ Play")
        self.timer.stop()

    def change_speed(self, idx):
        intervals = [1000, 500, 200, 100, 10]
        if idx < len(intervals): self.timer.setInterval(intervals[idx])

    def seek_frame(self, idx):
        self.current_index = idx
        self.load_frame(idx)

    def next_frame(self):
        if self.current_index < self.total_frames - 1:
            self.current_index += 1
            self.window.slider_seek.setValue(self.current_index)
            self.load_frame(self.current_index)
        else:
            self.pause_playback()

    def prev_frame(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.window.slider_seek.setValue(self.current_index)
            self.load_frame(self.current_index)

    def load_frame(self, idx):
        self.window.lbl_index.setText(f"{idx + 1} / {self.total_frames}")
        if hasattr(self.window.trend_view, 'update_indicator'):
            self.window.trend_view.update_indicator(idx)

        # Detect sequential navigation vs jump
        is_sequential = False
        if self.last_frame_index >= 0:
            # Sequential if next or previous frame
            if idx == self.last_frame_index + 1 or idx == self.last_frame_index - 1:
                is_sequential = True
        
        # Reset smoothing state if jump detected
        if not is_sequential and self.last_frame_index >= 0:
            for sid in range(6):
                self.distance_history[sid].clear()

        first_valid_info = True
        for sid in range(6):
            if sid not in self.timeline_folders or idx >= len(self.timeline_folders[sid]): continue
            folder_path = self.timeline_folders[sid][idx]

            json_path = os.path.join(folder_path, "info.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                    widget = self.window.stacked_meta_info.widget(sid)
                    if widget:
                        widget.lbl_obj_temp.setText(f"{info.get('objectTemp', '-')}")
                        widget.lbl_amb_temp.setText(f"{info.get('temp', '-')} °C")
                        widget.lbl_humi.setText(f"{info.get('humi', '-')} %")
                        widget.lbl_pres.setText(f"{info.get('pressure', '-')} hPa")
                        
                        # Calculate distance if metadata tab is visible
                        if self.window.left_tabs.currentIndex() == 1:  # Metadata tab index
                            current_sensor_idx = self.window.stacked_meta_info.currentIndex()
                            if sid == current_sensor_idx:
                                distance_result = self._calculate_distance(
                                    sid, folder_path, info.get('temp'), info.get('humi')
                                )
                                if distance_result['status'] == 'OK':
                                    widget.lbl_distance.setText(f"{distance_result['distance_mm']:.3f} mm")
                                    widget.lbl_tof.setText(f"{distance_result['tof_us']:.1f} μs")
                                    widget.lbl_processing_time.setText(f"{distance_result['processing_time_ms']:.2f} ms")
                                else:
                                    widget.lbl_distance.setText("Not Detected")
                                    widget.lbl_tof.setText("Not Detected" if distance_result['tof_us'] == "Not Detected" else f"{distance_result['tof_us']:.1f} μs")
                                    widget.lbl_processing_time.setText(f"{distance_result['processing_time_ms']:.2f} ms")

                    if first_valid_info:
                        self.window.lbl_current_file.setText(os.path.basename(folder_path))
                        self.window.lbl_log_time.setText(info.get('loggedAt', '-'))
                        gps = info.get('gps', {})
                        self.window.lbl_gps_lat.setText(f"{gps.get('lat', '-')}")
                        self.window.lbl_gps_lon.setText(f"{gps.get('lng', '-')}")
                        self.window.lbl_gps_spd.setText(f"{gps.get('speed', '-')} km/h")
                        first_valid_info = False
                except:
                    pass

            # 현재 탭이 Detail View일 때만 신호 처리
            if self.window.right_main_tabs.currentIndex() == 0:
                self._process_and_draw_signal(sid, folder_path)
        
        # Update last frame index after processing
        self.last_frame_index = idx

    def reset_all(self):
        """Reset all loaded sensors and application state."""
        # Stop any ongoing playback or analysis
        if self.timer.isActive():
            self.timer.stop()
        
        if self.scanner is not None:
            self.scanner.stop()
            self.scanner.wait()
            self.scanner = None
        
        # Clear all sensor data
        self.sensor_dirs = {}
        self.timeline_folders = {}
        self.total_frames = 0
        self.current_index = 0
        
        # Reset distance tracking state
        for sid in range(6):
            self.distance_history[sid].clear()
        self.last_frame_index = -1
        
        # Update UI - sensor buttons
        for i in range(6):
            self.window.set_sensor_ready(i, False)
        
        # Reset playback controls
        self.window.slider_seek.setRange(0, 0)
        self.window.slider_seek.setValue(0)
        self.window.lbl_index.setText("0 / 0")
        self.window.lbl_current_file.setText("No File Loaded")
        self.window.btn_play.setChecked(False)
        self.window.btn_play.setText("▶ Play")

        # Clear all graphs
        for sensor_view in self.window.sensor_graphs:
            sensor_view.tx_plot['curve'].clear()
            sensor_view.rx_plot['curve'].clear()
            sensor_view.fft_plot['curve'].clear()
            sensor_view.p_fft_plot['curve'].clear()
            sensor_view.stft_widget['img'].clear()

        # Clear trend view
        if hasattr(self.window, 'trend_view'):
            for opt_data in self.window.trend_view.active_plots.values():
                for curve in opt_data['curves']:
                    curve.clear()

        # Reset metadata displays
        self.window.lbl_log_time.setText("-")
        self.window.lbl_gps_lat.setText("-")
        self.window.lbl_gps_lon.setText("-")
        self.window.lbl_gps_spd.setText("-")

        for info_view in [self.window.stacked_meta_info.widget(i) for i in range(6)]:
            info_view.lbl_obj_temp.setText("-")
            info_view.lbl_amb_temp.setText("-")
            info_view.lbl_humi.setText("-")
            info_view.lbl_pres.setText("-")
            info_view.lbl_distance.setText("-")
            info_view.lbl_tof.setText("-")
            info_view.lbl_processing_time.setText("-")

        print("[Reset] Application state reset complete.", flush=True)

    def _process_and_draw_signal(self, sid, folder_path):
        """
        [수정됨] UI에서 설정한 STFT 파라미터(Window Size, Overlap, Zero Padding) 적용
        [EXPERIMENTAL] Calculate ToF and TX centroid for visualization
        """
        rx_path = os.path.join(folder_path, "rx_plain.dat")
        tx_path = os.path.join(folder_path, "tx_plain.dat")
        rx_vol, tx_vol = None, None
        f_fft, m_fft = None, None
        t_stft, f_stft, z_stft = None, None, None
        f_pfft, m_pfft = None, None
        # [EXPERIMENTAL] ToF and TX start for markers
        tof_us_marker = None
        tx_start_us_marker = None

        try:
            # 1. RX Data Load
            if os.path.exists(rx_path):
                with open(rx_path, 'r') as f:
                    raw = []
                    for line in f:
                        try:
                            raw.append(float(line.strip()))
                        except:
                            pass
                    raw = np.array(raw)

                if len(raw) > 0:
                    rx_vol = raw - np.mean(raw)
                    n_points = len(rx_vol)

                    # [BPF 적용] 상세 뷰에서도 필터 적용
                    bpf_low_val = None
                    bpf_high_val = None
                    if self.window.chk_bpf_apply.isChecked():
                        low = self.window.spin_bpf_low.value()
                        high = self.window.spin_bpf_high.value()
                        bpf_low_val = low
                        bpf_high_val = high
                        try:
                            rx_vol = butter_bandpass_filter(rx_vol, low, high, FS)
                        except:
                            pass

                    # [EXPERIMENTAL] Calculate ToF for marker visualization (use TX if available for noise baseline)
                    try:
                        tx_signal_for_marker = None
                        if os.path.exists(tx_path):
                            tx_signal_for_marker = self.distance_calculator.load_adc_data(tx_path)
                        filtered_rx = self.distance_calculator.bandpass_filter(raw - np.mean(raw), bpf_low_val, bpf_high_val)
                        envelope_rx = self.distance_calculator.envelope_detection(filtered_rx)
                        tof_sample = self.distance_calculator.detect_tof(envelope_rx, tx_signal_for_marker)
                        if tof_sample != "Not Detected":
                            tof_us_marker = float(tof_sample)  # Already in μs since FS = 1MHz
                    except Exception as e:
                        pass

                    # --- 1. Full FFT ---
                    n_pad = max(n_points * 4, 1024)
                    yf = np.fft.fft(rx_vol, n=n_pad)
                    xf = np.fft.fftfreq(n_pad, 1 / FS)

                    # 상세 보기 그래프는 1kHz 이상 전체 표시
                    mask = (xf >= 1000)
                    if np.any(mask):
                        f_fft = xf[mask]
                        m_fft = np.abs(yf[mask]) / n_points

                    # --- 2. STFT Logic with UI Parameters ---

                    # 1) UI 값 가져오기
                    nperseg_ui = 256
                    overlap_pct = 50
                    pad_factor = 1

                    if hasattr(self.window, 'combo_stft_window'):
                        try:
                            win_str = self.window.combo_stft_window.currentText()
                            nperseg_ui = int(win_str)
                        except:
                            pass

                    if hasattr(self.window, 'spin_stft_overlap'):
                        try:
                            overlap_pct = self.window.spin_stft_overlap.value()
                        except:
                            pass

                    if hasattr(self.window, 'combo_stft_padding'):
                        try:
                            pad_str = self.window.combo_stft_padding.currentText()
                            pad_factor = int(pad_str.split('x')[0])
                        except:
                            pass

                    # 2) 데이터 길이에 맞춰 적용
                    if n_points < 32:
                        t_stft, f_stft, z_stft = [], [], []
                    else:
                        # UI 설정값과 실제 데이터 길이 중 작은 값 선택
                        nperseg = min(nperseg_ui, n_points)
                        if nperseg % 2 != 0: nperseg -= 1
                        if nperseg < 16: nperseg = 16

                        noverlap = int(nperseg * overlap_pct / 100.0)
                        if noverlap >= nperseg: noverlap = nperseg - 1

                        nfft = nperseg * pad_factor

                        f, t, Zxx = signal.stft(rx_vol, FS, nperseg=nperseg, noverlap=noverlap, nfft=nfft)

                        Zxx_mag = np.abs(Zxx)
                        t_stft = t
                        f_stft = f
                        z_stft = Zxx_mag

                        # --- 3. Partial FFT ---
                    p_s_ms = self.window.pfft_start_spin.value()
                    p_e_ms = self.window.pfft_end_spin.value()
                    i_s_target = int(p_s_ms * FS / 1000.0)
                    i_e_target = int(p_e_ms * FS / 1000.0)

                    i_s = min(i_s_target, n_points - 1)
                    i_e = min(i_e_target, n_points)

                    if i_e > i_s + 10:
                        seg = rx_vol[i_s:i_e]
                        n_sg = len(seg)
                        n_sg_pad = max(n_sg * 4, 512)
                        yf_p = np.fft.fft(seg, n=n_sg_pad)
                        xf_p = np.fft.fftfreq(n_sg_pad, 1 / FS)

                        mask_p = (xf_p >= 1000)

                        if np.any(mask_p):
                            f_pfft = xf_p[mask_p]
                            m_pfft = np.abs(yf_p[mask_p]) / n_sg

            # 2. TX Data Load
            if os.path.exists(tx_path):
                with open(tx_path, 'r') as f:
                    raw_tx = []
                    for line in f:
                        try:
                            raw_tx.append(float(line.strip()))
                        except:
                            pass
                    raw_tx = np.array(raw_tx)
                if len(raw_tx) > 0:
                    tx_vol = raw_tx - np.mean(raw_tx)
                    # [EXPERIMENTAL] Calculate TX start time for marker visualization
                    try:
                        tx_start_sample = self.distance_calculator.detect_tx_start(raw_tx)
                        if tx_start_sample != "Not Detected":
                            tx_start_us_marker = float(tx_start_sample)  # Already in μs since FS = 1MHz
                    except Exception as e:
                        pass

            # 그래프 업데이트 호출 [EXPERIMENTAL: Added ToF and TX start markers]
            self.window.update_graph(sid, None, rx_vol, tx_vol, f_fft, m_fft, t_stft, z_stft, f_pfft, m_pfft, f_stft, 
                                    tof_us_marker, tx_start_us_marker)

        except Exception as e:
            print(f"[Error in signal processing] {e}")
            traceback.print_exc()
            pass


def main():
    # 고해상도 모니터(HiDPI) 지원
    try:
        from PyQt5.QtCore import Qt
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    except Exception:
        pass

    app = QApplication(sys.argv)
    controller = PlayerController()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()