# -*- coding: utf-8 -*-
import os
import math
import numpy as np
from scipy import signal
from datetime import datetime


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
    """Calculate distance using ToF detection, TX start time, and LSE model."""
    
    # LSE Model parameters
    BETA_0 = -14.057  # mm 
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
        
        # Cache for last valid distance (to handle empty file edge-case)
        self.last_valid_distance = None  # Store last successful distance prediction
    
    def load_adc_data(self, file_path):
        """Load raw ADC data from file. Returns None for empty files."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check if file is empty (0 bytes) - skip processing
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
        
        # If no valid data found after reading, return None
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
        Detect Time-of-Flight from envelope signal using TX-derived noise baseline.
        
        Args:
            envelope: RX envelope signal
            tx_signal: Raw TX signal (optional, for stable noise estimation)
            
        Returns:
            ToF sample index, or "Not Detected" if not found
        """
        if len(envelope) <= self.crosstalk_skip + self.min_detection_distance:
            return "Not Detected"
        
        # Use TX post-burst noise for stable, distance-independent baseline
        if tx_signal is not None and len(tx_signal) >= 2000:
            # Extract TX envelope for noise estimation
            tx_centered = tx_signal - np.median(tx_signal)
            tx_analytic = signal.hilbert(tx_centered)
            tx_env = np.abs(tx_analytic)
            
            # STABLE NOISE ZONE: Samples 1500-2000 of TX (after burst, before echoes)
            noise_region = tx_env[1500:2000]
        else:
            # Fallback: use RX-based noise from early region (before most echoes)
            # Use 1500-2000 μs zone (safer than 2500-3500 for close targets)
            noise_start = 1500
            noise_end = 2000
            if len(envelope) < noise_end:
                return "Not Detected"
            noise_region = envelope[noise_start:noise_end]
        
        noise_mean = np.mean(noise_region)
        noise_std = np.std(noise_region)
        noise_median = np.median(noise_region)
        noise_max = np.max(noise_region)
        
        # Calculate adaptive threshold using multiple strategies
        threshold_median = noise_median + 5 * noise_std
        threshold_percentile = np.percentile(noise_region, 90) + 4 * noise_std
        threshold_max = noise_max + 3 * noise_std
        
        # Use the median of thresholds
        adaptive_threshold = np.median([threshold_median, threshold_percentile, threshold_max])
        
        # SAFETY FLOOR: Never allow threshold below minimum (prevents false positives from too-quiet TX)
        MIN_THRESHOLD = 30  # ADC units
        adaptive_threshold = max(adaptive_threshold, MIN_THRESHOLD)
        
        # Search region: after crosstalk zone
        search_signal = envelope[self.crosstalk_skip:]
        
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
                # Reduced backtracking: 100 samples max (was 400)
                # More conservative to avoid overshooting
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
        
        # Method 2: Energy-Based Detection (Fallback)
        window_size = 100  # 100 μs window
        hop_size = 20  # 20 μs hop
        
        max_energy = 0
        max_energy_idx = None
        
        for i in range(self.crosstalk_skip, len(envelope) - window_size, hop_size):
            window = envelope[i:i+window_size]
            energy = np.sum(window ** 2)
            
            if energy > max_energy:
                max_energy = energy
                max_energy_idx = i
        
        noise_energy = np.sum(noise_region ** 2) / len(noise_region) * window_size
        energy_ratio_db = 10 * np.log10(max_energy / (noise_energy + 1e-10))
        
        if energy_ratio_db > self.snr_threshold and max_energy_idx is not None:
            if max_energy_idx - self.crosstalk_skip >= self.min_detection_distance:
                if envelope[max_energy_idx] > self.min_absolute_amplitude:
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
        Detect TX burst start time with minimal backtracking to avoid overshooting.
        
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
        
        # Use a higher threshold to avoid picking up noise edges
        # TX signals are strong, so require 6 * noise_std above max
        threshold = noise_max + 6 * noise_std
        
        # Find first threshold crossing
        threshold_crossings = np.where(tx_env > threshold)[0]
        
        if len(threshold_crossings) == 0:
            return "Not Detected"
        
        first_crossing = threshold_crossings[0]
        
        # MINIMAL backtracking: only look back 5 samples max (half a microsecond)
        # TX signals have SHARP rise, so overshoot is minimal
        start_idx = first_crossing
        lookback_limit = max(0, first_crossing - 5)
        
        # Only backtrack if we find a point that is SIGNIFICANTLY lower
        # and shows a clear noise-to-signal separation (strict condition)
        for i in range(first_crossing - 1, lookback_limit, -1):
            # STRICT: point must be below 50% of threshold to be considered "noise"
            if tx_env[i] < threshold * 0.5:
                # Found clear separation from burst - this is the start
                start_idx = i + 1  # Start is right after noise region
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
            Distance in mm, last valid distance if detection fails, or "Not Detected" if no previous valid distance
        """
        # Edge case: empty rx_plain.dat or tx_plain.dat files
        # Return last valid distance instead of "Not Detected"
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

        # Cache this valid distance for future use
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


def parse_timestamp(timestamp_str):
    """
    Parse timestamp string to datetime object.
    
    Supports multiple formats:
    - YYYYMMDDHHMMSS_milliseconds (e.g., "20260105104916_350")
    - Standard datetime formats
    - ISO format strings
    
    Args:
        timestamp_str: Timestamp string from info.json 'loggedAt' field
        
    Returns:
        datetime object or None if parsing fails
    """
    if timestamp_str is None or timestamp_str == '':
        return None
    
    try:
        # Try format: YYYYMMDDHHMMSS_milliseconds
        if '_' in timestamp_str:
            parts = timestamp_str.split('_')
            dt = datetime.strptime(parts[0], '%Y%m%d%H%M%S')
            if len(parts) > 1:
                milliseconds = int(parts[1])
                dt = dt.replace(microsecond=milliseconds * 1000)
            return dt
    except (ValueError, AttributeError):
        pass
    
    # Try standard datetime formats
    formats = [
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%S',
        '%Y/%m/%d %H:%M:%S',
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(str(timestamp_str), fmt)
        except (ValueError, AttributeError):
            continue
    
    return None

