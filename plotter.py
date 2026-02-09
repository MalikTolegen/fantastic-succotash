import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

FS = 1000000.0  # 1 MHz


def load_adc_data(file_path):
	if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
		return None
	data = []
	with open(file_path, "r", errors="replace") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				data.append(float(line))
			except ValueError:
				pass
	if len(data) == 0:
		return None
	return np.array(data)


class DistanceCalculator:
	def __init__(self, sampling_rate=1e6):
		self.fs = sampling_rate
		self.nyquist = sampling_rate / 2.0

		self.bandpass_low = 20e3
		self.bandpass_high = 60e3
		self.envelope_cutoff = 5e3
		self.filter_order = 4

		self.crosstalk_skip = 2500
		self.min_detection_distance = 200
		self.noise_estimation_start = 2500
		self.noise_estimation_length = 1000
		self.snr_threshold = 12
		self.min_absolute_amplitude = 50

	def bandpass_filter(self, signal_data, lowcut=None, highcut=None):
		low = lowcut if lowcut is not None else self.bandpass_low
		high = highcut if highcut is not None else self.bandpass_high

		dc_offset = np.median(signal_data)
		signal_dc_removed = signal_data - dc_offset

		low_norm = low / self.nyquist
		high_norm = high / self.nyquist

		low_norm = np.clip(low_norm, 0.001, 0.999)
		high_norm = np.clip(high_norm, low_norm + 0.001, 0.999)

		b, a = signal.butter(self.filter_order, [low_norm, high_norm], btype="band")
		return signal.filtfilt(b, a, signal_dc_removed)

	def envelope_detection(self, signal_data):
		rectified = np.abs(signal_data)

		cutoff_norm = self.envelope_cutoff / self.nyquist
		cutoff_norm = np.clip(cutoff_norm, 0.001, 0.999)

		b, a = signal.butter(self.filter_order, cutoff_norm, btype="low")
		return signal.filtfilt(b, a, rectified)

	def detect_tof(self, envelope, tx_signal=None):
		if len(envelope) <= self.crosstalk_skip + self.min_detection_distance:
			return "Not Detected"

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

		threshold_median = noise_median + 5 * noise_std
		threshold_percentile = np.percentile(noise_region, 90) + 4 * noise_std
		threshold_max = noise_max + 3 * noise_std

		adaptive_threshold = np.median([threshold_median, threshold_percentile, threshold_max])
		MIN_THRESHOLD = 30
		adaptive_threshold = max(adaptive_threshold, MIN_THRESHOLD)

		search_signal = envelope[self.crosstalk_skip :]
		threshold_crossings = np.where(search_signal > adaptive_threshold)[0]

		if len(threshold_crossings) > 0:
			first_crossing = threshold_crossings[0]

			if first_crossing < self.min_detection_distance:
				valid_crossings = threshold_crossings[
					threshold_crossings >= self.min_detection_distance
				]
				if len(valid_crossings) == 0:
					first_crossing = None
				else:
					first_crossing = valid_crossings[0]

			if first_crossing is not None:
				sustained = True
				for i in range(first_crossing, min(first_crossing + 5, len(search_signal))):
					if search_signal[i] <= adaptive_threshold * 0.8:
						sustained = False
						break
				if not sustained:
					first_crossing = None

			if first_crossing is not None:
				start_idx = first_crossing
				lookback_limit = max(self.min_detection_distance, first_crossing - 100)

				for i in range(first_crossing - 1, lookback_limit, -1):
					if i + 5 < len(search_signal):
						next_points = search_signal[i + 1 : i + 6]
						if search_signal[i] < np.min(next_points):
							start_idx = i
							break

				tof_idx = start_idx + self.crosstalk_skip
				if envelope[tof_idx] > self.min_absolute_amplitude:
					return int(tof_idx)

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
					first_crossing_rx, min(first_crossing_rx + 5, len(search_signal))
				):
					if search_signal[i] <= adaptive_threshold_rx * 0.8:
						sustained = False
						break
				if sustained:
					tof_idx_rx = first_crossing_rx + self.crosstalk_skip
					if envelope[tof_idx_rx] > self.min_absolute_amplitude:
						return int(tof_idx_rx)

		rise_len = 100
		rise_delta_min = 10
		rise_end_min = 50
		slope_threshold = np.tan(np.deg2rad(22.5))

		if len(search_signal) > rise_len + 1:
			for start_idx in range(0, len(search_signal) - rise_len - 1):
				end_idx = start_idx + rise_len
				rising = True
				for j in range(start_idx, end_idx):
					if search_signal[j + 1] <= search_signal[j]:
						rising = False
						break
				if not rising:
					continue

				if (search_signal[end_idx] - search_signal[start_idx]) < rise_delta_min:
					continue
				if search_signal[end_idx] < rise_end_min:
					continue

				onset_idx = None
				for j in range(start_idx, end_idx):
					if (search_signal[j + 1] - search_signal[j]) > slope_threshold:
						onset_idx = j
						break

				if onset_idx is not None:
					tof_idx_basic = onset_idx + self.crosstalk_skip
					return int(tof_idx_basic)

		window_size = 100
		hop_size = 20
		energy_snr_threshold = 6
		energy_onset_k = 1.8
		energy_sustain_len = 5
		energy_lookback_factor = 4

		energy_noise_region = None
		if tx_env is not None and len(tx_env) >= 3000:
			energy_noise_region = tx_env[2500:3000]

		max_energy = 0
		max_energy_idx = None

		for i in range(self.crosstalk_skip, len(envelope) - window_size, hop_size):
			window = envelope[i : i + window_size]
			energy = np.sum(window**2)
			if energy > max_energy:
				max_energy = energy
				max_energy_idx = i

		if energy_noise_region is None or len(energy_noise_region) == 0:
			return "Not Detected"

		noise_energy = (
			np.sum(energy_noise_region**2) / len(energy_noise_region) * window_size
		)
		energy_ratio_db = 10 * np.log10(max_energy / (noise_energy + 1e-10))

		if energy_ratio_db > energy_snr_threshold and max_energy_idx is not None:
			if max_energy_idx - self.crosstalk_skip >= self.min_detection_distance:
				if envelope[max_energy_idx] > self.min_absolute_amplitude:
					energy_search_start = max(
						self.crosstalk_skip,
						max_energy_idx - (window_size * energy_lookback_factor),
					)
					energy_segment = envelope[energy_search_start : max_energy_idx + 1]
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

						if (
							onset_idx is not None
							and envelope[onset_idx] > self.min_absolute_amplitude
						):
							return int(onset_idx)
					return int(max_energy_idx)

		derivative = np.diff(search_signal)
		significant_rise = derivative > (noise_std * 2.5)

		if np.any(significant_rise):
			rise_indices = np.where(significant_rise)[0]
			valid_rises = rise_indices[rise_indices >= self.min_detection_distance]

			if len(valid_rises) > 0:
				tof_idx = valid_rises[0] + self.crosstalk_skip
				if (
					envelope[tof_idx] > adaptive_threshold * 0.7
					and envelope[tof_idx] > self.min_absolute_amplitude
				):
					return int(tof_idx)

		return "Not Detected"

	def detect_tx_start(self, tx_signal):
		dc_offset = np.median(tx_signal)
		tx_centered = tx_signal - dc_offset

		analytic_signal = signal.hilbert(tx_centered)
		tx_env = np.abs(analytic_signal)

		if len(tx_env) < 100:
			return "Not Detected"

		noise_region = tx_env[:50]
		noise_std = np.std(noise_region)
		noise_max = np.max(noise_region)

		threshold = noise_max + 6 * noise_std
		threshold_crossings = np.where(tx_env > threshold)[0]
		if len(threshold_crossings) == 0:
			return "Not Detected"

		first_crossing = threshold_crossings[0]
		start_idx = first_crossing
		lookback_limit = max(0, first_crossing - 5)

		for i in range(first_crossing - 1, lookback_limit, -1):
			if tx_env[i] < threshold * 0.5:
				start_idx = i + 1
				break

		return float(start_idx)


def plot_folder(folder_path, output_dir, calculator):
	tx_path = os.path.join(folder_path, "tx_plain.dat")
	rx_path = os.path.join(folder_path, "rx_plain.dat")

	tx_signal = load_adc_data(tx_path)
	rx_signal = load_adc_data(rx_path)

	if tx_signal is None and rx_signal is None:
		return False

	tof_sample = "Not Detected"
	rx_start_sample = "Not Detected"
	tx_start_sample = "Not Detected"

	if tx_signal is not None and len(tx_signal) > 0:
		tx_start_sample = calculator.detect_tx_start(tx_signal)

	if rx_signal is not None and len(rx_signal) > 0:
		filtered = calculator.bandpass_filter(rx_signal)
		envelope = calculator.envelope_detection(filtered)
		tof_sample = calculator.detect_tof(envelope, tx_signal)
		rx_start_sample = tof_sample

	fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

	if tx_signal is not None:
		x_tx = np.arange(len(tx_signal))
		axes[0].plot(x_tx, tx_signal, color="#1f77b4", linewidth=1.0, label="TX")
		if tx_start_sample != "Not Detected":
			axes[0].axvline(
				tx_start_sample,
				color="#2ca02c",
				linestyle=":",
				linewidth=1.2,
				label="TX start",
			)
			ymax = np.max(tx_signal)
			axes[0].text(
				tx_start_sample,
				ymax,
				f"{tx_start_sample:.1f} us",
				color="#2ca02c",
				fontsize=9,
				va="bottom",
				ha="left",
			)
		axes[0].set_title("TX Signal")
		axes[0].set_xlabel("Sample (us)")
		axes[0].set_ylabel("Amplitude")
		axes[0].grid(True, alpha=0.2)
		axes[0].legend(loc="upper right")
	else:
		axes[0].set_title("TX Signal (missing)")
		axes[0].axis("off")

	if rx_signal is not None:
		x_rx = np.arange(len(rx_signal))
		axes[1].plot(x_rx, rx_signal, color="#ff7f0e", linewidth=1.0, label="RX")

		if rx_start_sample != "Not Detected":
			axes[1].axvline(
				rx_start_sample,
				color="#2ca02c",
				linestyle=":",
				linewidth=1.2,
				label="RX start",
			)
			ymax = np.max(rx_signal)
			axes[1].text(
				rx_start_sample,
				ymax,
				f"{rx_start_sample:.1f} us",
				color="#2ca02c",
				fontsize=9,
				va="bottom",
				ha="left",
			)

		axes[1].set_title("RX Signal")
		axes[1].set_xlabel("Sample (us)")
		axes[1].set_ylabel("Amplitude")
		axes[1].grid(True, alpha=0.2)
		axes[1].legend(loc="upper right")
	else:
		axes[1].set_title("RX Signal (missing)")
		axes[1].axis("off")

	folder_name = os.path.basename(folder_path)
	fig.suptitle(folder_name)
	fig.tight_layout(rect=[0, 0.03, 1, 0.95])

	output_path = os.path.join(output_dir, f"{folder_name}.png")
	fig.savefig(output_path, dpi=150)
	plt.close(fig)
	return True


def main():
	script_dir = os.path.dirname(os.path.abspath(__file__))
	data_root = os.path.join(script_dir, "260109", "Sensor_1")
	output_dir = os.path.join(script_dir, "260109_plots")

	os.makedirs(output_dir, exist_ok=True)

	folders = sorted(
		[
			path
			for path in glob.glob(os.path.join(data_root, "*"))
			if os.path.isdir(path)
		]
	)

	if not folders:
		print(f"No folders found under {data_root}")
		return

	if len(folders) > 100:
		folders = random.sample(folders, 100)
		folders.sort()

	calculator = DistanceCalculator(FS)

	processed = 0
	for folder in folders:
		if plot_folder(folder, output_dir, calculator):
			processed += 1

	print(f"Saved {processed} plots to {output_dir}")


if __name__ == "__main__":
	main()
