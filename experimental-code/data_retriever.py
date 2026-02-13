# -*- coding: utf-8 -*-
"""Data retriever for 20260212/Sensor_1.

Outputs CSV with:
- folder_name
- gps_speed
- ground_truth_dist
- predicted_dist_raw
- predicted_dist_filtered
"""

import os
import sys
import json
import glob
import csv
import math
import numpy as np

# Import from data-viewer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data-viewer'))
from main import DistanceCalculator, hampel_filter
from scipy import signal

# Constants from data-viewer/main.py
FS = 1e6  # 1 MHz sampling rate


def _to_float(value):
    """Convert value to float, return None if invalid."""
    try:
        if value is None or value == "-" or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def process_sensor_folder(sensor_path, output_csv="result.csv"):
    """
    Process all frame folders in sensor directory and output CSV.
    
    Args:
        sensor_path: Path to sensor folder (e.g., "20260212/Sensor_1")
        output_csv: Output CSV filename
    """
    print(f"[INFO] Processing sensor folder: {sensor_path}")

    # Initialize distance calculator
    calc = DistanceCalculator(FS)

    # Find all frame folders
    frame_folders = sorted(glob.glob(os.path.join(sensor_path, "*")))
    frame_folders = [f for f in frame_folders if os.path.isdir(f)]

    if not frame_folders:
        print(f"[ERROR] No frame folders found in {sensor_path}")
        return

    print(f"[INFO] Found {len(frame_folders)} frames to process")

    results = []
    distance_history = []  # For Hampel filtering

    for idx, folder_path in enumerate(frame_folders):
        folder_name = os.path.basename(folder_path)

        row = {
            "folder_name": folder_name,
            "gps_speed": None,
            "ground_truth_dist": None,
            "predicted_dist_raw": None,
            "predicted_dist_filtered": None,
        }

        try:
            # Read info.json
            info_path = os.path.join(folder_path, "info.json")
            if not os.path.exists(info_path):
                print(f"[WARNING] Frame {folder_name}: info.json not found")
                results.append(row)
                continue

            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)

            # Extract environmental data
            temp_C = _to_float(info.get("temp"))
            humi_percent = _to_float(info.get("humi"))
            
            # Extract GPS speed (from gps.speed_kmh)
            gps_data = info.get("gps", {})
            gps_speed = _to_float(gps_data.get("speed_kmh"))
            row["gps_speed"] = gps_speed

            # Extract ground truth distance (from laser.dist)
            laser_data = info.get("laser", {})
            ground_truth_dist = _to_float(laser_data.get("dist"))
            row["ground_truth_dist"] = ground_truth_dist

            # Load TX and RX signals
            tx_path = os.path.join(folder_path, "tx_plain.dat")
            rx_path = os.path.join(folder_path, "rx_plain.dat")

            if not os.path.exists(tx_path) or not os.path.exists(rx_path):
                print(f"[WARNING] Frame {folder_name}: Signal files not found")
                results.append(row)
                continue

            tx_signal = calc.load_adc_data(tx_path)
            rx_signal = calc.load_adc_data(rx_path)

            if tx_signal is None or rx_signal is None:
                print(f"[WARNING] Frame {folder_name}: Failed to load signals")
                results.append(row)
                continue

            if len(rx_signal) == 0 or len(tx_signal) == 0:
                print(f"[WARNING] Frame {folder_name}: Empty signals")
                results.append(row)
                continue

            # Detect TX start time
            tx_start = calc.detect_tx_start(tx_signal)
            if tx_start == "Not Detected":
                print(f"[WARNING] Frame {folder_name}: TX start not detected")
                results.append(row)
                continue

            tx_start_us = float(tx_start)

            # Extract envelope from RX signal for ToF detection
            rx_centered = rx_signal - np.median(rx_signal)
            rx_analytic = signal.hilbert(rx_centered)
            rx_envelope = np.abs(rx_analytic)

            # Detect ToF
            tof_start = calc.detect_tof(rx_envelope, tx_signal)
            if tof_start == "Not Detected":
                print(f"[WARNING] Frame {folder_name}: ToF not detected")
                results.append(row)
                continue

            tof_us = float(tof_start)

            # Calculate raw distance using DistanceCalculator's method
            distance_raw = calc.predict_distance(tof_us, tx_start_us, temp_C, humi_percent)
            
            if distance_raw != "Not Detected" and distance_raw is not None:
                row["predicted_dist_raw"] = distance_raw
                row["predicted_dist_raw"] = distance_raw
                
                # Add to history for filtering
                distance_history.append(distance_raw)
                
                # Apply Hampel filter if we have enough history
                if len(distance_history) >= 5:
                    history_array = np.array(distance_history)
                    filtered_array = hampel_filter(history_array, window=5, n_sigma=3.0)
                    distance_filtered = filtered_array[-1]  # Use last value
                    row["predicted_dist_filtered"] = distance_filtered
                else:
                    # Not enough history for filtering, use raw value
                    row["predicted_dist_filtered"] = distance_raw

        except Exception as e:
            print(f"[ERROR] Frame {folder_name}: {e}")
            import traceback
            traceback.print_exc()

        results.append(row)

        if (idx + 1) % 50 == 0 or idx == len(frame_folders) - 1:
            print(f"[PROGRESS] Processed {idx + 1}/{len(frame_folders)} frames")

    # Write results to CSV
    print(f"[INFO] Writing results to {output_csv}")

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "folder_name",
            "gps_speed",
            "ground_truth_dist",
            "predicted_dist_raw",
            "predicted_dist_filtered",
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"[SUCCESS] Wrote {len(results)} rows to {output_csv}")

    # Print summary statistics
    valid_raw = [r["predicted_dist_raw"] for r in results if r["predicted_dist_raw"] is not None]
    valid_filtered = [r["predicted_dist_filtered"] for r in results if r["predicted_dist_filtered"] is not None]
    valid_gt = [r["ground_truth_dist"] for r in results if r["ground_truth_dist"] is not None]

    print(f"\n[STATS] Summary:")
    print(f"  Total frames: {len(results)}")
    print(f"  Valid ground truth: {len(valid_gt)}")
    print(f"  Valid predictions (raw): {len(valid_raw)}")
    print(f"  Valid predictions (filtered): {len(valid_filtered)}")
    
    if valid_raw:
        print(f"  Raw distance range: {min(valid_raw):.3f} - {max(valid_raw):.3f} mm")
    if valid_filtered:
        print(f"  Filtered distance range: {min(valid_filtered):.3f} - {max(valid_filtered):.3f} mm")
    if valid_gt:
        print(f"  Ground truth range: {min(valid_gt):.3f} - {max(valid_gt):.3f} mm")


def main():
    """Main entry point."""
    sensor_path = os.path.join("20260212", "Sensor_1")

    if len(sys.argv) > 1:
        sensor_path = sys.argv[1]

    if not os.path.exists(sensor_path):
        print(f"[ERROR] Sensor path does not exist: {sensor_path}")
        sys.exit(1)

    output_csv = "result.csv"
    if len(sys.argv) > 2:
        output_csv = sys.argv[2]

    process_sensor_folder(sensor_path, output_csv=output_csv)


if __name__ == "__main__":
    main()
