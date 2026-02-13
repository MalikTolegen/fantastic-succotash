# -*- coding: utf-8 -*-
"""Plot predicted distance vs ground truth distance."""

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt


def plot_distances(csv_file="result.csv", output_file="distance_plot.png"):
    """
    Plot predicted and ground truth distances.
    
    Args:
        csv_file: Path to result CSV file
        output_file: Path to save plot image
    """
    print(f"[INFO] Loading data from {csv_file}...")
    
    # Read CSV file
    predicted = []
    ground_truth = []
    frame_indices = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            try:
                pred = float(row['predicted_dist_filtered'])
                gt = float(row['ground_truth_dist']) * 1000  # Convert to mm
                predicted.append(pred)
                ground_truth.append(gt)
                frame_indices.append(idx)
            except (ValueError, KeyError):
                continue
    
    predicted = np.array(predicted)
    ground_truth = np.array(ground_truth)
    frame_indices = np.array(frame_indices)
    
    print(f"[INFO] Loaded {len(predicted)} data points")
    
    # Create plot
    plt.figure(figsize=(14, 6))
    
    plt.plot(frame_indices, predicted, 'b.', markersize=4, label='Predicted distance', alpha=0.7)
    plt.plot(frame_indices, ground_truth, 'r.', markersize=4, label='Ground truth distance', alpha=0.7)
    
    plt.xlabel('Frame Index', fontsize=12)
    plt.ylabel('Distance (mm)', fontsize=12)
    plt.title('Predicted vs Ground Truth Distance', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[SUCCESS] Plot saved to {output_file}")
    
    # Show plot
    plt.show()


def main():
    """Main entry point."""
    csv_file = "result.csv"
    output_file = "distance_plot.png"
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    try:
        plot_distances(csv_file, output_file)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {csv_file}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
