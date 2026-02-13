# -*- coding: utf-8 -*-
"""Analyzer for distance prediction results.

Analyzes predicted_dist_raw against ground_truth_dist and computes error metrics.
"""

import sys
import csv
import numpy as np


def analyze_results(csv_file="result.csv"):
    """
    Analyze distance prediction results from CSV file.
    
    Args:
        csv_file: Path to result CSV file
    """
    print(f"[INFO] Loading data from {csv_file}...")
    
    # Read CSV file
    predicted = []
    ground_truth = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pred = float(row['predicted_dist_raw'])
                gt = float(row['ground_truth_dist']) * 1000  # Convert to mm
                predicted.append(pred)
                ground_truth.append(gt)
            except (ValueError, KeyError):
                continue
    
    predicted = np.array(predicted)
    ground_truth = np.array(ground_truth)
    
    # Calculate metrics
    avg_predicted = np.mean(predicted)
    avg_ground_truth = np.mean(ground_truth)
    
    # Errors
    errors = predicted - ground_truth
    avg_error = np.mean(errors)
    std_error = np.std(errors)
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(errors))
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs(errors / ground_truth)) * 100
    
    # RMSE
    rmse = np.sqrt(np.mean(errors**2))
    
    # Print results
    print("=" * 60)
    print("DISTANCE PREDICTION ANALYSIS")
    print("=" * 60)
    print(f"\nSample size: {len(predicted)}")
    print(f"\n{'Metric':<30} {'Predicted':>15} {'Ground Truth':>15}")
    print("-" * 60)
    print(f"{'Average (mm)':<30} {avg_predicted:>15.3f} {avg_ground_truth:>15.3f}")
    print(f"{'Std Dev (mm)':<30} {np.std(predicted):>15.3f} {np.std(ground_truth):>15.3f}")
    print(f"{'Min (mm)':<30} {np.min(predicted):>15.3f} {np.min(ground_truth):>15.3f}")
    print(f"{'Max (mm)':<30} {np.max(predicted):>15.3f} {np.max(ground_truth):>15.3f}")
    
    print(f"\n{'ERROR METRICS':<30}")
    print("-" * 60)
    print(f"{'Average Error (mm)':<30} {avg_error:>15.3f}")
    print(f"{'Std Dev Error (mm)':<30} {std_error:>15.3f}")
    print(f"{'MAE (mm)':<30} {mae:>15.3f}")
    print(f"{'MAPE (%)':<30} {mape:>15.2f}%")
    print(f"{'RMSE (mm)':<30} {rmse:>15.3f}")
    
    print("\n" + "=" * 60)
    
    # Additional percentile analysis
    print("\nERROR DISTRIBUTION (Absolute):")
    print("-" * 60)
    abs_errors = np.abs(errors)
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(abs_errors, p)
        print(f"  {p:2d}th percentile: {val:>8.3f} mm")
    
    print("\n" + "=" * 60)
    print("\nKEY FINDINGS:")
    print(f"  • Predicted distances are {'OVERESTIMATING' if avg_error > 0 else 'UNDERESTIMATING'} by {abs(avg_error):.3f} mm on average")
    print(f"  • 50% of predictions are within ±{np.percentile(abs_errors, 50):.3f} mm")
    print(f"  • 95% of predictions are within ±{np.percentile(abs_errors, 95):.3f} mm")
    print("=" * 60)
    
    # Return metrics as dictionary
    return {
        'sample_size': len(predicted),
        'avg_predicted': avg_predicted,
        'avg_ground_truth': avg_ground_truth,
        'avg_error': avg_error,
        'std_error': std_error,
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
        'median_abs_error': np.percentile(abs_errors, 50),
        'p95_abs_error': np.percentile(abs_errors, 95),
    }


def main():
    """Main entry point."""
    csv_file = "result.csv"
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    try:
        analyze_results(csv_file)
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
