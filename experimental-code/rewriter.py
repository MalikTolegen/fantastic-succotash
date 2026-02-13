"""
Batch convert info.json files from nested env format to flattened format.

Converts:
    "env": {"temp": X, "humi": Y, "pres": Z, "mlx": W}
To:
    "temp": X, "humi": Y, "pres": Z, "mlx": W (at top level)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List


def flatten_env(data: Dict) -> Dict:
    """
    Flatten the nested 'env' object to top-level fields.
    
    Args:
        data: Original JSON data with nested 'env' object
        
    Returns:
        Modified JSON data with env fields at top level
    """
    if 'env' not in data:
        return data
    
    # Extract env fields
    env = data.pop('env')
    
    # Add env fields to top level (preserving order: after loggedAt, before gps)
    result = {}
    result['sensorId'] = data.get('sensorId')
    result['loggedAt'] = data.get('loggedAt')
    
    # Add flattened env fields
    result['temp'] = env.get('temp')
    result['humi'] = env.get('humi')
    result['pres'] = env.get('pres')
    result['mlx'] = env.get('mlx')
    
    # Add remaining fields (gps, laser, etc.)
    for key, value in data.items():
        if key not in ['sensorId', 'loggedAt']:
            result[key] = value
    
    return result


def find_info_files(base_path: Path) -> List[Path]:
    """
    Find all info.json files in timestamped subdirectories.
    
    Args:
        base_path: Path to 20260212/Sensor_1/
        
    Returns:
        List of Path objects to info.json files
    """
    pattern = base_path / "*/info.json"
    files = sorted(base_path.glob("*/info.json"))
    return files


def validate_files(files: List[Path]) -> tuple[List[Path], List[tuple[Path, str]]]:
    """
    Validate that all files can be loaded as JSON.
    
    Args:
        files: List of file paths to validate
        
    Returns:
        Tuple of (valid_files, errors) where errors is list of (path, error_message)
    """
    valid = []
    errors = []
    
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                json.load(f)
            valid.append(filepath)
        except Exception as e:
            errors.append((filepath, str(e)))
    
    return valid, errors


def convert_file(filepath: Path, dry_run: bool = False) -> bool:
    """
    Convert a single info.json file.
    
    Args:
        filepath: Path to info.json file
        dry_run: If True, print preview without writing
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        # Read original
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if conversion needed
        if 'env' not in data:
            if dry_run:
                print(f"  SKIP: {filepath} (no 'env' object found)")
            return True
        
        # Convert
        converted = flatten_env(data)
        
        if dry_run:
            print(f"  PREVIEW: {filepath}")
            print(f"    Before: {list(data.keys())}")
            print(f"    After:  {list(converted.keys())}")
            print(f"    Moved:  temp={converted.get('temp')}, humi={converted.get('humi')}, "
                  f"pres={converted.get('pres')}, mlx={converted.get('mlx')}")
        else:
            # Write back
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(converted, f, indent=4)
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {filepath}: {e}", file=sys.stderr)
        return False


def main():
    """Main conversion routine."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Flatten env object in info.json files'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview conversions without modifying files'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of files to process (for testing)'
    )
    
    args = parser.parse_args()
    
    # Find files
    base_path = Path(__file__).parent / "20260212" / "Sensor_1"
    
    if not base_path.exists():
        print(f"ERROR: Directory not found: {base_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Searching for info.json files in {base_path}...")
    files = find_info_files(base_path)
    
    if not files:
        print("No info.json files found!")
        sys.exit(1)
    
    print(f"Found {len(files)} files")
    
    # Apply limit if specified
    if args.limit:
        files = files[:args.limit]
        print(f"Limited to first {len(files)} files")
    
    # Validate files first
    print("\nValidating files...")
    valid_files, errors = validate_files(files)
    
    if errors:
        print(f"\nWARNING: {len(errors)} files failed validation:")
        for filepath, error in errors[:10]:  # Show first 10 errors
            print(f"  - {filepath}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        
        response = input("\nContinue with valid files only? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(1)
    
    print(f"\n{len(valid_files)} files are valid")
    
    # Process files
    mode = "DRY RUN" if args.dry_run else "CONVERTING"
    print(f"\n{mode} - Processing {len(valid_files)} files...")
    
    success_count = 0
    for i, filepath in enumerate(valid_files, 1):
        if i % 50 == 0 or i == 1:
            print(f"Progress: {i}/{len(valid_files)}...")
        
        if convert_file(filepath, dry_run=args.dry_run):
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total files: {len(files)}")
    print(f"  Valid files: {len(valid_files)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {len(valid_files) - success_count}")
    
    if args.dry_run:
        print(f"\nDRY RUN completed. No files were modified.")
        print(f"Run without --dry-run to apply changes.")
    else:
        print(f"\nConversion completed!")
    
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
