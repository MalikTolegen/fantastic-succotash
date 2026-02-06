from __future__ import annotations

import shutil
from pathlib import Path


def find_and_delete_empty_timestamped_folders(base_dir: Path) -> list[str]:
	deleted: list[str] = []

	if not base_dir.exists():
		print(f"Base directory not found: {base_dir}")
		return deleted

	for entry in sorted(base_dir.iterdir()):
		if not entry.is_dir():
			continue

		info_path = entry / "info.json"
		tx_path = entry / "tx_plain.dat"
		rx_path = entry / "rx_plain.dat"

		if not (info_path.exists() and tx_path.exists() and rx_path.exists()):
			continue

		tx_empty = tx_path.stat().st_size == 0
		rx_empty = rx_path.stat().st_size == 0

		if tx_empty or rx_empty:
			shutil.rmtree(entry)
			deleted.append(entry.name)

	return deleted


def main() -> None:
	base_dir = Path(__file__).resolve().parent / "260206" / "Sensor_1"
	deleted = find_and_delete_empty_timestamped_folders(base_dir)

	if deleted:
		print(f"Deleted {len(deleted)} folder(s):")
		for name in deleted:
			print(f"- {name}")
	else:
		print("No folders deleted.")


if __name__ == "__main__":
	main()
