# Real-Time Data Saver and Viewer

A complete solution for real-time GPS and distance data collection and visualization. This project consists of two applications:
- **data-saver**: Collects real-time data from GPS and distance sensors
- **data-viewer**: Visualizes and analyzes the saved data

## Prerequisites

- Python 3.10 or higher
- Git
- pip (comes with Python)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/MalikTolegen/fantastic-succotash.git
cd fantastic-succotash
```

### 2. Create a Virtual Environment

Create a single virtual environment at the project root:

```bash
# On Windows
python -m venv .venv

# On macOS/Linux
python3 -m venv .venv
```

### 3. Activate the Virtual Environment

**Windows (PowerShell):**
```bash
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

You should see `(.venv)` at the beginning of your terminal prompt when activated.

### 4. Install Dependencies

Install all dependencies for both applications:

```bash
pip install --upgrade pip
pip install -r data-saver/requirements.txt
pip install -r data-viewer/requirements.txt
```

## Running the Applications

Once you have the virtual environment activated, you can run either application from the project root.

### Data Saver (Real-Time Data Collection)

Collects real-time data from GPS and ultrasonic distance sensors:

```bash
cd data-saver
python main.py
```

This application:
- Connects to GPS and distance sensors
- Saves real-time data to files
- Processes and logs sensor readings

### Data Viewer (Data Visualization & Analysis)

Visualizes and analyzes the collected data:

```bash
cd data-viewer
python main.py
```

This application:
- Loads saved data frames
- Provides signal processing and visualization tools
- Includes trend analysis and FFT capabilities
- Features bandpass filtering and envelope processing

## Project Structure

```
fantastic-succotash/
├── data-saver/
│   ├── main.py                  # Entry point for data collection
│   ├── MainWindow.py            # GUI for data saver
│   ├── CommWorker.py            # Communication handling
│   ├── GpsWorker.py             # GPS data processing
│   ├── DistanceCalculator.py    # Distance calculation
│   ├── ViewModel.py             # View model for GUI
│   ├── Protocol.py              # Protocol definitions
│   ├── requirements.txt         # Dependencies
│   └── ...
│
├── data-viewer/
│   ├── main.py                  # Entry point for data viewing
│   ├── MainWindow.py            # GUI for data viewer
│   ├── requirements.txt         # Dependencies
│   └── ...
│
├── .venv/                       # Virtual environment (auto-created)
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Dependencies

### Data Saver
- `pyserial` - Serial communication with sensors
- `numpy` - Numerical operations
- `pyqtgraph` - Real-time plotting
- `PyQt5` - GUI framework
- `scipy` - Signal processing

### Data Viewer
- `numpy` - Numerical operations
- `scipy` - Signal processing
- `PyQt5` or `PySide6` - GUI framework

## Troubleshooting

### PyQt5 Installation Issues
If PyQt5 fails to install on your platform, you can use PySide6 instead:

```bash
# For data-viewer, edit the requirements.txt and replace:
# PyQt5==5.15.11
# with:
# PySide6==6.6.1

pip install -r data-viewer/requirements.txt
```

### Virtual Environment Not Activating
Make sure you're running the activation script from the project root:
```bash
# Windows
.venv\Scripts\activate.bat    # or Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

### Module Not Found Errors
Ensure your virtual environment is activated and all requirements are installed:
```bash
pip list  # Check installed packages
pip install -r data-saver/requirements.txt
pip install -r data-viewer/requirements.txt
```

## Development

### Using Git
Once you have made changes:

```bash
git status              # Check what changed
git add .               # Stage all changes
git commit -m "Your message"  # Create a commit
git push origin main    # Push to GitHub
```

### Note on Virtual Environment
- The `.venv` folder is git-ignored and won't be pushed to GitHub
- When someone clones this repo, they need to create their own virtual environment using steps 2-4 above
- The virtual environment is shared across both applications in the project

## Contributing

1. Create a branch for your feature
2. Make your changes
3. Test both applications
4. Commit and push your changes
5. Create a pull request

## License

[Add your license here]

## Contact

For questions or issues, please open an issue on GitHub.
