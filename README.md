# VNBR-Road Layers

A Streamlit dashboard for visualizing and analyzing construction layer data (Embankment, Subgrade, GSB V, etc.) by chainage, bill number, and timeline.

## Overview

This app loads project data from a CSV file (`3 layers.csv`) and provides:

- **Summary metrics** — Total length, progress vs estimate, record counts
- **Overlap & gap analysis** — Merged chainages per layer, duplicate coverage (overlaps), and missing coverage (gaps)
- **Layer consistency check** — Validates that top-layer chainages lie within the layer below (hierarchy: Embankment → Subgrade → GSB V → GSB III → WMM → DBM → BC)
- **Layer coverage charts** — Per 1000 m chainage chunks, with bottom layer at bottom

## Features

| Feature | Description |
|--------|-------------|
| **Filters** | Filter by **Item** (layer) and **Bill No**; affects KPIs, charts, and the data table |
| **Merged chainages** | Overlapping/adjacent stretches per item are merged into single ranges |
| **Overlaps** | Pairs of segments (same item) with overlapping chainage, with length and bill info |
| **Gaps** | Missing chainage ranges per item (from 0 to project end) |
| **All layer chainages** | Expandable table of merged ranges used in the consistency check |
| **Layer consistency** | Violations where an upper layer has chainage not covered by the layer below |
| **Coverage per 1000 m** | One Gantt-style chart per 1000 m chunk; bars show coverage per layer |
| **Chunk bars** | Grouped bar chart: coverage length per layer in each 1000 m chunk |

## Data Format

The app expects a CSV file named **`3 layers.csv`** in the same folder as `app.py`, with columns:

| Column | Description |
|--------|-------------|
| Estimate | e.g. Main |
| Est Length | Project length in metres (e.g. 8000) |
| Bill No | e.g. Bill No. 07, Bill No. 08 |
| Item | Layer name: Embankment, Subgrade, GSB V, etc. |
| Stretch | Chainage range as `start-end` (e.g. 500-1000, 620-1000) |
| Length | Length in metres |
| Mbook | Book reference |
| Pages | Page range |
| Date | Date (formats: 6/27/2025, 05.08.2025, 27.09.2025) |

## Installation

1. **Python**  
   Use Python 3.8 or newer.

2. **Create and use the project environment (recommended)**  
   From the **VNBR-Road Layers** project folder (create the venv here so launchers point to the correct Python):

   ```bash
   # Create virtual environment
   python -m venv .venv

   # Activate it (Windows PowerShell)
   .venv\Scripts\Activate.ps1

   # Activate it (Windows CMD)
   .venv\Scripts\activate.bat

   # Activate it (macOS/Linux)
   source .venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt
   ```

   If PowerShell blocks the activation script, run once:  
   `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

   Or install globally (not recommended):

   ```bash
   pip install -r requirements.txt
   ```

## Running the App

From the project folder (where `app.py` and `3 layers.csv` are located).

**Option A — With environment activated:**

```bash
streamlit run app.py
```

**Option B — Without activating (avoids launcher path issues):**

```bash
# Windows
.venv\Scripts\python.exe -m streamlit run app.py

# macOS/Linux
.venv/bin/python -m streamlit run app.py
```

The app opens in your browser (default: http://localhost:8501). Use the sidebar to filter by **Item** and **Bill No**, and scroll the page to view all sections.

## Project Structure

```
VNBR-Road Layers/
├── .venv/            # Virtual environment (create with python -m venv .venv)
├── .gitignore        # Ignore .venv, __pycache__, etc.
├── app.py            # Streamlit application
├── 3 layers.csv      # Input data (required)
├── requirements.txt  # Python dependencies
├── README.md         # This file
└── USAGE.md          # Step-by-step usage instructions
```

## Requirements

- **streamlit** — Web app framework
- **pandas** — Data loading and processing
- **plotly** — Interactive charts (timeline, bar, subplots)

See [USAGE.md](USAGE.md) for detailed instructions on using the app.
