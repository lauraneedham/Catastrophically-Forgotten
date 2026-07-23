# Interactive Experiment Notebooks

This directory contains interactive [marimo](https://marimo.io) notebooks for running and comparing learning rules on catastrophic forgetting tasks.

## Notebooks Included

- **`pc_forgetting_experiments.py`**: Interactive side-by-side comparison of **Predictive Coding (`jpc`)** and **Backpropagation** on MNIST. Includes interactive controls to adjust training epochs, dataset subset size, and batch size.

---

## Getting Started

### 1. Install Dependencies

All dependencies (including `marimo` and `jpc`) are listed in `requirements.txt`. Activate your virtual environment and install them:

```bash
pip install -r requirements.txt
```

---

### 2. Launch the Notebook

To open and interact with the notebook in your browser, run:

```bash
marimo edit notebooks/pc_forgetting_experiments.py
```

Marimo will launch a local web server (usually at `http://127.0.0.1:2718`) and open the interactive notebook interface in your default web browser.

---

### 3. Play Around & Experiment

Once open in your browser:
- Adjust the **Epochs per Phase** slider to control training duration.
- Adjust **MNIST Subset Proportion** for fast quick-runs or thorough full-dataset runs.
- Click **`Re-run Experiment`** to re-train both models live and visualize updated accuracy curves and metrics side by side.
