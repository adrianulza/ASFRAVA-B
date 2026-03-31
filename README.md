# ASFRAVA-B
[![Latest Release](https://img.shields.io/github/v/release/adrianulza/ASFRAVA-B?label=Download%20Latest&sort=semver)](https://github.com/adrianulza/ASFRAVA-B/releases/latest)

## 📖 Overview
**ASFRAVA-B:** Automated Seismic Fragility and Vulnerability Assessment for Buildings

**Citation:**
- Ulza, A. et al. (2025). *Automated Seismic Fragility and Vulnerability Assessment for Buildings (ASFRAVA-B): Integrating Probabilistic Seismic Design into Performance-Based Engineering Practices*, International Journal of Disaster Risk Reduction, [https://doi.org/10.1016/j.ijdrr.2025.105679](https://doi.org/10.1016/j.ijdrr.2025.105679)

---

## ✨ What's New in v1.1.0

- **Multiple Intensity Measure (IM) methods** — choose from PGA, Sa(T), or Sa(avg) via a pre-analysis workflow setup dialog
- **Sa(T)** — scale ground motion records to spectral acceleration at a user-specified (or auto-derived) structural period
- **Sa(avg)** — scale to average spectral acceleration over the 0.2T–1.5T period band (10 periods, 5% damping)
- **J-MLE fragility fitting** — joint maximum likelihood estimation with a shared dispersion parameter; mathematically guarantees non-crossing fragility curves
- **IDA visualization** — incremental dynamic analysis curves 

---

## 🚀 Quick Start

### 📥 Downloading the App (Windows) for Engineer

Download the latest executable from the [Releases](https://github.com/adrianulza/ASFRAVA-B/releases/latest) page:

- **[ASFRAVA-B.exe](https://github.com/adrianulza/ASFRAVA-B/releases/download/v.1.1.0/ASFRAVA-B.zip)** *(after unzip, ~280 MB)*

### 🖥️ Running the App
1. Download and unzip the file.
2. Double-click `ASFRAVA-B.exe` to start the application.

---

## 🛠️ Developer Guide

### 📋 Requirements:
- Python 3.11+
- Windows 10/11

### 📦 Installation:
```bash
# create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
