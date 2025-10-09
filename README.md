# 🔥 Smoke Mapping for Prescribed Fires

This repository hosts open-source code, workflows, and datasets developed as part of **_Satellite Remote Sensing of Prescribed Burns in Canada_**.  
Content will be released incrementally throughout the project timeline (2025–2028).

---

## 📖 Overview

The goal of this project is to build transparent, reproducible pipelines linking **satellite observations**, **emission estimates**, and **dispersion forecasts** to map prescribed-fire smoke and its downwind impacts.

The workflow follows an end-to-end chain:

> **(A)** Satellite ignition context →  
> **(B)** Hourly PM₂.₅ emissions →  
> **(C)** Peak 3-hour mean surface concentrations →  
> **(D)** Exceedance-hours ≥ 25 µg m⁻³  

Panel A imagery is sourced directly from [USGS EarthExplorer](https://earthexplorer.usgs.gov/).  
Panels B–D are generated from the scripts in this repository.

---

## ⚙️ Installation

This project uses standard Python scientific packages:

```bash
pip install numpy pandas matplotlib xarray scipy
