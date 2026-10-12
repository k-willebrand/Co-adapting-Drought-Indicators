# Co-adapting-Drought-Indicators

This repository contains code and processed data accompanying the manuscript:

**“Co-adapting drought indicators and management to climate change”**  
*Submitted to* *Nature Water* (2026)

**Authors:**  
Keani Willebrand<sup>1*</sup>, Mofan Zhang<sup>1</sup>, Sebastian Vicuña<sup>2,3</sup>, Jorge Gironás<sup>2,3,4</sup>, Oscar Melo<sup>3,5</sup>, Danny Sallis<sup>1</sup>, Sarah Fletcher<sup>1,6*</sup>  

<sup>1</sup> Department of Civil and Environmental Engineering, Stanford University, Stanford, CA, USA

<sup>2</sup> Departamento de Ingeniería Hidráulica y Ambiental, Pontificia Universidad Católica de Chile, Santiago, Chile.

<sup>3</sup> Centro Interdisciplinario de Cambio Global, Pontificia Universidad Católica de Chile, Santiago, Chile

<sup>4</sup> Centro de Desarrollo Urbano Sustentable (CEDEUS), Santiago, Chile

<sup>5</sup> Department of Agricultural Economics, Pontificia Universidad Católica de Chile, Santiago, Chile

<sup>6</sup> Woods Institute for the Environment, Stanford University, Stanford, CA, USA

\* Corresponding authors  

---

## Overview

This repository provides the code and data required to reproduce the main results and figures from the manuscript.

The project develops a **simulation–optimization framework** to evaluate how drought indicators and water management strategies must co-adapt under climate change.

Key contributions:
- Joint design of **drought indicators** and **management policies**
- Evaluation across **plausible future climate scenarios**
- Integration of:
  - A river-basin simulation model using Pywr  
  - Multi-objective optimization using the Python wrapper for Borg MOEA  
- Assessment of trade-offs between:
  - Water supply reliability  
  - Economic impacts  

---

## Important Notes

- This repository integrates multiple external tools and scientific models.
- Initial setup can take **2–15 hours**, depending on your system and experience.
- Full-scale experiments were run on a **high-performance computing (HPC) cluster**.
- A **simplified demo** is provided for local machines (see below).
- This version of the repository was developed and tested on a **MacBook Pro with Apple Silicon (M2)**.
- Some scripts contain **absolute file paths**, which must be updated for your local machine.
- Some packages (e.g., Borg MOEA) require **local compilation**.

---

## Repository Structure

```
├── MAIPO_PYWR/        # Core simulation and optimization framework  
├── MAIPO_CC/          # Climate downscaling and scenario generation  
├── MOEAFramework/     # Post-processing and convergence diagnostics  
├── demo/              # Simplified demo for local execution  
```


### MAIPO_PYWR

`MAIPO_PYWR` is the central component of the framework. It contains the data, model configurations, and scripts required to simulate and optimize the Maipo basin water system using the PYWR modeling framework. 

This folder includes:
- Input datasets required to run the model
- Optimization routines (including Borg multi-objective optimization)
- Scripts to generate key figures presented in the manuscript
- Post-simulation analysis tools
- A requirements.txt file to define necessary packages

All core model simulations and optimization experiments described in the manuscript are derived from this directory.
Optimization in parallel was conducted with a high-performance computing (HPC) cluster with Python version 3.9.0 and Open MPI version 5.0.5. 
The optimization script is stored in the `dps_BORG` subfolder under the filename `sim_opt_NATWAT_3obj_2paramGamma_disc0_MIN_DeCost.py`. The script can be configured to specify whether management plans are optimized over the near-term (period A) 2020-2040 planning horizon or the end-of-century (period B) 2080-2100 planning horizon conditioned on alternative drought indicators (e.g., SPI-3).

The main analysis and result figures from the manuscript can be recreated in `plots_Manuscript_NatureV2.py`

---

### MAIPO_CC

`MAIPO_CC` generates the climate forcing scenarios used to drive the optimization framework. It:

- Downscales Global Climate Model (GCM) outputs
- Develops future climate scenarios for analysis
- Implements **unbiased quantile mapping (UQM)** for statistical downscaling

The outputs of `MAIPO_CC` serve as input climate scenarios for the optimization experiments conducted in `MAIPO_PYWR`.

---

### MOEAFramework

`MOEAFramework` supports post-processing and diagnostic analysis of optimization results generated in `MAIPO_PYWR`. Specifically, it:

- Processes output files from Borg multi-objective optimization
- Evaluates convergence across random seeds
- Assesses solution quality and stability
- Supports comparative analysis of optimization performance

This component ensures robustness of the optimization results presented in the manuscript.


### demo

`demo` supports a a simplified example problem setup to the optimization for use on a local desktop.

---

## System Requirements

### Supported Systems
- macOS (Intel and Apple Silicon with x86 emulation)
- Linux (recommended for full experiments)
- Windows (requires additional configuration)

This GitHub repository is configured for MacOS.

### Core Dependencies
- Python 3.9
- Pywr
- Borg MOEA (compiled locally)
- MOEA Framework (Java-based)

### Python Libraries
- numpy, pandas, scipy, matplotlib  
- seaborn, cython, networkx  
- mpi4py (for parallel runs)

Full package versions are provided in:
MAIPO_PYWR_packages.pdf

### Hardware
- **Demo:** standard laptop 
- **Full experiments:** HPC recommended  

---

## Installation Guide

Follow official documentation for setup of required tools:

- Pywr: https://pywr.github.io/pywr-docs/master/install.html  
- Borg MOEA Python Wrapper: https://reedgroup.github.io/Software/BorgMOEA.html  
- MOEA Framework: https://moeaframework.org/  

### Python Environment Setup

```
conda create -n maipo python=3.9
conda activate maipo
pip install -r requirements.txt
```

The file `MAIPO_PYWR_packages.pdf` additionally provides a list of Python packages previously installed locally to run the optimization on a local computer.

Apple Silicon users:
- Create an **x86-based environment** in Anaconda Navigator  
- Install Pywr dependencies within that environment  

### Optional (HPC Only)
- Install OpenMPI and mpi4py for parallel optimization

Estimated Setup Time:  
2–15 hours depending on system configuration.

---

## Quick Start Demo

A simplified demo is provided in the `demo/` folder. The main demo file to run is named `demo_optimization.py`.

### Purpose
The demo demonstrates the optimization workflow on a local machine.

It is designed to:
- Run in ~10 minutes
- Avoid MPI (serial execution only)
- Use reduced datasets

---

### Demo Configuration

The demo includes:
- One drought indicator (i.e., SPI-3)
- One planning horizon (2020–2040)
- Reduced number of climate scenarios for optimization
- 2 random seeds
- 100 function evaluations

The demo is **not expected to converge** and does not reproduce published results. To support ease of readability and reduced script length, `demo_optimization.py` loads the `make_model()` function from `make_model_function.py`.

---

### Running the Demo

Follow the installation guides and Python environment setup for your local computer. Then, run `demo_optimization.py` in Python.

---

### Expected Output

You should see:
- Optimization output files in `demo/outputs_demo/`
- Runtime logs
- Objective values
- Files such as:
  - `.runtime` (optimization progress)
  - `.set` (Pareto solution sets)

Exact results may vary slightly depending on random seeds utilized. You can choose to post-process the results using MOEAFramework. A quick start description of computing performance metrics (e.g., hypervolume) using MOEA Framework can be found in section 4 of this blog post: https://waterprogramming.wpcomstaging.com/2025/02/19/everything-you-need-to-run-borg-moea-and-python-wrapper-part-2/#4-streamlined-tools-for-computing-performance-metrics-eg-hypervolume-using-moeaframework.

---

### Expected Runtime

- The demo is expected to run ~10–15 minutes on a standard laptop  

---

## Workflow Overview

The full workflow consists of:

1. Climate Scenario Generation (`MAIPO_CC`)  
2. Hydrologic Simulation (WEAP model)  
3. Simulation–Optimization (`MAIPO_PYWR`)  
4. Post-processing (`MOEAFramework`)  

The demo focuses on Step 3 only, which is the central component of our framework.

---

## Instructions for Use

This repository can be used to evaluate drought indicators and management policies under user-defined climate scenarios.

### Running the Framework on Your Own Data

1. Prepare climate scenarios  
2. Simulate hydrology (WEAP)  
3. Prepare Pywr inputs  
4. Update file paths  
5. Configure optimization  
6. Run scripts  
7. Post-process results  

---

## Data Availability

Additional large data files are located here:
https://drive.google.com/drive/folders/1pwGsR5cXxacTHKjueLDrl5hS6YS4_gq1

These large data files are also available via the associated Zenodo dataset for this code with the following DOI: 
10.5281/zenodo.23310362

Note: this will be changed from a Google Drive folder to a permanent repository on Zenodo prior to publication.

---

## Reproducibility Notes

- Full experiments were conducted on an HPC cluster using parallel optimization with MPI.
- Results may vary depending on random seeds utilized, and number of function evaluations should be tuned for your specific problem.


---

## Contact

For questions regarding the repository or manuscript, please contact the corresponding authors.