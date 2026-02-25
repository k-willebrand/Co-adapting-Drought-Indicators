# Co-adapting-Drought-Indicators

This repo contains code and data accompanying the manuscript:

"Co-adapting drought indicators and management to climate change"  

Keani Willebrand $^{1*}$, Sebastian Vicuna $^2$, Jorge Gironás $^{3,4,5,6}$, Oscar Melo $^7$, Danny Sallis $^{1}$, Sarah Fletcher $^{1,8*}$

$^1$ Department of Civil and Environmental Engineering, Stanford University, Stanford, CA, USA

$^2$ Department of Hydraulics and Environmental Engineering, Pontificia Universidad Católica de Chile, Santiago, Chile

$^3$ Departamento de Ingeniería Hidráulica y Ambiental, Pontificia Universidad Católica de Chile, Santiago, Chile

$^4$ Centro Interdisciplinario de Cambio Global, Pontificia Universidad Católica de Chile, Santiago, Chile.

$^5$ Centro Interdisciplinario de Cambio Global, Pontificia Universidad Católica de Chile, Santiago, Chile.

$^6$ Centro de Desarrollo Urbano Sustentable (CEDEUS ANID/FONDAP/1523A0004), Santiago, Chile.

$^7$ Department of Agricultural Economics, Pontificia Universidad Católica de Chile, Santiago, Chile

$^8$ Woods Institute for the Environment, Stanford University, Stanford, CA, USA

\* Corresponding authors
 
Submitted to *Nature Climate Change* (2026)

---

## Overview

This repository contains the code and processed data required to reproduce the main results and figures presented in the manuscript.

The project investigates:
- How drought indicators used in water management must adapt under climate change to remain effective in triggering response actions.
- Its key contribution is a simulation–optimization framework that jointly designs drought indicators and management responses to maximize water supply reliability while minimizing economic cost impacts under future climate scenarios.
- The approach integrates a river-basin simulation using a PYWR model of Chile’s Maipo Basin and multi-objective optimization (Borg MOEA) to evaluate alternative drought indicator performance in drought plans across plausible climate futures.

---

## Repository Structure

## Repository Structure

The repository is organized into three primary components:

├── MAIPO_PYWR/        # Core simulation and optimization framework  
├── MAIPO_CC/          # Climate downscaling and scenario development  
└── MOEAFramework/     # Post-processing and convergence diagnostics  

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
Optimization scripts are stored in the `dps_BORG` subfolder under files starting with `example_sim_opt` and specifying
whether management plans are optimized over the near-term (20yrA) 2020-2040 planning horizon or the end-of-century (20yrB) 2080-2100
planning horizon conditioned on alternative drought indicators (e.g., SPI-3).

The main figures from the manuscript can be recreated in `plots_Manuscript_NatureV1_clean_Validation_pooled_seeds.py`

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
