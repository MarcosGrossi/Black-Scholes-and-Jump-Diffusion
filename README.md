# Jump-Diffusion vs Black-Scholes: Option Pricing Analysis

## Overview
This repository contains the code used in my thesis, which evaluates the viability of the **jump-diffusion model** as an alternative to the **Black-Scholes model** in explaining the implied volatility puzzle in U.S. oil-sector equity options.

The analysis focuses on comparing model performance in pricing options and assessing whether incorporating jump risk improves fit relative to the standard Black-Scholes framework.

## Data Access
The data used in this project is **not included** in this repository due to:
- Proprietary restrictions
- File size limitations

### Data Sources
The analysis combines proprietary and publicly available data:

- **WRDS (Wharton Research Data Services)** *(subscription required)*  
  - **CRSP**: stock price and return data  
  - **OptionMetrics Ivy DB US**: equity options data  

- **FRED (Federal Reserve Bank of St. Louis)** *(free access)*  
  - WTI crude oil prices and/or related series used for market context and robustness analysis 

### Important
To run the code in this repository, you must have:
- An active WRDS subscription  
- Valid WRDS credentials  

The data extraction scripts for WRDS use a combination of **Python and SQL queries**.

## Methodology
- Option prices are estimated under both the **Black-Scholes** and **jump-diffusion** models  
- Parameters are obtained using a **cross-sectional non-linear least squares (NLLS)** procedure  
- The objective is to minimize pricing residuals between model-implied and observed market prices  

## Repository Structure

### Notebooks
- `ret_analysis.ipynb`  
  Analysis of underlying log-returns, including jump behavior and volatility patterns  

- `opt_analysis.ipynb`  
  Cleaning and exploration of options data, including variables such as moneyness and maturity  

- `price_estimation.ipynb`  
  Implementation of Black-Scholes and jump-diffusion pricing models, along with cross-sectional parameter estimation  

- `robust_check1.ipynb`  
  First robustness check on model performance  

- `robust_check2.ipynb`  
  Second robustness check on model performance  

### Python Files
- `helper.py`  
  Contains helper functions used across notebooks (e.g., pricing routines, data processing utilities)

- `user_name.py` *(not included in repository)*  
  Stores sensitive credentials (e.g., WRDS username).  
  Users must create this file locally to run the code.

## Requirements
To run the notebooks, you will typically need:

- Python 3.x  
- Common scientific libraries:
  - `numpy`
  - `pandas`
  - `scipy`
  - `matplotlib`
- WRDS Python package (`wrds`)
- Jupyter Notebook or JupyterLab  

## Notes
- File paths and WRDS queries may need to be adapted to your local environment  
- Sensitive credentials (e.g., WRDS username) should **not** be stored in the repository  
- Large datasets should be stored locally and excluded via `.gitignore`  

## Reference
For full details on the theoretical framework, data construction, and empirical results, see the thesis:

[Jump-Diffusion Dynamics in Oil-Sector Equity Options](https://theses.lib.sfu.ca/show/8959)

## Disclaimer
This repository is intended for academic and research purposes only. Data access is subject to WRDS licensing agreements.
