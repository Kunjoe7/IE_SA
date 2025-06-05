# Interactive and Explainable Survival Analysis

This repository contains all code and processed data required for Interactive and Explainable Survival Analysis experiments. It includes:

- Five processed datasets (folders):  
  - `FLCHAIN`  
  - `GABS`  
  - `METABRIC`  
  - `NWTCO`  
  - `TCGA_Task3`  

- For each dataset folder:  
  - The main algorithm implementation (our proposed interactive and explainable survival model).  
  - Comparison experiment scripts (baseline and competing methods).  
  - Any database files required for data loading.  

All datasets have been pre-processed and are ready for direct use. Running the Python script inside a given dataset folder will automatically execute both the main algorithm and all comparison experiments for that dataset.

---

## Requirements

- **Python (December 2024 mainstream version)**:  
  - Python 3.10 or later (tested on 3.10–3.11)  
- **Key Packages** (install via `pip install <package>`):  
  - `numpy`  
  - `pandas`  
  - `scikit-learn`  
  - `lifelines`  
  - `torch` (for any deep‐learning components)  
  - `torchvision` (if image or tensor utilities are used)  
  - `matplotlib` (for plotting results)  
  - `scipy` (for statistical utilities)  
