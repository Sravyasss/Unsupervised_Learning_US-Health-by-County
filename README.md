# Unsupervised Learning - US Health by County
*Authors:** Badamgarav Battushig, Edwin Okwor, Sravyasri Murala  
**Course:** DATA 5322 — Statistical Machine Learning 2  
**Date:** May 2026

***

## Overview

Public health burden is not spread evenly across the United States, and the patterns are not always tied to geography in obvious ways. Two counties in different states can look more similar to each other on chronic disease and access measures than two neighboring counties in the same state.

This project applies unsupervised machine learning to discover natural community health profiles across all 3,144 U.S. counties using the CDC PLACES 2025 dataset, which provides model-based estimates of 40 health measures per county.

**Research question:** What natural groupings emerge when each U.S. county is treated as a point in 40-dimensional health space?

***

## Dataset

**Source:** [CDC PLACES: Local Data for Better Health, County Data 2025 Release](https://data.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-County-Data-20/swc5-untb)  
**Easy access:** Hosted on Hugging Face — [`kemba/oak-CDCHealthData`](https://huggingface.co/datasets/kemba/oak-CDCHealthData) *(notebook streams directly from here — no local file needed)*

| Item | Value |
|------|-------|
| Counties | 3,144 |
| Health measures | 40 |
| Missing entries | 11,144 |
| Missing percent | 8.86% |
| Survey years | 2022 and 2023 (varies by measure) |
| Geographic coverage | 50 states + Washington D.C. + 1 U.S. territory |

The 40 measures span six categories:
- **Disability** - mobility, cognition, hearing, vision, self-care, independent living
- **Health Outcomes** - diabetes, stroke, COPD, coronary heart disease, obesity, depression
- **Health Risk Behaviors** - smoking, binge drinking, physical inactivity, sleep
- **Health Status** — fair/poor self-rated health, mental distress, physical distress
- **Health-Related Social Needs** - food insecurity, housing insecurity, transportation barriers, loneliness
- **Prevention** - dental visits, cancer screening, cholesterol screening, mammography

***

## Methods

Four unsupervised learning methods were applied in sequence:

### 1. Matrix Completion (Algorithm 12.1)
Missing values were filled using iterative low-rank SVD (ISLR Ch. 12.5.2). The rank parameter M was tuned via held-out RMSE, with M=7 selected as optimal.

| Method | Held-out RMSE |
|--------|--------------|
| Mean imputation | 1.0037 |
| Median imputation | 1.0147 |
| k-NN (k=5) | 0.3386 |
| **Algorithm 12.1 (M=7)** | **0.4056** |

Algorithm 12.1 was chosen over k-NN because it fills missing entries using low-rank SVD structure, making the completed matrix directly consistent with the assumptions of PCA in the next step.

### 2. PCA / SVD
PCA was run on the standardized completed matrix. The first 6 principal components explain 90% of county-to-county variation.

| Component | Individual Variance | Cumulative Variance |
|-----------|--------------------|--------------------|
| PC1 | 60.4% | 60.4% |
| PC2 | 12.1% | 72.4% |
| PC3 | 9.8% | 82.2% |
| PC4 | 3.8% | 86.1% |
| PC5 | 3.1% | 89.2% |
| PC6 | 2.1% | 91.2% |

PC1 captures an overall chronic disease and disability burden axis. Measures like mobility disability, stroke, and self-care disability load positively (~0.19–0.20), while dental visits and colorectal cancer screening load negatively. PC2 separates counties where cancer and depression rates are elevated from counties where lack of health insurance is the dominant factor.

### 3. K-Means Clustering
Clustered on the first 6 PC scores using `n_init=20` random restarts. K=4 was selected based on the elbow method and silhouette analysis.

| Cluster | Size | Profile | Geography |
|---------|------|---------|-----------|
| 0 | 992 | Healthiest — low disease burden, high preventive care rates | Upper Midwest, Great Plains |
| 1 | 1,148 | Near-average — close to national average across all measures | Mid-continent |
| 2 | 751 | Moderate burden — consistently worse than national average | South, Appalachia |
| 3 | 253 | Highest burden — extreme social needs, lowest preventive care | Deep South (MS, AL, LA, GA) |

### 4. Hierarchical Clustering (Ward Linkage)
All four linkage methods were tested at K=4. Ward linkage was selected for its balanced, interpretable groups.

| Linkage | Silhouette | ARI vs K-Means | Smallest Cluster | Largest Cluster |
|---------|-----------|---------------|-----------------|----------------|
| Average | 0.460 | 0.066 | 5 | 2,999 |
| Complete | 0.372 | 0.099 | 20 | 2,891 |
| Single | 0.248 | 0.001 | 1 | 3,140 |
| **Ward** | **0.143** | **0.453** | 286 | 1,146 |

***

## Key Findings

- **One dominant axis:** PC1 alone explains 60.4% of all county-to-county variation — the 40 health measures are not 40 independent signals.
- **Low-rank structure:** Only 3 components needed for 80% of variance; 6 for 90%.
- **Four interpretable archetypes** consistently identified by both K-means and hierarchical clustering (ARI = 0.453).
- **Both methods agree most on the extremes** — the healthiest and most burdened counties are the most stable across methods. Disagreement concentrates in the near-average middle.
- **County health profiles cross state lines.** Texas alone has counties in all four clusters. Mississippi, Alabama, and Louisiana counties share the same health archetype regardless of state boundaries.
- **The most burdened cluster (n=253)** has stroke rates, food insecurity, transportation barriers, and utility shutoff threats all more than 2 standard deviations above the national average — and the lowest dental visit and cancer screening rates of any group.

***

## Repository Structure

```
├── README.md
├── Blog_Post.ipynb               # Main analysis notebook (data prep → matrix completion → PCA/SVD → K-means → hierarchical clustering)
├── Blog_Post.html                # Rendered blog post (HTML export)
└── figures/
    ├── fig_missingness.png
    ├── fig_matcomp_M_tuning.png
    ├── fig_matcomp_baseline_comparison.png
    ├── fig_scree_plot.png
    ├── fig_loadings_heatmap.png
    ├── fig_pc1_pc2_scatter.png
    ├── fig_biplot.png
    ├── fig_kmeans_tuning.png
    ├── fig_kmeans_pc_scatter.png
    ├── fig_kmeans_cluster_profiles.png
    ├── fig_hierarchical_dendrograms.png
    ├── fig_kmeans_vs_hierarchical_pc_space.png
    ├── fig_hierarchical_cluster_profiles.png
    └── fig_cluster_size_comparison.png
```

> **Note:** The raw CDC PLACES data file is large and not stored in this repo. The notebook streams it directly from Hugging Face — no download required.

***

## How to Run

1. **Clone this repository**
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy plotly kneed requests
   ```

3. **Open and run the notebook**
   ```bash
   jupyter notebook Blog_Post.ipynb
   ```
   Or open directly in **Google Colab** — the notebook will automatically stream the CDC PLACES dataset from Hugging Face and generate all figures.

> Run all cells in order. No local data file is needed.

***

## References

Centers for Disease Control and Prevention. (2025). *PLACES: Local Data for Better Health, County Data 2025 Release* [Dataset]. https://data.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-County-Data-20/swc5-untb

James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An introduction to statistical learning with applications in R* (2nd ed.). Springer.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830.

Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., … SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, *17*, 261–272. https://doi.org/10.1038/s41592-019-0686-2

Centers for Disease Control and Prevention. *Behavioral Risk Factor Surveillance System (BRFSS)*. https://www.cdc.gov/brfss
