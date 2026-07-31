# US Transportation Safety Analysis

Multi-Criteria Decision Making (MCDM) framework for evaluating and clustering US state-level transportation safety performance across 2010, 2013, 2016, 2019, and 2022.

**Core model:** IDOCRIW (weighting) → DOBI (aggregation) → DPC (clustering, enhanced with KNN-density and adaptive thresholding)

## Repository Structure

```
├── thesis/                  # Paper (LaTeX source)
├── src/                     # Analysis pipeline (Python)
├── data/
│   ├── raw/                 # Original transport safety data
│   └── processed/           # Normalized, weighted, aggregated, and grouped results
├── outputs/
│   ├── figures/             # Generated charts (clustering, decomposition, deconstruction)
│   └── robustness/          # Robustness analysis outputs
└── docs/
    ├── improvement/         # Algorithm improvement notes (DOBI, DPC)
    └── presentations/       # Slide decks
```

## Analysis Pipeline

Scripts in `src/` follow the MCDM workflow:

| Stage | Scripts | Description |
|-------|---------|-------------|
| Preprocessing | `raw_data_labeling.py`, `preprocess.py` | Label raw indicators, combine normalization + weighting |
| Normalization | `normalization_*.py` | Vector, MinMax, Z-score normalization |
| Weighting | `weighting_*.py` | IDOCRIW, entropy, CILOS objective weights |
| Aggregation | `aggregating_*.py` | DOBI, TOPSIS, VIKOR, RSR, ARLON |
| Clustering | `clustering_*.py` | K-means, DBSCAN, HDBSCAN, OPTICS, DPC (original + modified) |
| Grouping | `grouping_*.py` | Collect cluster results across methods |
| Decomposition | `decomposition_*.py`, `decomoposition_indicators.py` | Radar charts by group / indicator |
| Deconstruction | `deconstruction_*.py` | Performance ranking and temporal change |
| Robustness | `*_robustness.py` | Sensitivity analysis across methods |

Run scripts from the `src/` directory:

```bash
cd src
python normalization_vector.py
python weighting_IDOCRIW.py
python aggregating_DOBI.py
python clustering_DPC_remodified.py
# ...
```

## Data

- **Raw data:** NHTSA Traffic Safety Facts (2010–2022), 8 Safety Performance Indicators (X1–X8) for 50 US states + DC
- **Years analyzed:** 2010, 2013, 2016, 2019, 2022
- **Processed data:** ~275 Excel files covering all normalization × weighting × aggregation combinations

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Outputs

Pre-generated figures are in `outputs/figures/`:

- `DBSCAN/`, `DPC/`, `HDBSCAN/`, `OPTICS/`, `kmeans/` — clustering visualizations
- `decompose_group/`, `decompose_indicator/` — group/indicator radar charts
- `deconstruct_change/`, `deconstruct_performance/` — temporal and ranking analysis

Robustness analysis results are in `outputs/robustness/`.

## Author

Xiuyuan Lu
