# US Transportation Safety Analysis

**Thesis (LaTeX source):** [`thesis/`](thesis/) — [`thesis.tex`](thesis/thesis.tex)

Multi-Criteria Decision Making (MCDM) framework for evaluating and clustering US state-level transportation safety performance across 2010, 2013, 2016, 2019, and 2022.

**Core model:** IDOCRIW (weighting) → DOBI (aggregation) → DPC (clustering, enhanced with KNN-density and adaptive thresholding)

---

## Research Overview

Full paper title: *Innovative IDOCRIW-DOBI-DPC model with KNN-Density and Adaptive Thresholding: An application to transport safety planning for US States* (author: Xiuyuan Lu). Source and figures live in [`thesis/`](thesis/).

### Background & Motivation

Decision-making is central to management, and Multiple Criteria Decision Making (MCDM) provides a stable framework for real-world problems—including road safety, where robust and well-justified decisions are essential. Choosing among MCDM methods remains difficult because of input sensitivity and model uncertainty.

Prior work on MCDM in road safety leaves several gaps:

1. No universal evaluation framework for diverse decision contexts
2. Most studies cover single countries or small regions, not cross-national regulatory and cultural diversity
3. Few studies unify weighting, aggregation, and clustering in one pipeline (grouping, deconstruction, and decomposition are often treated separately)
4. Classic DPC relies on manual center selection and is noise-sensitive on small-to-medium datasets

This project addresses those gaps with an **IDOCRIW–DOBI–DPC** pipeline enhanced by **KNN-density** and **adaptive thresholding (KAT-DPC)**.

### Contributions

- A set of **regional Safety Performance Indicators (SPIs)** for comprehensive US road-safety assessment
- **DPC enhancements** (KNN-based density + adaptive thresholds) that improve automation and noise handling
- **Policy-oriented insights** for US states to design data-driven, context-aware safety strategies

### Paper Structure

| Section | Topic |
|---------|-------|
| 1 | Introduction |
| 2 | Literature review (institutional capacity, MCDM procedure, research gaps) |
| 3 | Data (index construction, sources, cleaning) |
| 4 | Methodology (IDOCRIW, DOBI, KAT-DPC, full pipeline) |
| 5 | Results & robustness analysis |
| 6 | Practical guidance (indicator dynamics, de-construction, de-composition, benchmarking) |
| 7 | Conclusions & future work |

### Safety Performance Indicators (SPIs)

Eight indicators **X1–X8** evaluate each US state (50 states + DC). **X1–X7** are benefit criteria; **X8** is a cost criterion (handled via inverse transformation during normalization). The index hierarchy is documented in [`thesis/index.png`](thesis/index.png) (also editable via [`thesis/index.drawio`](thesis/index.drawio)).

### Data Sources

Primary source: NHTSA *Traffic Safety Facts* (motor vehicle crash data). Additional indicators draw on U.S. Bureau of Transportation Statistics and NHTSA research/testing databases. Raw files are in [`data/raw/`](data/raw/), including:

- `US Transport Data_2010-2022.xlsx` — consolidated multi-year workbook
- Year-specific extracts (`2010_raw.xlsx`, …, `2022_raw.xlsx`)
- NHTSA annual reports (PDF)

Suggested citation for the 2022 report:

> National Center for Statistics and Analysis. (2024, December). *Traffic safety facts 2022: A compilation of motor vehicle traffic crash data* (Report No. DOT HS 813 656). National Highway Traffic Safety Administration.

### Methodology

#### IDOCRIW (Integrated Determination of Objective CRIteria Weights)

Hybrid objective weighting (Zavadskas & Podvezko, 2016) combining **Entropy** and **CILOS** to capture both data dispersion and criterion interdependencies. Steps:

1. Build the decision matrix (alternatives × criteria)
2. Vector-normalize the matrix
3. Compute entropy weights
4. Build a square benchmark matrix from best-performing alternatives per criterion
5. Compute impact losses for non-optimal alternatives
6. Solve the weight system for CILOS (interdependency-aware) weights
7. Combine entropy and CILOS into final weights

#### DOBI (DOmbi Bonferroni)

Nonlinear MCDM aggregation using Dombi norms and Bonferroni mean operators. Handles criterion interdependencies and risk preference via adjustable parameters; produces integrated scores **ℜ** ranked in descending order. Compared with TOPSIS/MABAC, DOBI preserves proportionality and is more robust to extreme values.

#### KAT-DPC (DPC + KNN-Density + Adaptive Thresholding)

Density Peaks Clustering (Rodriguez & Laio, 2014) finds cluster centers as high-density, well-separated points. Original DPC weaknesses—manual center picking and cutoff-distance sensitivity—are mitigated by:

- **KNN-based density:** ρᵢ = 1 / mean(dᵢ, KNN)
- **Delta (separation):** minimum distance to a higher-density point (or max distance if none exists)
- **Automated center selection:** γᵢ = ρᵢ × δᵢ, elbow detection via KneeLocator
- **K-means initialization** for assignment, with reassignment of unclustered points to nearest centers

#### Full Pipeline (6 Steps)

1. **Construct** the decision matrix A (m alternatives × n criteria)
2. **Normalize** via vector normalization (inverse transform for cost criterion X8)
3. **Weight** criteria with IDOCRIW (ωⱼ from combined entropy + CILOS weights)
4. **Aggregate** performance with DOBI (Dombi–Bonferroni averaging + geometric functions → integrated ℜᵢ)
5. **Estimate density** with KNN
6. **Cluster** via automated γ-based center selection and K-means assignment

Implementation: see scripts in [`src/`](src/) (especially `weighting_IDOCRIW.py`, `aggregating_DOBI.py`, `clustering_DPC_remodified.py`).

### Key Results

**Ranking (2016, 2019, 2022):** California, Texas, and Florida consistently rank among the top states; DC, Alaska, and Vermont persistently rank lowest with limited improvement over time.

**Grouping:** States stratify into stable high performers (e.g., CA, FL, TX, IL, PA) and stable low performers (e.g., AK, DE, DC, HI, VT). Some states (e.g., UT, VA, WV) show significant group-assignment volatility across years.

**Robustness:** Sensitivity tested across normalization methods, weighting schemes, and aggregation alternatives—covering initial sensitivity, intermediate uncertainty, and transverse stability for both rankings and groupings. Outputs in [`outputs/robustness/`](outputs/robustness/).

---

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
