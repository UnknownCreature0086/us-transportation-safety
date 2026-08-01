# US Transportation Safety Analysis

**Thesis:** [`thesis/thesis.pdf`](thesis/thesis.pdf)

Multi-Criteria Decision Making (MCDM) framework for evaluating and clustering US state-level transportation safety performance. Core pipeline: **IDOCRIW** (weighting) → **DOBI** (aggregation) → **KAT-DPC** (clustering with KNN-density and adaptive thresholding).

Full title: *Innovative IDOCRIW-DOBI-DPC model with KNN-Density and Adaptive Thresholding: An application to transport safety planning for US States* (Xiuyuan Lu).

---

## Research Overview

### Motivation

MCDM is widely used for road-safety planning, but method choice remains difficult because of input sensitivity and model uncertainty. Prior work leaves several gaps:

1. No universal evaluation framework across diverse decision contexts
2. Most studies cover single countries or small regions
3. Weighting, aggregation, and clustering are rarely unified (grouping / deconstruction / decomposition often treated separately)
4. Classic DPC depends on manual center selection and is noise-sensitive on small-to-medium datasets

This project addresses those gaps with an **IDOCRIW–DOBI–KAT-DPC** pipeline.

### Contributions

- Regional **Safety Performance Indicators (SPIs)** for US road-safety assessment
- **KAT-DPC**: KNN-based density, knee-point center selection, noise-adaptive reassignment, and K-means refinement
- Policy-oriented insights for benchmarking and context-aware safety strategies

### Paper Structure

| Section | Topic | Status in draft |
|---------|-------|-----------------|
| 1 | Introduction | Complete |
| 2 | Literature review (institutional capacity; MCDM uncertainty; DPC) | Complete |
| 3 | Data (SPI index, NHTSA/BTS sources) | Complete |
| 4 | Methodology (IDOCRIW, DOBI, KAT-DPC, full 9-step pipeline) | Complete |
| 5 | Results & robustness (ranking, grouping, sensitivity) | Complete |
| 6 | Practical guidance (SPI dynamics, deconstruction, decomposition, benchmarking) | Complete |
| 7 | Conclusion, contributions, limitations, future work | Complete |
| — | Abstract & References | Complete |

---

## Safety Performance Indicators (SPIs)

Eight indicators (**X1–X8**) for 50 US states + DC:

| Code | Indicator | Type |
|------|-----------|------|
| X1 | Highway safety expenditures (HSE) | Benefit |
| X2 | Registered vehicles (RV) | Benefit |
| X3 | Licensed drivers (LD) | Benefit |
| X4 | Vehicle miles traveled (VMT) | Benefit |
| X5 | Road length (RL) | Benefit |
| X6 | Road condition (RC) | Benefit |
| X7 | Seat belt usage (SB) | Benefit |
| X8 | Fatality rate per 100M VMT (FR) | Cost |

Index hierarchy: [`outputs/figures/index/index.png`](outputs/figures/index/index.png).

### Data Sources

- **X1–X6:** U.S. Bureau of Transportation Statistics (BTS)
- **X7:** NHTSA *Seat Belt Use — Use Rates in the States and Territories*
- **X8:** NHTSA crash / fatality databases

Raw files live in [`data/raw/`](data/raw/) (multi-year workbook, year extracts, NHTSA annual reports). Main result years in the thesis: **2016, 2019, 2022** (pipeline also covers 2010 and 2013).

---

## Methodology

### IDOCRIW (Zavadskas & Podvezko, 2016)

Hybrid objective weighting combining **Entropy** and **CILOS**: vector normalization → entropy weights → impact-loss / CILOS weights → combined ωⱼ.

### DOBI (DOmbi Bonferroni)

Nonlinear aggregation via Dombi norms and Bonferroni means. Produces weighted averaging (Z1) and geometric (Z2) scores, then an integrated ℜᵢ ranked descending. Adjustable risk parameters (ψ₁, ψ₂, ζ, δ); more robust to extremes than TOPSIS/AHP-style linear schemes.

### KAT-DPC

Enhancements over Rodriguez & Laio (2014) DPC:

- **KNN density:** ρᵢ = 1 / mean(dᵢ,ₖₙₙ) (k = 5)
- **Delta:** minimum distance to a higher-density point
- **Automated centers:** γᵢ = ρᵢ × δᵢ, elbow via KneeLocator (target ~8–9 clusters)
- **K-means refinement** with density-aware noise reassignment
- Groups relabeled by average DOBI score (Group 1 = best)

### Full Pipeline (9 Steps)

1. Build decision matrix A
2. Vector-normalize (inverse transform for cost criterion X8)
3. Weight with IDOCRIW
4. Aggregate with DOBI → rank by ℜᵢ
5. KNN density estimation
6. Delta calculation
7. Automated center selection
8. K-means assignment / noise reassignment
9. Reorder cluster labels by mean DOBI score

Implementation: [`src/`](src/) (`weighting_IDOCRIW.py`, `aggregating_DOBI.py`, `clustering_DPC_remodified.py`, …).

---

## Key Results

**Ranking (2016 / 2019 / 2022):** California, Texas, and Florida stay in the top tier; DC, Alaska, and Vermont remain near the bottom with little improvement.

**Grouping:** Stable high performers include CA, FL, TX, IL, PA; stable low performers include AK, DE, DC, HI, VT. Some states (e.g. UT, VA, WV) show large group volatility across years.

**Robustness:**

- *Initial sensitivity* — MinMax / Vector / Z-score normalization: correlations mostly > 0.90
- *Intermediate uncertainty* — IDOCRIW / CILOS / Entropy: correlations ≳ 0.95
- *Transverse stability* — DOBI vs TOPSIS / VIKOR: strong agreement on top/bottom states
- *Grouping* — DPC vs KAT-DPC vs k-means, with V-measure tables

**Practical guidance:** SPI time paths, DOBI deconstruction, 2016→2022 score-change decomposition, and within-group benchmarking templates.

---

## Repository Structure

```
├── thesis/
│   └── thesis.pdf          # Thesis PDF
├── src/                    # Analysis pipeline (Python)
├── data/
│   ├── raw/                # Original transport safety data
│   └── processed/          # Normalized / weighted / aggregated / grouped results
├── outputs/
│   ├── figures/            # Charts used in the thesis
│   └── robustness/         # Sensitivity analysis outputs
└── docs/
    ├── improvement/        # Algorithm notes (DOBI, DPC)
    └── presentations/      # Slide decks
```

## Analysis Pipeline

| Stage | Scripts | Description |
|-------|---------|-------------|
| Preprocessing | `raw_data_labeling.py`, `preprocess.py` | Label indicators; combine normalization + weighting |
| Normalization | `normalization_*.py` | Vector, MinMax, Z-score |
| Weighting | `weighting_*.py` | IDOCRIW, entropy, CILOS |
| Aggregation | `aggregating_*.py` | DOBI, TOPSIS, VIKOR, RSR, ARLON |
| Clustering | `clustering_*.py` | K-means, DBSCAN, HDBSCAN, OPTICS, DPC (+ modified) |
| Grouping | `grouping_*.py` | Collect cluster labels across methods |
| Decomposition | `decomposition_*.py`, `decomoposition_indicators.py` | Radar charts by group / indicator |
| Deconstruction | `deconstruction_*.py` | Performance ranking and temporal change |
| Robustness | `*_robustness.py` | Cross-method sensitivity |

```bash
cd src
python normalization_vector.py
python weighting_IDOCRIW.py
python aggregating_DOBI.py
python clustering_DPC_remodified.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Author

Xiuyuan Lu
