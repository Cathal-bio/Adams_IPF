# Adams Aberrant Basaloid Cell Analysis Pipeline

## Quick Start

Simply run the pipeline without any arguments:

```bash
python Adams_visualisation.py
```

## What it does

The pipeline automatically:

1. **Searches for existing data** in local directories
2. **Creates sample data** if no real data is found
3. **Runs both analysis methods**:
   - Adams et al. 2020 exact method
   - Extended scoring method with multiple markers
4. **Generates comprehensive visualizations**
5. **Saves all results** to `aberrant_basaloid_results/` directory

## Output Structure

```
aberrant_basaloid_results/
├── adams_method/
│   ├── plots/                          # All Adams method visualizations
│   ├── adams_report.txt               # Human-readable summary
│   ├── adams_statistics.json         # Analysis statistics
│   └── adams_aberrant_cells_metadata.csv
├── extended_scoring/
│   ├── plots/                          # Extended method visualizations
│   ├── extended_statistics.json      # Extended analysis stats
│   └── pca_components.csv            # PCA component weights
└── comparison/
    ├── overlapping_cells.csv         # Cells identified by both methods
    └── comparison_statistics.json    # Method comparison stats
```

## Using Your Own Data

Place your single-cell data file (.h5ad format) in one of these locations:

- `./cache/05_final.h5ad`
- `./data/05_final.h5ad`
- `./05_final.h5ad`

The pipeline will automatically detect and use your data.

## Requirements

- Python 3.8+
- scanpy
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn

## Sample Data

If no real data is found, the pipeline creates realistic sample data with:

- 5,000 cells and 2,000 genes
- Proper expression patterns for marker genes
- Realistic cell type and sample metadata
- UMAP embedding for visualization

## Analysis Methods

### Adams Method (2020)
- Uses KRT17, TP63, VIM as positive markers
- Uses KRT5 as negative marker
- Identifies cells above 99th percentile threshold
- Includes senescence analysis with CDKN1A, CDKN2A

### Extended Scoring Method
- Comprehensive marker database (30+ genes)
- Multiple scoring strategies
- Consensus approach combining different thresholds
- Statistical outlier detection
- Gaussian mixture model clustering

## Results

The pipeline identifies aberrant basaloid cells and generates:

- **Score distributions and correlations**
- **UMAP plots** showing aberrant cell locations
- **Cell type and sample breakdowns**
- **Statistical comparisons** between methods
- **Expression pattern analysis**
- **Comprehensive reports and summaries**

## Troubleshooting

If you encounter issues:

1. Check that all required packages are installed
2. Ensure your data file is in .h5ad format
3. Verify gene symbols are properly mapped
4. Check that the working directory is writable

For questions or issues, please refer to the script documentation.
