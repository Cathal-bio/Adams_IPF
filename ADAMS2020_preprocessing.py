#!/usr/bin/env python3
"""
Preprocess Adams et al. 2020 raw data from GEO into H5AD format
This script loads the raw data files and creates the required .h5ad files
"""

import os
import gzip
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import mmread
import anndata as ad
from datetime import datetime

os.chdir('/home/cathal/Desktop/working/Single_Cell/scrnaseq_analysis')

def load_adams_raw_data(data_dir):
    """Load the raw Adams et al. data from GEO files"""

    print("Loading raw data files from GEO...")

    # Define file paths
    matrix_file = os.path.join(data_dir, "GSE136831_RawCounts_Sparse.mtx.gz")
    barcodes_file = os.path.join(data_dir, "GSE136831_AllCells.cellBarcodes.txt.gz")
    genes_file = os.path.join(data_dir, "GSE136831_AllCells.GeneIDs.txt.gz")
    metadata_file = os.path.join(data_dir, "GSE136831_AllCells.Samples.CellType.MetadataTable.txt.gz")

    # Check if all files exist
    files = [matrix_file, barcodes_file, genes_file, metadata_file]
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Required file not found: {f}")

    # Load barcodes first
    print("Loading cell barcodes...")
    with gzip.open(barcodes_file, 'rt') as f:
        barcodes = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(barcodes)} cell barcodes")

    # Load and parse gene IDs (single read)
    print("Loading gene IDs...")
    with gzip.open(genes_file, 'rt') as f:
        genes_raw = [line.strip() for line in f if line.strip()]

    # Parse gene file (handle both single and multi-column formats)
    lines = [line.split('\t') for line in genes_raw]
    
    # Debug: Show first few lines
    print(f"First 3 lines of gene file: {genes_raw[:3]}")
    print(f"First parsed line structure: {lines[0] if lines else 'No lines'}")

    # Check if first line is a header
    if lines and (
            lines[0][0].startswith('Ensembl') or lines[0][0].startswith('Gene') or not lines[0][0].startswith('ENS')):
        print("Detected header in gene file, removing...")
        lines = lines[1:]

    if len(lines[0]) > 1:
        # We have both Ensembl IDs and gene symbols
        gene_ids = [line[0] for line in lines]  # Ensembl IDs
        gene_symbols = [line[1] if len(line) > 1 else line[0] for line in lines]  # Gene symbols
        print(f"Gene file has {len(lines[0])} columns: Ensembl IDs and gene symbols")
    else:
        gene_ids = [line[0] for line in lines]
        gene_symbols = gene_ids
        print(f"Gene file has 1 column")

    print(f"Loaded {len(gene_ids)} genes")
    print(f"First 3 gene IDs: {gene_ids[:3]}")
    if gene_ids != gene_symbols:
        print(f"First 3 gene symbols: {gene_symbols[:3]}")

    # Load sparse matrix
    # MTX format is always genes × cells, so transpose unconditionally to get cells × genes
    print("Loading sparse matrix...")
    with gzip.open(matrix_file, 'rb') as f:
        matrix = mmread(f).T.tocsr()  # Read and transpose to cells × genes
    
    print(f"Matrix shape after transpose (cells × genes): {matrix.shape}")

    # Validate dimensions
    if matrix.shape[0] != len(barcodes):
        print(f"Warning: Cell count mismatch!")
        print(f"  Matrix has {matrix.shape[0]} cells")
        print(f"  Barcode file has {len(barcodes)} barcodes")
        if len(barcodes) > matrix.shape[0]:
            print(f"  → Trimming barcode list to match matrix size...")
            barcodes = barcodes[:matrix.shape[0]]
        else:
            print(f"  → ERROR: Matrix has more cells than barcode file!")
            raise ValueError("Matrix has more cells than provided in barcode file")

    if matrix.shape[1] != len(gene_ids):
        print(f"Warning: Gene count mismatch!")
        print(f"  Matrix has {matrix.shape[1]} genes")
        print(f"  Gene file has {len(gene_ids)} genes")
        if len(gene_ids) > matrix.shape[1]:
            print(f"  → Trimming gene list to match matrix size...")
            gene_ids = gene_ids[:matrix.shape[1]]
            gene_symbols = gene_symbols[:matrix.shape[1]]
        else:
            print(f"  → ERROR: Matrix has more genes than gene file!")
            raise ValueError("Matrix has more genes than provided in gene file")

    # Load metadata
    print("\nLoading metadata...")
    metadata = pd.read_csv(metadata_file, sep='\t', compression='gzip')
    print(f"Metadata shape: {metadata.shape}")
    print(f"Metadata columns: {list(metadata.columns)}")

    # Check if cell barcodes match
    if 'CellBarcode' in metadata.columns:
        print("Found CellBarcode column in metadata")
        metadata = metadata.set_index('CellBarcode')
    elif metadata.index.name != 'CellBarcode':
        # Try to match by order
        if len(metadata) == len(barcodes):
            print("Metadata rows match number of cells, using order-based matching")
            metadata.index = barcodes
        else:
            print(f"Warning: Metadata has {len(metadata)} rows but we have {len(barcodes)} cells")
            print("Metadata will be matched by cell barcode where possible")

    # Create AnnData object
    print("\nCreating AnnData object...")

    # Ensure dimensions match
    n_cells_matrix, n_genes_matrix = matrix.shape

    if n_cells_matrix != len(barcodes):
        print(f"Adjusting cell barcodes: matrix has {n_cells_matrix}, barcodes has {len(barcodes)}")
        if len(barcodes) > n_cells_matrix:
            barcodes = barcodes[:n_cells_matrix]
        else:
            # Pad with dummy barcodes
            for i in range(len(barcodes), n_cells_matrix):
                barcodes.append(f"DUMMY_CELL_{i}")

    if n_genes_matrix != len(gene_ids):
        print(f"Adjusting gene list: matrix has {n_genes_matrix}, gene list has {len(gene_ids)}")
        if len(gene_ids) > n_genes_matrix:
            gene_ids = gene_ids[:n_genes_matrix]
            gene_symbols = gene_symbols[:n_genes_matrix]
        else:
            # Pad with dummy genes
            for i in range(len(gene_ids), n_genes_matrix):
                gene_ids.append(f"DUMMY_GENE_{i}")
                gene_symbols.append(f"DUMMY_GENE_{i}")

    # Now create the AnnData object
    adata = ad.AnnData(X=matrix)

    # Add cell information
    adata.obs_names = barcodes

    # Add gene information
    adata.var_names = gene_ids
    if gene_ids != gene_symbols:
        adata.var['gene_symbol'] = gene_symbols

    # Add metadata
    print("\nAdding metadata to AnnData object...")

    # First, let's see what's in the metadata
    print(f"First few metadata rows:")
    print(metadata.head(3))

    # Match metadata to cells
    if 'CellBarcode' in metadata.columns or metadata.index.name == 'CellBarcode':
        # Metadata has cell barcodes
        common_cells = list(set(barcodes) & set(metadata.index))
        print(f"Found metadata for {len(common_cells)} cells out of {len(barcodes)}")
    else:
        # Try to match by position
        common_cells = []
        print("No CellBarcode column found, will try position-based matching")

    # Add available metadata columns
    metadata_cols = [col for col in metadata.columns if col not in ['CellBarcode', 'Unnamed: 0']]
    print(f"Metadata columns to add: {metadata_cols}")

    for col in metadata_cols:
        # Initialize with 'Unknown' for all cells
        adata.obs[col] = 'Unknown'

        if common_cells:
            # Fill in known values
            for cell in common_cells:
                if cell in adata.obs_names and cell in metadata.index:
                    adata.obs.loc[cell, col] = metadata.loc[cell, col]
        elif len(metadata) == len(barcodes):
            # Position-based matching
            adata.obs[col] = metadata[col].values
        else:
            print(f"Warning: Could not match metadata for column {col}")

    print("\nAnnData object created:")
    print(f"  Cells: {adata.n_obs:,}")
    print(f"  Genes: {adata.n_vars:,}")
    print(f"  Metadata columns: {list(adata.obs.columns)}")

    # Print some metadata statistics
    if 'Disease_Identity' in adata.obs.columns:
        print("\nDisease distribution:")
        print(adata.obs['Disease_Identity'].value_counts())

    if 'CellType' in adata.obs.columns:
        print("\nCell type distribution (top 10):")
        print(adata.obs['CellType'].value_counts().head(10))
    elif 'Cell_Type' in adata.obs.columns:
        print("\nCell type distribution (top 10):")
        print(adata.obs['Cell_Type'].value_counts().head(10))

    # Check for any other potentially useful columns
    for col in adata.obs.columns:
        if col not in ['Disease_Identity', 'CellType', 'Cell_Type'] and adata.obs[col].nunique() < 50:
            print(f"\n{col} distribution:")
            print(adata.obs[col].value_counts())

    return adata


def preprocess_for_pipeline(adata, output_dir):
    """Preprocess the data to create the files expected by the pipeline"""

    print("\n" + "=" * 60)
    print("Preprocessing data for Adams pipeline...")
    print("=" * 60)

    # Create output directory
    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Save a copy as the initial raw data
    print("\nSaving raw data...")
    raw_path = os.path.join(cache_dir, "00_raw.h5ad")
    adata.write(raw_path)
    print(f"Saved to: {raw_path}")

    # Basic quality control metrics
    print("\nCalculating QC metrics...")

    # Calculate mitochondrial gene percentage
    # Try both human (MT-) and mouse (mt-) prefixes
    adata.var['mt'] = adata.var_names.str.startswith('MT-') | adata.var_names.str.startswith('mt-')
    if 'gene_symbol' in adata.var.columns:
        adata.var['mt'] = adata.var['mt'] | adata.var['gene_symbol'].str.startswith('MT-') | adata.var[
            'gene_symbol'].str.startswith('mt-')

    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # Add number of genes per cell
    adata.obs['n_genes'] = (adata.X > 0).sum(axis=1)
    if hasattr(adata.obs['n_genes'], 'A1'):
        adata.obs['n_genes'] = adata.obs['n_genes'].A1

    print(f"Mean genes per cell: {adata.obs['n_genes'].mean():.1f}")
    print(f"Mean mitochondrial %: {adata.obs['pct_counts_mt'].mean():.1f}%")

    # Save as "01_qc.h5ad"
    qc_path = os.path.join(cache_dir, "01_qc.h5ad")
    adata.write(qc_path)
    print(f"Saved QC data to: {qc_path}")

    # Create the "02_scored.h5ad" file
    # This should be the data after initial QC but before heavy filtering
    print("\nCreating scored data file...")
    adata_scored = adata.copy()

    # Basic filtering (very gentle to preserve most cells)
    print("Applying gentle QC filters...")
    sc.pp.filter_cells(adata_scored, min_genes=200)  # Remove cells with < 200 genes
    sc.pp.filter_genes(adata_scored, min_cells=3)  # Remove genes in < 3 cells

    # Save scored data
    scored_path = os.path.join(cache_dir, "02_scored.h5ad")
    adata_scored.write(scored_path)
    print(f"Saved scored data to: {scored_path}")
    print(f"  Cells: {adata_scored.n_obs:,}")
    print(f"  Genes: {adata_scored.n_vars:,}")

    # Create the "05_final.h5ad" file
    # This should have UMAP coordinates and clustering
    print("\nCreating final processed data...")
    adata_final = adata_scored.copy()

    # Adams et al. specific QC
    print("Applying Adams et al. QC criteria...")
    initial_cells = adata_final.n_obs

    # Filter cells
    sc.pp.filter_cells(adata_final, min_genes=1000)  # Adams criteria
    print(f"  After min 1000 genes: {adata_final.n_obs} cells (removed {initial_cells - adata_final.n_obs})")

    initial_cells = adata_final.n_obs
    adata_final = adata_final[adata_final.obs.pct_counts_mt < 20, :]  # Adams criteria
    print(f"  After <20% mitochondrial: {adata_final.n_obs} cells (removed {initial_cells - adata_final.n_obs})")

    print(f"Total after QC: {adata_final.n_obs:,} cells, {adata_final.n_vars:,} genes")

    # Normalization
    print("\nNormalizing data...")
    adata_final.raw = adata_final  # Save raw counts
    sc.pp.normalize_total(adata_final, target_sum=1e4)
    sc.pp.log1p(adata_final)

    # Find highly variable genes
    print("Finding highly variable genes...")
    sc.pp.highly_variable_genes(adata_final, min_mean=0.0125, max_mean=3, min_disp=0.5)
    print(f"Found {adata_final.var.highly_variable.sum()} highly variable genes")

    # Keep only HVGs for downstream analysis
    adata_final_hvg = adata_final[:, adata_final.var.highly_variable].copy()

    # Scale data
    print("Scaling data...")
    sc.pp.scale(adata_final_hvg, max_value=10)

    # PCA
    print("Running PCA...")
    sc.tl.pca(adata_final_hvg, svd_solver='arpack')

    # Copy PCA to full object
    adata_final.obsm['X_pca'] = adata_final_hvg.obsm['X_pca']

    # Compute neighbors
    print("Computing neighbor graph...")
    sc.pp.neighbors(adata_final, n_neighbors=10, n_pcs=40)

    # UMAP
    print("Computing UMAP...")
    sc.tl.umap(adata_final, min_dist=0.3)

    # Clustering
    print("Running Leiden clustering...")
    for resolution in [0.5, 0.8, 1.0, 1.2]:
        sc.tl.leiden(adata_final, resolution=resolution, key_added=f'leiden_{resolution}')
    sc.tl.leiden(adata_final, resolution=1.0)  # Default

    # Restore full gene set but keep computed results
    print("\nRestoring full gene set...")
    adata_full = adata_scored.copy()

    # Transfer computed results from cells that passed QC
    print("Transferring results to full dataset...")

    # Initialize columns that will be transferred
    adata_full.obs['leiden'] = 'Filtered_Out'
    for resolution in [0.5, 0.8, 1.0, 1.2]:
        adata_full.obs[f'leiden_{resolution}'] = 'Filtered_Out'

    # Find common cells (those that passed QC)
    common_cells = list(set(adata_final.obs_names) & set(adata_full.obs_names))
    print(f"Transferring results for {len(common_cells):,} cells that passed QC")

    # Transfer clustering results
    for col in ['leiden', 'leiden_0.5', 'leiden_0.8', 'leiden_1.0', 'leiden_1.2']:
        if col in adata_final.obs.columns:
            for cell in common_cells:
                adata_full.obs.loc[cell, col] = adata_final.obs.loc[cell, col]

    # Transfer UMAP coordinates
    adata_full.obsm['X_umap'] = np.full((adata_full.n_obs, 2), np.nan)

    # Get indices for efficient transfer
    final_cell_to_idx = {cell: idx for idx, cell in enumerate(adata_final.obs_names)}
    full_cell_to_idx = {cell: idx for idx, cell in enumerate(adata_full.obs_names)}

    for cell in common_cells:
        final_idx = final_cell_to_idx[cell]
        full_idx = full_cell_to_idx[cell]
        adata_full.obsm['X_umap'][full_idx] = adata_final.obsm['X_umap'][final_idx]

    # Add QC status
    adata_full.obs['passed_qc'] = False
    for cell in common_cells:
        adata_full.obs.loc[cell, 'passed_qc'] = True

    # Save final data
    final_path = os.path.join(cache_dir, "05_final.h5ad")
    adata_full.write(final_path)
    print(f"\nSaved final data to: {final_path}")
    print(f"  Total cells: {adata_full.n_obs:,}")
    print(f"  Cells that passed QC: {adata_full.obs['passed_qc'].sum():,}")
    print(f"  Cells with UMAP: {(~np.isnan(adata_full.obsm['X_umap'][:, 0])).sum():,}")

    # Print final statistics
    print("\nFinal dataset summary:")
    print(f"  Shape: {adata_full.shape}")
    if 'Disease_Identity' in adata_full.obs.columns:
        print(f"\n  Disease distribution (cells that passed QC):")
        qc_cells = adata_full.obs['passed_qc']
        print(adata_full.obs.loc[qc_cells, 'Disease_Identity'].value_counts())

    return scored_path, final_path


def main():
    """Main preprocessing function"""

    import sys

    print("=" * 60)
    print("Adams et al. 2020 Data Preprocessing")
    print("Converting raw GEO data to H5AD format")
    print("=" * 60)

    # Get data directory from command line or use current directory
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "."

    data_dir = os.path.abspath(data_dir)
    print(f"\nData directory: {data_dir}")

    # List files in directory
    print("\nFiles in directory:")
    files = os.listdir(data_dir)
    for f in sorted(files):
        if f.endswith('.gz') or f.endswith('.h5ad'):
            print(f"  {f}")

    # Check if processed files already exist
    cache_dir = os.path.join(data_dir, "cache")
    scored_path = os.path.join(cache_dir, "02_scored.h5ad")
    final_path = os.path.join(cache_dir, "05_final.h5ad")

    if os.path.exists(scored_path) and os.path.exists(final_path):
        print(f"\n⚠️  Processed files already exist:")
        print(f"  {scored_path}")
        print(f"  {final_path}")
        response = input("\nOverwrite existing files? (y/n): ")
        if response.lower() != 'y':
            print("Preprocessing cancelled.")
            return 0

    try:
        # Load raw data
        adata = load_adams_raw_data(data_dir)

        # Preprocess for pipeline
        scored_path, final_path = preprocess_for_pipeline(adata, data_dir)

        print("\n" + "=" * 60)
        print("✅ Preprocessing completed successfully!")
        print("=" * 60)
        print(f"\nCreated files:")
        print(f"  {scored_path}")
        print(f"  {final_path}")
        print(f"\nYou can now run the main analysis pipeline:")
        print(f"  python adams_pipeline.py {data_dir}")

        # Save a summary file
        summary_path = os.path.join(cache_dir, "preprocessing_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Adams et al. 2020 Data Preprocessing Summary\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data directory: {data_dir}\n")
            f.write(f"\nOriginal data:\n")
            f.write(f"  Cells: {adata.n_obs:,}\n")
            f.write(f"  Genes: {adata.n_vars:,}\n")
            f.write(f"\nProcessed data:\n")
            f.write(f"  Scored file: {scored_path}\n")
            f.write(f"  Final file: {final_path}\n")
            if 'Disease_Identity' in adata.obs.columns:
                f.write(f"\nDisease distribution:\n")
                for disease, count in adata.obs['Disease_Identity'].value_counts().items():
                    f.write(f"  {disease}: {count:,}\n")

        print(f"\nPreprocessing summary saved to: {summary_path}")

    except FileNotFoundError as e:
        print(f"\n❌ Required file not found: {str(e)}")
        print("\nPlease ensure you have the following files in the directory:")
        print("  - GSE136831_RawCounts_Sparse.mtx.gz")
        print("  - GSE136831_AllCells.cellBarcodes.txt.gz")
        print("  - GSE136831_AllCells.GeneIDs.txt.gz")
        print("  - GSE136831_AllCells.Samples.CellType.MetadataTable.txt.gz")
        return 1
    except Exception as e:
        print(f"\n❌ Preprocessing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
