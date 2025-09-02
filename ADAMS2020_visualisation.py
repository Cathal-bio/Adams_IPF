#!/usr/bin/env python3
"""
Focused Analysis: Target Gene Expression in Aberrant Basaloid and AT2 Cells
Focus on PTCHD4, EDA2R, MUC3A, ITGBL1, and PLAUR across disease states
With separated senescence marker categories and improved plot spacing
Added: All cell type analysis with hierarchical clustering
"""

import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Ellipse, Patch
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal, chi2_contingency
from statsmodels.stats.multitest import multipletests
import warnings
from datetime import datetime
import json
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import gc

os.chdir('/home/cathal/Desktop/working/Single_Cell/scrnaseq_analysis')

warnings.filterwarnings('ignore')
sc.settings.verbosity = 1

# High-quality plot settings with better spacing
plt.rcParams.update({
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 10,
    'axes.titlesize': 11,  # Slightly smaller titles
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 14,
    'axes.titlepad': 10  # More padding for titles
})


class TargetGeneAnalysis:
    """Focused analysis of target genes in aberrant basaloid and AT2 cells"""

    def __init__(self, adata, output_dir):
        self.adata = adata.copy()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)

        # TARGET GENES OF INTEREST
        self.target_genes = ['PTCHD4', 'EDA2R', 'MUC3A', 'ITGBL1', 'PLAUR']

        # AT2 and senescence markers - SEPARATED BY CATEGORY
        self.at2_markers = ['SFTPC', 'SFTPB', 'SFTPA1', 'SFTPA2', 'SFTPD',
                            'ABCA3', 'SLC34A2', 'NAPSA', 'PGC', 'CLDN18']

        # Separate senescence marker categories
        self.senescence_categories = {
            'cell_cycle_inhibitors': ['CDKN1A', 'CDKN2A', 'CDKN2B'],
            'tumor_suppressors': ['TP53', 'RB1'],
            'sasp_cytokines': ['IL6', 'IL8', 'IL1A', 'IL1B'],
            'sasp_chemokines': ['CXCL1', 'CXCL2', 'CCL2'],
            'sasp_proteases': ['MMP1', 'MMP3', 'MMP10'],
            'growth_factors': ['IGFBP3', 'IGFBP7', 'GDF15'],
            'other_markers': ['GLB1', 'LMNB1', 'HMGB1']
        }

        # Flatten for compatibility
        self.senescence_markers = []
        for markers in self.senescence_categories.values():
            self.senescence_markers.extend(markers)

        # Find metadata columns
        self.manuscript_col = 'Manuscript_Identity' if 'Manuscript_Identity' in self.adata.obs.columns else None
        self.cell_type_col = 'CellType_Category' if 'CellType_Category' in self.adata.obs.columns else None
        self.subclass_col = 'Subclass_Cell_Identity' if 'Subclass_Cell_Identity' in self.adata.obs.columns else None
        self.disease_col = 'Disease_Identity' if 'Disease_Identity' in self.adata.obs.columns else None
        self.subject_col = 'Subject_Identity' if 'Subject_Identity' in self.adata.obs.columns else None

        print(f"\nFocused Target Gene Analysis")
        print(f"Target genes: {', '.join(self.target_genes)}")
        print(f"Total cells: {self.adata.n_obs}")

        # Check which target genes are available
        self.available_target_genes = [g for g in self.target_genes if g in self.adata.var_names]
        self.missing_target_genes = [g for g in self.target_genes if g not in self.adata.var_names]

        if self.missing_target_genes:
            print(f"WARNING: Missing target genes: {', '.join(self.missing_target_genes)}")
        print(f"Available target genes: {', '.join(self.available_target_genes)}")

        # Check other markers
        self.available_at2_markers = [g for g in self.at2_markers if g in self.adata.var_names]

        # Check senescence markers by category
        self.available_senescence_by_category = {}
        for category, markers in self.senescence_categories.items():
            available = [g for g in markers if g in self.adata.var_names]
            if available:
                self.available_senescence_by_category[category] = available
                print(f"Available {category}: {', '.join(available)}")

    def _get_gene_expression(self, gene, mask=None):
        """Extract gene expression efficiently"""
        idx = list(self.adata.var_names).index(gene)
        expr = self.adata.X[:, idx]
        if hasattr(expr, 'toarray'):
            expr = expr.toarray().flatten()
        else:
            expr = expr.flatten()

        if mask is not None:
            return expr[mask]
        return expr

    def identify_cell_populations(self):
        """Identify aberrant basaloid and AT2 cells"""
        print("\nIdentifying cell populations...")

        # 1. Identify aberrant basaloid cells
        self.adata.obs['is_aberrant_basaloid'] = False

        aberrant_identifiers = ['Aberrant_Basaloid', 'aberrant_basaloid', 'Aberrant Basaloid']

        if self.manuscript_col:
            for identifier in aberrant_identifiers:
                mask = self.adata.obs[self.manuscript_col].str.contains(identifier, case=False, na=False)
                self.adata.obs.loc[mask, 'is_aberrant_basaloid'] = True

        # 2. Identify AT2 cells
        self.adata.obs['is_at2'] = False

        at2_identifiers = ['ATII', 'AT2', 'Alveolar_Type_II', 'Alveolar Type II', 'ATII_Cell']

        if self.manuscript_col:
            for identifier in at2_identifiers:
                mask = self.adata.obs[self.manuscript_col].str.contains(identifier, case=False, na=False)
                self.adata.obs.loc[mask, 'is_at2'] = True

        # If not enough AT2 cells, use marker score
        if self.adata.obs['is_at2'].sum() < 100 and self.available_at2_markers:
            print("  Using AT2 marker score to identify additional AT2 cells...")
            sc.tl.score_genes(
                self.adata,
                self.available_at2_markers[:5],
                score_name='at2_marker_score',
                use_raw=False
            )

            at2_threshold = self.adata.obs['at2_marker_score'].quantile(0.95)
            high_at2_score = self.adata.obs['at2_marker_score'] > at2_threshold
            self.adata.obs.loc[high_at2_score, 'is_at2'] = True

        # Count cells
        self.n_aberrant = self.adata.obs['is_aberrant_basaloid'].sum()
        self.n_at2 = self.adata.obs['is_at2'].sum()

        # Create combined categories
        self.adata.obs['cell_category'] = 'Other'
        self.adata.obs.loc[self.adata.obs['is_aberrant_basaloid'], 'cell_category'] = 'Aberrant_Basaloid'
        self.adata.obs.loc[self.adata.obs['is_at2'], 'cell_category'] = 'AT2'

        # Check for overlap
        overlap = (self.adata.obs['is_aberrant_basaloid'] & self.adata.obs['is_at2']).sum()
        if overlap > 0:
            print(f"  Warning: {overlap} cells classified as both aberrant basaloid and AT2")
            # Prioritize aberrant basaloid classification
            self.adata.obs.loc[self.adata.obs['is_aberrant_basaloid'] & self.adata.obs[
                'is_at2'], 'cell_category'] = 'Aberrant_Basaloid'

        print(f"Results:")
        print(f"  Aberrant basaloid cells: {self.n_aberrant} ({self.n_aberrant / self.adata.n_obs * 100:.2f}%)")
        print(f"  AT2 cells: {self.n_at2} ({self.n_at2 / self.adata.n_obs * 100:.2f}%)")

        # Calculate scores
        self._calculate_scores()

    def _calculate_scores(self):
        """Calculate target gene and separated senescence scores"""
        print("\nCalculating expression scores...")

        # Target gene score
        if self.available_target_genes:
            sc.tl.score_genes(
                self.adata,
                self.available_target_genes,
                score_name='target_gene_score',
                use_raw=False
            )

        # Individual target gene scores
        for gene in self.available_target_genes:
            expr = self._get_gene_expression(gene)
            self.adata.obs[f'{gene}_expr'] = expr

        # Calculate separate scores for each senescence category
        for category, markers in self.available_senescence_by_category.items():
            score_name = f'{category}_score'
            sc.tl.score_genes(
                self.adata,
                markers,
                score_name=score_name,
                use_raw=False
            )
            print(f"  Calculated {score_name} using {len(markers)} markers")

    def _find_cell_regions(self):
        """Find regions containing AT2 and aberrant basaloid cells in UMAP"""
        at2_mask = self.adata.obs['is_at2']
        aberrant_mask = self.adata.obs['is_aberrant_basaloid']

        regions = {}

        # AT2 region
        if at2_mask.sum() > 10:
            at2_coords = self.adata.obsm['X_umap'][at2_mask]
            at2_center = np.mean(at2_coords, axis=0)
            at2_distances = np.sqrt(((at2_coords - at2_center) ** 2).sum(axis=1))
            at2_radius = np.percentile(at2_distances, 90)
            regions['at2'] = {'center': at2_center, 'radius': at2_radius}

        # Aberrant region
        if aberrant_mask.sum() > 10:
            aberrant_coords = self.adata.obsm['X_umap'][aberrant_mask]
            aberrant_center = np.mean(aberrant_coords, axis=0)
            aberrant_distances = np.sqrt(((aberrant_coords - aberrant_center) ** 2).sum(axis=1))
            aberrant_radius = np.percentile(aberrant_distances, 90)
            regions['aberrant'] = {'center': aberrant_center, 'radius': aberrant_radius}

        return regions

    def create_focused_visualizations(self):
        """Create focused visualizations on target genes"""
        print("\nCreating focused visualizations...")

        # Find cell regions for highlighting
        self.regions = self._find_cell_regions()

        # Create each visualization
        self._plot_target_gene_overview()
        self._plot_all_cell_types_analysis()  # NEW ANALYSIS
        self._plot_correlation_analyses()  # NEW CORRELATION ANALYSES
        self._plot_disease_stratified_comparison()
        self._plot_senescence_categories_analysis()
        self._plot_enhanced_umaps()
        self._plot_individual_gene_analysis()

    def _plot_all_cell_types_analysis(self):
        """Analyze target genes and senescence across all cell types with disease intersection"""
        print("\nAnalyzing all cell types...")

        if not self.manuscript_col:
            print("Cell type information not available")
            return

        # Get ALL cell types (not just >100 cells) for complete analysis
        cell_type_counts = self.adata.obs[self.manuscript_col].value_counts()
        all_cell_types = cell_type_counts.index.tolist()
        major_cell_types = cell_type_counts[cell_type_counts > 100].index.tolist()

        print(f"Found {len(all_cell_types)} total cell types")
        print(f"Found {len(major_cell_types)} cell types with >100 cells")

        # Create figure with multiple panels
        fig = plt.figure(figsize=(28, 24))
        gs = GridSpec(4, 4, figure=fig, hspace=0.5, wspace=0.5)

        # 0. Senescence category legend panel - moved to avoid overlap
        ax0 = fig.add_subplot(gs[0, 3])
        ax0.axis('off')

        legend_text = "Senescence Marker Categories\n" + "=" * 30 + "\n\n"

        for category, markers in self.available_senescence_by_category.items():
            legend_text += f"{category.replace('_', ' ').title()}:\n"
            marker_count = 0
            for marker in markers:
                if marker in self.adata.var_names:
                    legend_text += f"  • {marker}\n"
                    marker_count += 1
                else:
                    legend_text += f"  • {marker} (missing)\n"
                # Limit to 3 markers per category in legend to save space
                if marker_count >= 3 and len(markers) > 3:
                    legend_text += f"  ... (+{len(markers) - 3} more)\n"
                    break
            legend_text += "\n"

        ax0.text(0.05, 0.95, legend_text, transform=ax0.transAxes,
                 verticalalignment='top', fontfamily='monospace', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

        # 1. Hierarchical clustered heatmap of target gene expression
        ax1 = fig.add_subplot(gs[0, :2])

        # Create expression matrix for ALL cell types
        expr_matrix = []
        cell_type_labels = []

        for ct in all_cell_types:
            ct_mask = self.adata.obs[self.manuscript_col] == ct
            row = []
            for gene in self.available_target_genes:
                if ct_mask.sum() > 0:
                    expr = self._get_gene_expression(gene, ct_mask)
                    row.append(np.mean(expr))
                else:
                    row.append(0)  # Use 0 for missing cell types
            expr_matrix.append(row)
            cell_type_labels.append(f"{ct} (n={ct_mask.sum()})")

        expr_array = np.array(expr_matrix)

        # Row-scale (z-score normalize each row) - handle rows with no variance
        expr_scaled = np.zeros_like(expr_array)
        for i in range(expr_array.shape[0]):
            if np.std(expr_array[i, :]) > 0:
                expr_scaled[i, :] = (expr_array[i, :] - np.mean(expr_array[i, :])) / np.std(expr_array[i, :])
            else:
                expr_scaled[i, :] = 0

        # Only cluster cell types with >100 cells
        major_indices = [i for i, ct in enumerate(all_cell_types) if ct in major_cell_types]

        if len(major_indices) > 1:
            # Hierarchical clustering on major cell types only
            major_expr_scaled = expr_scaled[major_indices, :]
            row_linkage = linkage(major_expr_scaled, method='ward')
            col_linkage = linkage(expr_scaled.T, method='ward')

            # Create clustered heatmap for major cell types
            g = sns.clustermap(major_expr_scaled,
                               row_linkage=row_linkage,
                               col_linkage=col_linkage,
                               xticklabels=self.available_target_genes,
                               yticklabels=[cell_type_labels[i] for i in major_indices],
                               cmap='RdBu_r',
                               center=0,
                               figsize=(10, 12),
                               cbar_kws={'label': 'Z-score'},
                               dendrogram_ratio=(0.1, 0.2))

            # Save clustermap separately
            g.savefig(os.path.join(self.output_dir, 'plots', 'cell_type_target_gene_clustermap.png'), dpi=300)
            plt.close(g.fig)

            # Get ordering from clustering
            row_order = dendrogram(row_linkage, no_plot=True)['leaves']
            major_ordered_indices = [major_indices[i] for i in row_order]

            # Add other cell types at the end
            all_ordered_indices = major_ordered_indices + [i for i in range(len(all_cell_types)) if
                                                           i not in major_indices]
        else:
            all_ordered_indices = list(range(len(all_cell_types)))

        # Plot ALL cell types with consistent ordering
        sns.heatmap(expr_scaled[all_ordered_indices, :],
                    xticklabels=self.available_target_genes,
                    yticklabels=[cell_type_labels[i] for i in all_ordered_indices],
                    cmap='RdBu_r',
                    center=0,
                    ax=ax1,
                    cbar_kws={'label': 'Z-score'})
        ax1.set_title('Target Gene Expression Across All Cell Types (Row-scaled)', fontsize=14, pad=15)

        # Draw line to separate major cell types from others
        if len(major_indices) > 0 and len(major_indices) < len(all_cell_types):
            ax1.axhline(y=len(major_indices), color='black', linewidth=2)

        # 2. Senescence signature heatmap
        ax2 = fig.add_subplot(gs[1, :])

        # Create senescence score matrix for ALL cell types
        sen_matrix = []

        for ct in all_cell_types:
            ct_mask = self.adata.obs[self.manuscript_col] == ct
            row = []
            for category in self.available_senescence_by_category.keys():
                score_name = f'{category}_score'
                if score_name in self.adata.obs.columns:
                    if ct_mask.sum() > 0:
                        row.append(self.adata.obs[ct_mask][score_name].mean())
                    else:
                        row.append(0)
            sen_matrix.append(row)

        sen_array = np.array(sen_matrix)

        # Create shortened labels for display
        shortened_labels = []
        for idx in all_ordered_indices:
            ct = all_cell_types[idx]
            ct_count = (self.adata.obs[self.manuscript_col] == ct).sum()
            # Shorten long cell type names
            if len(ct) > 25:
                ct_short = ct[:22] + "..."
            else:
                ct_short = ct
            shortened_labels.append(f"{ct_short} (n={ct_count})")

        # Use same ordering and labels as target genes
        im2 = ax2.imshow(sen_array[all_ordered_indices, :],
                         aspect='auto',
                         cmap='YlOrRd')

        # Set ticks and labels
        ax2.set_xticks(range(len(self.available_senescence_by_category.keys())))
        ax2.set_xticklabels([cat.replace('_', ' ').title() for cat in self.available_senescence_by_category.keys()],
                            rotation=45, ha='right', fontsize=10)
        ax2.set_yticks(range(len(all_ordered_indices)))
        ax2.set_yticklabels(shortened_labels, fontsize=8)

        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Mean Score', fontsize=10)

        ax2.set_title('Senescence Signatures Across All Cell Types', fontsize=14, pad=15)

        # Draw line to separate major cell types
        if len(major_indices) > 0 and len(major_indices) < len(all_cell_types):
            ax2.axhline(y=len(major_indices) - 0.5, color='red', linewidth=2)

        # 3. Disease-Cell Type intersection analysis
        if self.disease_col:
            ax3 = fig.add_subplot(gs[2, 0])

            # Calculate disease composition for ALL cell types
            disease_comp = []
            diseases = self.adata.obs[self.disease_col].unique()

            for ct in all_cell_types:
                ct_mask = self.adata.obs[self.manuscript_col] == ct
                ct_data = self.adata.obs[ct_mask]

                row = []
                for disease in diseases:
                    if ct_mask.sum() > 0:
                        disease_count = (ct_data[self.disease_col] == disease).sum()
                        row.append(disease_count / ct_mask.sum() * 100)
                    else:
                        row.append(0)
                disease_comp.append(row)

            disease_array = np.array(disease_comp)

            # Use same ordering, but only show major cell types for clarity
            major_ordered = [i for i in all_ordered_indices if all_cell_types[i] in major_cell_types]

            # Create stacked bar plot
            bottom = np.zeros(len(major_ordered))
            colors = plt.cm.Set3(np.linspace(0, 1, len(diseases)))

            for i, disease in enumerate(diseases):
                values = [disease_array[idx, i] for idx in major_ordered]
                ax3.bar(range(len(major_ordered)), values,
                        bottom=bottom, label=disease, color=colors[i])
                bottom += values

            ax3.set_xticks(range(len(major_ordered)))
            ax3.set_xticklabels([all_cell_types[i] for i in major_ordered], rotation=90, ha='right')
            ax3.set_ylabel('Percentage')
            ax3.set_title('Disease Composition by Cell Type (>100 cells)', pad=10)
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 4. Target gene enrichment in disease-cell type combinations
        if self.disease_col:
            ax4 = fig.add_subplot(gs[2, 1:])

            # Calculate mean target gene score for ALL disease-cell type combinations
            enrichment_data = []

            # Use consistent cell type list
            ct_list = [all_cell_types[i] for i in all_ordered_indices[:30]]  # Top 30 for visibility

            for ct in ct_list:
                ct_mask = self.adata.obs[self.manuscript_col] == ct

                for disease in diseases:
                    disease_mask = self.adata.obs[self.disease_col] == disease
                    combined_mask = ct_mask & disease_mask

                    if combined_mask.sum() > 10:  # At least 10 cells
                        mean_score = self.adata.obs[combined_mask]['target_gene_score'].mean()
                        enrichment_data.append({
                            'Cell_Type': ct,
                            'Disease': disease,
                            'Target_Score': mean_score,
                            'N_Cells': combined_mask.sum()
                        })
                    else:
                        # Add zero for missing combinations
                        enrichment_data.append({
                            'Cell_Type': ct,
                            'Disease': disease,
                            'Target_Score': 0,
                            'N_Cells': combined_mask.sum()
                        })

            if enrichment_data:
                enrich_df = pd.DataFrame(enrichment_data)
                pivot_df = enrich_df.pivot(index='Cell_Type', columns='Disease', values='Target_Score')

                # Ensure all cell types are present
                pivot_df = pivot_df.reindex(ct_list, fill_value=0)

                # Use raw scores, no z-score normalization
                sns.heatmap(pivot_df,
                            cmap='YlOrRd',
                            ax=ax4,
                            cbar_kws={'label': 'Target Gene Score'},
                            yticklabels=True)
                ax4.set_title('Target Gene Score by Disease and Cell Type', pad=10)

        plt.suptitle('Cell Type Analysis: Target Genes and Senescence', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(os.path.join(self.output_dir, 'plots', 'all_cell_types_analysis.png'), dpi=300)
        plt.close()

        # Create a simple combined heatmap instead
        self._create_simple_combined_heatmap()

    def _create_simple_combined_heatmap(self):
        """Create a heatmap combining target genes and senescence markers with gene normalization and clustering"""
        if not self.manuscript_col:
            return

        # Get top 40 cell types by count for better visibility
        cell_type_counts = self.adata.obs[self.manuscript_col].value_counts()
        top_cell_types = cell_type_counts.head(40).index.tolist()

        # Combine all genes we want to show
        all_genes = self.available_target_genes + self.senescence_markers
        available_genes = [g for g in all_genes if g in self.adata.var_names]

        # Create expression matrix
        expr_matrix = []
        gene_labels = []

        for gene in available_genes:
            gene_expr = []
            for ct in top_cell_types:
                ct_mask = self.adata.obs[self.manuscript_col] == ct
                if ct_mask.sum() > 0:
                    expr = self._get_gene_expression(gene, ct_mask)
                    gene_expr.append(np.mean(expr))
                else:
                    gene_expr.append(0)
            expr_matrix.append(gene_expr)
            gene_labels.append(gene)

        if not expr_matrix:
            return

        # Convert to numpy array - already in genes x cell types format
        expr_array = np.array(expr_matrix)

        # Gene-wise normalization (z-score each gene across cell types)
        expr_normalized = np.zeros_like(expr_array)
        for i in range(expr_array.shape[0]):
            gene_expr = expr_array[i, :]
            if np.std(gene_expr) > 0:
                expr_normalized[i, :] = (gene_expr - np.mean(gene_expr)) / np.std(gene_expr)
            else:
                expr_normalized[i, :] = 0

        # Create cell type labels
        cell_labels = []
        for ct in top_cell_types:
            if len(ct) > 30:
                ct_short = ct[:27] + "..."
            else:
                ct_short = ct
            ct_count = (self.adata.obs[self.manuscript_col] == ct).sum()
            cell_labels.append(f"{ct_short} (n={ct_count})")

        # Create DataFrame - genes as rows, cell types as columns
        expr_df = pd.DataFrame(expr_normalized,
                               index=gene_labels,
                               columns=cell_labels)

        # Create gene colors
        gene_colors = []
        for gene in gene_labels:
            if gene in self.target_genes:
                gene_colors.append('red')
            else:
                gene_colors.append('blue')

        # Create clustermap with dendrograms on both sides
        g = sns.clustermap(expr_df,
                           method='ward',
                           metric='euclidean',
                           cmap='RdBu_r',
                           center=0,
                           vmin=-3,
                           vmax=3,
                           figsize=(max(14, len(cell_labels) * 0.3),
                                    max(12, len(gene_labels) * 0.25)),
                           row_colors=gene_colors,
                           cbar_kws={'label': 'Z-score'},
                           dendrogram_ratio=(0.15, 0.12),
                           linewidths=0.2,
                           xticklabels=True,
                           yticklabels=True,
                           cbar_pos=(0.02, 0.83, 0.03, 0.15))

        # Adjust label properties
        plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize=9, ha='right')
        plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), fontsize=9)

        # Color the gene names based on their type
        yticks = g.ax_heatmap.get_yticklabels()
        for i, label in enumerate(yticks):
            gene_name = label.get_text()
            if gene_name in self.target_genes:
                label.set_color('red')
                label.set_weight('bold')
            else:
                label.set_color('blue')

        # Add legend for gene colors
        legend_elements = [Patch(facecolor='red', label='Target Genes'),
                           Patch(facecolor='blue', label='Senescence Markers')]
        g.ax_row_colors.legend(handles=legend_elements,
                               loc='center left',
                               bbox_to_anchor=(1.1, 0.5),
                               frameon=False,
                               fontsize=10)

        # Add title
        g.fig.suptitle('Target Genes and Senescence Markers Expression\n(Gene-normalized, Hierarchical Clustering)',
                       fontsize=14, y=0.99)

        # Save
        g.savefig(os.path.join(self.output_dir, 'plots', 'combined_gene_heatmap.png'),
                  dpi=300, bbox_inches='tight')
        plt.close()

        # Also create a log-transformed version for comparison
        plt.figure(figsize=(max(12, len(cell_labels) * 0.3),
                            max(10, len(gene_labels) * 0.2)))

        # Create log expression matrix
        expr_log_matrix = []
        for gene in available_genes:
            gene_expr = []
            for ct in top_cell_types:
                ct_mask = self.adata.obs[self.manuscript_col] == ct
                if ct_mask.sum() > 0:
                    expr = self._get_gene_expression(gene, ct_mask)
                    gene_expr.append(np.log1p(np.mean(expr)))
                else:
                    gene_expr.append(0)
            expr_log_matrix.append(gene_expr)

        expr_log_df = pd.DataFrame(expr_log_matrix, index=gene_labels, columns=cell_labels)

        # Create simple heatmap without clustering
        ax = sns.heatmap(expr_log_df,
                         cmap='Reds',
                         cbar_kws={'label': 'log1p(mean expression)'},
                         linewidths=0.5,
                         xticklabels=True,
                         yticklabels=True,
                         vmin=0)  # Ensure white starts at 0

        # Color the gene names
        yticks = ax.get_yticklabels()
        for i, label in enumerate(yticks):
            gene_name = label.get_text()
            if gene_name in self.target_genes:
                label.set_color('red')
                label.set_weight('bold')
            else:
                label.set_color('blue')

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, fontsize=9, ha='right')
        plt.setp(ax.yaxis.get_majorticklabels(), fontsize=9)

        plt.title('Target Genes (red) and Senescence Markers (blue) - Log Expression', pad=20, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'combined_gene_heatmap_log.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_target_gene_overview(self):
        """Overview of target gene expression in populations - UPDATED without z-score"""
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

        at2_mask = self.adata.obs['is_at2']
        aberrant_mask = self.adata.obs['is_aberrant_basaloid']
        other_mask = ~(at2_mask | aberrant_mask)

        # 1. Raw expression values heatmap
        ax1 = fig.add_subplot(gs[0, :2])

        # Create expression matrix with log values
        expr_matrix = []
        populations = ['Other', 'AT2', 'Aberrant_Basaloid']

        for pop, mask in [('Other', other_mask), ('AT2', at2_mask), ('Aberrant_Basaloid', aberrant_mask)]:
            if mask.sum() > 0:
                row = []
                for gene in self.available_target_genes:
                    expr = self._get_gene_expression(gene, mask)
                    row.append(np.log1p(np.mean(expr)))
                expr_matrix.append(row)

        expr_array = np.array(expr_matrix)

        # Plot heatmap with log values
        sns.heatmap(expr_array,
                    xticklabels=self.available_target_genes,
                    yticklabels=populations[:len(expr_matrix)],
                    annot=True, fmt='.3f',
                    cmap='viridis',
                    vmin=0,
                    ax=ax1, cbar_kws={'label': 'log1p(mean expression)'})
        ax1.set_title('Target Gene Expression by Population', pad=15)

        # 2. Percentage expressing
        ax2 = fig.add_subplot(gs[0, 2])

        pct_data = []
        for gene in self.available_target_genes:
            expr = self._get_gene_expression(gene)

            pct_data.append({
                'Gene': gene,
                'Other': (expr[other_mask] > 0).sum() / other_mask.sum() * 100,
                'AT2': (expr[at2_mask] > 0).sum() / at2_mask.sum() * 100 if at2_mask.sum() > 0 else 0,
                'Aberrant': (expr[
                                 aberrant_mask] > 0).sum() / aberrant_mask.sum() * 100 if aberrant_mask.sum() > 0 else 0
            })

        pct_df = pd.DataFrame(pct_data)

        # Plot grouped bar
        x = np.arange(len(pct_df))
        width = 0.25

        ax2.bar(x - width, pct_df['Other'], width, label='Other', color='lightgray')
        ax2.bar(x, pct_df['AT2'], width, label='AT2', color='blue')
        ax2.bar(x + width, pct_df['Aberrant'], width, label='Aberrant', color='red')

        ax2.set_ylabel('% Cells Expressing')
        ax2.set_title('Percentage of Cells Expressing Target Genes', pad=15)
        ax2.set_xticks(x)
        ax2.set_xticklabels(pct_df['Gene'], rotation=45, ha='right')
        ax2.legend()

        # 3-7. Individual gene violin plots
        for i, gene in enumerate(self.available_target_genes):
            ax = fig.add_subplot(gs[1 + i // 3, i % 3])

            expr = self._get_gene_expression(gene)

            # Create violin data
            violin_data = []

            # Subsample other cells
            other_subsample = np.random.choice(np.where(other_mask)[0],
                                               min(5000, other_mask.sum()),
                                               replace=False)

            for val in np.log1p(expr[other_subsample]):
                violin_data.append({'Population': 'Other', 'Expression': val})

            for val in np.log1p(expr[at2_mask]):
                violin_data.append({'Population': 'AT2', 'Expression': val})

            for val in np.log1p(expr[aberrant_mask]):
                violin_data.append({'Population': 'Aberrant', 'Expression': val})

            violin_df = pd.DataFrame(violin_data)

            sns.violinplot(data=violin_df, x='Population', y='Expression',
                           palette={'Other': 'lightgray', 'AT2': 'blue', 'Aberrant': 'red'},
                           ax=ax)

            # Add statistical tests
            if at2_mask.sum() > 5 and other_mask.sum() > 5:
                _, p_at2 = mannwhitneyu(expr[other_mask], expr[at2_mask])
                ax.text(1, ax.get_ylim()[1], f'p={p_at2:.2e}', ha='center', fontsize=8)

            if aberrant_mask.sum() > 5 and other_mask.sum() > 5:
                _, p_aberrant = mannwhitneyu(expr[other_mask], expr[aberrant_mask])
                ax.text(2, ax.get_ylim()[1], f'p={p_aberrant:.2e}', ha='center', fontsize=8)

            ax.set_ylabel(f'log1p({gene})')
            ax.set_title(f'{gene} Expression', pad=10)

        plt.suptitle('Target Gene Expression Overview', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'target_gene_overview.png'), dpi=300)
        plt.close()

    def _plot_correlation_analyses(self):
        """Create comprehensive correlation analyses for target genes"""
        print("\nCreating correlation analyses...")

        # 1. Gene expression heatmap with group labels and row scaling
        self._create_gene_expression_by_groups()

        # 2. Target gene correlations
        self._create_target_gene_correlations()

        # 3. Target genes vs cell types correlations
        self._create_target_vs_celltypes()

        # 4. Target genes vs disease correlations
        self._create_target_vs_disease()

        # 5. Population-specific correlation heatmaps
        self._create_population_specific_correlations()

    def _create_gene_expression_by_groups(self):
        """Create gene expression heatmap with all genes grouped by category"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 20))

        # Prepare all genes with categories
        gene_categories = []
        gene_list = []

        # Add target genes
        for gene in self.available_target_genes:
            gene_list.append(gene)
            gene_categories.append('Target')

        # Add senescence genes by category
        for category, markers in self.available_senescence_by_category.items():
            for gene in markers:
                if gene in self.adata.var_names:
                    gene_list.append(gene)
                    gene_categories.append(category.replace('_', ' ').title())

        # Create expression matrix for a subset of cells
        n_cells = min(5000, self.adata.n_obs)
        cell_indices = np.random.choice(self.adata.n_obs, n_cells, replace=False)

        expr_matrix = []
        for gene in gene_list:
            expr = self._get_gene_expression(gene)[cell_indices]
            expr_matrix.append(expr)

        expr_array = np.array(expr_matrix)

        # Row-scale (z-score each gene)
        expr_scaled = np.zeros_like(expr_array)
        for i in range(expr_array.shape[0]):
            if np.std(expr_array[i, :]) > 0:
                expr_scaled[i, :] = (expr_array[i, :] - np.mean(expr_array[i, :])) / np.std(expr_array[i, :])

        # Create color map for categories
        unique_categories = list(dict.fromkeys(gene_categories))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_categories)))
        category_colors = [colors[unique_categories.index(cat)] for cat in gene_categories]

        # Hierarchical clustering
        row_linkage = linkage(expr_scaled, method='ward')
        col_linkage = linkage(expr_scaled.T, method='ward')

        # Create clustermap
        g = sns.clustermap(expr_scaled,
                           row_linkage=row_linkage,
                           col_linkage=col_linkage,
                           cmap='RdBu_r',
                           center=0,
                           figsize=(16, 20),
                           yticklabels=gene_list,
                           xticklabels=False,
                           row_colors=category_colors,
                           cbar_kws={'label': 'Z-score (within gene)'},
                           dendrogram_ratio=(0.15, 0.05))

        # Add legend for categories
        legend_elements = [Patch(facecolor=colors[i], label=cat)
                           for i, cat in enumerate(unique_categories)]
        g.ax_row_colors.legend(handles=legend_elements, title='Gene Category',
                               bbox_to_anchor=(1.2, 1), loc='upper left')

        g.fig.suptitle('All Genes Expression Patterns (Row-scaled)', fontsize=16, y=0.98)
        g.savefig(os.path.join(self.output_dir, 'plots', 'all_genes_expression_heatmap.png'), dpi=300)
        plt.close()

    def _create_target_gene_correlations(self):
        """Create correlation heatmap between target genes"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Get expression data for target genes
        expr_data = {}
        for gene in self.available_target_genes:
            expr_data[gene] = self._get_gene_expression(gene)

        expr_df = pd.DataFrame(expr_data)

        # 1. Pearson correlation
        corr_pearson = expr_df.corr(method='pearson')

        # Hierarchical clustering
        linkage_matrix = linkage(squareform(1 - corr_pearson), method='ward')
        dendro = dendrogram(linkage_matrix, no_plot=True)
        order = dendro['leaves']

        # Reorder correlation matrix
        corr_pearson_ordered = corr_pearson.iloc[order, order]

        # Plot
        sns.heatmap(corr_pearson_ordered,
                    annot=True,
                    fmt='.3f',
                    cmap='RdBu_r',
                    center=0,
                    vmin=-1,
                    vmax=1,
                    square=True,
                    ax=ax1,
                    cbar_kws={'label': 'Pearson Correlation'})
        ax1.set_title('Target Gene Pearson Correlations', pad=10)

        # 2. Spearman correlation
        corr_spearman = expr_df.corr(method='spearman')
        corr_spearman_ordered = corr_spearman.iloc[order, order]

        sns.heatmap(corr_spearman_ordered,
                    annot=True,
                    fmt='.3f',
                    cmap='RdBu_r',
                    center=0,
                    vmin=-1,
                    vmax=1,
                    square=True,
                    ax=ax2,
                    cbar_kws={'label': 'Spearman Correlation'})
        ax2.set_title('Target Gene Spearman Correlations', pad=10)

        plt.suptitle('Target Gene Expression Correlations', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'target_gene_correlations.png'), dpi=300)
        plt.close()

    def _create_target_vs_celltypes(self):
        """Create correlation heatmaps of target genes vs cell types"""
        if not self.manuscript_col:
            return

        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

        # 1. All cell types
        ax1 = fig.add_subplot(gs[0, :])

        # Get cell types
        cell_type_counts = self.adata.obs[self.manuscript_col].value_counts()
        all_cell_types = cell_type_counts.index.tolist()

        # Create expression matrix
        expr_matrix = []
        for ct in all_cell_types:
            ct_mask = self.adata.obs[self.manuscript_col] == ct
            if ct_mask.sum() > 0:
                row = []
                for gene in self.available_target_genes:
                    expr = self._get_gene_expression(gene, ct_mask)
                    row.append(np.mean(expr))
                expr_matrix.append(row)

        expr_df = pd.DataFrame(expr_matrix,
                               index=all_cell_types,
                               columns=self.available_target_genes)

        # Calculate correlation
        corr_matrix = expr_df.T.corr()

        # Plot subset for visibility
        if len(all_cell_types) > 50:
            # Select top variable cell types
            ct_variance = expr_df.var(axis=1).sort_values(ascending=False)
            top_cts = ct_variance.head(50).index
            corr_subset = corr_matrix.loc[top_cts, top_cts]
        else:
            corr_subset = corr_matrix

        sns.heatmap(corr_subset,
                    cmap='RdBu_r',
                    center=0,
                    vmin=-1,
                    vmax=1,
                    ax=ax1,
                    cbar_kws={'label': 'Correlation'})
        ax1.set_title('Cell Type Correlations based on Target Gene Expression', pad=10)

        # 2. Focus on AT2, Aberrant, and Other
        ax2 = fig.add_subplot(gs[1, 0])

        at2_mask = self.adata.obs['is_at2']
        aberrant_mask = self.adata.obs['is_aberrant_basaloid']
        other_mask = ~(at2_mask | aberrant_mask)

        # Create focused expression matrix
        focus_expr = []
        focus_labels = []

        for label, mask in [('Other', other_mask), ('AT2', at2_mask), ('Aberrant Basaloid', aberrant_mask)]:
            if mask.sum() > 0:
                row = []
                for gene in self.available_target_genes:
                    expr = self._get_gene_expression(gene, mask)
                    row.append(np.mean(expr))
                focus_expr.append(row)
                focus_labels.append(label)

        focus_df = pd.DataFrame(focus_expr,
                                index=focus_labels,
                                columns=self.available_target_genes)

        # Heatmap of expression
        sns.heatmap(focus_df,
                    annot=True,
                    fmt='.3f',
                    cmap='viridis',
                    ax=ax2,
                    cbar_kws={'label': 'Mean Expression'})
        ax2.set_title('Target Gene Expression in Key Populations', pad=10)

        # 3. Correlation between key populations
        ax3 = fig.add_subplot(gs[1, 1])

        pop_corr = focus_df.T.corr()

        sns.heatmap(pop_corr,
                    annot=True,
                    fmt='.3f',
                    cmap='RdBu_r',
                    center=0,
                    vmin=-1,
                    vmax=1,
                    square=True,
                    ax=ax3,
                    cbar_kws={'label': 'Correlation'})
        ax3.set_title('Population Correlations based on Target Genes', pad=10)

        plt.suptitle('Target Genes vs Cell Types Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'target_vs_celltypes.png'), dpi=300)
        plt.close()

    def _create_target_vs_disease(self):
        """Create correlation analysis of target genes vs disease conditions"""
        if not self.disease_col:
            return

        # Calculate required rows for GridSpec - simplified approach
        n_genes = len(self.available_target_genes)
        n_rows = 1 + ((n_genes + 1) // 2)  # Main plot + gene plots in pairs

        fig = plt.figure(figsize=(20, 5 * n_rows))
        gs = GridSpec(n_rows, 2, figure=fig, hspace=0.5, wspace=0.4)

        diseases = sorted(self.adata.obs[self.disease_col].unique())

        # 1. Overall target genes vs disease
        ax1 = fig.add_subplot(gs[0, :])

        # Create expression matrix
        expr_matrix = []
        for disease in diseases:
            disease_mask = self.adata.obs[self.disease_col] == disease
            row = []
            for gene in self.available_target_genes:
                expr = self._get_gene_expression(gene, disease_mask)
                row.append(np.mean(expr))
            expr_matrix.append(row)

        expr_df = pd.DataFrame(expr_matrix,
                               index=diseases,
                               columns=self.available_target_genes)

        # Plot simple version in the main figure with better formatting
        sns.heatmap(expr_df.T,
                    cmap='viridis',
                    ax=ax1,
                    cbar_kws={'label': 'Mean Expression'},
                    annot=True,
                    fmt='.2f')
        ax1.set_ylabel('Target Genes')
        ax1.set_xlabel('Disease')
        ax1.set_title('Target Gene Expression by Disease', pad=15)

        # Save clustered version separately with larger figure
        expr_df_copy = expr_df.copy()
        g = sns.clustermap(expr_df_copy.T,
                           cmap='viridis',
                           figsize=(14, 10),
                           cbar_kws={'label': 'Mean Expression'},
                           dendrogram_ratio=(0.15, 0.2),
                           annot=True,
                           fmt='.2f',
                           linewidths=0.5)
        g.ax_heatmap.set_ylabel('Target Genes', fontsize=12)
        g.ax_heatmap.set_xlabel('Disease', fontsize=12)
        g.fig.suptitle('Target Gene Expression by Disease (Clustered)', fontsize=14, y=0.95)
        plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
        g.savefig(os.path.join(self.output_dir, 'plots', 'target_genes_by_disease_clustered.png'),
                  dpi=300, bbox_inches='tight')
        plt.close(g.fig)

        # Individual gene plots
        for i, gene in enumerate(self.available_target_genes):
            row = 1 + i // 2
            col = i % 2
            ax = fig.add_subplot(gs[row, col])

            # Get expression by disease and population
            plot_data = []

            at2_mask = self.adata.obs['is_at2']
            aberrant_mask = self.adata.obs['is_aberrant_basaloid']
            other_mask = ~(at2_mask | aberrant_mask)

            for disease in diseases:
                disease_mask = self.adata.obs[self.disease_col] == disease

                for pop, pop_mask, color in [('Other', other_mask, 'gray'),
                                             ('AT2', at2_mask, 'blue'),
                                             ('Aberrant', aberrant_mask, 'red')]:
                    combined_mask = disease_mask & pop_mask
                    if combined_mask.sum() > 0:
                        expr = self._get_gene_expression(gene, combined_mask)
                        plot_data.append({
                            'Disease': disease,
                            'Population': pop,
                            'Expression': np.mean(expr),
                            'Color': color
                        })

            if plot_data:
                plot_df = pd.DataFrame(plot_data)

                # Create grouped bar plot
                try:
                    pivot_df = plot_df.pivot(index='Disease', columns='Population', values='Expression')

                    # Ensure column order for consistent colors
                    column_order = []
                    color_order = []
                    if 'Aberrant' in pivot_df.columns:
                        column_order.append('Aberrant')
                        color_order.append('red')
                    if 'AT2' in pivot_df.columns:
                        column_order.append('AT2')
                        color_order.append('blue')
                    if 'Other' in pivot_df.columns:
                        column_order.append('Other')
                        color_order.append('gray')

                    if column_order:  # Only plot if we have data
                        pivot_df = pivot_df[column_order]
                        pivot_df.plot(kind='bar', ax=ax, color=color_order, width=0.8)
                        ax.set_ylabel('Mean Expression', fontsize=10)
                        ax.set_title(f'{gene} Expression by Disease and Population', pad=10, fontsize=11)
                        ax.legend(title='Population', fontsize=9, title_fontsize=9)
                        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
                    else:
                        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                                transform=ax.transAxes, fontsize=12)
                        ax.set_title(f'{gene} - No data', fontsize=11)
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center',
                            transform=ax.transAxes, fontsize=10)
                    ax.set_title(f'{gene} - Error', fontsize=11)
            else:
                ax.text(0.5, 0.5, 'No expression data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{gene} - No data', fontsize=11)

        plt.suptitle('Individual Target Gene Analysis by Disease', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(os.path.join(self.output_dir, 'plots', 'target_vs_disease_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _create_population_specific_correlations(self):
        """Create separate correlation heatmaps for AT2, Aberrant, and Other populations"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))

        at2_mask = self.adata.obs['is_at2']
        aberrant_mask = self.adata.obs['is_aberrant_basaloid']
        other_mask = ~(at2_mask | aberrant_mask)

        populations = [
            ('Other', other_mask, axes[0, 0], axes[1, 0]),
            ('AT2', at2_mask, axes[0, 1], axes[1, 1]),
            ('Aberrant Basaloid', aberrant_mask, axes[0, 2], axes[1, 2])
        ]

        for pop_name, mask, ax_pearson, ax_spearman in populations:
            if mask.sum() < 10:
                ax_pearson.text(0.5, 0.5, f'Insufficient {pop_name} cells',
                                ha='center', va='center', transform=ax_pearson.transAxes)
                ax_spearman.text(0.5, 0.5, f'Insufficient {pop_name} cells',
                                 ha='center', va='center', transform=ax_spearman.transAxes)
                continue

            # Get expression data for this population
            expr_data = {}
            for gene in self.available_target_genes:
                expr_data[gene] = self._get_gene_expression(gene, mask)

            expr_df = pd.DataFrame(expr_data)

            # Pearson correlation
            corr_pearson = expr_df.corr(method='pearson')

            sns.heatmap(corr_pearson,
                        annot=True,
                        fmt='.2f',
                        cmap='RdBu_r',
                        center=0,
                        vmin=-1,
                        vmax=1,
                        square=True,
                        ax=ax_pearson,
                        cbar_kws={'label': 'Pearson r'})
            ax_pearson.set_title(f'{pop_name} - Pearson Correlation', pad=10)

            # Spearman correlation
            corr_spearman = expr_df.corr(method='spearman')

            sns.heatmap(corr_spearman,
                        annot=True,
                        fmt='.2f',
                        cmap='RdBu_r',
                        center=0,
                        vmin=-1,
                        vmax=1,
                        square=True,
                        ax=ax_spearman,
                        cbar_kws={'label': 'Spearman ρ'})
            ax_spearman.set_title(f'{pop_name} - Spearman Correlation', pad=10)

        axes[0, 0].set_ylabel('Pearson', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Spearman', fontsize=14, fontweight='bold')

        plt.suptitle('Population-Specific Target Gene Correlations', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'population_specific_correlations.png'), dpi=300)
        plt.close()

    def _plot_disease_stratified_comparison(self):
        """Compare target gene expression stratified by disease"""
        if not self.disease_col:
            print("Disease information not available")
            return

        fig = plt.figure(figsize=(20, 16))

        diseases = sorted(self.adata.obs[self.disease_col].unique())
        n_diseases = len(diseases)

        at2_mask = self.adata.obs['is_at2']
        aberrant_mask = self.adata.obs['is_aberrant_basaloid']

        # Create subplot for each target gene with more spacing
        for i, gene in enumerate(self.available_target_genes):
            ax = plt.subplot(3, 2, i + 1)

            expr = self._get_gene_expression(gene)

            # Collect data for plotting - include all disease-population combinations
            plot_data = []
            all_groups = []

            for disease in diseases:
                disease_mask = self.adata.obs[self.disease_col] == disease

                for pop, pop_mask, pop_name in [
                    ('Other', ~(at2_mask | aberrant_mask), 'Other'),
                    ('AT2', at2_mask, 'AT2'),
                    ('Aberrant', aberrant_mask, 'Aberrant')
                ]:
                    combined_mask = disease_mask & pop_mask
                    group_name = f'{disease}_{pop_name}'
                    all_groups.append(group_name)

                    if combined_mask.sum() > 0:
                        for val in expr[combined_mask]:
                            plot_data.append({
                                'Disease': disease,
                                'Population': pop_name,
                                'Expression': np.log1p(val),
                                'Group': group_name
                            })

            if plot_data:
                plot_df = pd.DataFrame(plot_data)

                # Ensure all groups are represented
                for group in all_groups:
                    if group not in plot_df['Group'].unique():
                        # Add dummy data for missing groups
                        plot_df = pd.concat([plot_df, pd.DataFrame({
                            'Disease': [group.split('_')[0]],
                            'Population': [group.split('_')[1]],
                            'Expression': [np.nan],
                            'Group': [group]
                        })], ignore_index=True)

                # Color mapping
                colors = []
                for group in all_groups:
                    if 'Other' in group:
                        colors.append('lightgray')
                    elif 'AT2' in group:
                        colors.append('blue')
                    else:
                        colors.append('red')

                # Plot with all groups
                sns.boxplot(data=plot_df, x='Group', y='Expression',
                            order=all_groups, palette=colors, ax=ax)

                # Rotate labels
                ax.set_xticklabels([g.replace('_', '\n') for g in all_groups],
                                   rotation=45, ha='right')
                ax.set_ylabel(f'log1p({gene})')
                ax.set_title(f'{gene} Expression by Disease and Population', pad=15)

        # Add summary subplot
        ax_summary = plt.subplot(3, 2, 6)

        # Calculate fold changes for each gene and disease
        fc_data = []

        for gene in self.available_target_genes:
            expr = self._get_gene_expression(gene)

            for disease in diseases:
                disease_mask = self.adata.obs[self.disease_col] == disease

                # Calculate means
                other_mask = disease_mask & ~(at2_mask | aberrant_mask)
                other_mean = np.mean(expr[other_mask]) if other_mask.sum() > 0 else 0.01

                at2_disease_mask = disease_mask & at2_mask
                at2_mean = np.mean(expr[at2_disease_mask]) if at2_disease_mask.sum() > 0 else 0.01

                aberrant_disease_mask = disease_mask & aberrant_mask
                aberrant_mean = np.mean(expr[aberrant_disease_mask]) if aberrant_disease_mask.sum() > 0 else 0.01

                fc_data.append({
                    'Gene': gene,
                    'Disease': disease,
                    'AT2_vs_Other': np.log2(at2_mean / other_mean),
                    'Aberrant_vs_Other': np.log2(aberrant_mean / other_mean)
                })

        fc_df = pd.DataFrame(fc_data)

        # Plot heatmap of fold changes
        pivot_at2 = fc_df.pivot(index='Gene', columns='Disease', values='AT2_vs_Other')
        pivot_aberrant = fc_df.pivot(index='Gene', columns='Disease', values='Aberrant_vs_Other')

        # Combine for plotting
        combined = pd.concat([pivot_at2, pivot_aberrant], keys=['AT2 vs Other', 'Aberrant vs Other'])

        sns.heatmap(combined, annot=True, fmt='.1f', cmap='RdBu_r', center=0,
                    ax=ax_summary, cbar_kws={'label': 'log2(FC)'})
        ax_summary.set_title('Fold Changes by Disease', pad=15)

        plt.suptitle('Disease-Stratified Target Gene Expression', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(os.path.join(self.output_dir, 'plots', 'disease_stratified_comparison.png'), dpi=300)
        plt.close()

    def _plot_senescence_categories_analysis(self):
        """Analyze correlation between target genes and different senescence categories"""
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

        at2_mask = self.adata.obs['is_at2']
        aberrant_mask = self.adata.obs['is_aberrant_basaloid']

        # 1. Overview heatmap of all senescence category scores
        ax1 = fig.add_subplot(gs[0, :])

        # Create score matrix
        score_matrix = []
        score_labels = []

        for pop, mask in [('Other', ~(at2_mask | aberrant_mask)),
                          ('AT2', at2_mask),
                          ('Aberrant', aberrant_mask)]:
            if mask.sum() > 0:
                row = []
                for category in self.available_senescence_by_category.keys():
                    score_name = f'{category}_score'
                    if score_name in self.adata.obs.columns:
                        row.append(self.adata.obs[mask][score_name].mean())
                score_matrix.append(row)
                score_labels.append(pop)

        if score_matrix:
            score_array = np.array(score_matrix)

            # Plot heatmap with actual scores
            sns.heatmap(score_array,
                        xticklabels=[cat.replace('_', ' ').title() for cat in
                                     self.available_senescence_by_category.keys()],
                        yticklabels=score_labels,
                        annot=True, fmt='.3f',
                        cmap='YlOrRd',
                        ax=ax1, cbar_kws={'label': 'Mean Score'})
            ax1.set_title('Senescence Category Scores by Population', pad=15)

        # 2-7. Individual category correlations with target gene score
        category_idx = 0
        for category, markers in self.available_senescence_by_category.items():
            if category_idx >= 6:  # Only plot first 6 categories
                break

            ax = fig.add_subplot(gs[1 + category_idx // 3, category_idx % 3])

            score_name = f'{category}_score'
            if score_name in self.adata.obs.columns and 'target_gene_score' in self.adata.obs.columns:
                # Subsample for plotting
                plot_mask = np.random.choice(len(self.adata),
                                             min(10000, len(self.adata)),
                                             replace=False)

                # Color by population
                colors = []
                for i in plot_mask:
                    if aberrant_mask[i]:
                        colors.append('red')
                    elif at2_mask[i]:
                        colors.append('blue')
                    else:
                        colors.append('lightgray')

                ax.scatter(self.adata.obs.iloc[plot_mask]['target_gene_score'],
                           self.adata.obs.iloc[plot_mask][score_name],
                           c=colors, s=1, alpha=0.5, rasterized=True)

                # Add trend lines for each population
                for pop, mask, color in [('AT2', at2_mask, 'blue'),
                                         ('Aberrant', aberrant_mask, 'red')]:
                    if mask.sum() > 10:
                        x = self.adata.obs[mask]['target_gene_score']
                        y = self.adata.obs[mask][score_name]

                        # Calculate correlation
                        corr = np.corrcoef(x, y)[0, 1]

                        # Fit line
                        z = np.polyfit(x, y, 1)
                        p = np.poly1d(z)

                        x_line = np.linspace(x.min(), x.max(), 100)
                        ax.plot(x_line, p(x_line), color=color, linewidth=2,
                                label=f'{pop}: r={corr:.3f}')

                ax.set_xlabel('Target Gene Score')
                ax.set_ylabel(f'{category.replace("_", " ").title()} Score')
                ax.set_title(f'{category.replace("_", " ").title()}', pad=10)
                ax.legend(fontsize=8)

                # Add marker list
                marker_text = f"Markers: {', '.join(markers[:3])}"
                if len(markers) > 3:
                    marker_text += f"... (+{len(markers) - 3})"
                ax.text(0.02, 0.02, marker_text, transform=ax.transAxes,
                        fontsize=7, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            category_idx += 1

        # 8. Summary comparison across categories
        ax8 = fig.add_subplot(gs[2, 1])

        # Calculate correlations for each category
        corr_data = []

        for category in self.available_senescence_by_category.keys():
            score_name = f'{category}_score'
            if score_name in self.adata.obs.columns and 'target_gene_score' in self.adata.obs.columns:
                for pop, mask in [('AT2', at2_mask), ('Aberrant', aberrant_mask)]:
                    if mask.sum() > 10:
                        corr = np.corrcoef(self.adata.obs[mask]['target_gene_score'],
                                           self.adata.obs[mask][score_name])[0, 1]

                        corr_data.append({
                            'Category': category.replace('_', ' ').title(),
                            'Population': pop,
                            'Correlation': corr
                        })

        if corr_data:
            corr_df = pd.DataFrame(corr_data)
            pivot_df = corr_df.pivot(index='Category', columns='Population', values='Correlation')

            pivot_df.plot(kind='bar', ax=ax8, color=['red', 'blue'])
            ax8.set_ylabel('Correlation with Target Gene Score')
            ax8.set_title('Senescence Category Correlations', pad=10)
            ax8.set_xlabel('')
            ax8.axhline(0, color='black', linestyle='-', alpha=0.3)
            ax8.legend()

        # 9. Statistical summary
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        summary_text = "Senescence Analysis Summary\n" + "=" * 30 + "\n\n"

        # Find highest scoring categories
        summary_text += "Highest scoring categories:\n"

        for pop in ['AT2', 'Aberrant']:
            if pop == 'AT2':
                mask = at2_mask
            else:
                mask = aberrant_mask

            if mask.sum() > 0:
                summary_text += f"\n{pop}:\n"

                scores = []
                for category in self.available_senescence_by_category.keys():
                    score_name = f'{category}_score'
                    if score_name in self.adata.obs.columns:
                        mean_score = self.adata.obs[mask][score_name].mean()
                        scores.append((category, mean_score))

                scores.sort(key=lambda x: x[1], reverse=True)

                for cat, score in scores[:3]:
                    summary_text += f"  {cat.replace('_', ' ')}: {score:.3f}\n"

        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
                 verticalalignment='top', fontfamily='monospace', fontsize=10)

        plt.suptitle('Senescence Categories Analysis', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(os.path.join(self.output_dir, 'plots', 'senescence_categories_analysis.png'), dpi=300)
        plt.close()

    def _plot_enhanced_umaps(self):
        """Create enhanced UMAP visualizations with better spacing"""
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

        at2_mask = self.adata.obs['is_at2']
        aberrant_mask = self.adata.obs['is_aberrant_basaloid']

        # 1. Main population UMAP
        ax1 = fig.add_subplot(gs[0, 0])

        # Plot all cells
        other_mask = ~(at2_mask | aberrant_mask)
        plot_sample = np.random.choice(np.where(other_mask)[0],
                                       min(20000, other_mask.sum()),
                                       replace=False)

        ax1.scatter(self.adata.obsm['X_umap'][plot_sample, 0],
                    self.adata.obsm['X_umap'][plot_sample, 1],
                    c='lightgray', s=0.1, alpha=0.3, rasterized=True)

        # Plot AT2 cells
        ax1.scatter(self.adata.obsm['X_umap'][at2_mask, 0],
                    self.adata.obsm['X_umap'][at2_mask, 1],
                    c='blue', s=2, alpha=0.6, label=f'AT2 (n={at2_mask.sum()})')

        # Plot aberrant cells
        ax1.scatter(self.adata.obsm['X_umap'][aberrant_mask, 0],
                    self.adata.obsm['X_umap'][aberrant_mask, 1],
                    c='red', s=2, alpha=0.8, label=f'Aberrant (n={aberrant_mask.sum()})')

        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        ax1.set_title('Cell Populations', pad=15)
        ax1.legend()
        ax1.set_aspect('equal')

        # 2-6. Individual target genes
        for i, gene in enumerate(self.available_target_genes):
            ax = fig.add_subplot(gs[(i + 1) // 3, (i + 1) % 3])

            expr = self._get_gene_expression(gene)
            expr_log = np.log1p(expr)

            # Plot expression
            plot_sample = np.random.choice(len(self.adata),
                                           min(30000, len(self.adata)),
                                           replace=False)

            scatter = ax.scatter(self.adata.obsm['X_umap'][plot_sample, 0],
                                 self.adata.obsm['X_umap'][plot_sample, 1],
                                 c=expr_log[plot_sample],
                                 cmap='viridis', s=0.5, alpha=0.6,
                                 rasterized=True)

            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title(f'{gene} Expression', pad=10)
            ax.set_aspect('equal')

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(f'log1p({gene})', fontsize=8)

        # 7. Target gene score
        ax7 = fig.add_subplot(gs[2, 0])

        if 'target_gene_score' in self.adata.obs.columns:
            plot_sample = np.random.choice(len(self.adata),
                                           min(30000, len(self.adata)),
                                           replace=False)

            scatter = ax7.scatter(self.adata.obsm['X_umap'][plot_sample, 0],
                                  self.adata.obsm['X_umap'][plot_sample, 1],
                                  c=self.adata.obs.iloc[plot_sample]['target_gene_score'],
                                  cmap='Reds', s=0.5, alpha=0.6,
                                  rasterized=True)

            ax7.set_xlabel('UMAP 1')
            ax7.set_ylabel('UMAP 2')
            ax7.set_title('Target Gene Score', pad=10)
            ax7.set_aspect('equal')

            cbar = plt.colorbar(scatter, ax=ax7)
            cbar.set_label('Target Gene Score', fontsize=8)

        # 8-9. Top senescence categories
        category_list = list(self.available_senescence_by_category.keys())
        for i, category in enumerate(category_list[:2]):  # Plot first two categories
            ax = fig.add_subplot(gs[2, 1 + i])

            score_name = f'{category}_score'
            if score_name in self.adata.obs.columns:
                plot_sample = np.random.choice(len(self.adata),
                                               min(30000, len(self.adata)),
                                               replace=False)

                scatter = ax.scatter(self.adata.obsm['X_umap'][plot_sample, 0],
                                     self.adata.obsm['X_umap'][plot_sample, 1],
                                     c=self.adata.obs.iloc[plot_sample][score_name],
                                     cmap='YlOrRd', s=0.5, alpha=0.6,
                                     rasterized=True)

                ax.set_xlabel('UMAP 1')
                ax.set_ylabel('UMAP 2')
                ax.set_title(f'{category.replace("_", " ").title()} Score', pad=10)
                ax.set_aspect('equal')

                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(f'{category.replace("_", " ").title()}', fontsize=8)

        plt.suptitle('Enhanced UMAP Visualizations', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(os.path.join(self.output_dir, 'plots', 'enhanced_umaps.png'), dpi=300)
        plt.close()

    def _plot_individual_gene_analysis(self):
        """Create detailed analysis for each target gene with better layout"""
        for gene in self.available_target_genes:
            fig = plt.figure(figsize=(18, 14))
            gs = GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

            expr = self._get_gene_expression(gene)
            at2_mask = self.adata.obs['is_at2']
            aberrant_mask = self.adata.obs['is_aberrant_basaloid']

            # 1. Expression distribution
            ax1 = fig.add_subplot(gs[0, 0])

            # Create histogram data
            hist_data = {
                'Other': np.log1p(expr[~(at2_mask | aberrant_mask)]),
                'AT2': np.log1p(expr[at2_mask]),
                'Aberrant': np.log1p(expr[aberrant_mask])
            }

            for pop, color in [('Other', 'lightgray'), ('AT2', 'blue'), ('Aberrant', 'red')]:
                if len(hist_data[pop]) > 0:
                    ax1.hist(hist_data[pop], bins=50, alpha=0.5, label=pop,
                             color=color, density=True)

            ax1.set_xlabel(f'log1p({gene})')
            ax1.set_ylabel('Density')
            ax1.set_title(f'{gene} Expression Distribution', pad=10)
            ax1.legend()

            # 2. UMAP with expression
            ax2 = fig.add_subplot(gs[0, 1])

            plot_sample = np.random.choice(len(self.adata),
                                           min(30000, len(self.adata)),
                                           replace=False)

            scatter = ax2.scatter(self.adata.obsm['X_umap'][plot_sample, 0],
                                  self.adata.obsm['X_umap'][plot_sample, 1],
                                  c=np.log1p(expr[plot_sample]),
                                  cmap='viridis', s=0.5, alpha=0.6,
                                  rasterized=True)

            ax2.set_xlabel('UMAP 1')
            ax2.set_ylabel('UMAP 2')
            ax2.set_title(f'{gene} Expression in UMAP', pad=10)
            ax2.set_aspect('equal')

            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label(f'log1p({gene})', fontsize=8)

            # 3. Disease comparison
            ax3 = fig.add_subplot(gs[0, 2])

            if self.disease_col:
                disease_data = []

                for disease in self.adata.obs[self.disease_col].unique():
                    disease_mask = self.adata.obs[self.disease_col] == disease

                    for pop, pop_mask in [('Other', ~(at2_mask | aberrant_mask)),
                                          ('AT2', at2_mask),
                                          ('Aberrant', aberrant_mask)]:
                        combined_mask = disease_mask & pop_mask

                        if combined_mask.sum() > 5:
                            disease_data.append({
                                'Disease': disease,
                                'Population': pop,
                                'Expression': np.mean(expr[combined_mask])
                            })

                if disease_data:
                    disease_df = pd.DataFrame(disease_data)
                    pivot_df = disease_df.pivot(index='Disease', columns='Population', values='Expression')

                    pivot_df.plot(kind='bar', ax=ax3,
                                  color=['red', 'blue', 'lightgray'])
                    ax3.set_ylabel('Mean Expression')
                    ax3.set_title(f'{gene} by Disease', pad=10)
                    ax3.legend(fontsize=8)

            # 4-6. Senescence category correlations
            for i, category in enumerate(list(self.available_senescence_by_category.keys())[:3]):
                ax = fig.add_subplot(gs[1, i])

                score_name = f'{category}_score'
                if score_name in self.adata.obs.columns:
                    # Subsample for plotting
                    plot_sample = np.random.choice(len(self.adata),
                                                   min(5000, len(self.adata)),
                                                   replace=False)

                    # Color by population
                    colors = []
                    for idx in plot_sample:
                        if aberrant_mask[idx]:
                            colors.append('red')
                        elif at2_mask[idx]:
                            colors.append('blue')
                        else:
                            colors.append('lightgray')

                    ax.scatter(np.log1p(expr[plot_sample]),
                               self.adata.obs.iloc[plot_sample][score_name],
                               c=colors, s=5, alpha=0.5)

                    ax.set_xlabel(f'log1p({gene})')
                    ax.set_ylabel(f'{category.replace("_", " ").title()} Score')
                    ax.set_title(f'{gene} vs {category.replace("_", " ").title()}', pad=10)

                    # Add correlations
                    at2_corr = np.corrcoef(expr[at2_mask],
                                           self.adata.obs[at2_mask][score_name])[
                        0, 1] if at2_mask.sum() > 10 else np.nan
                    aberrant_corr = np.corrcoef(expr[aberrant_mask],
                                                self.adata.obs[aberrant_mask][score_name])[
                        0, 1] if aberrant_mask.sum() > 10 else np.nan

                    ax.text(0.02, 0.98, f'AT2 r={at2_corr:.2f}\nAberrant r={aberrant_corr:.2f}',
                            transform=ax.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                            fontsize=8)

            # 7. Cell type specificity
            ax7 = fig.add_subplot(gs[2, 0])

            if self.manuscript_col:
                # Find top expressing cell types
                ct_expr = []

                for ct in self.adata.obs[self.manuscript_col].value_counts().head(30).index:
                    ct_mask = self.adata.obs[self.manuscript_col] == ct

                    if ct_mask.sum() > 10:
                        mean_expr = np.mean(expr[ct_mask])
                        pct_expr = (expr[ct_mask] > 0).sum() / ct_mask.sum() * 100

                        ct_expr.append({
                            'Cell_Type': ct,
                            'Mean_Expression': mean_expr,
                            'Pct_Expressing': pct_expr,
                            'Is_AT2': 'AT2' in ct or 'ATII' in ct,
                            'Is_Aberrant': 'Aberrant' in ct
                        })

                if ct_expr:
                    ct_df = pd.DataFrame(ct_expr).sort_values('Mean_Expression', ascending=False).head(15)

                    # Color by type
                    colors = []
                    for _, row in ct_df.iterrows():
                        if row['Is_AT2']:
                            colors.append('blue')
                        elif row['Is_Aberrant']:
                            colors.append('red')
                        else:
                            colors.append('gray')

                    ct_df.set_index('Cell_Type')['Mean_Expression'].plot(
                        kind='barh', ax=ax7, color=colors)
                    ax7.set_xlabel('Mean Expression')
                    ax7.set_title(f'Top Cell Types Expressing {gene}', pad=10)

            # 8-9. Summary statistics
            ax_summary = fig.add_subplot(gs[2, 1:])
            ax_summary.axis('off')

            summary_text = f"{gene} Summary Statistics\n" + "=" * 40 + "\n\n"

            # Expression statistics
            summary_text += "Mean expression:\n"
            summary_text += f"  Other: {np.mean(expr[~(at2_mask | aberrant_mask)]):.3f}\n"
            summary_text += f"  AT2: {np.mean(expr[at2_mask]):.3f}\n" if at2_mask.sum() > 0 else "  AT2: NA\n"
            summary_text += f"  Aberrant: {np.mean(expr[aberrant_mask]):.3f}\n\n" if aberrant_mask.sum() > 0 else "  Aberrant: NA\n\n"

            # Percentage expressing
            summary_text += "Percentage expressing:\n"
            summary_text += f"  Other: {(expr[~(at2_mask | aberrant_mask)] > 0).sum() / (~(at2_mask | aberrant_mask)).sum() * 100:.1f}%\n"
            summary_text += f"  AT2: {(expr[at2_mask] > 0).sum() / at2_mask.sum() * 100:.1f}%\n" if at2_mask.sum() > 0 else "  AT2: NA\n"
            summary_text += f"  Aberrant: {(expr[aberrant_mask] > 0).sum() / aberrant_mask.sum() * 100:.1f}%\n\n" if aberrant_mask.sum() > 0 else "  Aberrant: NA\n\n"

            # Fold changes
            if at2_mask.sum() > 0:
                fc_at2 = np.mean(expr[at2_mask]) / (np.mean(expr[~(at2_mask | aberrant_mask)]) + 0.01)
                summary_text += f"Fold change AT2/Other: {fc_at2:.2f}\n"

            if aberrant_mask.sum() > 0:
                fc_aberrant = np.mean(expr[aberrant_mask]) / (np.mean(expr[~(at2_mask | aberrant_mask)]) + 0.01)
                summary_text += f"Fold change Aberrant/Other: {fc_aberrant:.2f}\n"

            ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                            verticalalignment='top', fontfamily='monospace', fontsize=10)

            plt.suptitle(f'{gene} Detailed Analysis', fontsize=16, y=0.98)
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            plt.savefig(os.path.join(self.output_dir, 'plots', f'{gene}_detailed_analysis.png'), dpi=300)
            plt.close()

    def save_results(self):
        """Save analysis results"""
        print("\nSaving results...")

        # Save cell populations
        at2_cells = self.adata[self.adata.obs['is_at2']].copy()
        aberrant_cells = self.adata[self.adata.obs['is_aberrant_basaloid']].copy()

        # Save as h5ad (subsample if too large)
        if at2_cells.n_obs > 10000:
            subsample = np.random.choice(at2_cells.n_obs, 10000, replace=False)
            at2_cells = at2_cells[subsample]
        at2_cells.write(os.path.join(self.output_dir, 'at2_cells.h5ad'))

        if aberrant_cells.n_obs > 10000:
            subsample = np.random.choice(aberrant_cells.n_obs, 10000, replace=False)
            aberrant_cells = aberrant_cells[subsample]
        aberrant_cells.write(os.path.join(self.output_dir, 'aberrant_basaloid_cells.h5ad'))

        # Save metadata
        metadata_df = self.adata.obs.copy()
        metadata_df.to_csv(os.path.join(self.output_dir, 'all_cells_metadata.csv'))

        # Save detailed statistics
        stats = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_cells': int(self.adata.n_obs),
            'at2_cells': int(self.n_at2),
            'aberrant_cells': int(self.n_aberrant),
            'target_genes': {
                'requested': self.target_genes,
                'available': self.available_target_genes,
                'missing': self.missing_target_genes
            },
            'expression_statistics': {},
            'senescence_statistics': {},
            'disease_statistics': {}
        }

        # Add expression statistics for each target gene
        for gene in self.available_target_genes:
            expr = self._get_gene_expression(gene)

            at2_mask = self.adata.obs['is_at2']
            aberrant_mask = self.adata.obs['is_aberrant_basaloid']
            other_mask = ~(at2_mask | aberrant_mask)

            stats['expression_statistics'][gene] = {
                'mean_expression': {
                    'other': float(np.mean(expr[other_mask])),
                    'at2': float(np.mean(expr[at2_mask])) if at2_mask.sum() > 0 else None,
                    'aberrant': float(np.mean(expr[aberrant_mask])) if aberrant_mask.sum() > 0 else None
                },
                'pct_expressing': {
                    'other': float((expr[other_mask] > 0).sum() / other_mask.sum() * 100),
                    'at2': float((expr[at2_mask] > 0).sum() / at2_mask.sum() * 100) if at2_mask.sum() > 0 else None,
                    'aberrant': float((expr[
                                           aberrant_mask] > 0).sum() / aberrant_mask.sum() * 100) if aberrant_mask.sum() > 0 else None
                },
                'fold_changes': {
                    'at2_vs_other': float(
                        np.mean(expr[at2_mask]) / (np.mean(expr[other_mask]) + 0.01)) if at2_mask.sum() > 0 else None,
                    'aberrant_vs_other': float(np.mean(expr[aberrant_mask]) / (
                            np.mean(expr[other_mask]) + 0.01)) if aberrant_mask.sum() > 0 else None
                }
            }

        # Add senescence category statistics
        for category, markers in self.available_senescence_by_category.items():
            score_name = f'{category}_score'
            if score_name in self.adata.obs.columns:
                stats['senescence_statistics'][category] = {
                    'markers': markers,
                    'mean_scores': {
                        'other': float(self.adata.obs[other_mask][score_name].mean()),
                        'at2': float(self.adata.obs[at2_mask][score_name].mean()) if at2_mask.sum() > 0 else None,
                        'aberrant': float(
                            self.adata.obs[aberrant_mask][score_name].mean()) if aberrant_mask.sum() > 0 else None
                    }
                }

        # Save statistics
        with open(os.path.join(self.output_dir, 'analysis_statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\nResults saved to: {self.output_dir}")
        print("Generated files:")
        print("  - at2_cells.h5ad")
        print("  - aberrant_basaloid_cells.h5ad")
        print("  - all_cells_metadata.csv")
        print("  - analysis_statistics.json")
        print("  - plots/ (multiple visualizations)")

    def run_analysis(self):
        """Run complete focused analysis"""
        self.identify_cell_populations()
        self.create_focused_visualizations()
        self.save_results()
        return self.adata


def main():
    """Main analysis pipeline"""
    # Set working directory
    base_dir = "/home/cathal/Desktop/working/Single_Cell/scrnaseq_analysis"

    print("=" * 60)
    print("FOCUSED TARGET GENE ANALYSIS")
    print("Target Genes: PTCHD4, EDA2R, MUC3A, ITGBL1, PLAUR")
    print("=" * 60)

    # Set paths
    data_dir = base_dir
    output_dir = os.path.join(base_dir, 'target_gene_focused_analysis_v7')

    # Load data
    print(f"\nWorking from: {base_dir}")
    print("Loading data...")
    cache_dir = os.path.join(data_dir, 'cache')
    adata_path = os.path.join(cache_dir, '05_final.h5ad')

    if not os.path.exists(adata_path):
        raise FileNotFoundError(f"Could not find {adata_path}")

    adata = sc.read_h5ad(adata_path)

    # Filter for QC passed cells
    if 'passed_qc' in adata.obs.columns:
        adata = adata[adata.obs['passed_qc']].copy()

    print(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")
    print(f"Output directory: {output_dir}")

    # Check and clean gene names
    print("\nProcessing gene names...")

    if 'gene_symbol' in adata.var.columns:
        print("Found gene_symbol in var annotations")
        gene_symbols = [str(symbol).strip('"') for symbol in adata.var['gene_symbol']]
        adata.var_names = gene_symbols
        adata.var_names_make_unique()
    else:
        print("Using existing gene names")
        cleaned_names = [str(name).strip('"') for name in adata.var_names]
        adata.var_names = cleaned_names
        adata.var_names_make_unique()

    print(f"Sample of available gene names: {list(adata.var_names)[:10]}")

    # Create output structure
    os.makedirs(output_dir, exist_ok=True)

    # Run focused analysis
    print("\n" + "=" * 60)
    print("RUNNING FOCUSED ANALYSIS")
    print("=" * 60)

    analysis = TargetGeneAnalysis(adata, output_dir)
    analysis.run_analysis()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated visualizations:")
    print("  1. target_gene_overview.png - Expression overview across populations")
    print("  2. all_cell_types_analysis.png - Analysis across ALL cell types with log expression")
    print("  3. cell_type_target_gene_clustermap.png - Hierarchical clustering with log expression")
    print("  4. combined_gene_heatmap.png - Combined heatmap with gene-wise normalization and dual dendrograms")
    print("  5. combined_gene_heatmap_log.png - Combined heatmap with log expression values")
    print("\n  NEW CORRELATION ANALYSES:")
    print("  6. all_genes_expression_heatmap.png - All genes with row scaling and group labels")
    print("  7. target_gene_correlations.png - Pearson and Spearman correlations between target genes")
    print("  8. target_vs_celltypes.png - Target gene expression patterns across cell types")
    print("  9. target_genes_by_disease_clustered.png - Clustered heatmap of genes by disease")
    print("  10. target_vs_disease_analysis.png - Individual gene analysis by disease and population")
    print("  11. population_specific_correlations.png - Separate correlations for AT2, Aberrant, and Other")
    print("\n  ORIGINAL ANALYSES:")
    print("  12. disease_stratified_comparison.png - Disease-specific expression patterns")
    print("  13. senescence_categories_analysis.png - Separated senescence marker analysis")
    print("  14. enhanced_umaps.png - UMAP visualizations with improved spacing")
    print("  15. [gene]_detailed_analysis.png - Individual analysis for each target gene")
    print("\nKey improvements:")
    print("  - All cell types included in heatmaps (even with zero expression)")
    print("  - Using log1p(expression) instead of z-scores to preserve biological differences")
    print("  - Consistent cell type ordering across all panels")
    print("  - Senescence marker category legend panel moved to avoid overlap")
    print("  - Red lines separate major (>100 cells) from minor cell types")
    print("  - NEW: Comprehensive correlation analyses between genes, cell types, and conditions")
    print("  - NEW: Combined heatmap with gene-wise normalization and hierarchical clustering on both axes")
    print("  - FIXED: Gene names colored by type (red=target, blue=senescence) and placed outside heatmap")
    print("  - FIXED: GridSpec sizing for variable number of target genes")
    print("  - FIXED: Overlapping labels in heatmaps with truncation and better spacing")
    print("  - FIXED: Improved figure sizes and font sizes for readability")
    print("  - FIXED: Better error handling for missing data")


if __name__ == "__main__":
    main()
