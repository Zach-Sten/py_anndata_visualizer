# Py_Anndata_Visualizer

> Inspired by Xenium Explorer, Napari, and CellxGene. Designed for scientific research and exploratory data analysis.

<p align="center">
  <img src="img/demo.png" width="800">
</p>

# 🔬 PyAnnData Visualizer - User Guide

**An interactive spatial plotting tool for single-cell data analysis**

---

## Getting Started

### Prerequisites
- Pre-annotated AnnData object with spatial coordinates
- For large datasets, consider subsetting to a region of interest (e.g., individual tissue section from TMA)
- Works with both spatial and standard single-cell data

## Recommended: Conda Installation
```bash
# 1. Create environment
conda create -n anndata-viz python=3.10
conda activate anndata-viz

# 2. Install dependencies via conda
conda install -c conda-forge scanpy jupyterlab ipywidgets

# 3. Install visualizer
pip install git+https://github.com/Zach-Sten/py_anndata_visualizer.git

# 4. Launch Jupyter
jupyter lab
```

### Basic Workflow

#### 1️⃣ Color by Cell Annotations
- Enter an `obs` column name (e.g., `"cell_type"`)
- Click **Apply** to visualize
- Existing color schemes will load automatically
- Default colors assigned if none exist
- Toggle categories on/off in the legend
- Save edited color schemes directly back to adata

#### 2️⃣ Visualize Gene Expression
- Type a gene name from `adata.var_names`
- Click **Add Gene** to overlay expression (viridis colormap)
- Click gene chips to toggle visualization
- Drag gene chips into group then click group to get geometric mean of multiple markers.
- Combine with obs coloring for dual-layer views

#### 3️⃣ Switch Embeddings
- Toggle between **Spatial**, **UMAP**, and **PCA** views and now custom layours.
- Smooth animated transitions preserve your color state
- Uses `adata.obsm['spatial']`, `adata.obsm['X_umap']`, `adata.obsm['X_pca']`

---

## Visualization Controls

### Navigation

| Action | Controls |
|--------|----------|
| **Zoom** | Mouse wheel / trackpad pinch |
| **Reset Zoom** | Double-click |
| **Pan** | Click + drag on plot |
| **Point Size** | Slider or ← → arrow keys |
| **Rotate** | Press `R`, move mouse, click to lock |
| **Reset Rotation** | Double-tap `R` or double-click |
| **Export Image** | Click 📷 camera button |

> 💡 **Tip**: The minimap (bottom-left) shows your current view position

---

## Selection Tools

### Creating Selections

1. **Choose a tool**: Lasso, Rectangle, Polygon, Circle
2. **Draw on plot**: Click/drag to create closed regions
3. **Important**: Only **visible points** are selectable (respect category toggles)
4. **Deselect tool**: Click tool again to return to pan mode
5. **Edit selection**: In a selection, drag edges, points, or rotations to edit.

### Managing Selections

| Action | How To |
|--------|--------|
| **Rename** | Click to rename |
| **Multi-select** | Drag into new group |
| **Save to AnnData** | Click **Save** on group folder |
| **Delete** | Click **×** on selection/group |

### Saving Groups to AnnData

- Grouped selections create a new `adata.obs` column
- Group name becomes the column name
- Individual selection names become category labels
- Unselected cells are labeled as `NaN`
- Color schemes are saved to adata.uns as '{color_by}_colors'
- Layouts are saved to adata.obsm as 'X_{layout}'

---

## 🔧 Tips & Best Practices

⚠️ **Note:**
- As of right now cells are loaded in 250,000 at a time. This along with the max memory can be adjusted for larger datasets. Check pyav.create_adata_intereface? for more details.

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `←` `→` | Adjust point size |
| `R` | Toggle rotation mode |
| `R` `R` | Reset rotation |
| `click` `click` | Reset zoom |

---

## 📦 Requirements

```
Python ≥3.8
numpy, pandas, scipy, squidpy
ipywidgets, IPython
```

---

## 🐛 Troubleshooting

**Controls not loading?**
- Try turning debug to TRUE and use fn + f12 to get a detailed report in your browser console.

**Performance issues?**
- Subset your data to <100K cells
- Use lower resolution embeddings

---

## 📝 Development Status

This tool is under active development! Features are evolving rapidly.
- Coming: Segmentation masks
- Coming: Size bars and measurements for distances.

---

## Example Usage

```python
from py_anndata_visualizer import create_adata_interface
import scanpy as sc

# Load your spatial dataset
adata = sc.datasets.visium_sge()

# Create interactive visualizer
create_adata_interface(adata, figsize=(900, 600), sample_id = 'core_id')
```

---

**Questions or feedback?** Open an issue on GitHub or contact the development team.


