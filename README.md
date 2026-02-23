# Py_Anndata_Visualizer (PyAV)

> Inspired by Xenium Explorer, Napari, and CellxGene. Designed for scientific research and exploratory data analysis.

<p align="center">
  <img src="img/demo_hc.gif" width="800">
</p>

# 🔬 PyAnnData Visualizer - User Guide

**An interactive spatial plotting tool for single-cell data analysis**

---

## What's New in 0.3.0

### Regions Tool
Programmatically mask large areas across tissue using DBSCAN clustering and alpha shape boundaries. Define a cell type column and category, tune clustering parameters (radius, min_cells), then generate smooth polygon masks. Region masks save as geometries to `adata.uns['region_masks']` and can be imported back. Region labels display at polygon centroids with per-region visibility toggles.

### Spatial Heatmap Ribbon
Draw a bezier ribbon path across tissue to visualize how gene expression changes spatially. A heatmap panel below the visualizer shows mean GEX per bin from start (S) to end (E). Adjustable bins (5–500), variable ribbon width, and automatic colormap matching. Filter bins by toggling cell type categories on/off.

### Manual Selection Persistence
Manual lasso/polygon/rectangle/circle selections can now be saved to `adata.uns['manual_masks']` and imported back, preserving groups and cell indices across sessions. Selection labels display at centroids on the canvas.

### UI Reorganization
The sidebar is reorganized with a **Color By** parent section containing **GEX (gene)** and **Observations** subsections. **Masks** contains **Manual Selection** and **Regions**. All subsections are independently collapsible.


#### Known problems:
Working to resolve an issue where when switching between layouts region maks and manual masks are not tracking with cores. There's also a small glitch in loading large datasets and occasionally a glitch requiring user to click at the bottom of the graph to extend the canvas out all the way. Heatmap tool can be glitchy and sometimes reappear when recoloring cells. Planning to resolve in 0.3.1 coming soon.

---

## Getting Started

### Prerequisites
- Pre-annotated AnnData object with spatial coordinates
- For large datasets, consider subsetting to a region of interest (e.g., individual tissue section from TMA)
- Works with both spatial and standard single-cell data

### Recommended: Conda Installation
```bash
# 1. Create environment
conda create -n anndata-viz python=3.10
conda activate anndata-viz

# 2. Install dependencies via mamba/conda. We reccomend using mamba 
mamba install -c conda-forge scanpy squidpy jupyterlab ipywidgets

# 3. Install visualizer
pip install git+https://github.com/Zach-Sten/py_anndata_visualizer.git

# 4. Launch Jupyter
jupyter lab
```

---

## Basic Workflow

#### 1️⃣ Color by Cell Annotations
- Enter an `obs` column name (e.g., `"cell_type"`)
- Click **Apply** to visualize
- Existing color schemes load automatically; defaults assigned if none exist
- Toggle categories on/off in the legend
- Save edited color schemes directly back to adata

#### 2️⃣ Visualize Gene Expression
- Type a gene name from `adata.var_names`
- Click **Add Gene** to overlay expression
- Choose from multiple colormaps (viridis, magma, inferno, etc.)
- Click gene chips to toggle visualization
- Drag gene chips into a group, then click the group to see geometric mean of multiple markers
- Combine with obs coloring for dual-layer views

#### 3️⃣ Switch Embeddings
- Toggle between **Spatial**, **UMAP**, and **PCA** views, plus custom layouts
- Smooth animated transitions preserve your color state
- Uses `adata.obsm['spatial']`, `adata.obsm['X_umap']`, `adata.obsm['X_pca']`

#### 4️⃣ Region Masking
- Open **Masks → Regions** in the sidebar
- Set a cell type column and category (e.g., `stroma_epi` / `Epi`)
- Tune DBSCAN parameters: `radius` and `min_cells`
- Click **Run DBSCAN** to identify clusters, then **Generate Masks** for alpha shape boundaries
- Save/import region masks to `adata.uns['region_masks']`
- Toggle region labels and adjust fill opacity / outline weight

#### 5️⃣ Spatial Heatmap
- Add genes of interest, then click **+ Heatmap** to create a heatmap group
- Click **Edit Ribbon** to place a bezier path across tissue
- Click to set start (S), then end (E), then drag control points and width handles
- The heatmap panel renders mean GEX per bin along the ribbon
- Adjust bins with the slider; toggle cell categories to filter
- Heatmap colormap updates automatically when you change the GEX colormap

---

## Navigation Controls

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
3. **Important**: Only **visible points** are selectable (respects category toggles)
4. **Deselect tool**: Click the active tool again to return to pan mode
5. **Edit selection**: Drag edges, points, or rotation handles to adjust

### Managing Selections

| Action | How To |
|--------|--------|
| **Rename** | Click selection name to rename |
| **Group** | Drag selections into a group folder |
| **Save to adata.obs** | Click **Save** on a group folder |
| **Save to adata.uns** | Click **Save to .uns** to persist across sessions |
| **Import** | Click **Import from .uns** to restore saved selections |
| **Delete** | Click **×** on selection or group |

### Saving Groups to AnnData

- Grouped selections create a new `adata.obs` column
- Group name becomes the column name
- Individual selection names become category labels
- Unselected cells are labeled as `NaN`
- Color schemes are saved to `adata.uns` as `'{column}_colors'`
- Layouts are saved to `adata.obsm` as `'X_{layout}'`

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `←` `→` | Adjust point size |
| `R` | Toggle rotation mode |
| `R` `R` | Reset rotation |
| Double-click | Reset zoom |

---

## Data Persistence

| Data | Location | Method |
|------|----------|--------|
| Color schemes | `adata.uns['{col}_colors']` | **Save colors** button |
| Custom layouts | `adata.obsm['X_{name}']` | Layout **Save** button |
| Region masks | `adata.uns['region_masks']` | Regions **Save to .uns** |
| Manual selections | `adata.uns['manual_masks']` | Selections **Save to .uns** |
| Cell annotations | `adata.obs['{group}']` | Group **Save** button |

---

## 📦 Requirements

```
Python ≥3.8
numpy, pandas, scipy, scanpy, anndata, squidpy
ipywidgets, IPython
alphashape, scikit-learn
```

---

## 🐛 Troubleshooting

**Controls not loading?**
- Try setting `debug=True` and use F12 to open the browser console for detailed logs.

**Performance issues?**
- Cells load in chunks of 250,000. Adjust `chunk_size` and `max_result_size` in `create_adata_interface()`.
- For very large datasets, subset to a region of interest first.

**Widget "model not found" errors?**
- Restart the kernel, refresh the browser (Cmd+Shift+R), clear outputs, then re-run.

---

## 📝 Development Status

This tool is under active development. Features are evolving rapidly.

- ✅ Region masking with DBSCAN + alpha shapes
- ✅ Spatial heatmap ribbon tool
- ✅ Manual selection persistence to adata.uns
- 🔜 Segmentation mask overlays
- 🔜 Scale bars and distance measurements

---

## Example Usage

```python
from py_anndata_visualizer import create_adata_interface
import scanpy as sc

# Load your spatial dataset
adata = sc.datasets.visium_sge()

# Create interactive visualizer
create_adata_interface(adata, figsize=(900, 600), sample_id='core_id')
```

---

**Questions or feedback?** Open an issue on GitHub or contact the development team.
