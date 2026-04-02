"""
run_rough_annotation_classifer.py — Rank-gene XGBoost cell type classifier.

Trains an XGBoost classifier on rank-transformed gene expression from a labeled
reference h5ad, then predicts cell type labels on newly segmented data.

The rank transformation mirrors Geneformer's approach: within each cell, genes
are ranked by expression (rank 1 = most highly expressed, rank 0 = not detected).
This makes the feature space comparable across cells with different total counts
without requiring normalization.

Usage:
    python run_rough_annotation_classifer.py \
        --reference /path/to/reference.h5ad \
        --celltype-col cell_type \
        --query /path/to/segmented.h5ad \
        --output-dir /path/to/output \
        --sample-id XETG... \
        [--gpu]
"""

import os
import sys
import argparse
import warnings
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from scipy import sparse
from scipy.stats import rankdata as _rankdata
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from segger_functions.metrics import find_markers, find_mutually_exclusive_genes

warnings.filterwarnings("ignore")


# ── Rank transformation ──────────────────────────────────────────────────────

def counts_to_rank(adata, desc="Ranking genes"):
    """
    Convert a counts matrix to within-cell gene ranks.

    For each cell, genes are ranked by expression value (rank 1 = highest).
    Genes with zero counts receive rank 0 and contribute no signal.
    Uses adata.layers['counts'] if present, otherwise adata.X.

    Returns a dense float32 array of shape (n_cells, n_genes).
    """
    if "counts" in adata.layers:
        X = adata.layers["counts"]
    else:
        X = adata.X

    if sparse.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    ranks = np.zeros_like(X, dtype=np.float32)
    for i in tqdm(range(X.shape[0]), desc=f"  {desc}", leave=False):
        row = X[i]
        nonzero = row > 0
        if nonzero.any():
            # Dense rank descending: highest expression = rank 1
            ranks[i, nonzero] = _rankdata(-row[nonzero], method="dense").astype(np.float32)

    return ranks


# ── Gene alignment ───────────────────────────────────────────────────────────

def align_genes(ref, query):
    """
    Subset both AnnData objects to their shared genes in the same order.

    Returns (ref_subset, query_subset). Raises if there is no overlap,
    warns if overlap is suspiciously low (likely a gene name format mismatch).
    """
    common = ref.var_names.intersection(query.var_names)
    n_common = len(common)
    if n_common == 0:
        raise ValueError(
            "Reference and query share no gene names. "
            "Check that both use the same format (gene symbols vs Ensembl IDs)."
        )
    n_ref, n_q = len(ref.var_names), len(query.var_names)
    pct = 100 * n_common / min(n_ref, n_q)
    print(f"[INFO] Gene alignment: {n_ref} ref × {n_q} query → {n_common} shared ({pct:.0f}%)")
    if pct < 50:
        print("[WARN] Fewer than 50% of genes overlap — check gene name format.")
    return ref[:, common].copy(), query[:, common].copy()


# ── Classifier ───────────────────────────────────────────────────────────────

def train_classifier(X_train, y_train, use_gpu=False):
    """
    Encode labels and train XGBoost on rank-transformed reference data.

    Runs a quick 3-fold cross-validation for a sanity-check accuracy estimate,
    then fits the final model on all reference cells.

    Returns (clf, label_encoder).
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)
    sample_weights = compute_sample_weight("balanced", y_enc)

    xgb_params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        n_jobs=-1,
        device="cuda" if use_gpu else "cpu",
        random_state=42,
    )

    # Cross-val sanity check
    print("[INFO] Running 3-fold cross-validation on reference data...")
    cv_scores = cross_val_score(
        xgb.XGBClassifier(**xgb_params),
        X_train, y_enc,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring="balanced_accuracy",
        n_jobs=1,  # XGBoost already parallelizes; avoid nested parallelism
    )
    print(f"[INFO] CV balanced accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Final fit on full reference
    print("[INFO] Fitting final classifier on full reference data...")
    clf = xgb.XGBClassifier(**xgb_params)
    clf.fit(X_train, y_enc, sample_weight=sample_weights, verbose=False)
    print("[INFO] Training complete.")

    return clf, le


def save_classifier_cache(cache_dir: Path, clf, le, gene_list,
                          reference_path: str = "", celltype_col: str = ""):
    """Save model, label encoder, gene list, and reference metadata to cache_dir."""
    import json
    cache_dir.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(cache_dir / "model.json"))
    with open(cache_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    (cache_dir / "gene_list.txt").write_text("\n".join(gene_list))
    meta = {
        "reference_path": str(Path(reference_path).resolve()) if reference_path else "",
        "celltype_col":   celltype_col,
    }
    (cache_dir / "cache_info.json").write_text(json.dumps(meta, indent=2))
    print(f"[INFO] Classifier cached: {cache_dir}")


def save_marker_cache(cache_dir: Path, markers: dict, gene_pairs: list):
    """Save segger markers and gene_pairs to cache_dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "markers.pkl", "wb") as f:
        pickle.dump(markers, f)
    with open(cache_dir / "gene_pairs.pkl", "wb") as f:
        pickle.dump(gene_pairs, f)
    print(f"[INFO] Marker cache saved: {len(markers)} cell types, {len(gene_pairs)} gene pairs")


def load_marker_cache(cache_dir: Path):
    """Load markers and gene_pairs from cache_dir. Returns (None, None) if not found."""
    m_path  = cache_dir / "markers.pkl"
    gp_path = cache_dir / "gene_pairs.pkl"
    if not (m_path.exists() and gp_path.exists()):
        return None, None
    with open(m_path, "rb") as f:
        markers = pickle.load(f)
    with open(gp_path, "rb") as f:
        gene_pairs = pickle.load(f)
    print(f"[INFO] Loaded marker cache: {len(markers)} cell types, {len(gene_pairs)} gene pairs")
    return markers, gene_pairs


def load_classifier_cache(cache_dir: Path,
                          reference_path: str = "", celltype_col: str = ""):
    """Load model from cache_dir; returns None if not found or reference has changed."""
    import json
    model_path = cache_dir / "model.json"
    le_path    = cache_dir / "label_encoder.pkl"
    gene_path  = cache_dir / "gene_list.txt"
    if not all(p.exists() for p in [model_path, le_path, gene_path]):
        return None, None, None

    # Validate that the cache was built from the same reference + cell type column
    info_path = cache_dir / "cache_info.json"
    if info_path.exists() and reference_path:
        meta = json.loads(info_path.read_text())
        cached_ref = meta.get("reference_path", "")
        cached_col = meta.get("celltype_col", "")
        current_ref = str(Path(reference_path).resolve())
        if cached_ref != current_ref or (celltype_col and cached_col != celltype_col):
            print(f"[INFO] Cache reference mismatch — retraining.")
            print(f"[INFO]   cached:  {cached_ref}  col={cached_col}")
            print(f"[INFO]   current: {current_ref}  col={celltype_col}")
            return None, None, None

    clf = xgb.XGBClassifier()
    clf.load_model(str(model_path))
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    gene_list = gene_path.read_text().strip().splitlines()
    print(f"[INFO] Loaded cached classifier ({len(gene_list)} genes, {len(le.classes_)} classes)")
    return clf, le, gene_list


def predict_labels(clf, le, X_query):
    """
    Predict cell type labels and per-cell confidence score.

    Confidence = max class probability from predict_proba, so it reflects
    how decisive the classifier was, not just whether it was right.

    Returns (labels: ndarray[str], confidence: ndarray[float32]).
    """
    proba = clf.predict_proba(X_query)
    pred_enc = np.argmax(proba, axis=1)
    confidence = proba.max(axis=1).astype(np.float32)
    labels = le.inverse_transform(pred_enc)
    return labels, confidence


# ── Main ─────────────────────────────────────────────────────────────────────

def _predict_and_save(clf, le, gene_list, query_path: Path, output_dir: Path, sample_id: str, method: str):
    """Load one query h5ad, align to the training gene list (padding missing genes with
    rank 0), rank-transform, predict, and save CSV + annotated h5ad."""
    print(f"\n── {method} ──")
    query = sc.read_h5ad(query_path)
    print(f"[INFO] {query.n_obs} cells × {query.n_vars} genes")

    # Rank genes present in the query, then reindex to the full training gene list.
    # Missing genes get rank 0 (= not detected) — safe for rank-based classifiers.
    present = [g for g in gene_list if g in query.var_names]
    print(f"[INFO] Gene alignment: {len(present)} / {len(gene_list)} training genes present")
    X_present = counts_to_rank(query[:, present].copy(), desc=f"Ranking {method} genes")

    if len(present) < len(gene_list):
        gene_index = {g: i for i, g in enumerate(gene_list)}
        X_query = np.zeros((query.n_obs, len(gene_list)), dtype=np.float32)
        col_idx = [gene_index[g] for g in present]
        X_query[:, col_idx] = X_present
    else:
        X_query = X_present

    labels, confidence = predict_labels(clf, le, X_query)

    query.obs["predicted_cell_type"]            = labels
    query.obs["predicted_cell_type_confidence"] = confidence

    output_dir.mkdir(parents=True, exist_ok=True)
    query.write_h5ad(output_dir / f"{sample_id}_annotated.h5ad")
    csv_df = query.obs[["predicted_cell_type", "predicted_cell_type_confidence"]].copy()
    csv_df.index.name = "cell_id"
    csv_df.to_csv(output_dir / f"{sample_id}_predicted_celltypes.csv")
    print(f"[INFO] Saved: {output_dir / f'{sample_id}_predicted_celltypes.csv'}")

    # Xenium Explorer annotation export (cell_id, group, color)
    DITTO_COLORS = [
        "#E495A5", "#86B875", "#7DB0DD", "#9E66A7", "#C4A74A", "#D37B5A",
        "#B59BC4", "#7DA39B", "#FFCB57", "#9AD2F2", "#2CFFC6", "#F6EF8E",
        "#38B7FF", "#FF9B4D", "#E0AFCA", "#A3A3A3", "#8A5F00", "#1674A9",
        "#005F45", "#AA9F0D", "#00446B", "#803800", "#8D3666", "#3D3D3D",
    ]
    cell_types = sorted(query.obs["predicted_cell_type"].unique())
    ct_color_map = {ct: DITTO_COLORS[i % len(DITTO_COLORS)] for i, ct in enumerate(cell_types)}

    # For FastReseg: use sopa's sequential integer cell IDs (stored by Stage 4)
    # so the annotation CSV matches cells.zarr.zip in Xenium Explorer
    sopa_order_path = output_dir / f"{sample_id}_sopa_cell_order.csv"
    if method == "fastreseg" and sopa_order_path.exists():
        sopa_order = pd.read_csv(sopa_order_path).set_index("obs_name")
        cell_ids = [sopa_order.loc[n, "sopa_index"] if n in sopa_order.index else -1
                    for n in query.obs_names]
        valid = [i for i, cid in enumerate(cell_ids) if cid >= 0]
        explorer_df = pd.DataFrame({
            "cell_id": [cell_ids[i] for i in valid],
            "group":   query.obs["predicted_cell_type"].values[valid],
            "color":   [query.obs["predicted_cell_type"].values[i] for i in valid],
        })
        explorer_df["color"] = explorer_df["group"].map(ct_color_map)
        print(f"[INFO] FastReseg: using sopa integer cell IDs ({len(valid)}/{len(query)} cells mapped)")
    else:
        explorer_df = pd.DataFrame({
            "cell_id": query.obs_names,
            "group":   query.obs["predicted_cell_type"].values,
            "color":   query.obs["predicted_cell_type"].map(ct_color_map).values,
        })

    explorer_csv = output_dir / f"{sample_id}_xenium_explorer_annotations.csv"
    explorer_df.to_csv(explorer_csv, index=False)
    print(f"[INFO] Saved Xenium Explorer annotations: {explorer_csv.name}")

    counts_series = pd.Series(labels).value_counts()
    for ct, n in counts_series.items():
        bar = "█" * int(30 * n / len(labels))
        print(f"  {ct:<40} {n:>6}  {bar}")

    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Rank-gene XGBoost cell type classifier — trains once, predicts on all methods"
    )
    parser.add_argument("--reference",    required=True,
                        help="Path to reference h5ad with cell type labels")
    parser.add_argument("--celltype-col", required=True,
                        help="Column in reference .obs containing cell type labels")
    parser.add_argument("--data-dir",     required=True,
                        help="Root directory to search for *_reseg results "
                             "(experiment_dir, or slide dir for single-sample mode)")
    parser.add_argument("--gpu",          action="store_true",
                        help="Use GPU for XGBoost (requires CUDA, XGBoost >= 2.0)")
    parser.add_argument("--retrain",      action="store_true",
                        help="Ignore cached model and retrain from scratch")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Discover all reseg h5ads anywhere under data_dir.
    # Pattern: {anything}*_reseg/{sample_id}/{sample_id}.h5ad
    # Works for both single-sample and full-experiment layouts.
    queries = {}  # "{method}/{sample_id}" -> (h5ad_path, output_dir)
    for reseg_dir in sorted(data_dir.rglob("*_reseg")):
        if not reseg_dir.is_dir():
            continue
        method = reseg_dir.name.replace("_reseg", "")
        for sample_dir in sorted(reseg_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            sample_id = sample_dir.name
            h5ad = sample_dir / f"{sample_id}.h5ad"
            if h5ad.exists():
                queries[f"{method}/{sample_id}"] = (h5ad, sample_dir, method, sample_id)

    if not queries:
        print(f"[ERROR] No *_reseg/**/*.h5ad found under {data_dir}")
        sys.exit(1)

    print("=" * 60)
    print(f"  Classifier — {len(queries)} h5ad(s) discovered")
    print(f"  Reference:  {args.reference}")
    print(f"  Data dir:   {data_dir}")
    for key in queries:
        print(f"    {key}")
    print("=" * 60)

    ref_stem  = Path(args.reference).stem
    cache_dir = data_dir / f"classifier_cache_{ref_stem}"
    print(f"[INFO] Classifier cache: {cache_dir.name}")

    # ── Try loading cached classifier ──
    clf, le, gene_list = (None, None, None) if args.retrain else load_classifier_cache(
        cache_dir, reference_path=args.reference, celltype_col=args.celltype_col
    )

    if clf is None:
        # ── Load reference and train ──
        print("[INFO] Loading reference data...")
        ref = sc.read_h5ad(args.reference)
        print(f"[INFO]   {ref.n_obs} cells × {ref.n_vars} genes")

        if args.celltype_col not in ref.obs.columns:
            raise ValueError(
                f"Column '{args.celltype_col}' not found in reference .obs.\n"
                f"Available columns: {list(ref.obs.columns)}"
            )

        # Find the gene intersection across ALL queries so the classifier is trained
        # only on genes present in every method. Missing genes at prediction time
        # get rank 0 (not detected), which is handled gracefully.
        print("[INFO] Computing gene intersection across all queries...")
        common_genes = None
        for h5ad_path, _, _, _ in queries.values():
            q = sc.read_h5ad(h5ad_path, backed="r")
            genes = q.var_names
            common_genes = genes if common_genes is None else common_genes.intersection(genes)
        print(f"[INFO] Common genes across all queries: {len(common_genes)}")
        ref = ref[:, ref.var_names.intersection(common_genes)].copy()
        print(f"[INFO] Reference subset to {ref.n_vars} genes")

        print("[INFO] Rank-transforming reference counts...")
        X_train = counts_to_rank(ref, desc="Ranking reference genes")
        y_train = ref.obs[args.celltype_col].astype(str).values

        clf, le = train_classifier(X_train, y_train, use_gpu=args.gpu)
        gene_list = list(ref.var_names)
        print(f"[INFO] Classes ({len(le.classes_)}): {', '.join(le.classes_)}")

        save_classifier_cache(cache_dir, clf, le, gene_list,
                              reference_path=args.reference, celltype_col=args.celltype_col)

        # ── Compute and cache segger markers + gene pairs from reference ──
        print("[INFO] Computing segger markers from reference...")
        try:
            markers = find_markers(ref, args.celltype_col)
            gene_pairs = find_mutually_exclusive_genes(ref, markers, args.celltype_col)
            save_marker_cache(cache_dir, markers, gene_pairs)
        except Exception as e:
            print(f"[WARN] Could not compute segger markers: {e}")
    else:
        print(f"[INFO] Classes ({len(le.classes_)}): {', '.join(le.classes_)}")

        # ── Load or compute marker cache alongside loaded classifier ──
        markers, gene_pairs = load_marker_cache(cache_dir)
        if markers is None:
            print("[INFO] Marker cache missing — computing from reference...")
            try:
                ref = sc.read_h5ad(args.reference)
                if args.celltype_col in ref.obs.columns:
                    markers = find_markers(ref, args.celltype_col)
                    gene_pairs = find_mutually_exclusive_genes(ref, markers, args.celltype_col)
                    save_marker_cache(cache_dir, markers, gene_pairs)
            except Exception as e:
                print(f"[WARN] Could not compute segger markers: {e}")

    # ── Predict on every discovered h5ad ──
    print(f"\n[INFO] Predicting on {len(queries)} h5ad(s)...")
    total_annotated = 0
    for key, (h5ad_path, output_dir, method, sample_id) in queries.items():
        labels = _predict_and_save(clf, le, gene_list, h5ad_path, output_dir, sample_id, method)
        total_annotated += len(labels)

    print(f"\n[DONE] {total_annotated} cells annotated across {len(queries)} h5ad(s)")


if __name__ == "__main__":
    main()
