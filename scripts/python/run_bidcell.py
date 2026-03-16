"""
run_bidcell.py — BIDCell deep learning segmentation for a single sample.

Called by the generated SLURM script with sample-specific paths:
    python run_bidcell.py --config CONFIG --sample-dir /path/to/output-XETG... \
                          --output-dir /path/to/bidcell_reseg/XETG... --sample-id XETG...
"""

import os
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config_loader import load_config, get_method_config
from utils.data_io import configure_threads, save_run_metadata, timed


def main():
    parser = argparse.ArgumentParser(description="Run BIDCell segmentation (single sample)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--bidcell-config", default=None,
                        help="Pre-existing BIDCell YAML config (skips generation)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    method_cfg = get_method_config(cfg, "bidcell")
    params = method_cfg["params"]
    platform = cfg["data"]["platform"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_threads()
    t_start = time.time()

    print("=" * 60)
    print(f"  BIDCell — {args.sample_id}")
    print(f"  Input:  {args.sample_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # GPU check
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARN] No GPU — BIDCell training will be very slow")
    except ImportError:
        print("[ERROR] PyTorch not available")
        sys.exit(1)

    # Get or generate BIDCell config
    from bidcell import BIDCellModel

    if args.bidcell_config:
        bidcell_config = Path(args.bidcell_config)
    else:
        template = params.get("config_template", platform)
        bidcell_config = output_dir / f"bidcell_config_{template}.yaml"
        BIDCellModel.get_example_config(template)
        example_name = f"{template}_example_config.yaml"
        if os.path.exists(example_name):
            import shutil
            shutil.move(example_name, str(bidcell_config))
        print(f"[WARN] Review data paths in: {bidcell_config}")

    @timed("BIDCell full pipeline")
    def _run():
        os.chdir(str(output_dir))
        model = BIDCellModel(str(bidcell_config))
        model.run_pipeline()
    _run()

    print("[INFO] BIDCell outputs .tif label masks.")
    print("[INFO] Resize to DAPI dims with cv2.INTER_NEAREST before Explorer import.")

    elapsed = time.time() - t_start
    save_run_metadata(output_dir, "bidcell", method_cfg, elapsed)
    print(f"[DONE] BIDCell — {args.sample_id} — {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
