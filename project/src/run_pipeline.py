"""
run_pipeline.py
===============
Command-line entry point. Read this file to see the study's shape; read
experiments.py to see the study.

    python run_pipeline.py --smoke
        fast wiring check: one dataset, tiny settings, no DiCE

    python run_pipeline.py --datasets brfss --architectures mlp --seeds 0
        one full run

    python run_pipeline.py
        the paper: 3 datasets x 2 architectures x 3 seeds, all figures

Outputs land in project/results/tables/ and project/figures/.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import data as data_module
import experiments
import figures


def parse_args():
    parser = argparse.ArgumentParser(description="Conformal Selective Explanation")
    parser.add_argument("--datasets", nargs="+", default=config.DEFAULT_DATASETS,
                        choices=list(config.DATASETS))
    parser.add_argument("--architectures", nargs="+", default=["mlp", "ft_transformer"],
                        choices=["mlp", "ft_transformer"])
    parser.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    parser.add_argument("--n-explain", type=int, default=config.N_EXPLAIN)
    parser.add_argument("--ensemble-size", type=int, default=config.ENSEMBLE_SIZE)
    parser.add_argument("--uq", default="ensemble", choices=["ensemble", "mc_dropout"])
    parser.add_argument("--no-dice", action="store_true",
                        help="skip counterfactuals (much faster)")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--describe", action="store_true",
                        help="print the dataset table and exit")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny end-to-end wiring check")
    return parser.parse_args()


def main() -> dict:
    args = parse_args()
    config.ensure_dirs()

    if args.describe:
        table = data_module.dataset_table(args.datasets)
        print(table.to_string(index=False))
        return {"dataset_summary": table}

    if args.smoke:
        print("smoke test: support2, mlp, 1 seed, 2 ensemble members, 40 instances")
        config.MAX_EPOCHS = 3
        config.EARLY_STOPPING_PATIENCE = 2
        tables = experiments.run_all(
            dataset_keys=["support2"], architectures=("mlp",), seeds=[0],
            n_explain=40, ensemble_size=2, include_dice=False, save=True)
    else:
        tables = experiments.run_all(
            dataset_keys=args.datasets,
            architectures=tuple(args.architectures),
            seeds=args.seeds,
            n_explain=args.n_explain,
            ensemble_size=args.ensemble_size,
            uq_method=args.uq,
            include_dice=not args.no_dice,
            save=True)

    print("\ntables written to", config.TABLES_DIR)
    for name, table in sorted(tables.items()):
        print(f"  {name:<28} {table.shape}")

    if not args.no_figures:
        written = figures.make_all(tables)
        print("\nfigures written:")
        for path in written:
            print("  ", path)
    return tables


if __name__ == "__main__":
    main()


# ============================================================
# CHECKLIST
# - Single entry point for the whole study; --smoke runs a fast wiring
#   check, --describe prints the dataset table and exits
# - Flags for datasets, architectures, seeds, sample size, ensemble size,
#   UQ method, and skipping DiCE or figures
# - Calls experiments.run_all(), which writes every results table to
#   results/tables/, then figures.make_all()
# - Prints the shape of every table produced so a run is self-reporting
# ============================================================
