"""Select the co-occurrence route for one seed on the validation criterion.

Every fine-tuning task is run twice per seed: with the pre-trained co-occurrence
sub-layer active (default) and with it bypassed (--no_neighbor_attn). This script
reads the best validation value stored in each run's best checkpoint and prints
the checkpoint to evaluate. All trainers store a lower-is-better value
(validation loss, or the negated validation AUC / AUROC for social-link and ICU
mortality).

Usage::

    python scripts/select_route.py --active runs/<active>.pt --bypass runs/<bypass>.pt
"""
import argparse

import torch


def best_val(path: str) -> float:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return float(ckpt["best_val_loss"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--active", required=True, help="best checkpoint of the active-route run")
    ap.add_argument("--bypass", required=True, help="best checkpoint of the bypass-route run")
    args = ap.parse_args()

    va, vb = best_val(args.active), best_val(args.bypass)
    route, path = ("active", args.active) if va <= vb else ("bypass", args.bypass)
    print(f"active: best_val={va:.6f}  ({args.active})")
    print(f"bypass: best_val={vb:.6f}  ({args.bypass})")
    print(f"selected: {route}")
    print(path)


if __name__ == "__main__":
    main()
