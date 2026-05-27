import argparse

import matplotlib.pyplot as plt
import numpy as np

from ipie.analysis.autocorr import reblock_by_autocorr
from ipie.analysis.extraction import extract_observable, get_metadata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="estimates.0.h5")
    parser.add_argument("--column", default="ETotal")
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--save", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    data = extract_observable(args.file, "energy")
    metadata = get_metadata(args.file)
    params = metadata.get("params", {})
    dt = params.get("timestep", 1.0)
    steps_per_block = params.get("num_steps_per_block", 1)

    y = np.asarray(data[args.column], dtype=float)
    tau = np.arange(len(y)) * steps_per_block * dt

    try:
        rb = reblock_by_autocorr(y[args.start :], name=args.column)
        mean = rb[f"{args.column}_ac"].iloc[0]
        print(rb)
    except Exception:
        mean = np.mean(y[args.start :])
        print(mean)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(tau, y, color="lightblue", label="QMC")
    ax.axhline(mean, color="black", linestyle="dashed", label="reblock", lw=0.75)

    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$E$")
    ax.legend()
    plt.tight_layout()
    if args.save is None:
        plt.show()
    else:
        plt.savefig(args.save, bbox_inches="tight")


if __name__ == "__main__":
    main()
