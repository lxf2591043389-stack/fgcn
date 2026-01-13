import argparse
import os
import glob

import matplotlib.pyplot as plt

from config_utils import load_run_config

RUN_CONFIG_PATH = "run_config.json"


def read_loss_file(path):
    epochs = []
    train_losses = []
    eval_losses = []
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        epochs.append(int(parts[0]))
        train_losses.append(float(parts[1]))
        eval_losses.append(float(parts[2]))
    return epochs, train_losses, eval_losses


def main():
    cfg = load_run_config(RUN_CONFIG_PATH)
    common_cfg = cfg.get("common", {}) if isinstance(cfg, dict) else {}
    default_project_root = common_cfg.get("project_root", "experiments")
    default_datasets = common_cfg.get("datasets", "tofdc")

    parser = argparse.ArgumentParser(description="Plot stage losses")
    parser.add_argument("--project_root", default=default_project_root, type=str)
    parser.add_argument("--datasets", default=default_datasets, type=str)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--run_id", default="", type=str)
    args = parser.parse_args()

    result_dir = os.path.join(args.project_root, f"result_{args.datasets}")
    save_path = args.save_path or os.path.join(result_dir, "loss_summary.png")

    plt.figure(figsize=(9, 5))
    plotted = False
    for stage in ["a", "b", "c"]:
        path = ""
        if args.run_id:
            candidate = os.path.join(result_dir, f"loss_stage_{stage}_{args.run_id}.txt")
            if os.path.isfile(candidate):
                path = candidate
        else:
            candidates = glob.glob(os.path.join(result_dir, f"loss_stage_{stage}_*.txt"))
            if candidates:
                path = max(candidates, key=os.path.getmtime)
        if not path:
            fallback = os.path.join(result_dir, f"loss_stage_{stage}.txt")
            if os.path.isfile(fallback):
                path = fallback
        if not path:
            print(f"skip missing log for stage {stage}")
            continue
        epochs, train_losses, eval_losses = read_loss_file(path)
        if not epochs:
            print(f"skip empty log: {path}")
            continue
        plt.plot(epochs, train_losses, label=f"{stage.upper()} train")
        plt.plot(epochs, eval_losses, "--", label=f"{stage.upper()} eval")
        plotted = True

    if not plotted:
        print("no loss logs found, nothing to plot.")
        return

    title = "Stage Loss Curves"
    if args.run_id:
        title = f"Stage Loss Curves ({args.run_id})"
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
