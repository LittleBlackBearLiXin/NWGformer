import os, sys, subprocess, tempfile
from pathlib import Path
from shutil import disk_usage, rmtree
SCRIPT = "/aaaaaa/100M/nb-sample.py"
DATA_DIR = Path("/root/autodl-tmp/ogb_cache")

REQUIRED_FREE = 10 * (1024**3)

def human(n):
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"

def is_mounted(path: Path) -> bool:
    try:
        mounts = []
        with open("/proc/mounts", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    mounts.append(parts[1])
        p = path.resolve()
        candidates = [m for m in mounts if str(p).startswith(m.rstrip("/"))]
        if not candidates:
            return False
        mount_point = max(candidates, key=len)
        return True if mount_point else False
    except Exception:
        return True



def check_dir(path: Path):
    print(f"[CHECK] data root：{path}")
    path.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.NamedTemporaryFile(dir=path, delete=True) as _:
            pass
        print("[OK] ")
    except Exception as e:
        sys.exit(f"[ERR]：{e}")

    if is_mounted(path):
        print("[OK] ")
    else:
        print("[WARN] not data ")

    total, used, free = disk_usage(path)
    print(f"[INFO] total space: {human(total)} | useed: {human(used)} | use-able: {human(free)}")
    if free < REQUIRED_FREE:
        sys.exit(f"[ERR] use-able space:（{human(free)}），want ≥ {human(REQUIRED_FREE)}")

def main():
    check_dir(DATA_DIR)
    os.environ["OGB_CACHE_ROOT"] = str(DATA_DIR)
    tmpdir = DATA_DIR / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmpdir)
    py = sys.executable
    if not Path(SCRIPT).exists():
        sys.exit(f"[ERR] fund not bash：{SCRIPT}")

    pretrain_cmd = [
        py, SCRIPT,
        "--dataset", "ogbn-papers100M",
        "--method", "ours",
        "--lr", "0.001",
        "--num_layers", "3",
        "--hidden_channels", "256",
        "--dropout", "0.3",
        "--weight_decay", "1e-5",
        "--local_layers", "3",
        "--num_heads", "1",
        "--pre_ln",
        "--pre_linear",
        "--res",
        "--bn",
        "--gnn", "gcn",
        "--batch_size", "500",
        "--seed", "123",
        "--runs", "1",
        "--epochs", "40",
        "--display_step", "1",
        "--device", "0",
        "--save_model",
        "--data_dir", str(DATA_DIR),
    ]

    finetune_cmd = [
        py, SCRIPT,
        "--dataset", "ogbn-papers100M",
        "--data_dir", str(DATA_DIR),
        "--method", "ours",
        "--lr", "0.0001",
        "--num_layers", "1",
        "--hidden_channels", "256",
        "--dropout", "0.3",
        "--weight_decay", "1e-5",
        "--local_layers", "3",
        "--num_heads", "1",
        "--pre_ln",
        "--pre_linear",
        "--res",
        "--bn",
        "--gnn", "gcn",
        "--batch_size", "500",
        "--seed", "123",
        "--runs", "1",
        "--epochs", "10",
        "--display_step", "1",
        "--device", "0",
        "--save_model",
        "--use_pretrained",
        "--model_dir", "models/ogbn-papers100M_ours_23.pt",
    ]

    print("\n[RUN] Pretrain...")
    subprocess.run(pretrain_cmd, check=True)
    print("\n[RUN] Finetune...")
    subprocess.run(finetune_cmd, check=True)
    print("\n[DONE] All finished.")

if __name__ == "__main__":
    main()


