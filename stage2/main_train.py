"""NEO volume MAIN training (branch A) -> stage2/main/<run>/

As pilot_train.py (repo PretrainTetTraining, Ours-Large architecture, fine-tuned from the surface Ours-Large
weights), on data_neo2 (7 training families; c_tube held out) with a wall-clock cap of --hours.
The OneCycle schedule is sized with the EFFECTIVE step time seen in the pilot (0.51 s measured -> ~0.65 s per
step including loading/validation, the pilot stopped at 38-40 of 46 epochs), so it completes inside the cap.
Resumable: if <run>/last.ckpt exists, training continues from it (optimizer, schedule, timer restored).

    CUDA_VISIBLE_DEVICES=0 python stage2/main_train.py --run t4096 --data train_4096 --batch 32
    CUDA_VISIBLE_DEVICES=1 python stage2/main_train.py --run t2048 --data train_2048 --batch 64
"""
import argparse
import os
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
for var, sub in [("MPLCONFIGDIR", "mpl"), ("XDG_CACHE_HOME", ""), ("TMPDIR", "tmp"), ("TRITON_CACHE_DIR", "triton"),
                 ("TORCH_HOME", "torch"), ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("CUDA_CACHE_PATH", "nv")]:
    os.environ[var] = os.path.join(ROOT, ".cache", sub)
    os.makedirs(os.environ[var], exist_ok=True)

import open3d  # noqa: F401,E402  -- must precede h5py/torch/lightning in this env (segfault otherwise)
import torch  # noqa: E402
from lightning import Trainer, seed_everything  # noqa: E402
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint  # noqa: E402
from lightning.pytorch.loggers import CSVLogger  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from g2pt.training.common import load_partial_state_dict_strict  # noqa: E402
from g2pt.training.pretrain_tet import PretrainTetDataModule, PretrainTetTraining  # noqa: E402

CKPT = os.path.join(os.path.dirname(ROOT), "neo_check", "ckpts", "Ours-Large.ckpt")
STEP_S_EFF = 0.66                 # effective s/step (pilot: B64 x 2048; bench: B32 x 4096 costs the same, 503 ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True, help="subfolder of data_neo2, e.g. train_4096")
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--lr", type=float, default=3e-4)              # pilot: ft 3e-4 best (val 0.396 vs 0.423)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    BATCH, DATA = a.batch, os.path.join(ROOT, "data_neo2", a.data)
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "set CUDA_VISIBLE_DEVICES"

    seed_everything(42)
    torch.set_float32_matmul_precision("high")
    cfg = torch.load(CKPT, map_location="cpu", weights_only=False)["hyper_parameters"]["cfg"]
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.datamod = OmegaConf.create(dict(name="tet", data_dir=DATA, batch_size=BATCH, enable_rotate=6.283,
                                        num_workers=a.workers, pin_memory=True, prefetch_factor=4, split_ratio=0.9,
                                        targ_dim=int(cfg.targ_dim), seed=42))
    cfg.batch_size = BATCH
    cfg.optimizer.max_lr = a.lr
    dm = PretrainTetDataModule(cfg.datamod)
    steps_per_epoch = len(dm.train_indices) // BATCH
    epochs = max(1, int(0.95 * a.hours * 3600 / STEP_S_EFF / steps_per_epoch))
    cfg.epochs = epochs
    model = PretrainTetTraining(cfg)
    out = os.path.join(ROOT, "main", a.run)
    os.makedirs(out, exist_ok=True)
    last = os.path.join(out, "last.ckpt")
    resume = last if os.path.exists(last) else None
    if resume is None:
        keys = load_partial_state_dict_strict(model.model, ckpt_path=CKPT, key_prefix="model.", strict=True)
        print(f"loaded {len(keys)} tensors from Ours-Large", flush=True)
        OmegaConf.save(cfg, os.path.join(out, "cfg.yaml"))
    else:
        print(f"resuming from {resume}", flush=True)
        mc = os.path.join(out, "metrics.csv")                 # CSVLogger rewrites it on resume: keep the old log
        if os.path.exists(mc):
            os.replace(mc, os.path.join(out, f"metrics_before_{int(time.time())}.csv"))
    print(f"run {a.run}: data={a.data} batch={BATCH} lr={a.lr} epochs={epochs} steps/epoch={steps_per_epoch} "
          f"train={len(dm.train_indices)} val={len(dm.val_indices)}", flush=True)
    trainer = Trainer(accelerator="gpu", devices=1, precision="16-mixed", max_epochs=epochs,
                      max_time={"seconds": int(a.hours * 3600)}, gradient_clip_val=cfg.get("gradient_clip_val", None),
                      logger=CSVLogger(out, name="", version=""), default_root_dir=out,
                      callbacks=[LearningRateMonitor(logging_interval="step"),
                                 ModelCheckpoint(dirpath=out, filename="best", monitor="Val/total", mode="min",
                                                 save_last=True, every_n_epochs=1)],
                      enable_progress_bar=False, log_every_n_steps=20)
    trainer.fit(model=model, datamodule=dm, ckpt_path=resume, weights_only=False)   # own checkpoint (OmegaConf cfg inside)
    print(f"done: best {trainer.checkpoint_callback.best_model_score}", flush=True)


if __name__ == "__main__":
    main()
