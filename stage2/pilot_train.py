"""NEO volume pilot (decision 3): one training run, capped at 1 h of GPU time -> stage2/pilot/<run>/

Uses the repo's PretrainTetTraining (model + losses exactly as in NEO) with the Ours-Large architecture/config,
our procedural volume data (stage2/data_neo/train, NEO pretrain_tet h5 layout) and a Lightning Trainer with
max_time = 1 h.  The OneCycle schedule is sized to the step budget measured for batch 64 (~510 ms/step).

    CUDA_VISIBLE_DEVICES=0 python stage2/pilot_train.py --run ft_lr1e-4  --init large --lr 1e-4
    CUDA_VISIBLE_DEVICES=1 python stage2/pilot_train.py --run ft_lr3e-4  --init large --lr 3e-4
    CUDA_VISIBLE_DEVICES=2 python stage2/pilot_train.py --run scratch    --init scratch --lr 5e-4
"""
import argparse
import os

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
DATA = os.path.join(ROOT, "data_neo", "train")
BATCH = 64
STEP_S = 0.51                     # measured (bench_neo_step.py), batch 64, 2048 points, fp16
BUDGET_S = 3600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--init", choices=["large", "scratch"], required=True)
    ap.add_argument("--lr", type=float, required=True)
    a = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "set CUDA_VISIBLE_DEVICES"

    seed_everything(42)
    torch.set_float32_matmul_precision("high")
    cfg = torch.load(CKPT, map_location="cpu", weights_only=False)["hyper_parameters"]["cfg"]
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.datamod = OmegaConf.create(dict(name="tet", data_dir=DATA, batch_size=BATCH, enable_rotate=6.283,
                                        num_workers=4, pin_memory=True, prefetch_factor=4, split_ratio=0.9,
                                        targ_dim=int(cfg.targ_dim), seed=42))
    cfg.batch_size = BATCH
    cfg.optimizer.max_lr = a.lr
    dm = PretrainTetDataModule(cfg.datamod)
    steps_per_epoch = len(dm.train_indices) // BATCH
    epochs = max(1, int(0.92 * BUDGET_S / STEP_S / steps_per_epoch))   # ~8 % slack for validation / loading
    cfg.epochs = epochs
    model = PretrainTetTraining(cfg)
    if a.init == "large":
        keys = load_partial_state_dict_strict(model.model, ckpt_path=CKPT, key_prefix="model.", strict=True)
        print(f"loaded {len(keys)} tensors from Ours-Large", flush=True)

    out = os.path.join(ROOT, "pilot", a.run)
    os.makedirs(out, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(out, "cfg.yaml"))
    print(f"run {a.run}: init={a.init} lr={a.lr} epochs={epochs} steps/epoch={steps_per_epoch} "
          f"train={len(dm.train_indices)} val={len(dm.val_indices)}", flush=True)
    trainer = Trainer(accelerator="gpu", devices=1, precision="16-mixed", max_epochs=epochs,
                      max_time={"hours": 1}, gradient_clip_val=cfg.get("gradient_clip_val", None),
                      logger=CSVLogger(out, name="", version=""), default_root_dir=out,
                      callbacks=[LearningRateMonitor(logging_interval="step"),
                                 ModelCheckpoint(dirpath=out, filename="best", monitor="Val/total", mode="min",
                                                 save_last=True)],
                      enable_progress_bar=False, log_every_n_steps=20)
    trainer.fit(model=model, datamodule=dm)
    print(f"done: best {trainer.checkpoint_callback.best_model_score}", flush=True)


if __name__ == "__main__":
    main()
