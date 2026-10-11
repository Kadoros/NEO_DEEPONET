"""Re-scheduling after the resource check (runs on CPU, light polling).

  * runs 0-2 (started by the original queue processes) are left untouched while they train;
    their val@30k is taken from the logs so the early-stopping rule can use them as references.
  * run 0 violates the early-stopping rule -> its process is stopped (partial log kept).
  * when run 1 / run 2 finish (meta file appears), the original queue process on that GPU is stopped
    before it gets far into its next queued run, and a checkpointed queue worker takes the GPU.
  * P0 evaluation starts as soon as the P0 data build has finished (shares GPU 0).

    python stage1/don_orchestrate.py      (background)
"""
import json
import os
import re
import signal
import subprocess
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results_don_fair")
STATE = os.path.join(OUT, "state")
PY = "/home/cvmlserver7/anaconda3/envs/neo/bin/python"
OLD = {0: (1137822, 0), 1: (1137823, 1), 2: (1137824, 2)}      # gpu -> (pid, current run)
ES_FACTOR = 1.2


def log(msg):
    with open(os.path.join(ROOT, "logs", "orchestrate.log"), "a") as fh:
        fh.write(time.strftime("%H:%M:%S ") + msg + "\n")


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def val30k_from_log(gpu, run):
    name = {0: "p=64 lr=0.0003", 1: "p=64 lr=0.001", 2: "p=64 lr=0.003"}[run]
    for line in open(os.path.join(ROOT, "logs", f"don_fair_gpu{gpu}.log")):
        if name in line and "step 30000 " in line:
            return float(re.search(r"best ([0-9.]+)", line).group(1))
    return None


def start_worker(gpu):
    f = open(os.path.join(ROOT, "logs", f"don_worker_gpu{gpu}.log"), "a")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4")
    p = subprocess.Popen([PY, "train_don_fair.py", "--worker"], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    log(f"started queue worker pid {p.pid} on GPU {gpu}")
    return p


def main():
    os.makedirs(STATE, exist_ok=True)
    # references from the running (old) processes
    for gpu, (pid, run) in OLD.items():
        v = val30k_from_log(gpu, run)
        if v is not None:
            open(os.path.join(STATE, f"val30k_run{run}.txt"), "w").write(f"{v:.8f}")
            open(os.path.join(STATE, f"claim_run{run}"), "w").write(f"old-process {pid}")
            log(f"run {run}: val@30k {v:.4f} (from log)")
    for r in (3, 4, 5, 6, 7, 8, 9):                            # nothing of the queue has started yet
        pass
    ref = min(float(open(os.path.join(STATE, f"val30k_run{r}.txt")).read()) for r in (0, 1, 2))
    workers, p0_started = {}, False
    handled = set()
    while True:
        for gpu, (pid, run) in OLD.items():
            if gpu in handled:
                continue
            mine = float(open(os.path.join(STATE, f"val30k_run{run}.txt")).read())
            if mine > ES_FACTOR * ref:                          # early stop (rule decided in advance)
                os.kill(pid, signal.SIGTERM)
                json.dump(dict(name=f"run{run}", status=f"early-stopped (val@30k {mine:.4f} > {ES_FACTOR}x ref "
                                                        f"{ref:.4f})", best_val_at_30k=mine),
                          open(os.path.join(OUT, f"meta_run{run}_earlystop.json"), "w"), indent=2)
                log(f"run {run} early-stopped: val@30k {mine:.4f} > {ES_FACTOR} x {ref:.4f}; pid {pid} terminated")
                handled.add(gpu)
            elif os.path.exists(os.path.join(OUT, f"meta_run{run}.json")) or not alive(pid):
                if alive(pid):
                    os.kill(pid, signal.SIGTERM)                # its next queued run is re-claimed by the workers
                log(f"run {run} finished on GPU {gpu}; old queue process {pid} stopped")
                handled.add(gpu)
            if gpu in handled:
                time.sleep(3)
                workers[gpu] = start_worker(gpu)
        if not p0_started and os.path.exists(os.path.join(ROOT, "data_p0", "index.csv")):
            f = open(os.path.join(ROOT, "logs", "p0_eval.log"), "a")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="8")
            subprocess.Popen([PY, "p0_eval.py"], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            log("started P0 evaluation on GPU 0 (shared)")
            p0_started = True
        if len(handled) == 3 and p0_started and all(p.poll() is not None for p in workers.values()):
            log("all workers finished")
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
