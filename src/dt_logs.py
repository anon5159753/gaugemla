import json
import os

class _JsonlLogger:
    def __init__(self, path):
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        self.path = path

    def _write(self, fields):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(fields) + "\n")
            f.flush()
            os.fsync(f.fileno())

class TrainLogger(_JsonlLogger):

    def log_step(self, step, cum_examples, G, wall_clock_s, lr, loss_total, loss_ce, loss_kl,
                 grad_norm):
        self._write({"step": step, "cum_examples": cum_examples, "G": G,
                      "wall_clock_s": wall_clock_s, "lr": lr, "loss_total": loss_total,
                      "loss_ce": loss_ce, "loss_kl": loss_kl, "grad_norm": grad_norm})

class EvalLogger(_JsonlLogger):

    def log_eval(self, branch, G, timing, step, cum_examples, task, acc, ppl, n):
        self._write({"branch": branch, "G": G, "timing": timing, "step": step,
                      "cum_examples": cum_examples, "task": task, "acc": acc, "ppl": ppl, "n": n})

def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return out
