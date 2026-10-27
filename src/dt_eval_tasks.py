import argparse

import dt_data

DEFAULT_N_PERIODIC = 100
DEFAULT_N_ENDPOINT = 1000

SIZES = {
    "lambada":        (100, 1000, 0.27),
    "induction":      (100, 1000, 0.20),
    "kv_synth_N50":   ( 20,  500, 1.34),
    "arith_1d":       ( 71, 1000, 0.38),
    "piqa":           (100, 1000, 0.24),
    "openbookqa":     (100,  500, 0.21),
    "siqa":           ( 84, 1000, 0.32),
    "arc_easy":       ( 63, 1000, 0.43),
    "hellaswag":      ( 46, 1000, 0.58),
    "boolq":          ( 28, 1000, 0.97),
    "sciq":           ( 17, 1000, 1.59),
}

def n_for(task, phase="periodic"):
    i = 0 if phase == "periodic" else 1
    default = DEFAULT_N_PERIODIC if phase == "periodic" else DEFAULT_N_ENDPOINT
    return SIZES.get(task, (DEFAULT_N_PERIODIC, DEFAULT_N_ENDPOINT))[i] if task in SIZES else default

def describe(tasks=None, n_examples=40000, n_runs=None):
    tasks = list(tasks) if tasks is not None else list(dt_data.FEASIBILITY_TASKS)
    lines = [f"{'task':16}{'n_per':>7}{'n_end':>7}{'s/item':>8}{'sec/ckpt':>10}{'chance':>9}",
             "-" * 60]
    total = 0.0
    for t in tasks:
        npd, nend = n_for(t, "periodic"), n_for(t, "endpoint")
        s = SIZES.get(t, (0, 0, 0.0))[2]
        sec = s * npd
        total += sec
        ch = dt_data.CHANCE.get(t)
        lines.append(f"{t:16}{npd:>7}{nend:>7}{s:>8.2f}{sec:>10.1f}"
                     f"{('n/a' if ch is None else f'{ch:.3f}'):>9}")
    lines.append("-" * 60)
    lines.append(f"{'TOTAL':16}{'':>7}{'':>7}{'':>8}{total:>10.1f} sec per eval checkpoint "
                 f"({len(tasks)} tasks)")
    import dt_feasibility_train as _dft
    n_ck = len(_dft.eval_checkpoints(n_examples)) + 1
    if n_runs is None:
        n_runs = len(_dft.GROUP_SIZES) * len(_dft.ARMS) + 1
    lines.append(f"  x {n_ck} checkpoints (derived from dt_feasibility_train.eval_checkpoints at "
                 f"n_examples={n_examples}) x {n_runs} runs")
    lines.append(f"  = {total * n_ck * n_runs / 3600:.1f} h of eval across the whole experiment")
    return "\n".join(lines)

class EvalConfig:

    def __init__(self, tasks=None):
        self.tasks = list(tasks) if tasks is not None else list(dt_data.FEASIBILITY_TASKS)

    def n(self, task, phase="periodic"):
        return n_for(task, phase)

    def describe(self):
        return describe(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def __len__(self):
        return len(self.tasks)

    def __repr__(self):
        return f"EvalConfig({len(self.tasks)} tasks: {', '.join(self.tasks)})"

CONFIG = EvalConfig()

def configure(tasks=None, include=None, exclude=None):
    if include is not None or exclude is not None:
        CONFIG.tasks = dt_data.select_tasks(include=include, exclude=exclude)
    elif tasks is not None:
        CONFIG.tasks = dt_data.select_tasks(include=list(tasks))
    else:
        CONFIG.tasks = list(dt_data.FEASIBILITY_TASKS)
    return CONFIG.tasks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include", default=None, help="comma list")
    ap.add_argument("--exclude", default=None, help="comma list")
    a = ap.parse_args()
    tasks = dt_data.select_tasks(
        include=a.include.split(",") if a.include else None,
        exclude=a.exclude.split(",") if a.exclude else None)
    print(describe(tasks))

if __name__ == "__main__":
    main()
