import hashlib
import json
import time

def assert_cpu(model, what="conversion"):
    for n, p in model.named_parameters():
        if p.device.type != "cpu":
            raise RuntimeError(
                f"{what} requires a CPU-resident model, but {n} is on {p.device}. "
                "Order is: load on CPU -> convert -> restructure -> .to(device) -> build the optimiser."
            )

def sha256_of(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]

def timed(msg):
    class _T:
        def __enter__(self):
            self.t0 = time.time()
            print(f"  {msg} ...", end="", flush=True)
            return self

        def __exit__(self, *a):
            print(f" {time.time() - self.t0:.1f}s", flush=True)
    return _T()

def banner(msg):
    print(f"\n{'=' * 78}\n{msg}\n{'=' * 78}", flush=True)
