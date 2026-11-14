"""Run logging: what ran, when, how fast, on what hardware, with what result.

Every invocation of main.py gets its own folder, next to the results so it is
downloaded with them (and lands on the Modal volume):

    results/<model>/<job_id>/logs/<UTC start time>_<task>[_epN]/
        run.json        settings at the start; overwritten with the totals at the end
        events.jsonl    one JSON object per line: run/stage start and end, every
                        training epoch, every measured checkpoint
        stages.csv      one row per finished stage (pretrain, train, measure ep 30, ...)
        epochs.csv      one row per training epoch
        resources.csv   GPU / CPU / RAM sampled every --log-interval seconds
        console.log     a copy of everything printed by the main process

Folders are per invocation so the parallel measure containers on Modal
(one per checkpoint, same job id) never write to the same file. Every line is
flushed as it is written, so a crash or timeout still leaves the log up to
that point. JSONL/CSV are plain on purpose: pandas reads them, and each
events.jsonl line can be replayed into wandb.log() later.

GPU numbers come from NVML (pip package nvidia-ml-py) and CPU/RAM from psutil.
Either one missing just leaves those columns empty; nothing fails.
"""

import csv
import datetime as dt
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager

try:
    import psutil
except ImportError:  # optional
    psutil = None

try:
    import pynvml  # from the nvidia-ml-py package
except ImportError:  # optional
    pynvml = None


def _now_iso(t=None):
    return dt.datetime.fromtimestamp(t if t is not None else time.time(), dt.timezone.utc) \
        .isoformat(timespec="seconds")


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


class _Tee:
    """Writes to the real stream and to console.log."""

    def __init__(self, stream, f):
        self.stream, self.f = stream, f

    def write(self, s):
        self.stream.write(s)
        try:
            self.f.write(s)
            self.f.flush()
        except ValueError:  # file already closed at interpreter exit
            pass
        return len(s)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):  # isatty, fileno, encoding ... for tqdm
        return getattr(self.stream, name)


class _ResourceSampler(threading.Thread):
    """Background thread: one resources.csv row every `interval` seconds.

    GPU utilisation is NVML's figure: the % of the last sample period in which
    at least one kernel was running on the GPU. It says the GPU was busy, not
    how much of it was busy, so 100% does not mean every core was used.
    """

    FIELDS = ["time", "elapsed_s", "stage",
              "gpu_util_pct", "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_power_w", "gpu_temp_c",
              "torch_alloc_mb", "torch_peak_alloc_mb",
              "cpu_pct", "cpu_count", "ram_used_mb", "ram_total_mb", "proc_rss_mb"]

    def __init__(self, logger, interval):
        super().__init__(daemon=True)
        self.log, self.interval = logger, interval
        self._halt = threading.Event()
        self._gpus = []
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._gpus = [pynvml.nvmlDeviceGetHandleByIndex(i)
                              for i in range(pynvml.nvmlDeviceGetCount())]
            except Exception:
                self._gpus = []
        self._proc = psutil.Process() if psutil else None
        if psutil:
            psutil.cpu_percent(None)  # first call only primes the counter
        self.f = open(os.path.join(logger.dir, "resources.csv"), "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=self.FIELDS)
        if self.f.tell() == 0:
            self.w.writeheader()

    def sample(self):
        row = {"time": _now_iso(), "elapsed_s": round(time.time() - self.log.start, 1),
               "stage": self.log.current_stage}
        if self._gpus:
            # Summed over GPUs (there is one on every machine used so far);
            # utilisation is the mean.
            util = mem = total = power = 0.0
            temp = 0
            for h in self._gpus:
                try:
                    util += pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                    m = pynvml.nvmlDeviceGetMemoryInfo(h)
                    mem += m.used / 2**20
                    total += m.total / 2**20
                    power += pynvml.nvmlDeviceGetPowerUsage(h) / 1000
                    temp = max(temp, pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
                except Exception:
                    pass
            row.update(gpu_util_pct=round(util / len(self._gpus), 1), gpu_mem_used_mb=round(mem),
                       gpu_mem_total_mb=round(total), gpu_power_w=round(power, 1), gpu_temp_c=temp)
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
            row["torch_alloc_mb"] = round(torch.cuda.memory_allocated() / 2**20)
            row["torch_peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
        if psutil:
            vm = psutil.virtual_memory()
            # RSS of this process plus its children: DataLoader workers and
            # the --workers ReX processes all count.
            rss = 0
            try:
                procs = [self._proc] + self._proc.children(recursive=True)
                for p in procs:
                    try:
                        rss += p.memory_info().rss
                    except psutil.Error:
                        pass
            except psutil.Error:
                pass
            row.update(cpu_pct=psutil.cpu_percent(None), cpu_count=psutil.cpu_count(),
                       ram_used_mb=round((vm.total - vm.available) / 2**20),
                       ram_total_mb=round(vm.total / 2**20), proc_rss_mb=round(rss / 2**20))
        self.w.writerow(row)
        self.f.flush()
        self.log._observe(row)
        return row

    def run(self):
        while not self._halt.wait(self.interval):
            try:
                self.sample()
            except Exception as e:  # never let logging kill a run
                print(f"[runlog] resource sample failed: {e}", file=sys.__stderr__)

    def stop(self):
        self._halt.set()
        self.join(timeout=self.interval + 5)
        try:
            self.sample()  # one last row at the very end
        except Exception:
            pass
        self.f.close()


class _Stage:
    """Collects one stage's numbers. `count(n)` adds n processed items
    (batches, images) so the stage's throughput can be reported; the count
    also goes to the enclosing stage if it uses the same unit (so `measure`
    totals the images of its `measure_epNNN` stages)."""

    def __init__(self, name, unit, parent=None):
        self.name, self.unit, self.parent = name, unit, parent
        self.start = time.time()
        self.items = 0
        self.metrics = {}
        # filled by the resource sampler while this stage is current
        self.gpu_util = []
        self.cpu = []
        self.peak = {"gpu_mem_used_mb": 0, "ram_used_mb": 0, "proc_rss_mb": 0, "torch_peak_alloc_mb": 0}

    def count(self, n=1):
        self.items += n
        if self.parent is not None and self.parent.unit == self.unit:
            self.parent.count(n)

    def set(self, **metrics):
        self.metrics.update(metrics)


class RunLogger:
    def __init__(self, task, run_dir, interval=10.0, config=None, args=None):
        self.task = task
        self.dir = run_dir
        os.makedirs(self.dir, exist_ok=True)
        self.start = time.time()
        self.stages = []          # finished stage rows
        self._stack = []          # open stages, innermost last
        self._lock = threading.Lock()
        self._files = {}
        self.status = "running"

        self._console = open(os.path.join(self.dir, "console.log"), "a")
        sys.stdout = _Tee(sys.stdout, self._console)
        sys.stderr = _Tee(sys.stderr, self._console)

        self.info = {
            "task": task,
            "job_id": os.environ.get("JOB_ID", "local"),
            "smoke": os.environ.get("SMOKE", "0") == "1",
            "start": _now_iso(self.start),
            "argv": sys.argv,
            "args": args or {},
            "config": config or {},
            "git_commit": _git_commit(),
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
        torch = sys.modules.get("torch")
        if torch is not None:
            self.info["torch"] = torch.__version__
            if torch.cuda.is_available():
                self.info["gpu"] = torch.cuda.get_device_name(0)
                self.info["gpu_count"] = torch.cuda.device_count()
        self._write_json("run.json", self.info)
        self.event("run_start", **{k: self.info[k] for k in ("task", "job_id", "git_commit")})

        self.sampler = _ResourceSampler(self, interval)
        self.sampler.sample()
        self.sampler.start()
        print(f"[runlog] logging to {self.dir} (resources every {interval:g}s"
              f"{'' if pynvml else ', no GPU stats: pip install nvidia-ml-py'}"
              f"{'' if psutil else ', no CPU/RAM stats: pip install psutil'})")

    # ------------------------------------------------------------- files
    def _write_json(self, name, obj):
        path = os.path.join(self.dir, name)
        with open(path + ".partial", "w") as f:
            json.dump(obj, f, indent=2, default=str)
        os.replace(path + ".partial", path)

    def _append_csv(self, name, row):
        """Appends a row. The header comes from the first row written; later
        rows with extra keys get those keys dropped rather than breaking the file."""
        with self._lock:
            if name not in self._files:
                f = open(os.path.join(self.dir, name), "a", newline="")
                w = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
                if f.tell() == 0:
                    w.writeheader()
                self._files[name] = (f, w)
            f, w = self._files[name]
            w.writerow(row)
            f.flush()

    def _write_stages_csv(self):
        """Rewritten whole after every stage: stages carry different metric
        columns (training vs measure), so the header is the union of all."""
        fields = []
        for row in self.stages:
            fields += [k for k in row if k not in fields]
        path = os.path.join(self.dir, "stages.csv")
        with self._lock, open(path + ".partial", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(self.stages)
        os.replace(path + ".partial", path)

    def event(self, kind, **data):
        rec = {"time": _now_iso(), "elapsed_s": round(time.time() - self.start, 2),
               "event": kind, **data}
        with self._lock:
            with open(os.path.join(self.dir, "events.jsonl"), "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")

    # ------------------------------------------------------------ stages
    @property
    def current_stage(self):
        return self._stack[-1].name if self._stack else ""

    def _observe(self, row):
        for st in self._stack:  # a sample counts for every open (nested) stage
            if row.get("gpu_util_pct") is not None:
                st.gpu_util.append(row["gpu_util_pct"])
            if row.get("cpu_pct") is not None:
                st.cpu.append(row["cpu_pct"])
            for k in st.peak:
                if row.get(k) is not None:
                    st.peak[k] = max(st.peak[k], row[k])

    @contextmanager
    def stage(self, name, unit="items"):
        st = _Stage(name, unit, parent=self._stack[-1] if self._stack else None)
        self._stack.append(st)
        self.event("stage_start", stage=name)
        print(f"[runlog] stage {name} started {_now_iso(st.start)}")
        status = "ok"
        try:
            yield st
        except BaseException as e:
            status = f"failed: {type(e).__name__}: {e}"[:300]
            raise
        finally:
            try:
                self.sampler.sample()  # short stages still get a resource reading
            except Exception:
                pass
            self._stack.remove(st)
            end = time.time()
            dur = end - st.start
            mean = lambda xs: round(sum(xs) / len(xs), 1) if xs else None
            row = {
                "stage": name, "status": status,
                "start": _now_iso(st.start), "end": _now_iso(end), "duration_s": round(dur, 1),
                "items": st.items, "unit": st.unit,
                "items_per_s": round(st.items / dur, 3) if st.items and dur > 0 else None,
                "s_per_item": round(dur / st.items, 4) if st.items else None,
                "gpu_util_mean_pct": mean(st.gpu_util), "cpu_mean_pct": mean(st.cpu),
                "gpu_mem_peak_mb": st.peak["gpu_mem_used_mb"] or None,
                "torch_peak_alloc_mb": st.peak["torch_peak_alloc_mb"] or None,
                "ram_peak_mb": st.peak["ram_used_mb"] or None,
                "proc_rss_peak_mb": st.peak["proc_rss_mb"] or None,
                **st.metrics,
            }
            self.stages.append(row)
            self._write_stages_csv()
            self.event("stage_end", **row)
            print(f"[runlog] stage {name} {status} after {dur / 60:.1f} min"
                  + (f", {row['items_per_s']} {st.unit}/s" if row["items_per_s"] else ""))

    def count(self, n=1):
        """Add n processed items to the innermost open stage."""
        if self._stack:
            self._stack[-1].count(n)

    def set(self, **metrics):
        """Attach result metrics (accuracy, ...) to the innermost open stage."""
        if self._stack:
            self._stack[-1].set(**metrics)

    def epoch(self, **row):
        """One training epoch (see train.py for the fields)."""
        self._append_csv("epochs.csv", row)
        self.event("epoch", **row)

    # --------------------------------------------------------------- end
    def close(self, status="ok"):
        if self.status != "running":
            return
        self.status = status
        self.sampler.stop()
        end = time.time()
        self.info.update(end=_now_iso(end), duration_s=round(end - self.start, 1), status=status,
                         stages=self.stages)
        self._write_json("run.json", self.info)
        self.event("run_end", status=status, duration_s=round(end - self.start, 1))
        print(f"[runlog] run {status} after {(end - self.start) / 60:.1f} min; logs in {self.dir}")
        for f, _ in self._files.values():
            f.close()
        sys.stdout, sys.stderr = sys.stdout.stream, sys.stderr.stream
        self._console.close()


# A module-level logger, so train.py and evaluate.py can log without it
# being passed through every call. Code that runs without one (notebooks,
# analysis scripts, the ReX worker processes) gets a no-op stand-in.
_current = None


class _NullStage:
    def count(self, n=1):
        pass

    def set(self, **metrics):
        pass


class _NullLogger:
    current_stage = ""

    @contextmanager
    def stage(self, name, unit="items"):
        yield _NullStage()

    def count(self, n=1):
        pass

    def set(self, **metrics):
        pass

    def epoch(self, **row):
        pass

    def event(self, kind, **data):
        pass


def start(*args, **kwargs):
    global _current
    _current = RunLogger(*args, **kwargs)
    return _current


def get():
    return _current or _NullLogger()
