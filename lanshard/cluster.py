"""Single-host cluster launcher for development and emulated experiments.

A worker is a process, not a machine (proposal section 8): this starts a
coordinator in-process and N worker subprocesses, each optionally emulating a
slower device (``slowdown``), a smaller one (``mem_limit_gb``) or a worse link
(``netem``). Specs can come from a JSON/YAML-like dict list or the CLI.

    python -m lanshard.cluster --model Qwen/Qwen2.5-0.5B-Instruct \
        --worker id=gpu-fast --worker id=gpu-slow,slowdown=2 --worker id=cpu,device=cpu
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import subprocess
import sys
import time
from typing import Dict, Iterator, List, Optional

from .coordinator import Coordinator, DriftConfig, SchedulerConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_worker_spec(s: str) -> Dict:
    d: Dict = {}
    for part in s.split(","):
        if not part:
            continue
        k, v = part.split("=", 1)
        d[k.strip()] = v.strip()
    if "id" in d:
        d["node_id"] = d.pop("id")
    return d


def spawn_worker(coord_addr: str, spec: Dict, transport: str, log_dir: Optional[str] = None,
                 profile_budget: float = 5.0) -> subprocess.Popen:
    args = [sys.executable, "-m", "lanshard.worker", "--coordinator", coord_addr, "--transport", transport,
            "--host", "127.0.0.1", "--profile-budget", str(spec.get("profile_budget", profile_budget))]
    for key, flag in (("node_id", "--node-id"), ("device", "--device"), ("dtype", "--dtype"),
                      ("slowdown", "--slowdown"), ("mem_limit_gb", "--mem-limit-gb"), ("netem", "--netem"),
                      ("wire_dtype", "--wire-dtype")):
        if spec.get(key) not in (None, ""):
            args += [flag, str(spec[key])]
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""),
               PYTHONUNBUFFERED="1")
    out = subprocess.DEVNULL
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        out = open(os.path.join(log_dir, f"{spec.get('node_id', 'worker')}.log"), "w")
    return subprocess.Popen(args, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)


@contextlib.contextmanager
def local_cluster(model: str, workers: List[Dict], transport: str = "tcp", policy: str = "latency",
                  sched: Optional[SchedulerConfig] = None, drift: Optional[DriftConfig] = None,
                  coord_netem: Optional[str] = None, log_dir: Optional[str] = None, profile_budget: float = 5.0,
                  bootstrap: bool = True, device: Optional[str] = None, **coord_kw) -> Iterator[Coordinator]:
    coord = Coordinator(model, device=device, transport=transport, host="127.0.0.1", port=0, policy=policy,
                        sched=sched, drift=drift, netem=coord_netem, profile_budget_s=profile_budget,
                        **coord_kw).start()
    procs = [spawn_worker(coord.control_address, w, transport, log_dir, profile_budget) for w in workers]
    coord.worker_procs = {w.get("node_id"): p for w, p in zip(workers, procs)}  # for fault injection
    try:
        if bootstrap:
            coord.bootstrap(len(workers), timeout=900)
        yield coord
    finally:
        coord.shutdown()
        time.sleep(0.5)
        for p in procs:
            if p.poll() is None:
                p.kill()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run a LANShard cluster on one host")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--worker", action="append", required=True,
                    help="id=NAME[,device=cuda|cpu][,slowdown=2][,mem_limit_gb=2][,netem=wifi]")
    ap.add_argument("--transport", default="tcp")
    ap.add_argument("--policy", default="latency")
    ap.add_argument("--netem", help="coordinator outbound netem")
    ap.add_argument("--http", type=int, default=8000)
    ap.add_argument("--prompt", help="run one chat prompt and exit instead of serving")
    a = ap.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s [coord] %(levelname)s %(message)s")
    specs = [parse_worker_spec(w) for w in a.worker]
    with local_cluster(a.model, specs, a.transport, a.policy, coord_netem=a.netem,
                       log_dir=os.path.join(ROOT, "results", "logs")) as c:
        print(json.dumps(c.device_table(), indent=1))
        print(c.plans[c.active_version].summary())
        if a.prompt:
            print(c.chat([{"role": "user", "content": a.prompt}], max_new=200))
            return
        from .api import serve_api

        serve_api(c, "127.0.0.1", a.http)


if __name__ == "__main__":
    main()
