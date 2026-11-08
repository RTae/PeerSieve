"""Name every record: the Slurm job, the node, the GPUs and the code that produced it. This folder is copied to the
cluster for testing and is not a git repository yet, so the code is named by the SHA-256 of each source file."""
import hashlib
import os
import socket
import subprocess
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ("kvmend/hook.py", "kvmend/tags.py", "kvmend/tag_kernels.py", "kvmend/meta.py", "hook_site/sitecustomize.py",
         "run_poc.py", "bench_overhead.py", "bench_checker.py", "run_fair.py", "run_incident.py", "run_release.py",
         "run_hicache.py")


def run_meta():
    code = {}
    for f in FILES:
        path = os.path.join(HERE, f)
        if os.path.exists(path):
            with open(path, "rb") as fh:
                code[f] = hashlib.sha256(fh.read()).hexdigest()[:16]
    try:
        gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        gpus = None
    return dict(job=os.environ.get("SLURM_JOB_ID"), job_name=os.environ.get("SLURM_JOB_NAME"),
                node=socket.gethostname(), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), gpus=gpus,
                started=time.strftime("%Y-%m-%d %H:%M:%S %Z"), code_sha256=code)
