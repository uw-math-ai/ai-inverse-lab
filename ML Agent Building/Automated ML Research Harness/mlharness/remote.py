"""Remote GPU execution on SSH servers (e.g. the UW AMATH machines americano / doppio).

Only one command is ever executed remotely: the harness's own experiment runner,
``python -m mlharness.runner <job.json>``, inside a per-run work directory, under ``timeout``.
No model-generated shell is sent. Credentials are read from environment variables *by name*
(``SERVICE_USERNAME`` / ``SERVICE_PASSWORD`` by default, loaded from the repository ``.env`` file);
their values are never logged, written to artifacts or placed on a command line.

Flow per job: connect (cached) -> sync the ``mlharness`` package (content-hashed tarball, uploaded
once) -> sync the dataset and simulator file (content-hashed) -> upload a job.json with remote paths
-> run -> download result.json / val_pred.npy / val_idx.npy. Any connection, authentication or
environment failure returns ``status = remote_unavailable`` and the executor falls back to local
execution (recorded as a FAILURE artifact). After ``max_failures`` consecutive failures the runner
disables itself for the rest of the run (circuit breaker).

Requires ``paramiko`` (``pip install paramiko``) for password authentication.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import posixpath
import shlex
import tarfile
import threading
import time
from pathlib import Path

from . import HARNESS_ROOT
from .envfile import load_env_file

log = logging.getLogger("mlharness")

AMATH_SERVERS = {
    "americano": {"host": "americano.amath.washington.edu", "gpus": "2x NVIDIA Titan RTX", "cpus": 12, "ram_gb": 32},
    "cubano": {"host": "cubano.amath.washington.edu", "gpus": None, "cpus": 24, "ram_gb": 128},
    "doppio": {"host": "doppio.amath.washington.edu", "gpus": "2x TITAN X (Pascal) 12GB", "cpus": 40, "ram_gb": 512},
    "galao": {"host": "galao.amath.washington.edu", "gpus": None, "cpus": 12, "ram_gb": 32},
}

DEFAULT_REMOTE = {"enabled": False, "host": "americano", "port": 22, "username_env": "SERVICE_USERNAME",
                  "password_env": "SERVICE_PASSWORD", "env_file": "auto", "python": "python3",
                  "workdir": ".mlharness_remote", "use_for": "gpu", "connect_timeout": 20, "max_failures": 2,
                  "gpus": "auto"}


def resolve_host(name: str) -> str:
    return AMATH_SERVERS.get(name, {}).get("host", name)


def _tar_package() -> tuple[bytes, str]:
    buf = io.BytesIO()
    pkg = HARNESS_ROOT / "mlharness"
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in sorted(pkg.rglob("*.py")):
            if "__pycache__" in f.parts:
                continue
            info = tarfile.TarInfo(str(f.relative_to(HARNESS_ROOT)).replace(os.sep, "/"))
            data = f.read_bytes()
            info.size, info.mtime = len(data), 0
            tar.addfile(info, io.BytesIO(data))
    blob = buf.getvalue()
    return blob, hashlib.sha1(blob).hexdigest()[:12]


def _file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class RemoteRunner:
    def __init__(self, cfg: dict | None = None, client_factory=None):
        self.cfg = {**DEFAULT_REMOTE, **(cfg or {})}
        self.host = resolve_host(self.cfg["host"])
        self._client_factory = client_factory     # tests inject a fake SSH client
        self._client = None
        self._lock = threading.Lock()
        self._synced: set[str] = set()
        self._gpu_rr = 0
        self.failures = 0
        self.disabled = False
        self.last_error: str | None = None
        self.n_gpus: int | None = None

    # -- connection -------------------------------------------------------------------------
    def _credentials(self):
        load_env_file(self.cfg.get("env_file", "auto"))
        user = os.environ.get(self.cfg["username_env"])
        pwd = os.environ.get(self.cfg["password_env"]) if self.cfg.get("password_env") else None
        if not user:
            raise RuntimeError(f"environment variable {self.cfg['username_env']} is not set (see .env)")
        user = user.strip()
        if "@" in user:                 # UW NetID = the part of the e-mail address before '@'
            user = user.split("@", 1)[0]
        return user, (pwd.strip("\r\n") if pwd else pwd)

    def _connect(self):
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            self._client = self._client_factory(self.cfg)
            return self._client
        try:
            import paramiko
        except ImportError as exc:
            raise RuntimeError("remote execution needs paramiko: pip install paramiko") from exc
        user, pwd = self._credentials()
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        kh = Path.home() / ".mlharness" / "known_hosts"
        kh.parent.mkdir(parents=True, exist_ok=True)
        if kh.exists():
            client.load_host_keys(str(kh))
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())   # trust on first use, then pinned
        client.connect(self.host, port=int(self.cfg["port"]), username=user, password=pwd,
                       timeout=float(self.cfg["connect_timeout"]), allow_agent=pwd is None,
                       look_for_keys=pwd is None)
        try:
            client.save_host_keys(str(kh))
        except Exception:
            pass
        self._client = client
        return client

    def close(self):
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def _exec(self, cmd: str, timeout: float = 60.0) -> tuple[int, str, str]:
        client = self._connect()
        stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        return stdout.channel.recv_exit_status(), out, err

    def _sftp(self):
        return self._connect().open_sftp()

    def _home(self) -> str:
        code, out, _ = self._exec("echo $HOME", 20)
        return out.strip() or "."

    def _workdir(self) -> str:
        wd = self.cfg["workdir"]
        return wd if wd.startswith("/") else posixpath.join(self._home(), wd)

    # -- capability probe ---------------------------------------------------------------------
    def probe(self) -> dict:
        info = {"host": self.host, "ok": False}
        try:
            py = shlex.quote(self.cfg["python"])
            code, out, err = self._exec(
                "hostname; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu "
                "--format=csv,noheader 2>/dev/null || echo NO_GPU_TOOLS; "
                f"{py} -c \"import sys;print('python',sys.version.split()[0])\"; "
                f"{py} -c \"import numpy,scipy,sklearn;print('stack',numpy.__version__,scipy.__version__,sklearn.__version__)\" 2>&1 | tail -1; "
                f"{py} -c \"import torch;print('torch',torch.__version__,torch.cuda.is_available(),torch.cuda.device_count())\" 2>&1 | tail -1",
                60)
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            info["hostname"] = lines[0] if lines else None
            info["gpus"] = [ln for ln in lines[1:] if ln[:1].isdigit() and "," in ln]
            info["python"] = next((ln for ln in lines if ln.startswith("python ")), None)
            info["stack"] = next((ln for ln in lines if ln.startswith("stack ") or "Error" in ln), None)
            info["torch"] = next((ln for ln in lines if ln.startswith("torch ") or "No module named 'torch'" in ln), None)
            info["ok"] = bool(info["python"]) and bool(info["stack"]) and info["stack"].startswith("stack ")
            self.n_gpus = len(info["gpus"])
        except Exception as exc:
            info["error"] = f"{type(exc).__name__}: {exc}"
        return info

    # -- execution ----------------------------------------------------------------------------
    def accepts(self, job: dict) -> bool:
        if self.disabled or not self.cfg.get("enabled"):
            return False
        if job.get("mode") not in ("train_eval", "method_eval"):
            return False                     # the final fit + test evaluation always runs locally
        return self.cfg.get("use_for") == "all" or bool(job.get("wants_gpu"))

    def _ensure(self, sftp, local: Path, remote: str, key: str):
        if key in self._synced:
            return
        try:
            sftp.stat(remote)
        except IOError:
            sftp.put(str(local), remote)
        self._synced.add(key)

    def _sync_code(self, sftp, wd: str) -> str:
        blob, h = _tar_package()
        code_dir = posixpath.join(wd, "code", h)
        if ("code", h) not in self._synced:
            code, out, _ = self._exec(f"test -d {shlex.quote(code_dir)}/mlharness && echo present || echo absent", 20)
            if "present" not in out:
                self._exec(f"mkdir -p {shlex.quote(code_dir)}", 20)
                tar_remote = posixpath.join(code_dir, "pkg.tgz")
                with sftp.open(tar_remote, "wb") as f:
                    f.write(blob)
                self._exec(f"cd {shlex.quote(code_dir)} && tar xzf pkg.tgz", 60)
            self._synced.add(("code", h))
        return code_dir

    def _pick_gpu(self) -> int:
        n = self.n_gpus if self.n_gpus is not None else 1
        g = self._gpu_rr % max(1, n)
        self._gpu_rr += 1
        return g

    def run(self, job: dict, timeout: float) -> dict:
        with self._lock:                      # one SSH session; jobs serialise their control traffic
            try:
                if self.n_gpus is None:
                    self.probe()
                sftp = self._sftp()
                wd = self._workdir()
                self._exec(f"mkdir -p {shlex.quote(wd)}/data {shlex.quote(wd)}/sims {shlex.quote(wd)}/jobs", 20)
                code_dir = self._sync_code(sftp, wd)
                rjob = json.loads(json.dumps(job, default=str))
                data = Path(job["data_path"])
                dh = _file_hash(data)
                rdata = posixpath.join(wd, "data", f"{dh}.npz")
                self._ensure(sftp, data, rdata, ("data", dh))
                rjob["data_path"] = rdata
                sim = rjob.get("simulator") or {}
                if sim.get("name") == "python":
                    from .simulators import resolve_user_path
                    sp = resolve_user_path(sim["params"]["path"])
                    sh = _file_hash(sp)
                    rsim = posixpath.join(wd, "sims", f"{sh}_{sp.name}")
                    self._ensure(sftp, sp, rsim, ("sim", sh))
                    rjob["simulator"]["params"]["path"] = rsim
                rdir = posixpath.join(wd, "jobs", f"{int(time.time())}_{job['exp_id']}")
                self._exec(f"mkdir -p {shlex.quote(rdir)}", 20)
                rjob["job_dir"] = rdir
                g = self._pick_gpu()
                rjob["device"] = "cuda:0" if job.get("wants_gpu") else "cpu"
                with sftp.open(posixpath.join(rdir, "job.json"), "w") as f:
                    f.write(json.dumps(rjob, default=str))
            except Exception as exc:
                return self._fail(exc)
        threads = int(job.get("threads", 1))
        cmd = (f"cd {shlex.quote(code_dir)} && CUDA_VISIBLE_DEVICES={g} OMP_NUM_THREADS={threads} "
               f"MKL_NUM_THREADS={threads} PYTHONPATH={shlex.quote(code_dir)} timeout {int(timeout)} "
               f"{shlex.quote(self.cfg['python'])} -m mlharness.runner {shlex.quote(posixpath.join(rdir, 'job.json'))}")
        try:
            code, out, err = self._exec(cmd, timeout + 30)
        except Exception as exc:
            return self._fail(exc)
        if code == 124:
            return {"status": "timeout", "error": f"remote timeout after {timeout:.0f}s", "device": f"{self.host}:gpu{g}"}
        local_dir = Path(job["job_dir"])
        try:
            with self._lock:
                sftp = self._sftp()
                for name in ("result.json", "val_pred.npy", "val_idx.npy"):
                    try:
                        sftp.get(posixpath.join(rdir, name), str(local_dir / name))
                    except IOError:
                        pass
        except Exception as exc:
            return self._fail(exc)
        self.failures = 0
        return {"status": "done", "device": f"{self.host}:gpu{g}", "stderr": err[-3000:]}

    def _fail(self, exc) -> dict:
        self.failures += 1
        self.last_error = f"{type(exc).__name__}: {exc}"[:300]
        self.close()
        if self.failures >= int(self.cfg.get("max_failures", 2)):
            self.disabled = True
            log.warning("remote runner disabled after %d failures: %s", self.failures, self.last_error)
        return {"status": "remote_unavailable", "error": self.last_error}


def make_remote(cfg: dict | None):
    if not cfg or not cfg.get("enabled"):
        return None
    return RemoteRunner(cfg)
