"""Linux-only, identity-bound supervision for one V5 training process group."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

GIB = 1024 ** 3
PHYSICAL_GPUS = (3, 4, 5)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": int(pid), "state": fields[0], "pgid": int(fields[2]),
                "sid": int(fields[3]), "start_ticks": int(fields[19])}
    except (OSError, ValueError, IndexError):
        return None


def same_process(identity):
    current = process_identity(identity["pid"])
    return bool(current and current["state"] != "Z" and all(
        current[k] == identity[k] for k in ("pid", "pgid", "sid", "start_ticks")))


def group_members(identity):
    """A surviving descendant keeps its session/group reserved after leader exit."""
    result = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            p = process_identity(int(entry.name))
            if (p and p["state"] != "Z" and p["pgid"] == identity["pgid"]
                    and p["sid"] == identity["sid"] and p["start_ticks"] >= identity["start_ticks"]):
                result.append(p["pid"])
    return result


def group_rss(identity):
    total = 0
    for pid in group_members(identity):
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
        except OSError:
            pass
    return total


def terminate_owned(identity):
    if identity["pid"] != identity["pgid"] or identity["sid"] != identity["pgid"]:
        raise RuntimeError("Refusing a shared process group")
    for sig, grace in ((signal.SIGTERM, 2), (signal.SIGKILL, 2)):
        members = group_members(identity)
        if not members:
            return
        # A reused leader PID invalidates the group binding.
        leader = process_identity(identity["pid"])
        if leader and leader["start_ticks"] != identity["start_ticks"]:
            raise RuntimeError("Owned process identity was reused")
        try:
            os.killpg(identity["pgid"], sig)
        except ProcessLookupError:
            return
        until = time.monotonic() + grace
        while group_members(identity) and time.monotonic() < until:
            time.sleep(.05)


def probe(*, root, gpu, rss_limit_bytes, gpu_memory_budget_bytes=0, expected_host=None,
          storage_reserve_bytes=10 * GIB):
    """Contention is observed, never used as an admission threshold."""
    if gpu is not None and gpu not in PHYSICAL_GPUS:
        raise ValueError("V5 may only share physical GPUs 3,4,5")
    if rss_limit_bytes <= 0 or gpu_memory_budget_bytes < 0:
        raise ValueError("Invalid memory budget")
    reasons = []
    host = socket.gethostname()
    if expected_host and host != expected_host:
        reasons.append("hostname_mismatch")
    mem = {x.split(":")[0]: int(x.split()[1]) * 1024
           for x in Path("/proc/meminfo").read_text().splitlines()}
    available = mem["MemAvailable"]
    if available < rss_limit_bytes + 4 * GIB:
        reasons.append("insufficient_host_memory")
    root = Path(root)
    if not root.is_dir() or not os.access(root, os.W_OK):
        reasons.append("unwritable_output")
        disk_free = 0
    else:
        disk = os.statvfs(root)
        disk_free = disk.f_bavail * disk.f_frsize
        if disk_free < storage_reserve_bytes:
            reasons.append("insufficient_storage")
    result = {"hostname": host, "gpu": gpu, "memory_available_bytes": available,
              "storage_available_bytes": disk_free, "contention_is_observation_only": True,
              "load_average": list(os.getloadavg()), "at_unix": time.time()}
    if gpu is not None:
        raw = subprocess.check_output(["nvidia-smi", f"--id={gpu}",
            "--query-gpu=memory.total,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, timeout=15).strip()
        total, free, util = [x.strip() for x in raw.split(",")]
        result["gpu_observation"] = {"total_mib": int(total), "free_mib": int(free), "utilization_raw": util}
        if int(free) * 1024 ** 2 < gpu_memory_budget_bytes + 512 * 1024 ** 2:
            reasons.append("insufficient_gpu_memory")
    result.update(ready=not reasons, reasons=reasons)
    return result


def guard(binding_path):
    """Independent process remains responsible if the controller/supervisor exits."""
    binding = read_json(binding_path)
    identity = binding["identity"]
    deadline_monotonic = time.monotonic() + max(0, binding["deadline_unix"] - time.time())
    peak = 0
    reason = "process_group_exited"
    while group_members(identity):
        rss = group_rss(identity)
        peak = max(peak, rss)
        if rss > binding["rss_limit_bytes"]:
            reason = "rss_limit_exceeded"
        elif time.time() >= binding["deadline_unix"] or time.monotonic() >= deadline_monotonic:
            reason = "case_timeout"
        else:
            time.sleep(.25)
            continue
        terminate_owned(identity)
        break
    write_json(Path(binding_path).parent / "guard_result.json", {
        "reason": reason, "peak_group_rss_bytes": peak,
        "remaining_group_members": group_members(identity), "finished_unix": time.time()})


def supervise(spec_path, output):
    if os.name != "posix" or not Path("/proc").is_dir():
        raise RuntimeError("Linux /proc and POSIX sessions required")
    spec = read_json(spec_path)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({k: "2" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = "" if spec["gpu"] is None else str(spec["gpu"])
    observation = probe(root=root, gpu=spec["gpu"], rss_limit_bytes=spec["rss_limit_bytes"],
        gpu_memory_budget_bytes=spec["gpu_memory_budget_bytes"], expected_host=spec.get("expected_host"),
        storage_reserve_bytes=spec.get("storage_reserve_bytes", 10 * GIB))
    write_json(root / "admission.json", observation)
    if not observation["ready"]:
        write_json(root / "finished.json", {"status": "not_started_resource_wait", "exit_code": None,
            "reasons": observation["reasons"], "remaining_group_members": []})
        return 75
    started = time.time()
    process = None
    guard_process = None
    identity = None
    try:
        with (root / "stdout.log").open("wb") as stdout, (root / "stderr.log").open("wb") as stderr:
            process = subprocess.Popen(spec["command"], cwd=spec["cwd"], env=env,
                stdout=stdout, stderr=stderr, start_new_session=True)
            identity = process_identity(process.pid)
            if not identity or identity["pid"] != identity["pgid"] or identity["sid"] != process.pid:
                raise RuntimeError("Child did not create the owned standalone session")
            binding = {"identity": identity, "started_unix": started,
                "deadline_unix": started + spec["timeout_seconds"], "rss_limit_bytes": spec["rss_limit_bytes"],
                "command": spec["command"], "gpu": spec["gpu"], "supervisor_identity": process_identity(os.getpid())}
            write_json(root / "started.json", binding)
            guard_process = subprocess.Popen([sys.executable, "-m", "src.mt_supervise", "--guard", str(root / "started.json")],
                cwd=spec["cwd"], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True)
            write_json(root / "guard_started.json", {"identity": process_identity(guard_process.pid)})
            while process.poll() is None:
                if guard_process.poll() is not None:
                    if group_members(identity):
                        terminate_owned(identity)
                    break
                time.sleep(.25)
            code = process.wait()
        terminate_owned(identity)
        guard_process.wait(timeout=8)
        guard_result = read_json(root / "guard_result.json")
        status = "completed" if code == 0 and guard_result["reason"] == "process_group_exited" else "failed"
        result = {"status": status, "exit_code": code, "started_unix": started, "finished_unix": time.time(),
            "identity": identity, "remaining_group_members": group_members(identity), "guard": guard_result}
        write_json(root / "finished.json", result)
        return 0 if status == "completed" and not result["remaining_group_members"] else 1
    except BaseException as exc:
        if identity:
            terminate_owned(identity)
        if process:
            process.wait(timeout=8)
        if guard_process:
            guard_process.wait(timeout=8)
        write_json(root / "finished.json", {"status": "supervisor_error", "error": repr(exc),
            "exit_code": process.returncode if process else None,
            "remaining_group_members": group_members(identity) if identity else []})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec")
    parser.add_argument("--output")
    parser.add_argument("--guard")
    args = parser.parse_args()
    if args.guard:
        guard(args.guard)
    else:
        raise SystemExit(supervise(args.spec, args.output))


if __name__ == "__main__":
    main()
