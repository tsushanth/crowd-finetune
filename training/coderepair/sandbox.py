import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile

RESULT_MARK = "__CR_RESULT__"
KILL_MARK = "__CR_KILL__"

WRAPPER = r"""
import builtins, io, json, os, resource, signal, sys, traceback, types

sys.dont_write_bytecode = True

_TIMEOUT = __TIMEOUT__

_result_fd = os.dup(1)
_devnull = os.open(os.devnull, os.O_RDWR)
os.dup2(_devnull, 1)
sys.stdout = io.TextIOWrapper(io.FileIO(os.dup(_devnull), "w"), encoding="utf-8", write_through=True)
_result_out = io.TextIOWrapper(io.FileIO(_result_fd, "w"), encoding="utf-8", write_through=True)

def _result_write(text):
    _result_out.write(text)
    _result_out.flush()

def _die(sig, frm):
    try:
        _result_write("__CR_KILL__\n")
    except Exception:
        pass
    sys.exit(124)

try:
    signal.signal(signal.SIGALRM, _die)
    signal.alarm(_TIMEOUT)
except (ValueError, OSError):
    pass

def _set_rlimit(res, soft):
    try:
        _, hard = resource.getrlimit(res)
        if hard != resource.RLIM_INFINITY:
            soft = min(soft, hard)
        resource.setrlimit(res, (soft, hard))
    except (OSError, ValueError):
        pass

_set_rlimit(resource.RLIMIT_CPU, _TIMEOUT + 1)
_set_rlimit(resource.RLIMIT_AS, __MEM_MB__ << 20)
_set_rlimit(resource.RLIMIT_FSIZE, 0)

class _Blocker(types.ModuleType):
    def __getattr__(self, name):
        raise ImportError("blocked: " + name)

for _m in ("socket", "_socket", "ssl", "subprocess", "ctypes", "pty"):
    sys.modules[_m] = _Blocker(_m)

def _forbidden(*a, **k):
    raise RuntimeError("forbidden by sandbox")

for _n in ("system", "popen", "fork", "forkpty", "posix_spawn", "posix_spawnp",
           "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp",
           "execvpe", "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv",
           "spawnve", "spawnvp", "spawnvpe"):
    if hasattr(os, _n):
        setattr(os, _n, _forbidden)

for _n in ("open", "remove", "unlink", "rmdir", "mkdir", "chmod", "chown",
           "truncate", "rename", "replace", "link", "symlink", "mkfifo",
           "makedirs", "removedirs"):
    if hasattr(os, _n):
        setattr(os, _n, _forbidden)

_orig_open = builtins.open
def _safe_open(file, *args, **kwargs):
    mode = args[0] if args else kwargs.get("mode", "r")
    if isinstance(mode, str) and any(c in mode for c in "wax+"):
        raise PermissionError("writes blocked by sandbox")
    return _orig_open(file, *args, **kwargs)
builtins.open = _safe_open

_orig_io_open = io.open
def _safe_io_open(file, *args, **kwargs):
    mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
    if isinstance(mode, str) and any(c in mode for c in "wax+"):
        raise PermissionError("writes blocked by sandbox")
    return _orig_io_open(file, *args, **kwargs)
io.open = _safe_io_open

def _emit(obj):
    _result_write("__CR_RESULT__" + json.dumps(obj))

def _run():
    data = json.loads(sys.stdin.read())
    ns = {}
    for _imp in data.get("imports") or []:
        exec(compile(_imp, "<imports>", "exec"), ns)
    try:
        exec(compile(data["code"], "<candidate>", "exec"), ns)
    except BaseException:
        _emit({"passed": 0, "total": -1, "fraction": 0.0,
               "error": traceback.format_exc(limit=3)})
        return
    if data.get("candidate") and data["candidate"] in ns:
        ns["candidate"] = ns[data["candidate"]]
    passed = 0
    results = []
    for _t in data.get("tests") or []:
        try:
            exec(compile(_t, "<test>", "exec"), ns)
            passed += 1
            results.append(True)
        except BaseException:
            results.append(False)
    total = len(results)
    _emit({"passed": passed, "total": total,
           "fraction": (passed / total) if total else 0.0,
           "results": results})

try:
    _run()
    _result_write("\n")
except KeyboardInterrupt:
    raise
except BaseException as _exc:
    _emit({"passed": 0, "total": -2, "fraction": 0.0,
           "error": repr(_exc)[:300]})
    _result_write("\n")
"""


class Sandbox:
    def __init__(self, timeout=4.0, memory_mb=512, python=None):
        self.timeout = timeout
        self.memory_mb = memory_mb
        self.python = python or sys.executable

    def run(self, code, problem, timeout=None):
        timeout = timeout or self.timeout
        if not code or not code.strip():
            return {
                "passed": 0, "total": 0, "fraction": 0.0, "ok": False,
                "timeout": False, "error": "no code extracted",
            }
        payload = {
            "code": code,
            "imports": problem.get("imports") or [],
            "tests": problem.get("tests") or [],
            "candidate": problem.get("entry_point") or None,
        }
        wrapper = WRAPPER.replace("__TIMEOUT__", str(int(timeout))).replace(
            "__MEM_MB__", str(int(self.memory_mb))
        )
        cwd = tempfile.mkdtemp(prefix="cr-sandbox-")
        os.chmod(cwd, 0o555)
        proc = None
        try:
            proc = subprocess.run(
                [self.python, "-I", "-c", wrapper],
                input=json.dumps(payload),
                capture_output=True,
                encoding="utf-8",
                timeout=timeout,
                cwd=cwd,
                env={},
                start_new_session=True,
            )
        except subprocess.TimeoutExpired as exc:
            if proc and proc.pid:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except OSError:
                    pass
            return {
                "passed": 0, "total": 0, "fraction": 0.0, "ok": False,
                "timeout": True, "error": "timed out",
            }
        finally:
            try:
                os.chmod(cwd, 0o755)
                shutil.rmtree(cwd)
            except OSError:
                pass
        out = (proc.stdout or "") + (proc.stderr or "")
        for line in (proc.stdout or "").splitlines():
            if line.startswith(RESULT_MARK):
                try:
                    data = json.loads(line[len(RESULT_MARK):])
                except json.JSONDecodeError:
                    continue
                total = int(data.get("total", 0))
                passed = int(data.get("passed", 0))
                return {
                    "passed": passed,
                    "total": total,
                    "fraction": data.get("fraction", 0.0),
                    "ok": total > 0 and passed == total,
                    "timeout": False,
                    "error": data.get("error", ""),
                }
        return {
            "passed": 0, "total": 0, "fraction": 0.0, "ok": False,
            "timeout": False,
            "error": "candidate process exited rc=%s: %s"
            % (proc.returncode, out.strip()[-300:] if out.strip() else "no output"),
        }

    def run_completion(self, completion, problem, timeout=None):
        from . import formats

        return self.run(formats.extract_code(completion), problem, timeout=timeout)