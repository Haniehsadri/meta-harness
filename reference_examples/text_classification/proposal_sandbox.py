"""Linux filesystem write confinement for a proposer and all of its children.

Reads/network access are unchanged. Only the supplied workspace is writable.
Landlock ABI 1/2 need a small seccomp guard for their path-truncation gap.
Failure to install either required protection is fatal; never fall back silently.
"""
import argparse
import ctypes
import errno
import os
import platform
import sys
from pathlib import Path


def confine_writes(workspace):
    if sys.platform != "linux":
        raise RuntimeError("Adaptive proposal isolation requires Linux Landlock")
    architecture = platform.machine()
    if architecture not in ("x86_64", "aarch64"):
        raise RuntimeError(f"Unsupported sandbox architecture: {architecture}")
    libc = ctypes.CDLL(None, use_errno=True)

    def checked(result, operation):
        if result < 0:
            code = ctypes.get_errno()
            raise OSError(code, f"{operation}: {os.strerror(code)}")
        return result

    abi = checked(libc.syscall(444, 0, 0, 1), "query Landlock ABI")
    handled = sum(1 << bit for bit in (1, 4, 5, 6, 7, 8, 9, 10, 11, 12))
    if abi >= 2:
        handled |= 1 << 13  # REFER (cross-directory rename/link)
    if abi >= 3:
        handled |= 1 << 14  # TRUNCATE

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    ruleset = Ruleset(handled)
    fd = checked(libc.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0),
                 "create Landlock ruleset")
    try:
        for path, access in ((Path(workspace).resolve(strict=True), handled),
                             (Path("/dev/null"), 1 << 1)):
            path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(access, path_fd)
                checked(libc.syscall(445, fd, 1, ctypes.byref(rule), 0), "add Landlock path")
            finally:
                os.close(path_fd)
        checked(libc.prctl(38, 1, 0, 0, 0), "set no_new_privs")
        if abi < 3:
            _guard_legacy_truncation(libc, architecture, checked)
        checked(libc.syscall(446, fd, 0), "restrict Landlock writes")
    finally:
        os.close(fd)
    return abi


def _guard_legacy_truncation(libc, architecture, checked):
    """Block truncate(path), read-only O_TRUNC, and unsupported openat2 inspection.

    Writable open/ftruncate remains available inside the workspace. Landlock
    refuses obtaining writable descriptors to outside files. openat2 returns
    ENOSYS so standard libraries can fall back to guarded open/openat.
    """
    class Instruction(ctypes.Structure):
        _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte),
                    ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint32)]

    class Program(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ushort), ("filter", ctypes.POINTER(Instruction))]

    audit_arch, truncate, opens = ((0xC000003E, 76, [(2, 1), (257, 2)])
                                 if architecture == "x86_64" else
                                 (0xC00000B7, 45, [(56, 2)]))
    deny = 0x00050000 | errno.EPERM
    # Load/verify architecture; reject alternate syscall ABIs, including x32.
    instructions = [(0x20, 0, 0, 4), (0x15, 1, 0, audit_arch), (0x06, 0, 0, deny),
                    (0x20, 0, 0, 0), (0x35, 0, 1, 0x40000000), (0x06, 0, 0, deny),
                    (0x15, 0, 1, truncate), (0x06, 0, 0, deny),
                    (0x15, 0, 1, 437), (0x06, 0, 0, 0x00050000 | errno.ENOSYS)]
    for number, flags_argument in opens:
        instructions += [(0x15, 0, 4, number), (0x20, 0, 0, 16 + flags_argument * 8),
                         (0x54, 0, 0, os.O_ACCMODE | os.O_TRUNC),
                         (0x15, 0, 1, os.O_TRUNC), (0x06, 0, 0, deny),
                         (0x20, 0, 0, 0)]
    instructions.append((0x06, 0, 0, 0x7FFF0000))  # ALLOW
    array = (Instruction * len(instructions))(*(Instruction(*i) for i in instructions))
    program = Program(len(array), array)
    checked(libc.prctl(22, 2, ctypes.byref(program), 0, 0), "install truncation guard")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required")
    workspace = Path(args.workspace).resolve(strict=True)
    runtime = workspace / ".runtime"
    for name in ("tmp", "cache", "claude"):
        (runtime / name).mkdir(parents=True, exist_ok=True)
    # Keep runtime writes inside the workspace without repurposing HOME.
    os.environ.update(TMPDIR=str(runtime / "tmp"), XDG_CACHE_HOME=str(runtime / "cache"),
                      CLAUDE_CONFIG_DIR=str(runtime / "claude"), PYTHONDONTWRITEBYTECODE="1")
    try:
        abi = confine_writes(workspace)
    except (OSError, RuntimeError) as error:
        print(f"Proposer filesystem sandbox unavailable; refusing unprotected launch: {error}",
              file=sys.stderr, flush=True)
        return 125
    print(f"Proposer filesystem sandbox active (Landlock ABI {abi})", file=sys.stderr, flush=True)
    os.execvp(command[0], command)


if __name__ == "__main__":
    raise SystemExit(main())
