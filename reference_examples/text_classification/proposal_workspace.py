"""Proposal workspaces and content-addressed agent recovery, not selection policy."""
import ast
import json
import os
import tempfile
import shutil
from contextlib import contextmanager
from pathlib import Path


def write_new_source(path, content):
    """Publish complete bytes atomically without replacing any existing path."""
    descriptor, temporary = tempfile.mkstemp(prefix=".source-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        # A same-filesystem hard link publishes the complete temporary inode.
        # Unlike replace/rename, link refuses to overwrite an existing name.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)

def archive_source(root, name, source_hash, content):
    import hashlib
    if hashlib.sha256(content).hexdigest() != source_hash:
        raise ValueError(f"Source hash mismatch before archiving: {name}")
    directory = root / "source_archive"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{source_hash}.py"
    if path.exists():
        if path.is_symlink() or path.read_bytes() != content:
            raise ValueError(f"Source archive integrity failure: {path}")
    else:
        write_new_source(path, content)
    return path


def preserve_sources(root, code, records, file_hash, event):
    """Backfill verified archives; restore missing files only from matching originals."""
    for record in records:
        name, expected = record["name"], record["source_hash"]
        path = code / "agents" / f"{name}.py"
        archive = root / "source_archive" / f"{expected}.py"
        if not path.exists() and archive.is_file() and not archive.is_symlink():
            if file_hash(archive) != expected:
                raise ValueError(f"Source archive hash mismatch: {name}")
            write_new_source(path, archive.read_bytes())
            event(root, "candidate_source_restored", name=name, source_hash=expected)
        if not path.is_file() or path.is_symlink() or file_hash(path) != expected:
            raise ValueError(f"Missing or modified evaluated source: {name}; recover its original hash {expected}")
        archive_source(root, name, expected, path.read_bytes())


@contextmanager
def proposal_workspace(root, code, harness, iteration, datasets, file_hash, event):
    """Redirect all proposer paths to a disposable copy; restore globals on every exit."""
    directory = root / "proposals" / f"iteration-{iteration:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    attempt = max([0] + [int(p.name.split("-")[-1]) for p in directory.glob("attempt-*")
                          if p.name.split("-")[-1].isdigit()]) + 1
    workspace = directory / f"attempt-{attempt:03d}"
    work_code = workspace / "code" / "text_classification"
    shutil.copytree(code, work_code, ignore=shutil.ignore_patterns(
        "__pycache__", ".runtime", ".empty_plugins", ".venv", "*.log"))
    context = workspace / "context"
    context.mkdir()
    for name in ("frontier_val.json", "evolution_summary.jsonl"):
        source = root / name
        if source.exists():
            shutil.copy2(source, context / name)
    if (root / "reports").is_dir():
        shutil.copytree(root / "reports", context / "reports")
    existing = {str(p.relative_to(work_code)): file_hash(p)
                for p in work_code.rglob("*") if p.is_file()}
    attrs = {"EVOLVE_DIR": work_code, "LOGS_DIR": context,
             "PENDING_EVAL": context / "pending_eval.json",
             "FRONTIER_VAL": context / "frontier_val.json",
             "EVOLUTION_SUMMARY": context / "evolution_summary.jsonl",
             "PROPOSAL_WORKSPACE": workspace}
    missing = object()
    saved = {name: getattr(harness, name, missing) for name in attrs}
    for name, value in attrs.items():
        setattr(harness, name, value)
    event(root, "proposal_workspace_created", iteration=iteration, workspace=str(workspace),
          protected_agent_names=sorted(p.stem for p in (code / "agents").glob("*.py")))
    try:
        yield workspace, work_code, existing
    finally:
        for name, value in saved.items():
            if value is missing:
                delattr(harness, name)
            else:
                setattr(harness, name, value)


def accept_proposal(root, code, work_code, existing, pending, iteration, file_hash, save_json):
    """Validate the copy, archive accepted bytes, and durably prepare imports."""
    changed = [name for name, expected in existing.items()
               if not (work_code / name).is_file() or (work_code / name).is_symlink()
               or file_hash(work_code / name) != expected]
    if changed:
        raise ValueError(f"Proposal changed existing workspace files: {changed[:5]}")
    proposal = json.loads(pending.read_text(encoding="utf-8-sig"))
    candidates = proposal.get("candidates") or []
    if not candidates:
        raise ValueError("Proposer returned an empty candidate list")
    import re
    names = [c.get("name") for c in candidates]
    if any(not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in names):
        raise ValueError("Invalid candidate module name")
    if len(set(names)) != len(names):
        raise ValueError("Candidate names must be new and unique")
    reserved = {p.stem for p in (code / "agents").iterdir()}
    for candidate in candidates:
        name = candidate["name"]
        if name in reserved:
            raise ValueError(f"Candidate name already exists: {name}")
        path = work_code / "agents" / f"{name}.py"
        if path.is_symlink() or not path.is_file() or path.resolve().parent != (work_code / "agents").resolve():
            raise ValueError(f"Candidate must be a regular new agent module: {name}")
        content = path.read_bytes()
        ast.parse(content, filename=str(path))
        candidate["source_hash"] = file_hash(path)
        candidate["candidate_id"] = f"iter{iteration}:{name}:{candidate['source_hash']}"
    # Validate the complete manifest before archiving/importing any candidate.
    for candidate in candidates:
        name = candidate["name"]
        archive_source(root, name, candidate["source_hash"], (work_code / "agents" / f"{name}.py").read_bytes())
    proposal["iteration"] = iteration
    save_json(root / f"iteration-{iteration:04d}.json", proposal)
    return proposal


def install_prepared_sources(root, code, candidates, file_hash, event):
    """Restart-safe import: only archived, hash-matching bytes; never overwrite."""
    for candidate in candidates:
        path = code / "agents" / f"{candidate['name']}.py"
        expected = candidate["source_hash"]
        if path.exists():
            if path.is_symlink() or file_hash(path) != expected:
                raise ValueError(f"Refusing to overwrite existing agent: {path.name}")
            continue
        archive = root / "source_archive" / f"{expected}.py"
        if archive.is_symlink() or not archive.is_file() or file_hash(archive) != expected:
            raise ValueError(f"Prepared source archive missing or invalid: {candidate['name']}")
        write_new_source(path, archive.read_bytes())
        event(root, "candidate_source_imported", name=candidate["name"], source_hash=expected)
