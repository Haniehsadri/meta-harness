"""Protection boundaries: real Linux writes, source recovery, and staged rejection."""
import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import adaptive_evolution as e
from proposal_workspace import (accept_proposal, archive_source, install_prepared_sources,
                                preserve_sources, proposal_workspace, write_new_source)


class ProposalProtectionTests(unittest.TestCase):
    def prepare(self, parent):
        root = Path(parent) / "run"
        code = root / "code/text_classification"
        (code / "agents").mkdir(parents=True)
        (code / "agents/old.py").write_text("# original\n")
        (code / "config.yaml").write_text("datasets: [LawBench]\n")
        record = {"name": "old", "source_hash": e.file_hash(code / "agents/old.py")}
        harness = types.SimpleNamespace(EVOLVE_DIR=code, LOGS_DIR=root,
            PENDING_EVAL=root/"pending_eval.json", FRONTIER_VAL=root/"frontier_val.json",
            EVOLUTION_SUMMARY=root/"evolution_summary.jsonl")
        return root, code, record, harness

    def test_overwrite_or_delete_existing_copy_rejects_without_losing_canonical_source(self):
        for operation in ("overwrite", "delete"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as tmp:
                root, code, record, harness = self.prepare(tmp)
                with proposal_workspace(root, code, harness, 8, ["LawBench"], e.file_hash, e.event) as (_, work, existing):
                    target = work / "agents/old.py"
                    if operation == "delete":
                        target.unlink()
                    else:
                        target.write_text("# changed")
                    e.save_json(harness.PENDING_EVAL, {"candidates": [{"name":"old"}]})
                    with self.assertRaisesRegex(ValueError, "existing workspace files"):
                        accept_proposal(root, code, work, existing, harness.PENDING_EVAL, 8, e.file_hash, e.save_json)
                self.assertEqual(e.file_hash(code / "agents/old.py"), record["source_hash"])
                self.assertEqual(harness.EVOLVE_DIR, code)
                self.assertFalse(hasattr(harness, "PROPOSAL_WORKSPACE"))
                self.assertFalse((root/"iteration-0008.json").exists())

    def test_collision_symlink_and_invalid_python_cannot_be_imported(self):
        for kind in ("collision", "symlink", "syntax"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root, code, record, harness = self.prepare(tmp)
                with proposal_workspace(root, code, harness, 8, ["LawBench"], e.file_hash, e.event) as (_, work, existing):
                    name = "old" if kind == "collision" else "new"
                    if kind == "symlink":
                        (work/"agents/new.py").symlink_to(code/"agents/old.py")
                    elif kind == "syntax":
                        (work/"agents/new.py").write_text("def broken(")
                    e.save_json(harness.PENDING_EVAL, {"candidates":[{"name": name}]})
                    with self.assertRaises((ValueError, SyntaxError)):
                        accept_proposal(root, code, work, existing, harness.PENDING_EVAL, 8, e.file_hash, e.save_json)
                self.assertFalse((code/"agents/new.py").exists())
                self.assertFalse((root/"iteration-0008.json").exists())

    def test_accepted_sources_survive_interrupted_import_and_missing_file_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, code, record, harness = self.prepare(tmp)
            with proposal_workspace(root, code, harness, 8, ["LawBench"], e.file_hash, e.event) as (_, work, existing):
                (work/"agents/iter0008_new.py").write_text("# new implementation\n")
                e.save_json(harness.PENDING_EVAL, {"candidates":[{"name":"iter0008_new"}]})
                result = accept_proposal(root, code, work, existing, harness.PENDING_EVAL, 8, e.file_hash, e.save_json)
            self.assertFalse((code/"agents/iter0008_new.py").exists())
            self.assertTrue((root/"iteration-0008.json").exists())
            install_prepared_sources(root, code, result["candidates"], e.file_hash, e.event)
            install_prepared_sources(root, code, result["candidates"], e.file_hash, e.event)
            preserve_sources(root, code, [record], e.file_hash, e.event)
            (code/"agents/old.py").unlink()
            preserve_sources(root, code, [record], e.file_hash, e.event)
            self.assertEqual(e.file_hash(code/"agents/old.py"), record["source_hash"])
            (code/"agents/iter0008_new.py").write_text("# somebody changed it")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                install_prepared_sources(root, code, result["candidates"], e.file_hash, e.event)

    def test_failed_atomic_publication_leaves_no_partial_agent_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "agent.py"
            with patch("proposal_workspace.os.link", side_effect=OSError("interrupted publish")):
                with self.assertRaisesRegex(OSError, "interrupted publish"):
                    write_new_source(target, b"# complete")
            self.assertFalse(target.exists())
            write_new_source(target, b"# complete")
            with self.assertRaises(FileExistsError):
                write_new_source(target, b"# overwrite")
            self.assertEqual(target.read_bytes(), b"# complete")
            self.assertEqual(list(Path(tmp).glob(".source-*")), [])

    def test_corrupt_archive_is_never_used_to_reconstruct_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, code, record, _ = self.prepare(tmp)
            preserve_sources(root, code, [record], e.file_hash, e.event)
            (code/"agents/old.py").unlink()
            (root/"source_archive"/(record["source_hash"]+".py")).write_text("# wrong")
            with self.assertRaisesRegex(ValueError, "archive hash mismatch"):
                preserve_sources(root, code, [record], e.file_hash, e.event)
            self.assertFalse((code/"agents/old.py").exists())

    def test_workspace_globals_restored_when_proposer_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, code, _, harness = self.prepare(tmp)
            with self.assertRaisesRegex(RuntimeError, "failed"):
                with proposal_workspace(root, code, harness, 1, ["LawBench"], e.file_hash, e.event):
                    raise RuntimeError("failed")
            self.assertEqual(harness.EVOLVE_DIR, code)
            self.assertFalse(hasattr(harness,"PROPOSAL_WORKSPACE"))

    @unittest.skipUnless(sys.platform == "linux", "kernel sandbox is Linux-only")
    def test_real_root_and_child_write_delete_truncate_and_symlink_attacks_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp)
            workspace=parent/"workspace"
            workspace.mkdir()
            victim=parent/"evaluated.py"
            victim.write_text("original")
            (workspace/"escape").symlink_to(victim)
            script=workspace/"check.py"
            script.write_text("""
import os, subprocess, sys
from pathlib import Path
victim=Path(sys.argv[1]); workspace=Path(sys.argv[2])
def forbidden(operation):
    try: operation()
    except OSError: return
    raise AssertionError('outside mutation was allowed')
forbidden(lambda: victim.write_text('changed'))
forbidden(lambda: victim.unlink())
forbidden(lambda: os.truncate(victim,0))
forbidden(lambda: os.open(victim,os.O_RDONLY|os.O_TRUNC))
forbidden(lambda: (workspace/'escape').write_text('changed'))
forbidden(lambda: victim.rename(workspace/'stolen.py'))
child=subprocess.run([sys.executable,'-c','from pathlib import Path; import sys; Path(sys.argv[1]).write_text("changed")',str(victim)],capture_output=True)
assert child.returncode != 0
(workspace/'allowed.txt').write_text('allowed')
assert victim.read_text()=='original'
print('WRITE_BOUNDARY_OK')
""")
            sandbox=Path(e.__file__).with_name("proposal_sandbox.py")
            result=subprocess.run([sys.executable,str(sandbox),'--workspace',str(workspace),'--',
                sys.executable,'-B',str(script),str(victim),str(workspace)],capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('WRITE_BOUNDARY_OK',result.stdout)
            self.assertEqual(victim.read_text(),'original')

    def test_sandbox_failure_never_executes_target_command(self):
        import proposal_sandbox
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                sys,'argv',['sandbox','--workspace',tmp,'--','must-not-run']), patch.object(
                proposal_sandbox,'confine_writes',side_effect=RuntimeError('unavailable')), patch.object(
                proposal_sandbox.os,'execvp') as execute, patch('sys.stderr',io.StringIO()):
            self.assertEqual(proposal_sandbox.main(),125)
            execute.assert_not_called()
