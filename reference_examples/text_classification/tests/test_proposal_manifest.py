"""Known manifest-placement failures and strict acceptance of current submissions."""
import io
import json
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import adaptive_evolution as e
import meta_harness
from proposal_workspace import (accept_proposal, proposal_workspace, read_submission,
                                resolve_pending_manifest)


class ManifestTests(unittest.TestCase):
    def layout(self, parent):
        workspace=Path(parent)/"workspace"
        code=workspace/"code/text_classification"
        (code/"agents").mkdir(parents=True)
        (workspace/"context").mkdir()
        return workspace,code

    def payload(self, iteration=10, name="iter0010_candidate"):
        return {"iteration":iteration,"candidates":[{"name":name,"file":f"agents/{name}.py"}]}

    def test_all_known_locations_are_resolved_and_accepted(self):
        for location in ("cwd","context","workspace","package_parent"):
            with self.subTest(location=location),tempfile.TemporaryDirectory() as tmp:
                workspace,code=self.layout(tmp)
                paths={"cwd":code/"pending_eval.json","context":workspace/"context/pending_eval.json",
                       "workspace":workspace/"pending_eval.json","package_parent":code.parent/"pending_eval.json"}
                paths[location].write_text(json.dumps(self.payload()))
                (code/"agents/iter0010_candidate.py").write_text("# new source")
                path=resolve_pending_manifest(workspace,code,10)
                self.assertEqual(path,paths[location])
                root=Path(tmp)/"canonical"
                (root/"code/agents").mkdir(parents=True)
                result=accept_proposal(root,root/"code",code,{},path,10,e.file_hash,e.save_json)
                self.assertEqual(result["iteration"],10)
                self.assertTrue((root/"iteration-0010.json").exists())

    def test_conflicting_or_stale_manifests_are_not_silently_chosen(self):
        for alternate in (self.payload(name="different"),self.payload(iteration=9)):
            with tempfile.TemporaryDirectory() as tmp:
                workspace,code=self.layout(tmp)
                (code/"pending_eval.json").write_text(json.dumps(self.payload()))
                (workspace/"context/pending_eval.json").write_text(json.dumps(alternate))
                with self.assertRaises(ValueError):
                    resolve_pending_manifest(workspace,code,10)

    def test_identical_duplicates_use_cwd_without_ambiguity(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace,code=self.layout(tmp)
            for path in (code/"pending_eval.json",workspace/"context/pending_eval.json"):
                path.write_text(json.dumps(self.payload()))
            self.assertEqual(resolve_pending_manifest(workspace,code,10),code/"pending_eval.json")

    def test_missing_malformed_or_nonobject_manifests_fail_safely(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace,code=self.layout(tmp)
            self.assertIsNone(resolve_pending_manifest(workspace,code,10))
            path=code/"pending_eval.json"
            for value in ("broken","[]",json.dumps({"candidates":[{}]}),
                          json.dumps({"iteration":True,"candidates":[{}]}),
                          json.dumps({"iteration":10,"candidates":["wrong"]})):
                path.write_text(value)
                with self.assertRaises(ValueError):
                    resolve_pending_manifest(workspace,code,10)

    def test_symlink_and_other_attempt_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace,code=self.layout(tmp)
            outside=Path(tmp)/"older_pending_eval.json"
            outside.write_text(json.dumps(self.payload()))
            self.assertIsNone(resolve_pending_manifest(workspace,code,10))
            (code/"pending_eval.json").symlink_to(outside)
            with self.assertRaisesRegex(ValueError,"symlink"):
                resolve_pending_manifest(workspace,code,10)

    def test_inherited_pending_is_not_copied_into_new_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"run";code=root/"code/text_classification"
            (code/"agents").mkdir(parents=True)
            (code/"pending_eval.json").write_text(json.dumps(self.payload(iteration=9)))
            harness=types.SimpleNamespace(EVOLVE_DIR=code)
            with proposal_workspace(root,code,harness,10,["LawBench"],e.file_hash,e.event) as (workspace,work,_):
                self.assertEqual(harness.PENDING_EVAL,work/"pending_eval.json")
                self.assertFalse(harness.PENDING_EVAL.exists())
                self.assertIsNone(resolve_pending_manifest(workspace,work,10))

    def test_wrapper_success_checks_alternate_manifest_not_just_expected_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace,code=self.layout(tmp)
            expected=code/"pending_eval.json"
            def propose(**kwargs):
                self.assertEqual(kwargs["write_workspace"],workspace)
                (workspace/"context/pending_eval.json").write_text(json.dumps(self.payload()))
                return Mock(exit_code=0)
            with patch.object(meta_harness,"EVOLVE_DIR",code), patch.object(
                    meta_harness,"PROPOSAL_WORKSPACE",workspace,create=True), patch.object(
                    meta_harness,"PENDING_EVAL",expected), patch.object(
                    meta_harness.claude_wrapper,"run",side_effect=propose), redirect_stdout(io.StringIO()):
                self.assertTrue(meta_harness.propose_claude("prompt",10))
            self.assertFalse(expected.exists())

    def test_manifest_file_must_match_candidate_name_before_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace,code=self.layout(tmp)
            payload=self.payload();payload["candidates"][0]["file"]="../outside.py"
            pending=code/"pending_eval.json";pending.write_text(json.dumps(payload))
            (code/"agents/iter0010_candidate.py").write_text("# source")
            root=Path(tmp)/"run";(root/"code/agents").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,"does not match"):
                accept_proposal(root,root/"code",code,{},pending,10,e.file_hash,e.save_json)
            self.assertFalse((root/"iteration-0010.json").exists())

    def test_safe_relative_spellings_preserve_source_and_original_metadata(self):
        from proposal_workspace import normalize_candidate_submission
        with tempfile.TemporaryDirectory() as tmp:
            _, code = self.layout(tmp)
            name = "iter0009_damped_adaptive_k"
            source = code / "agents" / f"{name}.py"
            source.write_text("# unchanged candidate\n")
            before = source.read_bytes()
            for prefix in ("", "./", "text_classification/", "./text_classification/"):
                declared = f"{prefix}agents/{name}.py"
                result = normalize_candidate_submission({"name": name, "file": declared}, code, 9)
                self.assertEqual(result["file"], f"agents/{name}.py")
                self.assertEqual(source.read_bytes(), before)
                if prefix:
                    self.assertEqual(result["original_submission"]["file"], declared)

    def test_unsafe_relative_spellings_are_rejected(self):
        from proposal_workspace import normalize_candidate_submission
        with tempfile.TemporaryDirectory() as tmp:
            _, code = self.layout(tmp)
            for path in ("../agents/a.py", "text_classification/../agents/a.py",
                         "/agents/a.py", "././agents/a.py", "other/agents/a.py",
                         "agents/../a.py", "text_classification//agents/a.py"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    normalize_candidate_submission({"name": "a", "file": path}, code, 9)