from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import submission_format as fmt


class SubmissionFormatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.outer = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_package(
        self,
        *,
        wrapped: bool = True,
        seeds: tuple[int, ...] = (101, 202),
        task_type: str = "model_training",
        with_models: bool = True,
    ) -> Path:
        root = self.outer / "upload" / "case" if wrapped else self.outer
        harbor = root / "workspace" / "harbor_task"
        for directory in (
            harbor / "environment" / "starter",
            harbor / "solution",
            harbor / "tests" / "hidden_assets",
            root / "workspace" / "reference",
            root / "expert_evidence",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        (harbor / "instruction.md").write_text(
            "This task trains a model from initialization." if task_type == "model_training" else "This is a non-training task; no training is performed.",
            encoding="utf-8",
        )
        (harbor / "task.toml").write_text('name = "fixture"\n', encoding="utf-8")
        opt = root / "optimization_evidence"
        opt.mkdir()
        (opt / "训练证据说明.md").write_text("真实运行证据。\n", encoding="utf-8")
        values: dict[str, dict[str, float]] = {}
        for role, offset in (("baseline", 0.0), ("reference", -0.1)):
            role_dir = opt / f"{role}_runs"
            role_dir.mkdir()
            for seed in seeds:
                run = role_dir / f"seed_{seed}"
                run.mkdir()
                value = 1.0 + offset + seed / 10000
                (run / "result.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "fixture.v1",
                            "status": "COMPLETE",
                            "role": role,
                            "seed": seed,
                            "task_type": task_type,
                            "method": {"source_path": "fixture.py"},
                            "protocol": {"train_from_initialization": task_type == "model_training"},
                            "training": {"epochs": 1} if task_type == "model_training" else {},
                            "execution": {"exit_code": 0},
                            "metrics": {"primary": {"name": "loss", "direction": "minimize", "value": value}},
                            "quality_gate": {"valid": True},
                            "artifacts": {},
                        }
                    ),
                    encoding="utf-8",
                )
                (run / "run.log").write_text("completed\n", encoding="utf-8")
                if with_models:
                    model = run / "model"
                    model.mkdir()
                    (model / "model.pt").write_bytes(b"checkpoint")
                    (model / "artifact.json").write_text('{"sha256":"fixture"}\n', encoding="utf-8")
                    (model / "reload.log").write_text("reload ok\n", encoding="utf-8")
                values.setdefault(str(seed), {})[role] = value
        (opt / "comparison_summary.json").write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "metric": {"name": "loss", "direction": "minimize"},
                    "seeds": list(seeds),
                    "paired_results": values,
                    "statistics": {"baseline_mean": 1.0, "reference_mean": 0.9},
                    "normalized_score": {"baseline": 0.0, "reference": 0.25},
                    "significance_rule": {"required_baseline_sigma_multiple": 3, "passed": True},
                }
            ),
            encoding="utf-8",
        )
        return root

    def issue_codes(self, report):
        return {row["code"] for row in report["alignment_issues"]}

    def test_aligned_training_package_below_wrappers(self):
        package = self.make_package()
        report = fmt.collect(self.outer)
        self.assertEqual(report["submission_root"], str(package.resolve()))
        self.assertEqual(report["submission_root_relative"], "upload/case")
        self.assertEqual(report["wrapper_depth"], 2)
        self.assertEqual(report["status"], "aligned")
        self.assertEqual(report["issues"], [])
        self.assertEqual(report["missing"], [])
        self.assertIn("规范结构已对齐", report["summary"])
        opt = report["optimization_evidence"]
        self.assertEqual(opt["task_kind"], "training")
        self.assertTrue(opt["seed_pairing"]["paired"])
        self.assertEqual(opt["seed_pairing"]["paired_seeds"], ["101", "202"])
        run = opt["baseline_runs"]["runs"]["101"]
        self.assertEqual(run["result"]["role"], "baseline")
        self.assertEqual(run["result"]["seed"], 101)
        self.assertEqual(run["result"]["primary_metric"]["name"], "loss")
        self.assertTrue(run["model"]["artifact"])
        self.assertTrue(run["model"]["reload"])

    def test_missing_top_level_and_harbor_material_are_issues(self):
        root = self.outer / "case"
        (root / "workspace" / "harbor_task").mkdir(parents=True)
        report = fmt.collect(self.outer)
        codes = self.issue_codes(report)
        self.assertEqual(report["status"], "deviations")
        self.assertIn("MISSING_TOP_LEVEL", codes)
        self.assertIn("MISSING_HARBOR_FILE", codes)
        self.assertIn("MISSING_HARBOR_DIR", codes)
        self.assertIn("MISSING_REFERENCE", codes)

    def test_unpaired_seed_sets_are_reported(self):
        root = self.make_package(seeds=(101, 202))
        reference_202 = root / "optimization_evidence" / "reference_runs" / "seed_202"
        for path in sorted(reference_202.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            else:
                path.rmdir()
        reference_202.rmdir()
        report = fmt.collect(self.outer)
        self.assertIn("UNPAIRED_SEEDS", self.issue_codes(report))
        pairing = report["optimization_evidence"]["seed_pairing"]
        self.assertEqual(pairing["baseline_only"], ["202"])
        self.assertEqual(pairing["reference_only"], [])

    def test_required_seed_files_and_minimal_result_fields(self):
        root = self.make_package(seeds=(101,))
        run = root / "optimization_evidence" / "baseline_runs" / "seed_101"
        (run / "run.log").unlink()
        (run / "result.json").write_text('{"role":"reference"}', encoding="utf-8")
        report = fmt.collect(self.outer)
        codes = self.issue_codes(report)
        self.assertIn("MISSING_RUN_LOG", codes)
        self.assertIn("MISSING_RESULT_FIELDS", codes)
        self.assertIn("RESULT_ROLE_MISMATCH", codes)

    def test_seed_mismatch_and_invalid_json_are_reported(self):
        root = self.make_package(seeds=(101,))
        baseline = root / "optimization_evidence" / "baseline_runs" / "seed_101" / "result.json"
        data = json.loads(baseline.read_text())
        data["seed"] = 999
        baseline.write_text(json.dumps(data), encoding="utf-8")
        reference = root / "optimization_evidence" / "reference_runs" / "seed_101" / "result.json"
        reference.write_text("{", encoding="utf-8")
        report = fmt.collect(self.outer)
        codes = self.issue_codes(report)
        self.assertIn("RESULT_SEED_MISMATCH", codes)
        self.assertIn("INVALID_RESULT_JSON", codes)

    def test_summary_seed_set_must_match_paired_runs(self):
        root = self.make_package(seeds=(101, 202))
        path = root / "optimization_evidence" / "comparison_summary.json"
        data = json.loads(path.read_text())
        data["seeds"] = [101]
        path.write_text(json.dumps(data), encoding="utf-8")
        report = fmt.collect(self.outer)
        self.assertIn("SUMMARY_SEED_MISMATCH", self.issue_codes(report))

    def test_training_model_artifact_and_reload_are_required(self):
        root = self.make_package(seeds=(101,))
        model = root / "optimization_evidence" / "reference_runs" / "seed_101" / "model"
        (model / "model.pt").unlink()
        (model / "artifact.json").unlink()
        (model / "reload.log").unlink()
        report = fmt.collect(self.outer)
        codes = self.issue_codes(report)
        self.assertIn("MISSING_MODEL_ARTIFACT", codes)
        self.assertIn("MISSING_ARTIFACT_MANIFEST", codes)
        self.assertIn("MISSING_RELOAD_LOG", codes)

    def test_non_training_empty_or_absent_model_is_allowed(self):
        root = self.make_package(
            seeds=(101,), task_type="non_training", with_models=False
        )
        empty_model = root / "optimization_evidence" / "baseline_runs" / "seed_101" / "model"
        empty_model.mkdir()
        report = fmt.collect(self.outer)
        self.assertEqual(report["status"], "aligned_with_extras")
        self.assertEqual(report["optimization_evidence"]["task_kind"], "non_training")
        self.assertIn(
            "EMPTY_NON_TRAINING_MODEL_DIR",
            {row["code"] for row in report["suggestions"]},
        )

    def test_non_training_nonempty_model_is_only_a_suggestion(self):
        root = self.make_package(
            seeds=(101,), task_type="non_training", with_models=False
        )
        model = root / "optimization_evidence" / "baseline_runs" / "seed_101" / "model"
        model.mkdir()
        (model / "unexpected.pt").write_bytes(b"optional")
        report = fmt.collect(self.outer)
        self.assertEqual(report["status"], "aligned_with_extras")
        self.assertIn(
            "NON_TRAINING_MODEL_PRESENT",
            {row["code"] for row in report["suggestions"]},
        )
        self.assertIn(
            "optimization_evidence/baseline_runs/seed_101/model",
            {row["path"] for row in report["extras"]["extra_allowed"]},
        )

    def test_extra_items_are_classified_without_failing(self):
        root = self.make_package(seeds=(101,))
        opt = root / "optimization_evidence"
        (opt / "README.md").write_text("help\n", encoding="utf-8")
        (opt / "experiment_plan.json").write_text("{}", encoding="utf-8")
        (opt / "ablation").mkdir()
        (root / "notes.txt").write_text("extra\n", encoding="utf-8")
        report = fmt.collect(self.outer)
        self.assertEqual(report["status"], "aligned_with_extras")
        self.assertIn(
            "optimization_evidence/experiment_plan.json",
            {row["path"] for row in report["extras"]["merge_candidate"]},
        )
        self.assertIn(
            "optimization_evidence/ablation",
            {row["path"] for row in report["extras"]["misplaced"]},
        )
        allowed = {row["path"] for row in report["extras"]["extra_allowed"]}
        self.assertIn("optimization_evidence/README.md", allowed)
        self.assertIn("notes.txt", allowed)

    def test_legacy_trusted_and_runtime_are_suggestions_not_failures(self):
        root = self.make_package(seeds=(101,))
        harbor = root / "workspace" / "harbor_task"
        (harbor / "environment" / "trusted").mkdir()
        (harbor / "tests" / "runtime").mkdir()
        report = fmt.collect(self.outer)
        self.assertEqual(report["status"], "aligned_with_extras")
        self.assertTrue(report["harbor_task"]["legacy_trusted_present"])
        self.assertTrue(report["harbor_task"]["legacy_tests_runtime_present"])
        codes = {row["code"] for row in report["suggestions"]}
        self.assertIn("LEGACY_TRUSTED_DIR", codes)
        self.assertIn("LEGACY_TESTS_RUNTIME", codes)
        self.assertTrue(
            all(
                {"path", "classification", "recommendation"} <= set(row)
                for row in report["suggestions"]
            )
        )

    def test_invalid_input_returns_stable_result(self):
        report = fmt.collect(self.outer / "missing")
        self.assertEqual(report["status"], "manual")
        self.assertIsNone(report["submission_root"])
        self.assertEqual(report["alignment_issues"][0]["code"], "INVALID_INPUT")
        self.assertEqual(report["issues"], report["alignment_issues"])


if __name__ == "__main__":
    unittest.main()
