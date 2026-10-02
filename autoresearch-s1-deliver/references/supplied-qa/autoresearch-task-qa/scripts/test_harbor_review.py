import json
from pathlib import Path
import tempfile
import unittest

import harbor_review as harbor


def valid_review():
    """Reviewed fixture conclusions; tests check contracts, not semantic accuracy."""
    return {"target_version": "official-docs@2026-09-08", "version_basis": harbor.OFFICIAL_SOURCE,
            "path_contract": {"profile": "harbor-environment-v1", "profile_basis": harbor.OFFICIAL_SOURCE,
                              "manual_resolution": {"summary": "人工已读 task.toml 的 environment.docker_image，使用预构建镜像。", "evidence": ["task.toml"]}},
            "provider": "docker", "task_root": ".", "checks": [
                {"id": "H01", "status": "pass", "summary": "任务根目录与入口存在。", "evidence": ["task.toml", "instruction.md", "tests/test.sh"]},
                {"id": "H02", "status": "pass", "summary": "已按目标模型阅读配置字段。", "evidence": ["task.toml"]},
                {"id": "H03", "status": "pass", "summary": "使用镜像配置，无须强制 Dockerfile。", "evidence": ["task.toml"]},
                {"id": "H04", "status": "pass", "summary": "test.sh 写入标准 reward 文件。", "evidence": ["tests/test.sh"]},
                {"id": "H05", "status": "not_applicable", "summary": "单任务包未附批量调用配置。", "evidence": []},
                {"id": "H06", "status": "not_applicable", "summary": "未提供 Harness 运行记录，未进行实跑。", "evidence": []}]}


class HarborReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        (self.root / "task.toml").write_text('[environment]\ndocker_image = "test-fixture:local"\n')
        (self.root / "instruction.md").write_text("Fixture task")
        (self.root / "tests").mkdir()
        (self.root / "tests/test.sh").write_text("mkdir -p /logs/verifier\necho 1 > /logs/verifier/reward.txt\n")

    def tearDown(self):
        self.temp.cleanup()

    def facts(self):
        return harbor.collect(self.root, [p.relative_to(self.root).as_posix() for p in self.root.rglob('*') if p.is_file()])

    def apply(self, value):
        return harbor.apply_review(self.facts(), value, self.root)

    def test_inventory_is_not_a_pass(self):
        self.assertEqual(self.facts()["static_status"], "manual")
        self.assertEqual(self.facts()["observations"]["task_root"], ".")

    def test_static_pass_does_not_claim_execution(self):
        result = self.apply(valid_review())
        self.assertEqual(result["static_status"], "pass")
        self.assertEqual(result["runtime_status"], "not_run")

    def test_missing_six_part_review_cannot_pass(self):
        with self.assertRaises(ValueError):
            self.apply({})

    def test_unknown_version_cannot_pass_config_review(self):
        review = valid_review()
        review["target_version"] = "unknown"
        with self.assertRaises(ValueError):
            self.apply(review)

    def test_unknown_provider_cannot_pass_environment_review(self):
        review = valid_review()
        review["provider"] = "unknown"
        with self.assertRaises(ValueError):
            self.apply(review)

    def test_reward_interface_failure_is_qa17_failure(self):
        review = valid_review()
        review["checks"][3]["status"] = "fail"
        review["checks"][3]["summary"] = "入口仅输出 stdout，未实现 reward 文件写入。"
        self.assertEqual(self.apply(review)["qa17_status"], "fail")

    def test_required_checks_cannot_be_skipped(self):
        review = valid_review()
        review["checks"][2]["status"] = "not_applicable"
        with self.assertRaises(ValueError):
            self.apply(review)

    def test_path_escape_rejected(self):
        review = valid_review()
        review["task_root"] = ".."
        with self.assertRaises(ValueError):
            self.apply(review)

    def test_explicit_path_profile_required(self):
        review = valid_review()
        del review["path_contract"]
        with self.assertRaisesRegex(ValueError, "profile"):
            self.apply(review)

    def test_wrong_teaching_dockerfile_cannot_be_overridden(self):
        review = valid_review()
        review["path_contract"]["profile"] = "teaching-task-root-v1"
        result = self.apply(review)
        self.assertEqual(result["checks"][2]["status"], "fail")
        self.assertEqual(result["qa17_status"], "fail")

    def test_fake_manual_resolution_reference_rejected(self):
        review = valid_review()
        review["path_contract"]["manual_resolution"] = {"summary": "已复核。", "evidence": ["missing.txt"]}
        with self.assertRaisesRegex(ValueError, "not a file"):
            self.apply(review)

    def test_manual_resolution_cannot_override_definite_failure(self):
        review = valid_review()
        review["path_contract"] = {"profile": "teaching-task-root-v1", "profile_basis": "fixture",
                                  "manual_resolution": {"summary": "口头声称可以运行。", "evidence": ["task.toml"]}}
        result = self.apply(review)
        self.assertEqual(result["checks"][2]["status"], "fail")

    def trial(self, value=1.25):
        folder = self.root / "run-a"
        (folder / "verifier").mkdir(parents=True)
        (folder / "config.json").write_text('{"task":{"path":"fixture"}}')
        (folder / "result.json").write_text(json.dumps({"finished_at": "2026-09-08T06:00:00Z", "exception_info": None,
                                                     "verifier_result": {"rewards": {"reward": value}}}))
        (folder / "verifier/reward.txt").write_text(str(value))
        (folder / "trial.log").write_text("Fixture verifier completed")
        review = valid_review()
        review["checks"][5] = {"id": "H06", "status": "pass", "summary": "同一 trial 的结果与 reward 一致。",
                                "evidence": ["run-a/config.json", "run-a/result.json", "run-a/verifier/reward.txt", "run-a/trial.log"]}
        return review

    def test_above_one_reward_is_accepted(self):
        result = self.apply(self.trial(1.25))
        self.assertEqual(result["runtime_status"], "evidence_consistent")

    def test_negative_reward_is_not_harness_failure(self):
        self.assertEqual(self.apply(self.trial(-0.5))["runtime_status"], "evidence_consistent")

    def test_reward_mismatch_rejected(self):
        review = self.trial()
        (self.root / "run-a/verifier/reward.txt").write_text("0.0")
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.apply(review)

    def test_nan_rejected(self):
        review = self.trial(float('nan'))
        with self.assertRaises(ValueError):
            self.apply(review)

    def test_exception_cannot_be_successful_evidence(self):
        review = self.trial()
        path = self.root / "run-a/result.json"
        result = json.loads(path.read_text())
        result["exception_info"] = {"exception_type": "VerifierTimeoutError"}
        path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "exception_info"):
            self.apply(review)

    def test_fake_runtime_status_does_not_override_evidence(self):
        review = valid_review()
        review["runtime_status"] = "verified"
        self.assertEqual(self.apply(review)["runtime_status"], "not_run")

    def test_reward_json_precedence(self):
        review = self.trial()
        (self.root / "run-a/verifier/reward.json").write_text('{"score":1.25,"status":"ok"}')
        review["checks"][5]["evidence"].append("run-a/verifier/reward.json")
        with self.assertRaisesRegex(ValueError, "numeric"):
            self.apply(review)


if __name__ == '__main__':
    unittest.main()
