import tempfile
from pathlib import Path
import unittest

import docker_paths as paths


DOCKER = """FROM example:local
ENV EVFI_TASK_ROOT=/workspace
COPY environment/requirements.txt /tmp/requirements.txt
WORKDIR /workspace
COPY instruction.md task.toml ./
COPY environment/starter ./environment/starter
COPY solution ./solution
COPY tests ./tests
CMD ["bash"]
"""


class DockerPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.task = self.root / "wrapper/workspace/harbor_task"
        for name in ("environment/requirements.txt", "environment/starter/method.py", "solution/solve.sh", "tests/test.sh", "instruction.md", "task.toml"):
            path = self.task / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n")
        self.docker = self.task / "environment/Dockerfile"
        self.docker.write_text(DOCKER)

    def tearDown(self):
        self.temp.cleanup()

    def inspect(self, profile=paths.TEACHING, **extra):
        return paths.inspect(self.root, "wrapper/workspace/harbor_task", declaration={"profile": profile, "profile_basis": "fixture", **extra})

    def codes(self, result):
        return {f["code"] for f in result["findings"]}

    def test_teaching_example_paths_and_wrapper(self):
        result = self.inspect()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["build_context"], "wrapper/workspace/harbor_task")
        self.assertEqual(result["runtime_task_root"], "/workspace")
        self.assertEqual(result["test_entry"], "/workspace/tests/test.sh")

    def test_wrong_dockerfile_location_fails(self):
        self.docker.rename(self.task / "Dockerfile")
        result = self.inspect()
        self.assertEqual(result["status"], "fail")
        self.assertIn("dockerfile_missing", self.codes(result))

    def test_profiles_cannot_mix_contexts(self):
        result = self.inspect(paths.NATIVE)
        self.assertEqual(result["status"], "fail")
        self.assertIn("copy_source_missing", self.codes(result))

    def test_explicit_wrong_context_fails(self):
        result = self.inspect(build_context="wrapper/workspace/harbor_task/environment")
        self.assertIn("profile_path_mismatch", self.codes(result))

    def test_json_copy_and_continuations(self):
        self.docker.write_text(DOCKER.replace("COPY instruction.md task.toml ./", 'COPY ["instruction.md", "task.toml", "./"]').replace("COPY tests ./tests", "COPY --chown=1000:1000 \\\n+ tests ./tests".replace("\n+", "\n")))
        self.assertEqual(self.inspect()["status"], "pass")

    def test_dotdot_source_fails(self):
        self.docker.write_text(DOCKER + "COPY ../reference /private\n")
        self.assertIn("copy_source_escape", self.codes(self.inspect()))

    def test_private_source_fails(self):
        self.docker.write_text(DOCKER + "COPY reference /private\n")
        self.assertIn("private_copy_source", self.codes(self.inspect()))

    def test_environment_root_mismatch_fails(self):
        self.docker.write_text(DOCKER.replace("EVFI_TASK_ROOT=/workspace", "EVFI_TASK_ROOT=/harbor_task"))
        self.assertIn("runtime_root_mismatch", self.codes(self.inspect()))

    def test_workdir_mismatch_fails(self):
        self.docker.write_text(DOCKER.replace("WORKDIR /workspace", "WORKDIR /harbor_task"))
        self.assertIn("workdir_mismatch", self.codes(self.inspect()))

    def test_dynamic_source_manual_not_false_missing(self):
        self.docker.write_text(DOCKER + "COPY ${ASSET_DIR} /assets\n")
        self.assertEqual(self.inspect()["status"], "manual")

    def test_stage_source_not_local_source(self):
        self.docker.write_text(DOCKER + "COPY --from=builder /artifact /artifact\n")
        self.assertEqual(self.inspect()["status"], "manual")
        self.assertNotIn("copy_source_missing", self.codes(self.inspect()))

    def test_ignore_of_required_source_fails(self):
        (self.task / ".dockerignore").write_text("tests\n")
        self.assertIn("copy_source_ignored", self.codes(self.inspect()))

    def test_complex_ignore_requires_manual(self):
        (self.task / ".dockerignore").write_text("**\n!tests/**\n")
        self.assertEqual(self.inspect()["status"], "manual")

    def test_missing_runtime_test_entry_fails(self):
        self.docker.write_text(DOCKER.replace("COPY tests ./tests", "COPY tests /wrong_tests"))
        result = self.inspect()
        self.assertEqual(result["status"], "fail")
        self.assertIn("runtime_entry_unmapped", self.codes(result))

    def test_run_generated_runtime_entry_is_manual(self):
        self.docker.write_text(DOCKER.replace("COPY tests ./tests", "RUN generate-tests"))
        self.assertEqual(self.inspect()["status"], "manual")

    def test_native_test_is_harness_injected(self):
        self.docker.write_text("FROM example:local\nWORKDIR /app\nCOPY requirements.txt /tmp/requirements.txt\n")
        result = self.inspect(paths.NATIVE, runtime_task_root="/app")
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["test_entry"], "/tests/test.sh")

    def test_native_prebuilt_does_not_require_dockerfile(self):
        self.docker.unlink()
        result = paths.inspect(self.root, "wrapper/workspace/harbor_task", {"environment": {"docker_image": "fixture:local"}}, {"profile": paths.NATIVE})
        self.assertEqual(result["status"], "not_applicable")

    def test_context_symlink_escape_fails(self):
        outside = self.root / "outside.txt"
        outside.write_text("fixture")
        (self.task / "outside.txt").symlink_to(outside)
        self.docker.write_text(DOCKER + "COPY outside.txt /outside.txt\n")
        self.assertIn("copy_symlink_escape", self.codes(self.inspect()))

    def test_fixed_test_entry_cannot_be_replaced(self):
        result = paths.inspect(self.root, "wrapper/workspace/harbor_task", {"entrypoint": {"test_script": "tests/wrong.sh"}})
        self.assertIn("canonical_entry_mismatch", self.codes(result))

    def test_missing_cmd_script_fails(self):
        self.docker.write_text(DOCKER.replace('CMD ["bash"]', 'CMD ["bash", "/workspace/solution/missing.sh"]'))
        self.assertIn("runtime_command_unmapped", self.codes(self.inspect()))

    def test_valid_cmd_script_passes(self):
        self.docker.write_text(DOCKER.replace('CMD ["bash"]', 'CMD ["bash", "/workspace/solution/solve.sh"]'))
        self.assertEqual(self.inspect()["status"], "pass")

    def test_entry_symlink_escape_fails(self):
        outside = self.root / "outside.sh"
        outside.write_text("fixture")
        (self.task / "tests/test.sh").unlink()
        (self.task / "tests/test.sh").symlink_to(outside)
        self.assertIn("entry_path_escape", self.codes(self.inspect()))

    def test_unused_stage_does_not_cause_false_runtime_root_failure(self):
        self.docker.write_text("FROM example:local AS builder\nENV TASK_ROOT=/build\nCOPY unavailable /optional\n" + DOCKER)
        result = self.inspect()
        self.assertEqual(result["status"], "manual")
        self.assertNotIn("runtime_root_mismatch", self.codes(result))


if __name__ == "__main__":
    unittest.main()
