from pathlib import Path
import unittest

from src.config import build_vllm_command, load_yaml


class VllmCommandTests(unittest.TestCase):
    def test_preferred_parallel_layout_and_backend(self) -> None:
        config = load_yaml(Path("configs/model_qwen3_30b_a3b.yaml"))
        command = build_vllm_command(config, [])
        self.assertIn("--enable-expert-parallel", command)
        self.assertNotIn("--enable-eplb", command)
        self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "1")
        self.assertEqual(command[command.index("--data-parallel-size") + 1], "4")
        self.assertEqual(
            command[command.index("--all2all-backend") + 1],
            "allgather_reducescatter",
        )

    def test_extra_vllm_arguments_are_preserved(self) -> None:
        config = load_yaml(Path("configs/model_qwen3_30b_a3b.yaml"))
        command = build_vllm_command(config, ["--enforce-eager"])
        self.assertEqual(command[-1], "--enforce-eager")


if __name__ == "__main__":
    unittest.main()
