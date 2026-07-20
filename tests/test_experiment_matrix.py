from pathlib import Path

from scripts.run_locked_experiment_matrix import build_commands


def test_locked_matrix_contains_twenty_tabular_endpoints_and_survival():
    commands = build_commands(
        protocol_path=Path("protocol.json"),
        input_dir=Path("inputs"),
        output_dir=Path("outputs"),
        allow_draft=False,
    )
    assert len(commands) == 21
    assert sum("run_nested_tabular_probe.py" in " ".join(command) for command in commands) == 20
    assert sum("run_nested_survival_probe.py" in " ".join(command) for command in commands) == 1
    assert all(command.count("--feature-group") == 6 for command in commands)
