from __future__ import annotations

from pathlib import Path


def test_project_dependencies_include_typer_for_gradio_cli_import():
    # Gradio imports `gradio.cli` (which depends on `typer`) from `gradio.__init__`.
    # Some environments may miss transitive deps; we pin it explicitly in our project deps.
    txt = Path("pyproject.toml").read_text(encoding="utf-8")

    in_project = False
    in_deps = False
    deps_lines: list[str] = []

    for raw in txt.splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            in_project = line == "[project]"
            in_deps = False
            continue
        if not in_project:
            continue
        if line.startswith("dependencies"):
            in_deps = True
            deps_lines.append(line)
            continue
        if in_deps:
            deps_lines.append(line)
            if line.startswith("]"):
                break

    deps_block = "\n".join(deps_lines)
    assert '"typer"' in deps_block

