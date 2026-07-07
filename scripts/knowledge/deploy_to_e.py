"""deploy_to_e — mirror the version-controlled ops into the co-located E-drive operating layer.

Owner requirement: the operations must live in the SAME directory as the knowledge. The canonical
SOURCE is version-controlled here (work repo ``scripts/knowledge/``); this copies the ops modules +
README into ``$SPEECHRL_KB_DIR/ops/`` and stamps ``ops/VERSION`` with the current git sha so the
deployed layer is always traceable back to a commit.

    python scripts/knowledge/deploy_to_e.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def git_sha(repo_dir: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        dirty = subprocess.run(
            ["git", "-C", str(repo_dir), "status", "--porcelain"],
            capture_output=True, text=True,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception as e:
        return f"unknown ({type(e).__name__})"


def main() -> int:
    src = Path(__file__).resolve().parent
    kb_root = Path(os.environ.get("SPEECHRL_KB_DIR", "E:/speechrl-knowledge"))
    dst = kb_root / "ops"
    dst.mkdir(parents=True, exist_ok=True)

    copied = []
    for f in sorted(src.glob("*.py")) + sorted(src.glob("*.md")):
        if f.name == "deploy_to_e.py":
            continue
        shutil.copy2(f, dst / f.name)
        copied.append(f.name)

    sha = git_sha(src)
    (dst / "VERSION").write_text(
        f"deployed_from: {src}\ngit_sha: {sha}\nfiles: {', '.join(copied)}\n",
        encoding="utf-8",
    )
    print(f"[deploy_to_e] -> {dst}\n  git_sha={sha}\n  files={copied}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
