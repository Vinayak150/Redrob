"""Parse job_description.docx into config/jd_profile.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.utils import CONFIG_DIR, DATA_RAW, docx_to_text, load_yaml


def build_jd_profile() -> dict:
    existing = load_yaml(CONFIG_DIR / "jd_profile.yaml")
    docx_path = DATA_RAW / "job_description.docx"
    if docx_path.exists():
        existing["jd_text_full"] = docx_to_text(docx_path)
    return existing


def main() -> None:
    profile = build_jd_profile()
    out = CONFIG_DIR / "jd_profile.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(profile, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"Updated {out}")


if __name__ == "__main__":
    main()
