from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


COVER_TEMPLATES = [
    "Covering [something] with [something]",
    "Putting [number of] [something] onto [something]",
    "Putting [something] and [something] on the table",
    "Putting [something] on a surface",
    "Putting [something] onto a slanted surface but it doesn't glide down",
    "Putting [something] onto [something]",
    "Putting [something similar to other things that are already on the table]",
    "Putting [something] that can't roll onto a slanted surface, so it slides down",
    "Putting [something] that can't roll onto a slanted surface, so it stays where it is",
    "Putting [something] upright on the table",
    "Putting [something], [something] and [something] on the table",
]


def normalize_template(template: str) -> str:
    """Remove brackets so strings match the wording in test-answers.csv."""
    return template.replace("[", "").replace("]", "")


def load_test_ids(label_dir: Path, normalized_templates: set[str]) -> set[str]:
    """Pull clip ids whose text matches the templates from test-answers.csv."""
    ids: set[str] = set()
    answers_path = label_dir / "test-answers.csv"
    if not answers_path.exists():
        print(f"{answers_path} 缺失，测试集无法筛选")
        return ids
    with answers_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        for row in reader:
            if len(row) != 2:
                continue
            clip_id, template_text = row[0].strip(), row[1].strip()
            if template_text in normalized_templates:
                ids.add(clip_id)
    return ids


def sort_entries_by_id(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return entries sorted by numeric id (ascending)."""
    def id_key(entry: Dict[str, Any]) -> int:
        identifier = entry.get("id")
        try:
            return int(identifier)
        except (TypeError, ValueError):
            return int(1e12)
    return sorted(entries, key=id_key)


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    label_dir = repo_dir.parent / "labels"
    out_dir = label_dir / "cover_object_order"
    out_dir.mkdir(parents=True, exist_ok=True)

    normalized_templates = {normalize_template(t) for t in COVER_TEMPLATES}
    test_ids = load_test_ids(label_dir, normalized_templates)

    split_files = {
        "train": label_dir / "train.json",
        "validation": label_dir / "validation.json",
        "test": label_dir / "test.json",
    }
    filename_map = {"train": "train", "validation": "val", "test": "test"}

    stats: dict[str, int] = {}
    for split, file_path in split_files.items():
        if not file_path.exists():
            print(f"{file_path} 缺失，跳过 {split}")
            continue
        entries = json.loads(file_path.read_text(encoding="utf-8"))
        if split == "test":
            filtered = [e for e in entries if str(e.get("id")) in test_ids]
        else:
            filtered = [e for e in entries if e.get("template") in COVER_TEMPLATES]
        filtered = sort_entries_by_id(filtered)
        stats[split] = len(filtered)

        out_path = out_dir / f"{filename_map[split]}_cover_object.json"
        out_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{split}: {len(filtered)} entries -> {out_path}")

    print("cover_object_order_summary:", stats)


if __name__ == "__main__":
    main()
