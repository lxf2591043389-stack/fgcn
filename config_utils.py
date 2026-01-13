import json
import os


def load_run_config(path):
    if not path:
        return {}
    if not os.path.isabs(path):
        base_dir = os.path.dirname(__file__)
        path = os.path.join(base_dir, path)
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_defaults(defaults, cfg, sections):
    for key in sections:
        values = cfg.get(key, {})
        if not isinstance(values, dict):
            continue
        defaults.update(values)
    return defaults
