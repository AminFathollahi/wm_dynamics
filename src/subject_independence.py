import json
from functools import lru_cache
from pathlib import Path

REGISTRY_PATH = Path(__file__).parent.parent / "provenance" / "subject_independence_registry.json"


@lru_cache(maxsize=1)
def _alias_to_group() -> dict[str, str]:
    registry = json.loads(REGISTRY_PATH.read_text())
    mapping = {}
    for group in registry["groups"]:
        for member in group["member_corpus_identifiers"]:
            mapping[member] = group["group_name"]
    return mapping


def resolve_group(corpus_id: str) -> str:
    mapping = _alias_to_group()
    if corpus_id not in mapping:
        raise KeyError(
            f"corpus identifier {corpus_id!r} is not registered in "
            f"{REGISTRY_PATH}; add it to a group's member_corpus_identifiers "
            "before using it, do not default it to independent"
        )
    return mapping[corpus_id]


def count_independent_groups(corpus_ids) -> int:
    return len({resolve_group(c) for c in corpus_ids})
