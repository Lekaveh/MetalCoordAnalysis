import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import metalCoord


def _site_key(site: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        site.get("chain"),
        site.get("residue"),
        site.get("sequence"),
        site.get("icode"),
        site.get("altloc"),
        site.get("metal"),
        site.get("metalElement"),
    )


def _normalize_value(value: Any) -> Any:
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive conversion
            pass
    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        try:
            return value.tolist()
        except Exception:  # pragma: no cover - defensive conversion
            pass
    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(v) for v in value]
    return value


def _default_debug_json_path(output_path: str) -> str:
    return output_path + ".debug.json"


def _default_debug_md_path(output_path: str) -> str:
    return output_path + ".debug.md"


def resolve_debug_paths(
    output_path: str, override: Optional[str], multi_ligand: bool = False
) -> tuple[str, str]:
    if not override:
        return _default_debug_json_path(output_path), _default_debug_md_path(output_path)

    override_path = Path(override)
    if multi_ligand:
        if override_path.suffix:
            raise ValueError("For multi-ligand stats, --debug-output must be a directory.")
        return (
            str(override_path / Path(_default_debug_json_path(output_path)).name),
            str(override_path / Path(_default_debug_md_path(output_path)).name),
        )

    suffix = override_path.suffix.lower()
    if suffix == ".json":
        return str(override_path), str(override_path.with_suffix(".md"))
    if suffix == ".md":
        return str(override_path.with_suffix(".json")), str(override_path)
    if suffix:
        return str(override_path), str(override_path.with_name(override_path.name + ".md"))
    return (
        str(override_path / Path(_default_debug_json_path(output_path)).name),
        str(override_path / Path(_default_debug_md_path(output_path)).name),
    )


class DebugRecorder:
    def __init__(self, command: str, debug_level: str) -> None:
        self._debug_level = debug_level
        self._descriptor_candidates: list[dict[str, Any]] = []
        self._log_mark = 0
        self._json_path = ""
        self._md_path = ""
        self.payload: dict[str, Any] = {
            "meta": {
                "version": metalCoord.__version__,
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "command": command,
                "debug_level": debug_level,
                "status": "in_progress",
            },
            "inputs": {},
            "outputs": {},
            "analysis": {},
            "trace": {"structures": []},
            "descriptor_info": [],
            "domain_report": {},
            "logs": [],
            "errors": [],
        }

    def set_paths(self, json_path: str, md_path: str) -> None:
        self._json_path = json_path
        self._md_path = md_path
        self.payload["outputs"]["debug_json"] = json_path
        self.payload["outputs"]["debug_markdown"] = md_path

    @property
    def json_path(self) -> str:
        return self._json_path

    @property
    def md_path(self) -> str:
        return self._md_path

    def set_log_mark(self, mark: int) -> None:
        self._log_mark = mark

    @property
    def log_mark(self) -> int:
        return self._log_mark

    def set_status(self, status: str) -> None:
        self.payload["meta"]["status"] = status

    def set_inputs(self, values: dict[str, Any]) -> None:
        self.payload["inputs"] = _normalize_value(values)

    def set_outputs(self, values: dict[str, Any]) -> None:
        normalized = _normalize_value(values)
        normalized.update(self.payload["outputs"])
        self.payload["outputs"] = normalized

    def set_analysis(self, values: dict[str, Any]) -> None:
        self.payload["analysis"] = _normalize_value(values)

    def add_trace_structure(self, values: dict[str, Any]) -> None:
        self.payload["trace"]["structures"].append(_normalize_value(values))

    def add_descriptor_candidate(self, values: dict[str, Any]) -> None:
        self._descriptor_candidates.append(_normalize_value(values))

    def finalize_descriptor_info(
        self, chosen_by_site: dict[Tuple[Any, ...], Optional[str]]
    ) -> None:
        grouped: dict[Tuple[Any, ...], dict[str, dict[str, Any]]] = {}
        for record in self._descriptor_candidates:
            key = _site_key(record.get("metal_site", {}))
            clazz = record.get("class")
            if not key or not clazz:
                continue
            grouped.setdefault(key, {})
            previous = grouped[key].get(clazz)
            if previous is None:
                grouped[key][clazz] = record
            else:
                prev_p = previous.get("procrustes")
                curr_p = record.get("procrustes")
                if prev_p is None or (curr_p is not None and curr_p < prev_p):
                    grouped[key][clazz] = record

        result: list[dict[str, Any]] = []
        for key, class_map in grouped.items():
            records = list(class_map.values())
            records.sort(key=lambda x: (float("inf") if x.get("procrustes") is None else x["procrustes"], x.get("class", "")))
            chosen_class = chosen_by_site.get(key)

            selected: list[dict[str, Any]]
            if self._debug_level == "max":
                selected = records
            elif self._debug_level == "summary":
                selected = [r for r in records if r.get("class") == chosen_class]
                if not selected and records:
                    selected = [records[0]]
            else:
                top3 = records[:3]
                selected = list(top3)
                if chosen_class:
                    chosen_items = [r for r in records if r.get("class") == chosen_class]
                    if chosen_items and chosen_items[0] not in selected:
                        selected.append(chosen_items[0])
                unique = {}
                for item in selected:
                    unique[item.get("class")] = item
                selected = list(unique.values())
                selected.sort(key=lambda x: (float("inf") if x.get("procrustes") is None else x["procrustes"], x.get("class", "")))

            for rank, record in enumerate(selected, start=1):
                enriched = dict(record)
                enriched["rank"] = rank
                enriched["selected"] = chosen_class is not None and record.get("class") == chosen_class
                result.append(enriched)

        self.payload["descriptor_info"] = result

    def set_domain_report(self, values: dict[str, Any]) -> None:
        self.payload["domain_report"] = _normalize_value(values)

    def set_logs(self, logs: list[dict[str, Any]]) -> None:
        self.payload["logs"] = _normalize_value(logs)

    def add_error(self, reason: str) -> None:
        self.payload["errors"].append(
            {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "reason": reason,
            }
        )

    def write_json(self) -> None:
        if not self._json_path:
            raise ValueError("Debug JSON path is not set.")
        json_path = Path(self._json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(_normalize_value(self.payload), handle, indent=4, separators=(",", ": "))

    def write_markdown(self, markdown: str) -> None:
        if not self._md_path:
            raise ValueError("Debug markdown path is not set.")
        md_path = Path(self._md_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(markdown, encoding="utf-8")
