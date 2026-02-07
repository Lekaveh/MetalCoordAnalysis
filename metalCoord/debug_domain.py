from typing import Any, Dict, List, Optional, Tuple

import gemmi

from metalCoord.analysis.models import MetalPairStats, MetalStats, PdbStats


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


def _site_from_metal(metal: MetalStats) -> Dict[str, Any]:
    return {
        "metal": metal.metal,
        "metalElement": metal.metal_element,
        "chain": metal.chain,
        "residue": metal.residue,
        "sequence": metal.sequence,
        "icode": metal.insertion_code,
        "altloc": metal.altloc,
    }


class DomainReportBuilder:
    def __init__(
        self,
        pdb_stats: PdbStats,
        metal_pair_stats: List[MetalPairStats],
        config: Any,
        inputs: Dict[str, Any],
        trace: Dict[str, Any],
        descriptor_info: List[Dict[str, Any]],
        debug_level: str,
    ) -> None:
        self._pdb_stats = pdb_stats
        self._metal_pair_stats = metal_pair_stats
        self._config = config
        self._inputs = inputs
        self._trace = trace or {"structures": []}
        self._descriptor_info = descriptor_info
        self._debug_level = debug_level

    def _flags(self, metals: List[MetalStats]) -> List[str]:
        flags = []
        if not metals:
            flags.append("no metal found")
            return flags

        for metal in metals:
            best = metal.get_best_class()
            if best and best.count not in (-1, None) and best.count < self._config.min_sample_size:
                flags.append("sample size below threshold")

        for trace_item in self._trace.get("structures", []):
            chosen = trace_item.get("chosen_strategy")
            if chosen and "Covalent" in chosen:
                flags.append("fallback to covalent distances")

        return sorted(set(flags))

    def build(self) -> Dict[str, Any]:
        metals = list(self._pdb_stats.metals)
        trace_by_site = {
            _site_key(item.get("metal_site", {})): item
            for item in self._trace.get("structures", [])
        }

        steps = []
        metal_sites = [_site_from_metal(metal) for metal in metals]
        steps.append(
            {
                "step_id": 1,
                "name": "Metal Site Detection",
                "narrative": f"Detected {len(metal_sites)} metal coordination site(s) in the model.",
                "key_data": {"metal_sites": metal_sites, "count": len(metal_sites)},
            }
        )

        ligand_environment = []
        coordination_data = []
        geometry_candidates = []
        strategy_resolution = []
        distance_data = []
        angle_data = []
        final_data = []

        for metal in metals:
            site = _site_from_metal(metal)
            trace_item = trace_by_site.get(_site_key(site), {})
            ligand_environment.append(
                {
                    "metal_site": site,
                    "ligands": trace_item.get("ligand_environment", {}).get("ligands", []),
                }
            )
            coordination_data.append(
                {
                    "metal_site": site,
                    "coordination": trace_item.get("coordination", {}),
                }
            )
            geometry_candidates.append(
                {
                    "metal_site": site,
                    "candidates": trace_item.get("candidates", []),
                }
            )
            strategy_resolution.append(
                {
                    "metal_site": site,
                    "chosen_strategy": trace_item.get("chosen_strategy"),
                    "chosen_class": trace_item.get("chosen_class"),
                }
            )

            best = metal.get_best_class()
            distances = []
            angles = []
            if best:
                for bond in best.bonds:
                    cov = (
                        gemmi.Element(metal.metal_element).covalent_r
                        + gemmi.Element(bond.ligand.element).covalent_r
                    )
                    distances.append(
                        {
                            "ligand": bond.ligand.to_dict(),
                            "distance": bond.distance,
                            "std": bond.std,
                            "covalent_sum": round(cov, 3),
                        }
                    )
                for angle in best.angles:
                    angles.append(
                        {
                            "ligand1": angle.ligand1.to_dict(),
                            "ligand2": angle.ligand2.to_dict(),
                            "angle": angle.angle,
                            "std": angle.std,
                        }
                    )
            distance_data.append({"metal_site": site, "distances": distances})
            angle_data.append({"metal_site": site, "angles": angles})
            final_data.append(
                {
                    "metal_site": site,
                    "coordination_number": best.coordination if best else None,
                    "chosen_geometry": best.clazz if best else None,
                    "procrustes": float(best.procrustes) if best else None,
                    "sample_size": best.count if best else None,
                }
            )

        steps.append(
            {
                "step_id": 2,
                "name": "Ligand Environment",
                "narrative": "Characterized coordinating ligands and compared observed distances with covalent expectations.",
                "key_data": {"sites": ligand_environment},
            }
        )
        steps.append(
            {
                "step_id": 3,
                "name": "Coordination Number Determination",
                "narrative": "Determined ligand counts per metal site and tracked base vs extra ligands.",
                "key_data": {"sites": coordination_data},
            }
        )
        steps.append(
            {
                "step_id": 4,
                "name": "Geometry Candidates",
                "narrative": "Evaluated candidate coordination geometries using Procrustes fit.",
                "key_data": {"sites": geometry_candidates},
            }
        )
        steps.append(
            {
                "step_id": 5,
                "name": "Linear Descriptor Generation",
                "narrative": "Generated linear descriptors from class code and lexicographic atom ordering.",
                "key_data": {"descriptors": self._descriptor_info},
            }
        )
        steps.append(
            {
                "step_id": 6,
                "name": "Strategy Resolution",
                "narrative": "Resolved the final statistics path by selecting the first successful strategy per chosen class.",
                "key_data": {"sites": strategy_resolution},
            }
        )
        steps.append(
            {
                "step_id": 7,
                "name": "Distance Statistics",
                "narrative": "Computed metal-ligand distance means and uncertainties for the selected geometry.",
                "key_data": {"sites": distance_data},
            }
        )
        steps.append(
            {
                "step_id": 8,
                "name": "Angle Statistics",
                "narrative": "Computed ligand-ligand angle distributions to evaluate geometric consistency.",
                "key_data": {"sites": angle_data},
            }
        )
        steps.append(
            {
                "step_id": 9,
                "name": "Final Assessment",
                "narrative": "Summarized selected geometry quality and available metal-metal distance references.",
                "key_data": {
                    "sites": final_data,
                    "metal_metal": [item.to_dict() for item in self._metal_pair_stats],
                },
            }
        )

        quality_notes = []
        for item in final_data:
            proc = item.get("procrustes")
            if proc is None:
                continue
            if proc <= self._config.procrustes_threshold:
                quality_notes.append("Geometry fit within Procrustes threshold.")
            else:
                quality_notes.append("Geometry fit exceeds Procrustes threshold.")

        summary = {
            "coordination_number": final_data[0]["coordination_number"] if len(final_data) == 1 else None,
            "chosen_geometry": final_data[0]["chosen_geometry"] if len(final_data) == 1 else None,
            "procrustes": final_data[0]["procrustes"] if len(final_data) == 1 else None,
            "sample_size": final_data[0]["sample_size"] if len(final_data) == 1 else None,
            "quality_notes": quality_notes[0] if len(quality_notes) == 1 else quality_notes,
            "sites": final_data,
        }

        return {
            "title": f"Metal Coordination Analysis Report: {self._inputs.get('source', '')}",
            "inputs": {
                "source": self._inputs.get("source"),
                "ligand": self._inputs.get("ligand"),
                "pdb": self._inputs.get("pdb"),
                "class": self._inputs.get("class"),
                "thresholds": {
                    "distance": self._config.distance_threshold,
                    "procrustes": self._config.procrustes_threshold,
                    "min_sample_size": self._config.min_sample_size,
                    "metal_distance": self._config.metal_distance_threshold,
                },
            },
            "steps": steps,
            "summary": summary,
            "flags": self._flags(metals),
        }


def render_domain_markdown(domain_report: Dict[str, Any], logs: List[Dict[str, Any]]) -> str:
    lines = []
    title = domain_report.get("title", "Metal Coordination Analysis Report")
    lines.append(f"# {title}")
    lines.append("")

    inputs = domain_report.get("inputs", {})
    thresholds = inputs.get("thresholds", {})
    lines.append("## Inputs")
    lines.append(f"- Source: `{inputs.get('source')}`")
    lines.append(f"- Ligand: `{inputs.get('ligand')}`")
    lines.append(f"- PDB: `{inputs.get('pdb')}`")
    lines.append(f"- Class: `{inputs.get('class')}`")
    lines.append(
        f"- Thresholds: distance={thresholds.get('distance')}, procrustes={thresholds.get('procrustes')}, "
        f"min_sample_size={thresholds.get('min_sample_size')}, metal_distance={thresholds.get('metal_distance')}"
    )
    lines.append("")

    lines.append("## Analysis Steps")
    for step in domain_report.get("steps", []):
        lines.append(f"### {step.get('step_id')}. {step.get('name')}")
        lines.append(step.get("narrative", ""))
        lines.append("")
        lines.append("```json")
        lines.append(str(step.get("key_data", {})))
        lines.append("```")
        lines.append("")

    lines.append("## Summary")
    lines.append("```json")
    lines.append(str(domain_report.get("summary", {})))
    lines.append("```")
    lines.append("")

    flags = domain_report.get("flags", [])
    lines.append("## Flags")
    if flags:
        for flag in flags:
            lines.append(f"- {flag}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Logs")
    if logs:
        for item in logs:
            lines.append(
                f"- `{item.get('timestamp')}` `{item.get('level')}` {item.get('message')}"
            )
    else:
        lines.append("- no log records captured")
    lines.append("")
    return "\n".join(lines)
