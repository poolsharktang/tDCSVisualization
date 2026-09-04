"""Run a tDCS FEM simulation from a JSON config using SimNIBS.

This script is intended to be executed by the SimNIBS Python environment:

    "<SimNIBS python.exe>" simnibs_run_tdcs.py config.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _split_currents(total_ma: float, n_sources: int, n_sinks: int) -> list[float]:
    if n_sources < 1 or n_sinks < 1:
        raise ValueError("At least one source and one sink electrode are required.")
    if total_ma <= 0:
        raise ValueError("Total current must be greater than 0 mA.")

    total_a = total_ma * 1.0e-3
    source_a = total_a / n_sources
    sink_a = -total_a / n_sinks
    return [source_a] * n_sources + [sink_a] * n_sinks


def _add_electrode(tdcs, position: str, channel: int, config: dict):
    electrode = tdcs.add_electrode()
    electrode.channelnr = channel
    electrode.centre = position
    electrode.thickness = float(config.get("electrode_thickness_mm", 4))

    shape = str(config.get("electrode_shape", "ellipse")).lower()
    if shape in ("rect", "rectangle"):
        electrode.shape = "rect"
        width = float(config.get("electrode_width_mm", 50))
        height = float(config.get("electrode_height_mm", 50))
        electrode.dimensions = [width, height]
    else:
        electrode.shape = "ellipse"
        diameter = float(config.get("electrode_diameter_mm", 10))
        electrode.dimensions = [diameter, diameter]
    return electrode


def run_from_config(config: dict) -> None:
    from simnibs import sim_struct, run_simnibs

    sources = [str(name) for name in config.get("sources", [])]
    sinks = [str(name) for name in config.get("sinks", [])]
    total_ma = float(config.get("total_current_ma", 2.0))
    currents = _split_currents(total_ma, len(sources), len(sinks))

    session = sim_struct.SESSION()
    session.subpath = config.get("subpath", "m2m_ernie")
    session.pathfem = config.get("pathfem", "tdcs_gui_results")
    session.fields = config.get("fields", "veEjJ")
    session.open_in_gmsh = bool(config.get("open_in_gmsh", True))
    session.map_to_vol = bool(config.get("map_to_vol", False))

    tdcs = session.add_tdcslist()
    tdcs.currents = currents

    channel = 1
    for name in sources:
        _add_electrode(tdcs, name, channel, config)
        channel += 1
    for name in sinks:
        _add_electrode(tdcs, name, channel, config)
        channel += 1

    print("Starting SimNIBS tDCS simulation")
    print("=" * 48)
    print(f"Head model : {session.subpath}")
    print(f"Output     : {session.pathfem}")
    print(f"Fields     : {session.fields}")
    print(f"Open Gmsh  : {session.open_in_gmsh}")
    print(f"Map to vol : {session.map_to_vol}")
    print()
    print("刺激电极 (sink)")
    for name, amp in zip(sources, currents[: len(sources)]):
        print(f"  {name:<6} {amp * 1e3:+8.4f} mA")
    print("回流电极 (source)")
    for name, amp in zip(sinks, currents[len(sources) :]):
        print(f"  {name:<6} {amp * 1e3:+8.4f} mA")
    print()
    print(f"Total current : {total_ma:.4f} mA")
    print(f"Current sum   : {sum(currents) * 1e3:+.6f} mA")
    print("=" * 48)
    print()

    run_simnibs(session)

    print()
    print("Simulation finished.")
    print(f"Results saved to: {session.pathfem}")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("Usage: simnibs_run_tdcs.py <config.json>", file=sys.stderr)
        return 2

    config_path = Path(args[0]).expanduser().resolve()
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 2

    try:
        run_from_config(_load_config(config_path))
    except Exception as exc:
        print(f"Simulation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
