"""Make all figures:  python -m analysis   (from the repository root; writes analysis/figures/*.pdf)."""
import runpy
import sys

for name in ("pole_fibres", "hopf_fibration", "energy_orbit_sphere", "energy_target_sphere"):
    print(f"--- analysis.{name}")
    sys.argv = [f"analysis.{name}"]
    runpy.run_module(f"analysis.{name}", run_name="__main__")
