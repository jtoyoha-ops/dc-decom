"""Run the C-CN scenario only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dc_decom.load_inputs import load_global_parameters, load_regions, load_temperature_96h
from dc_decom.model import build_and_solve
from dc_decom.scenarios import C_CN


if __name__ == "__main__":
    result = build_and_solve(C_CN,
                              load_regions(),
                              load_global_parameters(),
                              load_temperature_96h())
    print(f"C-CN: status={result.status}, "
          f"objective={result.objective_yen/1e12:.2f} trillion JPY")
