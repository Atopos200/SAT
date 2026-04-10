"""Bridge module for CoT data building.

Current CLI implementation lives in `innovation.build_cot_data`.
"""

from innovation import build_cot_data as _legacy

process_split = _legacy.process_split
main = _legacy.main

