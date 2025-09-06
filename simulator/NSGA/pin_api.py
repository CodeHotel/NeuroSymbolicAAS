# simulator/pin_api.py
from dataclasses import dataclass
from typing import Optional, Iterable, Dict, Tuple

@dataclass
class PinSpec:
    job_id: str
    op_id: str
    machine: Optional[str] = None
    agv: Optional[str] = None

def decode_pins(pin_list: Iterable[PinSpec]) -> Dict[Tuple[str,str], PinSpec]:
    return {(p.job_id, p.op_id): p for p in pin_list}
