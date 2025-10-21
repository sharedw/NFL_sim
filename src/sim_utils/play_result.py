# plays.py or results.py
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

@dataclass
class PlayResult:
    """Result of a play execution with sensible defaults"""
    
    # Required fields (must be provided)
    yards: int = None
    #time_elapsed: float
    player: str = None
    
    # Optional fields with defaults
    rusher: Optional[object] = None
    rusher_id: Optional[str] = None
    rusher_name: Optional[str] = None
    
    receiver: Optional[object] = None
    receiver_id: Optional[str] = None
    receiver_name: Optional[str] = None

    passer: Optional[object] = None
    passer_id: Optional[str] = None
    passer_name: Optional[str] = None

    kicker: Optional[object] = None
    kicker_id: Optional[str] = None
    kicker_name: Optional[str] = None
    # Play outcome booleans
    incomplete_pass: bool = 0
    completion: bool = 0
    touchdown: bool = False
    first_down: bool = False
    turnover: bool = False
    fumble: bool = False
    interception: bool = False
    sack: bool = False
    penalty: bool = False
    safety: bool = False
    out_of_bounds: bool = False
    
    # Pass-specific
    air_yards: int = 0
    yac: int = 0
    
    # Additional context
    play_description: str = ""
    play_type: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging"""
        result = asdict(self)

        return result
