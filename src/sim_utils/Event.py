from dataclasses import dataclass, field
from typing import Dict, Any, List
from enum import Enum

class EventType(Enum):
    PLAY = "play"
    SCORE = "score"
    TURNOVER = "turnover"
    PENALTY = "penalty"
    TIMEOUT = "timeout"
    INJURY = "injury"
    QUARTER_END = "quarter_end"
    DRIVE_START = "drive_start"
    DRIVE_END = "drive_end"
    GAME_START = "game_start"
    GAME_END = "game_end"

class PlayType(Enum):
    RUN = "run"
    PASS = "pass"
    FIELD_GOAL = "field_goal"
    PUNT = "punt"
    KICKOFF = "kickoff"
    QB_KNEEL = "qb_kneel"
    QB_SPIKE = "qb_spike"

@dataclass
class GameEvent:
    """Base event - every action becomes one of these"""
    event_id: int
    game_id: int
    event_type: EventType
    timestamp: float  # Game clock in seconds
    quarter: int
    drive_id: int
    
    # Game state snapshot
    possession_team: str
    defense_team: str
    yardline: int
    down: int
    yards_to_go: int
    score_home: int
    score_away: int
    
    # Event-specific data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        return {
            'event_id': self.event_id,
            'game_id': self.game_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'quarter': self.quarter,
            'drive_id': self.drive_id,
            'possession_team': self.possession_team,
            'defense_team': self.defense_team,
            'yardline': self.yardline,
            'down': self.down,
            'yards_to_go': self.yards_to_go,
            'score_home': self.score_home,
            'score_away': self.score_away,
            **self.metadata
        }

@dataclass
class PlayEvent(GameEvent):
    """Specific event for plays"""
    play_type: PlayType = None
    yards_gained: int = 0
    player_id: str = None
    player_name: str = None
    is_complete: bool = True
    is_touchdown: bool = False
    is_first_down: bool = False
    time_elapsed: float = 0
    
    def __post_init__(self):
        self.event_type = EventType.PLAY
        self.metadata.update({
            'play_type': self.play_type.value if self.play_type else None,
            'yards_gained': self.yards_gained,
            'player_id': self.player_id,
            'player_name': self.player_name,
            'is_complete': self.is_complete,
            'is_touchdown': self.is_touchdown,
            'is_first_down': self.is_first_down,
            'time_elapsed': self.time_elapsed,
        })

class EventLog:
    """Stores and queries game events"""
    
    def __init__(self):
        self.events: List[GameEvent] = []
        self._event_counter = 0
    
    def add_event(self, event: GameEvent) -> int:
        """Add event and return its ID"""
        event.event_id = self._event_counter
        self.events.append(event)
        self._event_counter += 1
        return event.event_id