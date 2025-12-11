"""
Vehicle object class for tracking vehicles in video frames.
"""

from dataclasses import dataclass
from .box import Box


@dataclass
class Vehicle:
    """Tracks vehicles detected in video frames."""

    track_id: int
    box: Box
    frame_number: int
    class_name: str  # car, bus, truck, motorcycle
    last_seen_frame: int = 0

    def __init__(self, track_id: int, box: Box, frame_number: int, class_name: str = "vehicle"):
        self.track_id = track_id
        self.box = box
        self.frame_number = frame_number
        self.class_name = class_name
        self.last_seen_frame = frame_number

    def update(self, box: Box, frame_number: int) -> None:
        """Update vehicle position."""
        self.box = box
        self.last_seen_frame = frame_number
