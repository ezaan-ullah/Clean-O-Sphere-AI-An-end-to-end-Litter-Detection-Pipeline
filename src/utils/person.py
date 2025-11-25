"""
Person object class for tracking persons and detecting littering events.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict

from .box import Box


@dataclass
class LitterAssociation:
    """Stores overlap history and state for a person–litter pair."""

    history: deque = field(default_factory=lambda: deque(maxlen=40))
    last_frame: int = -1
    holding: bool = False
    event_reported: bool = False
    max_overlap: float = 0.0
    low_overlap_streak: int = 0
    high_overlap_frames: int = 0

    def record(self, overlap: float, frame_number: int) -> None:
        self.history.append(overlap)
        self.last_frame = frame_number
        self.max_overlap = max(self.max_overlap, overlap)

    @property
    def current_overlap(self) -> float:
        return self.history[-1] if self.history else 0.0


class Person:
    """Tracks persons and associated litter, detects when person has littered."""

    def __init__(self, track_id: int, box: Box, frame_number: int):
        self.track_id = track_id
        self.box = box
        self.frame_number = frame_number
        self.litter_links: Dict[int, LitterAssociation] = {}
        self.has_littered = False
        self.littering_frame = None
        self.last_seen_frame = frame_number
        self.image_captured = False  # Placeholder for future database/face-recognition hooks

    def update(self, box: Box, frame_number: int) -> None:
        self.box = box
        self.last_seen_frame = frame_number

    def record_overlap(self, litter_id: int, overlap: float, frame_number: int) -> LitterAssociation:
        if litter_id not in self.litter_links:
            self.litter_links[litter_id] = LitterAssociation()
        link = self.litter_links[litter_id]
        link.record(overlap, frame_number)
        return link

    def evaluate_litter_drop(
        self,
        litter_id: int,
        hold_threshold: float,
        drop_threshold: float,
        min_hold_frames: int,
        min_drop_frames: int,
        frame_number: int,
        visible: bool = True,
    ) -> bool:
        if self.has_littered:
            return False

        link = self.litter_links.get(litter_id)
        if not link or link.event_reported:
            return False

        if not link.holding and (
            link.max_overlap >= hold_threshold or link.high_overlap_frames >= min_hold_frames
        ):
            link.holding = True

        # Track consecutive low-overlap frames
        if link.current_overlap < drop_threshold:
            link.low_overlap_streak += 1
        else:
            link.low_overlap_streak = 0

        drop_detected = (
            link.holding
            and link.low_overlap_streak >= min_drop_frames
            and visible
            and link.max_overlap >= hold_threshold
        )

        if drop_detected:
            self.has_littered = True
            self.littering_frame = frame_number
            link.event_reported = True
            return True

        return False

    def __repr__(self):
        return f"Person(id={self.track_id}, has_littered={self.has_littered}, litter_links={len(self.litter_links)})"

