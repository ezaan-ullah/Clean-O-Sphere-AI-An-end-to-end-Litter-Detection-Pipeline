"""
Main tracking system for managing persons and litter objects.
"""

from typing import Dict, List, Set

from .person import Person
from .litter import Litter
from .vehicle import Vehicle

class Tracker:
    """Manages persons and litter objects, updates tracking IDs, detects littering events."""

    CLASS_MAPPING = {
        0: "person",
        1: "paper",
        2: "cardboard",
        3: "can",
        4: "plastic",
    }
    
    # COCO class IDs for vehicles (used by yolov8m.pt)
    VEHICLE_CLASS_MAPPING = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        association_threshold: float = 0.6,
        hold_threshold: float = 0.85,
        drop_threshold: float = 0.4,
        min_hold_frames: int = 2,
        min_drop_frames: int = 2,
        disassociate_threshold: float = 0.15,
        trace_threshold: float = 0.05,
        max_frames_missing: int = 60,
    ) -> None:
        self.association_threshold = association_threshold
        self.hold_threshold = hold_threshold
        self.drop_threshold = drop_threshold
        self.min_hold_frames = min_hold_frames
        self.min_drop_frames = min_drop_frames
        self.disassociate_threshold = disassociate_threshold
        self.trace_threshold = trace_threshold
        self.max_frames_missing = max_frames_missing

        self.persons: Dict[int, Person] = {}
        self.litter_items: Dict[int, Litter] = {}
        self.vehicles: Dict[int, Vehicle] = {}
        self.frame_number = 0
        self.littering_events: List[Dict] = []

    def update(self, detections: List[Dict], frame_number: int) -> None:
        """
        Update tracker with new detections.

        Args:
            detections: List of detections with track_id, class_id, class_name, box, confidence
            frame_number: Current frame number
        """
        self.frame_number = frame_number

        person_detections = [det for det in detections if det["class_id"] == 0]
        litter_detections = [det for det in detections if det["class_id"] != 0]

        seen_persons = self._update_persons(person_detections)
        seen_litter = self._update_litter(litter_detections)

        self._associate_and_detect_littering()
        self._mark_missing_tracks(seen_persons, seen_litter)
        self._cleanup_old_tracks()

    def _update_persons(self, detections: List[Dict]) -> Set[int]:
        seen: Set[int] = set()
        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue
            seen.add(track_id)
            if track_id in self.persons:
                self.persons[track_id].update(det["box"], self.frame_number)
            else:
                self.persons[track_id] = Person(track_id, det["box"], self.frame_number)
        return seen

    def _update_litter(self, detections: List[Dict]) -> Set[int]:
        seen: Set[int] = set()
        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue
            seen.add(track_id)
            class_id = det["class_id"]
            class_name = det.get("class_name") or self.CLASS_MAPPING.get(class_id, "litter")
            if track_id in self.litter_items:
                self.litter_items[track_id].update(det["box"], self.frame_number)
            else:
                self.litter_items[track_id] = Litter(
                    track_id, det["box"], class_id, class_name, self.frame_number
                )
        return seen

    def _associate_and_detect_littering(self) -> None:
        for litter_id, litter in self.litter_items.items():
            for person_id, person in self.persons.items():
                overlap = person.box.containment_ratio(litter.box)
                should_track = overlap >= self.trace_threshold or litter_id in person.litter_links
                if not should_track:
                    continue

                link = person.record_overlap(litter_id, overlap, self.frame_number)
                if overlap >= self.hold_threshold:
                    link.high_overlap_frames += 1
                else:
                    link.high_overlap_frames = max(0, link.high_overlap_frames - 1)

                if overlap >= self.association_threshold:
                    if litter.person_id != person_id:
                        litter.associate_person(person_id)
                elif litter.person_id == person_id and overlap < self.disassociate_threshold:
                    litter.associate_person(None)

                if person.evaluate_litter_drop(
                    litter_id,
                    hold_threshold=self.hold_threshold,
                    drop_threshold=self.drop_threshold,
                    min_hold_frames=self.min_hold_frames,
                    min_drop_frames=self.min_drop_frames,
                    frame_number=self.frame_number,
                    visible=not litter.occluded,
                ):
                    self._register_event(person_id, litter_id, litter.class_name)

    def _register_event(self, person_id: int, litter_id: int, litter_type: str) -> None:
        exists = any(
            event["person_id"] == person_id and event["litter_id"] == litter_id
            for event in self.littering_events
        )
        if exists:
            return

        self.littering_events.append(
            {
                "person_id": person_id,
                "litter_id": litter_id,
                "frame": self.frame_number,
                "litter_type": litter_type,
                "sent": False,
            }
        )

    def _mark_missing_tracks(self, seen_persons: Set[int], seen_litter: Set[int]) -> None:
        _ = seen_persons

        for litter_id, litter in self.litter_items.items():
            if litter_id not in seen_litter:
                litter.mark_missing()

    def _cleanup_old_tracks(self) -> None:
        current_frame = self.frame_number

        persons_to_remove = [
            pid for pid, person in self.persons.items()
            if current_frame - person.last_seen_frame > self.max_frames_missing
        ]
        for pid in persons_to_remove:
            del self.persons[pid]

        litter_to_remove = [
            lid for lid, litter in self.litter_items.items()
            if current_frame - litter.last_seen_frame > self.max_frames_missing
        ]
        for lid in litter_to_remove:
            del self.litter_items[lid]

    def get_littering_events(self) -> List[Dict]:
        return self.littering_events

    def get_pending_events(self) -> List[Dict]:
        return [event for event in self.littering_events if not event.get("sent")]

    def mark_event_sent(self, event: Dict) -> None:
        event["sent"] = True

    def reset(self) -> None:
        self.persons.clear()
        self.litter_items.clear()
        self.vehicles.clear()
        self.frame_number = 0
        self.littering_events.clear()
    
    def update_vehicles(self, detections: List[Dict], frame_number: int) -> None:
        """
        Update tracker with vehicle detections from the secondary model.
        
        Args:
            detections: List of vehicle detections with track_id, class_id, class_name, box
            frame_number: Current frame number
        """
        seen_vehicles: Set[int] = set()
        
        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue
            
            class_id = det["class_id"]
            # Only process vehicle classes
            if class_id not in self.VEHICLE_CLASS_MAPPING:
                continue
                
            seen_vehicles.add(track_id)
            class_name = self.VEHICLE_CLASS_MAPPING.get(class_id, "vehicle")
            
            if track_id in self.vehicles:
                self.vehicles[track_id].update(det["box"], frame_number)
            else:
                self.vehicles[track_id] = Vehicle(
                    track_id, det["box"], frame_number, class_name
                )
        
        # Cleanup old vehicle tracks
        vehicles_to_remove = [
            vid for vid, vehicle in self.vehicles.items()
            if frame_number - vehicle.last_seen_frame > self.max_frames_missing
        ]
        for vid in vehicles_to_remove:
            del self.vehicles[vid]

