"""
Litter object class for tracking litter items.
"""

from .box import Box

class Litter:
    """Tracks litter items with bounding boxes and associates with persons."""
    
    def __init__(self, track_id, box, class_id, class_name, frame_number):
        """
        Initialize litter object.
        
        Args:
            track_id: Tracking ID
            box: Box object
            class_id: Class ID (1=paper, 2=cardboard, 3=can, 4=plastic)
            class_name: Class name
            frame_number: Frame number when detected
        """
        self.track_id = track_id
        self.box = box
        self.class_id = class_id
        self.class_name = class_name
        self.frame_number = frame_number
        self.person_id = None  # Associated person ID
        self.occluded = False  # Occlusion state
        self.last_seen_frame = frame_number
        self.frames_since_seen = 0
    
    def update(self, box, frame_number):
        """
        Update litter position and frame.
        
        Args:
            box: New Box object
            frame_number: Current frame number
        """
        self.box = box
        self.last_seen_frame = frame_number
        self.frames_since_seen = 0
        self.occluded = False
    
    def set_occluded(self, occluded=True):
        """
        Set occlusion state.
        
        Args:
            occluded: Boolean indicating if litter is occluded
        """
        self.occluded = occluded
    
    def mark_missing(self):
        """Increment frames since seen and set occluded."""
        self.frames_since_seen += 1
        self.occluded = True
    
    def associate_person(self, person_id):
        """
        Associate litter with a person.
        
        Args:
            person_id: Person tracking ID
        """
        self.person_id = person_id
    
    def __repr__(self):
        return f"Litter(id={self.track_id}, class={self.class_name}, person_id={self.person_id}, occluded={self.occluded})"

