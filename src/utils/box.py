"""
Bounding box utility class for coordinate transformations and overlap detection.
"""


class Box:
    """Handles bounding box operations in xywh format."""
    
    def __init__(self, x, y, w, h):
        """
        Initialize bounding box.
        
        Args:
            x: x-coordinate of center
            y: y-coordinate of center
            w: width
            h: height
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def to_xyxy(self):
        """
        Convert from xywh to xyxy format.
        
        Returns:
            tuple: (x1, y1, x2, y2)
        """
        x1 = self.x - self.w / 2
        y1 = self.y - self.h / 2
        x2 = self.x + self.w / 2
        y2 = self.y + self.h / 2
        return (x1, y1, x2, y2)
    
    def from_xyxy(x1, y1, x2, y2):
        """
        Create Box from xyxy format.
        
        Args:
            x1, y1, x2, y2: bounding box coordinates
            
        Returns:
            Box: Box object
        """
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return Box(x, y, w, h)
    
    def area(self):
        """Calculate area of bounding box."""
        return self.w * self.h
    
    def overlap_ratio(self, other):
        """
        Calculate overlap ratio with another box.
        
        Args:
            other: Another Box object
            
        Returns:
            float: Overlap ratio (intersection / union)
        """
        # Convert to xyxy for easier calculation
        x1_1, y1_1, x2_1, y2_1 = self.to_xyxy()
        x1_2, y1_2, x2_2, y2_2 = other.to_xyxy()
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        union = self.area() + other.area() - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def containment_ratio(self, other):
        """
        Calculate the overlap relative to the smaller box.
        Mirrors the CleanCampus logic where a smaller box fully inside
        a larger one yields a ratio of 1.0.
        """
        x1_1, y1_1, x2_1, y2_1 = self.to_xyxy()
        x1_2, y1_2, x2_2, y2_2 = other.to_xyxy()
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        min_area = min(self.area(), other.area())
        
        if min_area == 0:
            return 0.0
        
        return intersection / min_area
    
    def iou(self, other):
        """Alias for overlap_ratio."""
        return self.overlap_ratio(other)
    
    def __repr__(self):
        return f"Box(x={self.x:.2f}, y={self.y:.2f}, w={self.w:.2f}, h={self.h:.2f})"

