"""Main video processing loop for litter detection and tracking."""

import cv2
import time
import requests
from pathlib import Path
import torch
import os
from typing import Optional, Dict, Any

_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    """Patched torch.load to handle ultralytics model loading."""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from ultralytics import YOLO
from src.utils.tracker import Tracker
from src.utils.box import Box

# Try to import Supabase client (optional dependency)
try:
    from src.utils.supabase_client import get_supabase_client, SupabaseClient
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("Supabase client not available. Events will only be stored locally.")


class VideoProcessor:
    """Processes video frames for litter detection and tracking."""
    
    def __init__(
        self,
        model_path,
        flask_server_url="http://localhost:5000",
        skip_frames=0,
        imgsz=1280,
        conf_threshold=0.25,
        iou_threshold=0.45,
        use_supabase=True,
    ):
        """Initialize video processor with litter detection and person/vehicle extraction."""
        self.model = YOLO(model_path)
        self.server_url = flask_server_url.rstrip("/")
        self.skip_frames = max(0, skip_frames)
        self.img_size = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.tracker = Tracker()
        self.frame_count = 0
        self.last_inference_time = 0
        
        # Supabase integration
        self.supabase: Optional[SupabaseClient] = None
        self.video_id: Optional[int] = None
        self.video_metadata: Optional[Dict[str, Any]] = None
        if use_supabase and SUPABASE_AVAILABLE:
            try:
                self.supabase = get_supabase_client()
                print("Supabase client initialized successfully")
            except Exception as e:
                print(f"Failed to initialize Supabase client: {e}")
                self.supabase = None
    
    def process_video(self, video_path, output_path=None, display=False):
        """
        Process video for litter detection.
        
        Args:
            video_path: Path to input video file
            output_path: Optional path to save output video
            display: Whether to display video during processing
            
        Returns:
            dict: Processing results with littering events
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Resolution: {width}x{height}, Total frames: {total_frames}")
        
        # Store video metadata (record will only be created if littering is detected)
        self.video_metadata = {
            "filename": os.path.basename(video_path),
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "duration_seconds": total_frames / fps if fps > 0 else 0,
        }
        
        # Reset video_id - will be set when first littering event is detected
        self.video_id = None
        
        self.tracker.reset()
        self.frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            results = self.model.track(
                frame,
                persist=True,
                verbose=False,
                imgsz=self.img_size,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
            )
            
            detections = self._extract_detections(results[0]) if results else []
            self.tracker.update(detections, self.frame_count)
            
            annotated_frame = self._draw_annotations(frame.copy())
            self._handle_littering_events(annotated_frame)
            
            self.last_inference_time = time.time() - start_time
            
            if writer:
                writer.write(annotated_frame)
            
            if display:
                cv2.imshow('Litter Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            self.frame_count += 1

            # ======================================================== #
            print("debugging frame count:", self.frame_count)
            # ======================================================== #

            if self.frame_count % max(1, fps) == 0:
                print(f"Processed {self.frame_count}/{total_frames} frames")
        
        # ============================================================== #
        print("Finished processing video. While loop ho gya hai exit")
        # ============================================================== #

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        
        # Upload processed video to Supabase and update record
        # NOTE: Video uploads commented out due to large file sizes
        processed_video_url = None
        original_video_url = None
        # if self.supabase and self.video_id and output_path:
        #     try:
        #         # Upload processed video
        #         storage_path = f"videos/{self.video_id}/processed_{os.path.basename(output_path)}"
        #         processed_video_url = self.supabase.upload_video(output_path, storage_path)
        #         
        #         # Upload original video
        #         original_storage_path = f"videos/{self.video_id}/original_{os.path.basename(video_path)}"
        #         original_video_url = self.supabase.upload_video(str(video_path), original_storage_path)
        #         
        #         # Update video record with URLs and final metadata
        #         final_metadata = {
        #             **video_metadata,
        #             "processed_frames": self.frame_count,
        #             "littering_events_count": len(self.tracker.get_littering_events()),
        #             "persons_with_littering": len([p for p in self.tracker.persons.values() if p.has_littered]),
        #             "total_litter_items": len(self.tracker.litter_items),
        #         }
        #         self.supabase.update_video_record(
        #             self.video_id,
        #             original_video_url=original_video_url,
        #             processed_video_url=processed_video_url,
        #             metadata=final_metadata,
        #         )
        #         print(f"Uploaded videos to Supabase storage")
        #     except Exception as e:
        #         print(f"Failed to upload videos to Supabase: {e}")
        
        # Get results summary
        return {
            'total_frames': self.frame_count,
            'littering_events': self.tracker.get_littering_events(),
            'persons_with_littering': len([p for p in self.tracker.persons.values() if p.has_littered]),
            'total_litter_items': len(self.tracker.litter_items),
            'video_id': self.video_id,
            'processed_video_url': processed_video_url,
            'original_video_url': original_video_url,
        }
    
    def _extract_detections(self, result):
        """
        Extract detections from YOLO result.
        
        Args:
            result: YOLO result object
            
        Returns:
            list: List of detections [x, y, w, h, confidence, class_id]
        """
        detections = []
        
        if result.boxes is not None and result.boxes.xyxy is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            track_ids = (
                result.boxes.id.cpu().numpy().astype(int)
                if result.boxes.id is not None
                else [None] * len(boxes)
            )
            
            for i in range(len(boxes)):
                track_id = track_ids[i]
                if track_id is None:
                    continue
                x1, y1, x2, y2 = boxes[i]
                conf = float(confidences[i])
                class_id = int(class_ids[i])
                class_name = Tracker.CLASS_MAPPING.get(class_id, 'litter')
                
                detections.append({
                    'track_id': int(track_id),
                    'box': Box.from_xyxy(x1, y1, x2, y2),
                    'confidence': conf,
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        return detections
    
    def _draw_annotations(self, frame):
        """
        Draw annotations on frame.
        
        Args:
            frame: Input frame
            
        Returns:
            numpy.ndarray: Annotated frame
        """
        h, w = frame.shape[:2]
        
        # Draw persons
        for person_id, person in self.tracker.persons.items():
            x1, y1, x2, y2 = person.box.to_xyxy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            color = (0, 0, 255) if person.has_littered else (0, 100, 255)
            thickness = 5 if person.has_littered else 4
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            label = f"id:{person_id} person"
            if person.has_littered:
                label += " LITTERED"
            
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
            )
            label_y = max(y1 - 15, text_height + 15)
            cv2.rectangle(
                frame,
                (x1, label_y - text_height - 12),
                (x1 + text_width + 20, label_y + 8),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (x1 + 10, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )
        
        # Draw litter
        for litter_id, litter in self.tracker.litter_items.items():
            x1, y1, x2, y2 = litter.box.to_xyxy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            color = (0, 255, 255)
            thickness = 4
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            label = f"id:{litter_id} {litter.class_name}"
            if litter.person_id is not None:
                label += f" -> person {litter.person_id}"
            
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            label_y = max(y1 - 12, text_height + 12)
            cv2.rectangle(
                frame,
                (x1, label_y - text_height - 10),
                (x1 + text_width + 16, label_y + 6),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (x1 + 8, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                lineType=cv2.LINE_AA,
            )
        
        return frame
    
    def _handle_littering_events(self, annotated_frame):
        """Publish annotated frames and extract faces/plates for new littering events."""
        pending_events = self.tracker.get_pending_events()
        
        for event in pending_events:
            snapshot_url = None
            face_url = None
            
            # Create video record on first littering event (lazy creation)
            if self.supabase and self.video_id is None:
                try:
                    video_record = self.supabase.create_video_record(metadata=self.video_metadata)
                    self.video_id = video_record.get("id")
                    print(f"Littering detected! Created video record in Supabase with ID: {self.video_id}")
                except Exception as e:
                    print(f"Failed to create video record in Supabase: {e}")
            
            # Upload snapshot to Supabase
            if self.supabase and self.video_id:
                try:
                    storage_path = f"events/{self.video_id}/event_p{event['person_id']}_l{event['litter_id']}_f{event['frame']}.jpg"
                    snapshot_url = self.supabase.upload_frame(annotated_frame, storage_path)
                except Exception as e:
                    print(f"Failed to upload snapshot to Supabase: {e}")
            
            # Extract face/person crop and upload
            face_url = self._extract_face_plate_from_event(event, annotated_frame)
            
            # Save event to Supabase database
            if self.supabase and self.video_id:
                try:
                    self.supabase.create_event(
                        video_id=self.video_id,
                        snapshot_url=snapshot_url,
                        face_url=face_url,
                        label=event.get("litter_type", "unknown"),
                        confidence=None,  # Could be added from detection
                        extra={
                            "person_id": event["person_id"],
                            "litter_id": event["litter_id"],
                            "frame_number": event["frame"],
                            "license_plate": event.get("license_plate"),
                        },
                    )
                    print(f"Saved littering event to Supabase: person {event['person_id']}, litter {event['litter_id']}")
                except Exception as e:
                    print(f"Failed to save event to Supabase: {e}")
            
            # Also send to Flask server for backward compatibility
            self._send_litter_event_image(event, annotated_frame)
            self.tracker.mark_event_sent(event)
    
    def _extract_face_plate_from_event(self, event, frame):
        """Extract person crops from littering event frame using general object detection.
        
        Returns:
            str: URL of the first person crop uploaded to Supabase, or None
        """
        face_url = None
        try:
            event_dir = f"static/litter_events/event_p{event['person_id']}_l{event['litter_id']}"
            os.makedirs(event_dir, exist_ok=True)
            os.makedirs("static/faces", exist_ok=True)
            os.makedirs("static/number_plates", exist_ok=True)
            
            frame_path = os.path.join(event_dir, f"frame_{event['frame']}.jpg")
            cv2.imwrite(frame_path, frame)
            
            from ultralytics import YOLO
            yolo_model = YOLO("yolov8m.pt")
            results = yolo_model(frame_path, conf=0.3, device=self.device, verbose=False)
            result = results[0]
            
            person_crops = []
            vehicle_crops = []
            vehicle_details = []
            
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    cls = int(box.cls.item())
                    conf = box.conf.item()
                    
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), x2, y2
                    
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        
                        if cls == 0:
                            filename = f"person_{event['person_id']}_{len(person_crops)}.jpg"
                            filepath = os.path.join("static/faces", filename)
                            cv2.imwrite(filepath, crop)
                            person_crops.append(filepath)
                            
                            # Upload first person crop to Supabase
                            if self.supabase and self.video_id and face_url is None:
                                try:
                                    storage_path = f"faces/{self.video_id}/{filename}"
                                    face_url = self.supabase.upload_file(filepath, storage_path)
                                except Exception as e:
                                    print(f"Failed to upload face to Supabase: {e}")
                                    
                        elif cls in [2, 5, 7]:
                            filename = f"vehicle_{event['person_id']}_{len(vehicle_crops)}.jpg"
                            filepath = os.path.join("static/number_plates", filename)
                            cv2.imwrite(filepath, crop)
                            vehicle_crops.append(filepath)
                            
                            plate_text = self._extract_plate_text(crop)
                            vehicle_details.append({
                                'crop': filepath,
                                'plate_number': plate_text,
                                'confidence': conf
                            })
                            
                            if not event.get('license_plate'):
                                event['license_plate'] = plate_text
            
            print(f"Extracted {len(person_crops)} person(s) and {len(vehicle_crops)} vehicle(s)")
            if person_crops:
                print(f"  Persons: {person_crops}")
            if vehicle_details:
                print(f"  Vehicles:")
                for detail in vehicle_details:
                    print(f"    - Plate: {detail['plate_number']} (conf: {detail['confidence']:.2f})")
                    
            return face_url
        except Exception as e:
            print(f"Error extracting person/vehicle from event: {e}")
            return None
    
    def _extract_plate_text(self, vehicle_crop):
        """Extract license plate number from vehicle crop using OCR."""
        try:
            import easyocr
            
            reader = easyocr.Reader(['en'], gpu=True if self.device == 0 else False)
            results = reader.readtext(vehicle_crop)
            
            if results:
                text = ' '.join([detection[1] for detection in results]).upper()
                return text if len(text) > 2 else "NO-TEXT"
            else:
                return "NO-TEXT"
        except Exception as e:
            print(f"OCR Error: {e}")
            return "ERROR"
    
    def _send_litter_event_image(self, event, frame):
        """
        Send the annotated frame for a littering event to the Flask server.
        """
        try:
            filename = f"event_p{event['person_id']}_l{event['litter_id']}_f{event['frame']}.jpg"
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                print("Failed to encode event frame for upload.")
                return False
            
            files = {
                'image': (filename, buffer.tobytes(), 'image/jpeg')
            }
            data = {
                'person_id': str(event['person_id']),
                'litter_id': str(event['litter_id']),
                'frame_number': str(event['frame']),
                'litter_type': event.get('litter_type', 'unknown'),
                'filename': filename
            }
            
            response = requests.post(
                f"{self.server_url}/api/addImage",
                files=files,
                data=data,
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"Uploaded annotated littering frame for person {event['person_id']}")
                return True
            else:
                print(f"Failed to upload littering frame: {response.status_code} - {response.text}")
                return False
        except Exception as exc:
            print(f"Error sending littering event image: {exc}")
            return False

