"""
Supabase client for database and storage operations.
Handles video metadata, littering events, and media uploads.
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from supabase import create_client, Client
import cv2
import numpy as np

load_dotenv()


class SupabaseClient:
    """Client for interacting with Supabase database and storage."""

    def __init__(self):
        """Initialize Supabase client with environment variables."""
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        self.bucket = os.getenv("SUPABASE_BUCKET", "events-media")

        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")

        # Handle both PostgreSQL connection string and Supabase API URL
        if self.url.startswith("postgresql://"):
            # Extract project ref from connection string for API URL
            # Format: postgresql://postgres:password@db.{project_ref}.supabase.co:5432/postgres
            try:
                project_ref = self.url.split("@db.")[1].split(".supabase.co")[0]
                self.api_url = f"https://{project_ref}.supabase.co"
            except IndexError:
                raise ValueError("Invalid Supabase PostgreSQL connection string format")
        else:
            self.api_url = self.url

        self.client: Client = create_client(self.api_url, self.key)

    # ==================== VIDEO OPERATIONS ====================

    def create_video_record(
        self,
        original_video_url: Optional[str] = None,
        processed_video_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new video record in the database.

        Args:
            original_video_url: URL to the original uploaded video
            processed_video_url: URL to the processed video with annotations
            metadata: Additional metadata (fps, resolution, total_frames, etc.)

        Returns:
            dict: Created video record with id
        """
        data = {
            "original_video_url": original_video_url,
            "processed_video_url": processed_video_url,
            "metadata": metadata or {},
        }

        response = self.client.table("videos").insert(data).execute()
        return response.data[0] if response.data else {}

    def update_video_record(
        self,
        video_id: int,
        original_video_url: Optional[str] = None,
        processed_video_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Update an existing video record.

        Args:
            video_id: ID of the video to update
            original_video_url: URL to the original video
            processed_video_url: URL to the processed video
            metadata: Additional metadata to merge

        Returns:
            dict: Updated video record
        """
        data = {}
        if original_video_url is not None:
            data["original_video_url"] = original_video_url
        if processed_video_url is not None:
            data["processed_video_url"] = processed_video_url
        if metadata is not None:
            data["metadata"] = metadata

        if not data:
            return {}

        response = self.client.table("videos").update(data).eq("id", video_id).execute()
        return response.data[0] if response.data else {}

    def get_video(self, video_id: int) -> Optional[Dict[str, Any]]:
        """Get a video record by ID."""
        response = self.client.table("videos").select("*").eq("id", video_id).execute()
        return response.data[0] if response.data else None

    def get_all_videos(self) -> List[Dict[str, Any]]:
        """Get all video records."""
        response = self.client.table("videos").select("*").order("created_at", desc=True).execute()
        return response.data or []

    # ==================== EVENT OPERATIONS ====================

    def create_event(
        self,
        video_id: int,
        snapshot_url: Optional[str] = None,
        face_url: Optional[str] = None,
        label: Optional[str] = None,
        confidence: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a littering event record.

        Args:
            video_id: ID of the parent video
            snapshot_url: URL to the annotated frame snapshot
            face_url: URL to the extracted face/person crop
            label: Litter type (paper, cardboard, can, plastic)
            confidence: Detection confidence score
            extra: Additional data (person_id, litter_id, frame_number, etc.)

        Returns:
            dict: Created event record
        """
        data = {
            "video_id": video_id,
            "snapshot_url": snapshot_url,
            "face_url": face_url,
            "label": label,
            "confidence": confidence,
            "extra": extra or {},
        }

        response = self.client.table("events").insert(data).execute()
        return response.data[0] if response.data else {}

    def get_events_by_video(self, video_id: int) -> List[Dict[str, Any]]:
        """Get all events for a specific video."""
        response = (
            self.client.table("events")
            .select("*")
            .eq("video_id", video_id)
            .order("event_time", desc=False)
            .execute()
        )
        return response.data or []

    def get_all_events(self) -> List[Dict[str, Any]]:
        """Get all littering events."""
        response = (
            self.client.table("events")
            .select("*, videos(id, metadata)")
            .order("event_time", desc=True)
            .execute()
        )
        return response.data or []

    # ==================== STORAGE OPERATIONS ====================

    def upload_file(
        self,
        file_path: str,
        storage_path: str,
        content_type: str = "image/jpeg",
    ) -> Optional[str]:
        """
        Upload a file to Supabase storage.

        Args:
            file_path: Local path to the file
            storage_path: Destination path in the bucket (e.g., "videos/123/frame.jpg")
            content_type: MIME type of the file

        Returns:
            str: Public URL of the uploaded file, or None if failed
        """
        try:
            with open(file_path, "rb") as f:
                file_data = f.read()

            response = self.client.storage.from_(self.bucket).upload(
                storage_path,
                file_data,
                file_options={"content-type": content_type, "upsert": "true"},
            )

            # Get public URL
            public_url = self.client.storage.from_(self.bucket).get_public_url(storage_path)
            return public_url

        except Exception as e:
            print(f"Error uploading file to Supabase: {e}")
            return None

    def upload_frame(
        self,
        frame: np.ndarray,
        storage_path: str,
    ) -> Optional[str]:
        """
        Upload an OpenCV frame directly to Supabase storage.

        Args:
            frame: OpenCV frame (numpy array)
            storage_path: Destination path in the bucket

        Returns:
            str: Public URL of the uploaded image, or None if failed
        """
        try:
            # Encode frame to JPEG
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                print("Failed to encode frame")
                return None

            file_data = buffer.tobytes()

            response = self.client.storage.from_(self.bucket).upload(
                storage_path,
                file_data,
                file_options={"content-type": "image/jpeg", "upsert": "true"},
            )

            # Get public URL
            public_url = self.client.storage.from_(self.bucket).get_public_url(storage_path)
            return public_url

        except Exception as e:
            print(f"Error uploading frame to Supabase: {e}")
            return None

    def upload_video(
        self,
        video_path: str,
        storage_path: str,
    ) -> Optional[str]:
        """
        Upload a video file to Supabase storage.

        Args:
            video_path: Local path to the video file
            storage_path: Destination path in the bucket

        Returns:
            str: Public URL of the uploaded video, or None if failed
        """
        # Determine content type based on extension
        ext = os.path.splitext(video_path)[1].lower()
        content_types = {
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
            ".mkv": "video/x-matroska",
        }
        content_type = content_types.get(ext, "video/mp4")

        return self.upload_file(video_path, storage_path, content_type)

    def delete_file(self, storage_path: str) -> bool:
        """Delete a file from storage."""
        try:
            self.client.storage.from_(self.bucket).remove([storage_path])
            return True
        except Exception as e:
            print(f"Error deleting file from Supabase: {e}")
            return False


# Singleton instance for easy import
_client: Optional[SupabaseClient] = None


def get_supabase_client() -> SupabaseClient:
    """Get or create a Supabase client singleton."""
    global _client
    if _client is None:
        _client = SupabaseClient()
    return _client
