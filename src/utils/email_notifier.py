"""
Email notification service for littering events.
Sends email alerts to admin when littering is detected.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()


class EmailNotifier:
    """Sends email notifications for littering events."""

    def __init__(self):
        """Initialize email notifier with environment variables."""
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.sender_email = os.getenv("SENDER_EMAIL")
        self.sender_password = os.getenv("SENDER_PASSWORD")  # App password for Gmail
        self.admin_email = os.getenv("ADMIN_EMAIL")
        
        self.enabled = all([self.sender_email, self.sender_password, self.admin_email])
        
        if not self.enabled:
            print("Email notifications disabled. Set SENDER_EMAIL, SENDER_PASSWORD, and ADMIN_EMAIL in .env")

    def send_littering_alert(
        self,
        event: Dict[str, Any],
        video_id: Optional[int] = None,
        snapshot_url: Optional[str] = None,
        face_url: Optional[str] = None,
        local_snapshot_path: Optional[str] = None,
    ) -> bool:
        """
        Send email alert for a littering event.

        Args:
            event: Littering event dictionary with person_id, litter_id, frame, litter_type
            video_id: ID of the video in database
            snapshot_url: URL to the event snapshot image
            face_url: URL to the person's face/crop image
            local_snapshot_path: Local path to snapshot (for attachment)

        Returns:
            bool: True if email sent successfully, False otherwise
        """
        if not self.enabled:
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"🚨 Littering Alert - {event.get('litter_type', 'Unknown')} Detected"
            msg["From"] = self.sender_email
            msg["To"] = self.admin_email

            # Build email content
            html_content = self._build_email_html(event, video_id, snapshot_url, face_url)
            text_content = self._build_email_text(event, video_id, snapshot_url, face_url)

            msg.attach(MIMEText(text_content, "plain"))
            msg.attach(MIMEText(html_content, "html"))

            # Attach local image if available
            if local_snapshot_path and os.path.exists(local_snapshot_path):
                with open(local_snapshot_path, "rb") as img_file:
                    img = MIMEImage(img_file.read())
                    img.add_header("Content-Disposition", "attachment", filename="event_snapshot.jpg")
                    msg.attach(img)

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, self.admin_email, msg.as_string())

            print(f"Email alert sent for littering event: person {event['person_id']}, litter {event['litter_id']}")
            return True

        except Exception as e:
            print(f"Failed to send email alert: {e}")
            return False

    def _build_email_html(
        self,
        event: Dict[str, Any],
        video_id: Optional[int],
        snapshot_url: Optional[str],
        face_url: Optional[str],
    ) -> str:
        """Build HTML email content."""
        snapshot_section = ""
        if snapshot_url:
            snapshot_section = f"""
            <h3>Event Snapshot:</h3>
            <p><a href="{snapshot_url}">View Snapshot Image</a></p>
            <img src="{snapshot_url}" alt="Event Snapshot" style="max-width: 600px; border: 1px solid #ddd; border-radius: 8px;">
            """

        face_section = ""
        if face_url:
            face_section = f"""
            <h3>Person Image:</h3>
            <p><a href="{face_url}">View Person Image</a></p>
            <img src="{face_url}" alt="Person Image" style="max-width: 300px; border: 1px solid #ddd; border-radius: 8px;">
            """

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 700px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #dc3545; color: white; padding: 20px; border-radius: 8px 8px 0 0; }}
                .content {{ background: #f8f9fa; padding: 20px; border: 1px solid #ddd; }}
                .details {{ background: white; padding: 15px; border-radius: 8px; margin: 15px 0; }}
                .detail-row {{ display: flex; padding: 8px 0; border-bottom: 1px solid #eee; }}
                .detail-label {{ font-weight: bold; width: 150px; color: #666; }}
                .footer {{ background: #343a40; color: #aaa; padding: 15px; text-align: center; border-radius: 0 0 8px 8px; font-size: 12px; }}
                a {{ color: #007bff; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚨 Littering Event Detected</h1>
                </div>
                <div class="content">
                    <p>A littering event has been detected and recorded by the Clean-O-Sphere system.</p>
                    
                    <div class="details">
                        <h3>Event Details:</h3>
                        <div class="detail-row">
                            <span class="detail-label">Litter Type:</span>
                            <span><strong>{event.get('litter_type', 'Unknown').upper()}</strong></span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Person ID:</span>
                            <span>{event.get('person_id', 'N/A')}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Litter ID:</span>
                            <span>{event.get('litter_id', 'N/A')}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Frame Number:</span>
                            <span>{event.get('frame', 'N/A')}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Video ID:</span>
                            <span>{video_id or 'N/A'}</span>
                        </div>
                    </div>
                    
                    {snapshot_section}
                    {face_section}
                    
                </div>
                <div class="footer">
                    <p>This is an automated message from Clean-O-Sphere Litter Detection System</p>
                </div>
            </div>
        </body>
        </html>
        """

    def _build_email_text(
        self,
        event: Dict[str, Any],
        video_id: Optional[int],
        snapshot_url: Optional[str],
        face_url: Optional[str],
    ) -> str:
        """Build plain text email content."""
        text = f"""
LITTERING EVENT DETECTED
========================

A littering event has been detected by the Clean-O-Sphere system.

EVENT DETAILS:
- Litter Type: {event.get('litter_type', 'Unknown').upper()}
- Person ID: {event.get('person_id', 'N/A')}
- Litter ID: {event.get('litter_id', 'N/A')}
- Frame Number: {event.get('frame', 'N/A')}
- Video ID: {video_id or 'N/A'}

LINKS:
"""
        if snapshot_url:
            text += f"- Event Snapshot: {snapshot_url}\n"
        if face_url:
            text += f"- Person Image: {face_url}\n"

        text += """
---
This is an automated message from Clean-O-Sphere Litter Detection System
"""
        return text


# Singleton instance
_email_notifier: Optional[EmailNotifier] = None


def get_email_notifier() -> EmailNotifier:
    """Get or create the email notifier singleton."""
    global _email_notifier
    if _email_notifier is None:
        _email_notifier = EmailNotifier()
    return _email_notifier
