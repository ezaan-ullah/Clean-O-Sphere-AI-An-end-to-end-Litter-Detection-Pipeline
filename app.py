"""
Flask backend server for litter detection system.
"""

import os
import time
import shutil
from flask import Flask, request, render_template, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
from src.video_processor import VideoProcessor

# Try to import Supabase client
try:
    from src.utils.supabase_client import get_supabase_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'mov', 'mp4', 'avi', 'mkv'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Model path configuration
MODEL_PATH = 'model/v8mBestWeights.pt'


def cleanup_previous_request():
    """
    Clean up files from the previous request:
    - Original videos in uploads/
    - Processed videos in outputs/
    - Event images in static/litter_events/
    - Face images in static/faces/
    - Number plate images in static/number_plates/
    """
    folders_to_clean = [
        app.config['UPLOAD_FOLDER'],
        app.config['OUTPUT_FOLDER'],
        os.path.join('static', 'litter_events'),
        os.path.join('static', 'faces'),
        os.path.join('static', 'number_plates'),
    ]
    
    for folder in folders_to_clean:
        if os.path.exists(folder):
            try:
                # Remove all contents but keep the folder
                for item in os.listdir(folder):
                    item_path = os.path.join(folder, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                print(f"Cleaned up: {folder}")
            except Exception as e:
                print(f"Error cleaning {folder}: {e}")

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Home page."""
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Error loading template: {str(e)}", 500

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and processing."""

    # =============================================================================== #
    # Clean up files from previous request before starting new one
    cleanup_previous_request()
    
    # =============================================================================== #
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: mov, mp4, avi, mkv'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"Processing video: {filename}")
        
        processor = VideoProcessor(MODEL_PATH, flask_server_url=request.url_root, detect_vehicles=False)
        output_filename = f"processed_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        results = processor.process_video(filepath, output_path=output_path, display=False)
        
        print(f"Processing complete. Results: {results}")
        
        if not isinstance(results, dict):
            results = {
                'total_frames': getattr(results, 'total_frames', 0),
                'littering_events': getattr(results, 'littering_events', []),
                'persons_with_littering': getattr(results, 'persons_with_littering', 0),
                'total_litter_items': getattr(results, 'total_litter_items', 0)
            }
        
        return jsonify({
            'success': True,
            'message': 'Video processed successfully',
            'total_frames': results.get('total_frames', 0),
            'littering_events': len(results.get('littering_events', [])),
            'persons_littered': results.get('persons_with_littering', 0),
            'total_litter': results.get('total_litter_items', 0),
            'output_video': output_filename,
            'video_id': results.get('video_id'),
            'processed_video_url': results.get('processed_video_url'),
            'original_video_url': results.get('original_video_url'),
        })
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error processing video: {error_trace}")
        return jsonify({
            'success': False,
            'error': f'Processing failed: {str(e)}'
        }), 500

@app.route('/api/report_littering', methods=['POST'])
def report_littering():
    """
    Receive person image when littering is detected.
    This endpoint is called by the video processor.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    person_id = request.form.get('person_id')
    frame_number = request.form.get('frame_number')
    
    image_file = request.files['image']
    
    image_dir = os.path.join('static', 'person_images')
    os.makedirs(image_dir, exist_ok=True)
    
    image_filename = f"person_{person_id}_frame_{frame_number}.jpg"
    image_path = os.path.join(image_dir, image_filename)
    image_file.save(image_path)
    
    # person_data = {
    #     'person_id': person_id,
    #     'frame_number': frame_number,
    #     'image_path': image_path,
    #     'timestamp': datetime.now()
    # }
    # db.persons.insert_one(person_data)
    
    print(f"Received littering report: Person {person_id} at frame {frame_number}")
    
    return jsonify({
        'success': True,
        'message': 'Littering report received',
        'person_id': person_id,
        'frame_number': frame_number
    })

@app.route('/api/addImage', methods=['POST'])
def add_image():
    """
    Receive an annotated littering frame and persist it locally for review.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filename = request.form.get('filename') or image_file.filename or f"event_{int(time.time())}.jpg"
    safe_filename = secure_filename(filename)
    if not safe_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        safe_filename = f"{safe_filename}.jpg"
    
    events_dir = os.path.join('static', 'litter_events')
    os.makedirs(events_dir, exist_ok=True)
    
    save_path = os.path.join(events_dir, safe_filename)
    image_file.save(save_path)
    
    metadata = {
        'person_id': request.form.get('person_id'),
        'litter_id': request.form.get('litter_id'),
        'litter_type': request.form.get('litter_type'),
        'frame_number': request.form.get('frame_number'),
    }
    
    image_url = url_for('serve_static', filename=f"litter_events/{safe_filename}", _external=True)
    
    return jsonify({
        'success': True,
        'message': 'Annotated image stored successfully',
        'image_url': image_url,
        'metadata': metadata
    })

@app.route('/api/littering_events', methods=['GET'])
def get_littering_events():
    """Get all littering events from Supabase."""
    if SUPABASE_AVAILABLE:
        try:
            client = get_supabase_client()
            events = client.get_all_events()
            return jsonify({
                'success': True,
                'events': events,
                'count': len(events)
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'events': []
            }), 500
    else:
        return jsonify({
            'success': False,
            'events': [],
            'message': 'Supabase not configured'
        })


@app.route('/api/videos', methods=['GET'])
def get_videos():
    """Get all processed videos from Supabase."""
    if SUPABASE_AVAILABLE:
        try:
            client = get_supabase_client()
            videos = client.get_all_videos()
            return jsonify({
                'success': True,
                'videos': videos,
                'count': len(videos)
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'videos': []
            }), 500
    else:
        return jsonify({
            'success': False,
            'videos': [],
            'message': 'Supabase not configured'
        })


@app.route('/api/videos/<int:video_id>/events', methods=['GET'])
def get_video_events(video_id):
    """Get all events for a specific video."""
    if SUPABASE_AVAILABLE:
        try:
            client = get_supabase_client()
            events = client.get_events_by_video(video_id)
            return jsonify({
                'success': True,
                'video_id': video_id,
                'events': events,
                'count': len(events)
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'events': []
            }), 500
    else:
        return jsonify({
            'success': False,
            'events': [],
            'message': 'Supabase not configured'
        })


@app.route('/outputs/<filename>')
@app.route('/download/<filename>')
def download_output(filename):
    """Download processed video."""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory('static', filename)

@app.route('/test')
def test():
    """Test route to verify Flask is working."""
    return "Flask is working! Template folder: " + str(app.template_folder)


if __name__ == '__main__':
    print("Starting Flask server...")
    print(f"Template folder: {app.template_folder}")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Supabase available: {SUPABASE_AVAILABLE}")
    app.run(debug=True, host='0.0.0.0', port=5000)

