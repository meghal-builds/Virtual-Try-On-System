"""Flask API for Virtual Try-On System"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any

import cv2
import numpy as np
import werkzeug
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from src.image_utils import load_image, save_image, get_image_dimensions
from src.validation import validate_image
from src.model_layer import load_models
from src.segmentation import segment_body
from src.pose_detection import detect_pose
from src.measurement_inference import infer_measurements, validate_measurements
from src.garment_manager import list_available_garments, load_garment_metadata, load_garment_image
from src.size_recommendation import recommend_size


# Configuration
UPLOAD_FOLDER = Path("database/data/uploads")
if not UPLOAD_FOLDER.parent.exists() and Path("data").exists():
    UPLOAD_FOLDER = Path("data/uploads")

# Flask 3.0 test client expects this attribute with older Werkzeug combinations.
if not hasattr(werkzeug, "__version__"):
    werkzeug.__version__ = "3"
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
CORS(app)

# Create upload folder if it doesn't exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Load models on startup
try:
    seg_model, pose_model = load_models()
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    seg_model = None
    pose_model = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_session_id() -> str:
    """Generate unique session ID"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_") + os.urandom(4).hex()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': seg_model is not None and pose_model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/upload', methods=['POST'])
def upload_image():
    """
    Upload and validate user image
    
    Returns:
        {
            'success': bool,
            'session_id': str,
            'image_path': str,
            'dimensions': (height, width),
            'validation': {...}
        }
    """
    try:
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'File type not allowed'}), 400
        
        # Generate session ID
        session_id = generate_session_id()
        
        # Save file
        filename = secure_filename(f"{session_id}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Validate image
        validation_result = validate_image(filepath)
        
        if not validation_result.is_valid:
            os.remove(filepath)
            return jsonify({
                'success': False,
                'error': 'Image validation failed',
                'errors': validation_result.errors,
                'warnings': validation_result.warnings
            }), 400
        
        # Get dimensions
        height, width = get_image_dimensions(filepath)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'image_path': filepath,
            'dimensions': {'height': height, 'width': width},
            'validation': {
                'is_valid': validation_result.is_valid,
                'warnings': validation_result.warnings
            }
        }), 200
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/process', methods=['POST'])
def process_image():
    """
    Process image: segment body, detect pose, infer measurements
    
    Request:
        {
            'image_path': str
        }
    
    Returns:
        {
            'success': bool,
            'session_id': str,
            'measurements': {...},
            'pose': {...},
            'segmentation': {...}
        }
    """
    try:
        if seg_model is None or pose_model is None:
            return jsonify({'success': False, 'error': 'Models not loaded'}), 500
        
        data = request.get_json()
        image_path = data.get('image_path')
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({'success': False, 'error': 'Invalid image path'}), 400
        
        # Load image
        image = load_image(image_path)
        
        # Run segmentation
        seg_result = segment_body(image, seg_model)
        
        # Run pose detection
        try:
            pose_result = detect_pose(image, pose_model)
        except RuntimeError as e:
            return jsonify({
                'success': False,
                'error': 'Pose detection failed',
                'details': str(e)
            }), 400
        
        # Infer measurements
        measurements = infer_measurements(pose_result, seg_result)
        
        # Validate measurements
        is_valid, errors = validate_measurements(measurements)
        
        if not is_valid:
            return jsonify({
                'success': False,
                'error': 'Measurement validation failed',
                'details': errors
            }), 400
        
        return jsonify({
            'success': True,
            'measurements': {
                'shoulder_width_cm': round(measurements.shoulder_width_cm, 2),
                'chest_circumference_cm': round(measurements.chest_circumference_cm, 2),
                'torso_length_cm': round(measurements.torso_length_cm, 2),
                'confidence': round(measurements.confidence, 2)
            },
            'pose': {
                'shoulder_width_px': round(pose_result.shoulder_width_px, 2),
                'is_frontal': pose_result.is_frontal,
                'keypoint_count': len(pose_result.keypoints),
                'warnings': pose_result.warnings
            },
            'segmentation': {
                'confidence': round(seg_result.confidence, 2),
                'torso_percentage': round(seg_result.torso_percentage, 2),
                'warnings': seg_result.warnings
            }
        }), 200
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/recommend', methods=['POST'])
def recommend_size_endpoint():
    """
    Recommend sizes based on measurements
    
    Request:
        {
            'measurements': {
                'shoulder_width_cm': float,
                'chest_circumference_cm': float,
                'torso_length_cm': float
            },
            'garment_id': str
        }
    
    Returns:
        {
            'success': bool,
            'recommendation': {
                'size': str,
                'confidence': float,
                'fit_scores': {...},
                'recommended_sizes': [...]
            }
        }
    """
    try:
        from src.models import Measurements
        
        data = request.get_json()
        
        # Parse measurements
        meas_data = data.get('measurements', {})
        measurements = Measurements(
            shoulder_width_cm=float(meas_data.get('shoulder_width_cm', 0)),
            chest_circumference_cm=float(meas_data.get('chest_circumference_cm', 0)),
            torso_length_cm=float(meas_data.get('torso_length_cm', 0)),
            source='inferred',
            confidence=float(meas_data.get('confidence', 0.8))
        )
        
        # Get garment
        garment_id = data.get('garment_id')
        if not garment_id:
            return jsonify({'success': False, 'error': 'garment_id required'}), 400
        
        try:
            metadata = load_garment_metadata(garment_id)
        except FileNotFoundError:
            return jsonify({'success': False, 'error': f'Garment not found: {garment_id}'}), 404
        
        size_chart = metadata.get('size_chart', {})
        
        # Get recommendation
        recommendation = recommend_size(measurements, size_chart)
        
        return jsonify({
            'success': True,
            'recommendation': {
                'size': recommendation.size,
                'confidence': round(recommendation.confidence, 2),
                'recommended_sizes': recommendation.recommended_sizes,
                'fit_scores': {
                    size: round(score, 2)
                    for size, score in recommendation.fit_scores.items()
                }
            }
        }), 200
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/garments', methods=['GET'])
def get_garments():
    """
    List all available garments
    
    Returns:
        {
            'success': bool,
            'garments': [
                {
                    'id': str,
                    'name': str,
                    'category': str,
                    'brand': str
                },
                ...
            ]
        }
    """
    try:
        garment_ids = list_available_garments()
        
        garments = []
        for garment_id in garment_ids:
            try:
                metadata = load_garment_metadata(garment_id)
                garments.append({
                    'id': metadata.get('id'),
                    'name': metadata.get('name'),
                    'category': metadata.get('category'),
                    'brand': metadata.get('brand'),
                    'price_usd': metadata.get('price_usd', 0)
                })
            except Exception:
                pass
        
        return jsonify({
            'success': True,
            'count': len(garments),
            'garments': garments
        }), 200
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/garments/<garment_id>', methods=['GET'])
def get_garment_details(garment_id: str):
    """
    Get detailed information about a garment
    
    Returns:
        {
            'success': bool,
            'garment': {...}
        }
    """
    try:
        metadata = load_garment_metadata(garment_id)
        
        return jsonify({
            'success': True,
            'garment': {
                'id': metadata.get('id'),
                'name': metadata.get('name'),
                'category': metadata.get('category'),
                'brand': metadata.get('brand'),
                'description': metadata.get('description', ''),
                'material': metadata.get('material', ''),
                'price_usd': metadata.get('price_usd', 0),
                'available_colors': metadata.get('available_colors', []),
                'size_chart': metadata.get('size_chart', {})
            }
        }), 200
    
    except FileNotFoundError:
        return jsonify({'success': False, 'error': f'Garment not found: {garment_id}'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'success': False, 'error': 'File too large (max 10MB)'}), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Run development server
    app.run(debug=True, host='0.0.0.0', port=5000)
