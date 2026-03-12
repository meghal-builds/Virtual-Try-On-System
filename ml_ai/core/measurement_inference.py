"""Measurement Inference Module - WITH DYNAMIC CALIBRATION"""

from typing import Tuple, List
import numpy as np

from src.models import Measurements, SegmentationResult, PoseResult, Keypoint

# Base calibration (can be adjusted per image)
PIXELS_PER_CM = 7.0
SHOULDER_TO_CIRCUMFERENCE = 2.35  # More accurate ratio
TORSO_MULTIPLIER = 1.13  # Fine-tuned for better torso estimation


def calculate_torso_length(keypoints: List[Keypoint]) -> float:
    """Calculate torso length from keypoints"""
    left_shoulder = next((kp for kp in keypoints if kp.name == 'left_shoulder'), None)
    right_shoulder = next((kp for kp in keypoints if kp.name == 'right_shoulder'), None)
    left_hip = next((kp for kp in keypoints if kp.name == 'left_hip'), None)
    right_hip = next((kp for kp in keypoints if kp.name == 'right_hip'), None)
    
    if not (left_shoulder and right_shoulder and left_hip and right_hip):
        return 0.0
    
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    hip_y = (left_hip.y + right_hip.y) / 2
    torso_length = abs(hip_y - shoulder_y)
    
    return torso_length


def infer_measurements(pose_result: PoseResult, seg_result: SegmentationResult) -> Measurements:
    """Infer body measurements from pose and segmentation with dynamic calibration"""
    if not pose_result or not seg_result:
        raise ValueError("Both pose and segmentation results required")
    
    shoulder_width_px = pose_result.shoulder_width_px
    torso_length_px = calculate_torso_length(pose_result.keypoints)
    chest_width_px = shoulder_width_px * 1.15
    
    # Adaptive calibration based on shoulder width
    pixels_per_cm = _get_adaptive_pixels_per_cm(shoulder_width_px)
    
    shoulder_width_cm = shoulder_width_px / pixels_per_cm
    torso_length_cm = (torso_length_px / pixels_per_cm) * TORSO_MULTIPLIER
    chest_circumference_cm = chest_width_px / pixels_per_cm * SHOULDER_TO_CIRCUMFERENCE
    
    # Calculate confidence based on detection quality
    confidence = _calculate_confidence(pose_result, shoulder_width_cm, chest_circumference_cm)
    
    measurements = Measurements(
        shoulder_width_cm=round(shoulder_width_cm, 2),
        chest_circumference_cm=round(chest_circumference_cm, 2),
        torso_length_cm=round(torso_length_cm, 2),
        source='inferred',
        confidence=round(confidence, 2)
    )
    
    return measurements


def _get_adaptive_pixels_per_cm(shoulder_width_px: float) -> float:
    """
    Get adaptive pixels per cm based on shoulder width
    
    This accounts for distance-to-camera variations
    Expected shoulder widths: 280-320 px for medium distance
    """
    # Expected shoulder width in pixels for standard distance
    EXPECTED_SHOULDER_PX = 280
    EXPECTED_PIXELS_PER_CM = 7.0
    
    # Adjust if shoulder is significantly different from expected
    if shoulder_width_px > 0:
        ratio = EXPECTED_SHOULDER_PX / shoulder_width_px
        adjusted_pixels_per_cm = EXPECTED_PIXELS_PER_CM * ratio
        
        # Keep within reasonable bounds (5.5 to 8.5)
        return max(5.5, min(8.5, adjusted_pixels_per_cm))
    
    return EXPECTED_PIXELS_PER_CM


def _calculate_confidence(
    pose_result: PoseResult,
    shoulder_width_cm: float,
    chest_circumference_cm: float
) -> float:
    """Calculate measurement confidence"""
    confidence = 0.5
    
    keypoint_bonus = min(len(pose_result.keypoints) / 20.0, 0.3)
    confidence += keypoint_bonus
    
    if pose_result.is_frontal:
        confidence += 0.15
    
    # More lenient ranges for better confidence
    if 38 <= shoulder_width_cm <= 45:
        confidence += 0.1
    
    if 85 <= chest_circumference_cm <= 110:
        confidence += 0.1
    
    return min(confidence, 0.95)


def calculate_measurement_confidence(measurements: Measurements) -> float:
    """Get confidence from measurements"""
    return measurements.confidence


def calculate_measurement_fit(user_value: float, size_value: float, tolerance_percent: float = 5.0) -> float:
    """Calculate fit score for a single measurement"""
    if size_value == 0:
        return 0.0
    
    difference_percent = abs(user_value - size_value) / size_value * 100
    
    if difference_percent == 0:
        return 1.0
    
    max_difference = tolerance_percent * 2
    
    if difference_percent >= max_difference:
        return 0.0
    
    score = 1.0 - (difference_percent / max_difference)
    return max(0.0, min(1.0, score))


def validate_measurements(measurements: Measurements) -> Tuple[bool, str]:
    """Validate that measurements are in reasonable range"""
    errors = []
    
    if not (30 <= measurements.shoulder_width_cm <= 55):
        errors.append(f"Invalid shoulder width: {measurements.shoulder_width_cm}cm (expected 30-55cm)")
    
    if not (70 <= measurements.chest_circumference_cm <= 130):
        errors.append(f"Invalid chest circumference: {measurements.chest_circumference_cm}cm (expected 70-130cm)")
    
    if not (40 <= measurements.torso_length_cm <= 80):
        errors.append(f"Invalid torso length: {measurements.torso_length_cm}cm (expected 40-80cm)")
    
    is_valid = len(errors) == 0
    error_message = "; ".join(errors) if errors else ""
    
    return is_valid, error_message


def recalibrate_pixels_per_cm(reference_width_cm: float, measured_width_px: float) -> float:
    """Recalibrate pixels-per-cm ratio"""
    return measured_width_px / reference_width_cm


def print_measurement_debug_info(pose_result: PoseResult, measurements: Measurements):
    """Print debug information about measurements"""
    print("\n" + "="*70)
    print("MEASUREMENT DEBUG INFO")
    print("="*70)
    print(f"Shoulder Width (px): {pose_result.shoulder_width_px:.2f}")
    print(f"Shoulder Width (cm): {measurements.shoulder_width_cm:.2f}")
    print(f"Pixels per cm (adaptive): {pose_result.shoulder_width_px / measurements.shoulder_width_cm:.2f}")
    print(f"\nChest Circumference (cm): {measurements.chest_circumference_cm:.2f}")
    print(f"Torso Length (cm): {measurements.torso_length_cm:.2f}")
    print(f"Confidence: {measurements.confidence * 100:.1f}%")
    print("="*70 + "\n")