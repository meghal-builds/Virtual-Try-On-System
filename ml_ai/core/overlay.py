"""Image Overlay and Composition Module"""

from typing import Tuple, Optional
import numpy as np
import cv2


def overlay_garment(
    background_image: np.ndarray,
    garment_image: np.ndarray,
    position: Tuple[int, int],
    alpha: float = 0.8
) -> np.ndarray:
    """
    Overlay garment onto background image
    
    Args:
        background_image: Background image (person)
        garment_image: Garment image to overlay
        position: (x, y) position to place garment
        alpha: Transparency of garment (0-1)
        
    Returns:
        Composite image with garment overlaid
    """
    if not isinstance(background_image, np.ndarray):
        raise TypeError("Background image must be numpy array")
    
    if not isinstance(garment_image, np.ndarray):
        raise TypeError("Garment image must be numpy array")
    
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    # Make a copy to avoid modifying original
    result = background_image.copy()
    
    x, y = position
    g_height, g_width = garment_image.shape[:2]
    
    # Check bounds
    if x < 0 or y < 0:
        raise ValueError("Position must be non-negative")
    
    if x + g_width > result.shape[1] or y + g_height > result.shape[0]:
        raise ValueError("Garment extends beyond image bounds")
    
    # Extract region where garment will be placed
    roi = result[y:y + g_height, x:x + g_width]
    
    # Handle different garment image formats
    if len(garment_image.shape) == 3 and garment_image.shape[2] == 4:
        # Garment has alpha channel
        garment_bgr = garment_image[:, :, :3]
        garment_alpha = garment_image[:, :, 3].astype(float) / 255.0
        
        # Blend using alpha
        blended = (garment_bgr.astype(float) * garment_alpha[:, :, None] +
                   roi.astype(float) * (1 - garment_alpha[:, :, None]))
        result[y:y + g_height, x:x + g_width] = blended.astype(np.uint8)
    else:
        # No alpha channel - use transparency parameter
        garment_bgr = garment_image if len(garment_image.shape) == 3 else cv2.cvtColor(garment_image, cv2.COLOR_GRAY2BGR)
        
        # Blend
        blended = (garment_bgr.astype(float) * alpha +
                   roi.astype(float) * (1 - alpha))
        result[y:y + g_height, x:x + g_width] = blended.astype(np.uint8)
    
    return result


def composite_multiple_garments(
    background_image: np.ndarray,
    garments: list,
    positions: list,
    alphas: Optional[list] = None
) -> np.ndarray:
    """
    Overlay multiple garments onto background
    
    Args:
        background_image: Background image
        garments: List of garment images
        positions: List of (x, y) positions for each garment
        alphas: List of alpha values (optional)
        
    Returns:
        Composite image with all garments
    """
    if len(garments) != len(positions):
        raise ValueError("Number of garments must match positions")
    
    if alphas is None:
        alphas = [0.8] * len(garments)
    elif len(alphas) != len(garments):
        raise ValueError("Number of alphas must match garments")
    
    result = background_image.copy()
    
    for garment, position, alpha in zip(garments, positions, alphas):
        result = overlay_garment(result, garment, position, alpha)
    
    return result


def blend_images(
    image1: np.ndarray,
    image2: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Blend two images together
    
    Args:
        image1: First image
        image2: Second image
        alpha: Weight of first image (0-1)
        
    Returns:
        Blended image
    """
    if image1.shape != image2.shape:
        raise ValueError("Images must have same dimensions")
    
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    blended = (image1.astype(float) * alpha +
               image2.astype(float) * (1 - alpha))
    
    return blended.astype(np.uint8)