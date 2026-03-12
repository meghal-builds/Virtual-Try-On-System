"""Garment Management Module"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


class GarmentManager:
    """Manages garment loading and validation"""
    
    def __init__(self, garment_base_path: str = "database/data/garments"):
        """
        Initialize Garment Manager
        
        Args:
            garment_base_path: Base path to garments directory
        """
        if garment_base_path == "database/data/garments" and not Path(garment_base_path).exists() and Path("data/garments").exists():
            garment_base_path = "data/garments"

        self.garment_base_path = Path(garment_base_path)
        if not self.garment_base_path.exists():
            raise FileNotFoundError(f"Garment path not found: {garment_base_path}")
    
    def load_garment_metadata(self, garment_id: str) -> Dict:
        """
        Load garment metadata from JSON
        
        Args:
            garment_id: Garment identifier
            
        Returns:
            Dictionary with garment metadata
        """
        metadata_path = self.garment_base_path / garment_id / "metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {garment_id}: {e}")
    
    def load_garment_image(self, garment_id: str) -> np.ndarray:
        """
        Load garment image
        
        Args:
            garment_id: Garment identifier
            
        Returns:
            Image as numpy array
        """
        metadata = self.load_garment_metadata(garment_id)
        image_filename = metadata.get("image_filename", "image.png")
        image_path = self.garment_base_path / garment_id / image_filename
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        return image
    
    def validate_garment_file(self, garment_id: str) -> bool:
        """
        Validate that a garment has all required files
        
        Args:
            garment_id: Garment identifier
            
        Returns:
            True if valid, False otherwise
        """
        try:
            garment_path = self.garment_base_path / garment_id
            
            if not garment_path.exists():
                return False
            
            # Check metadata
            metadata = self.load_garment_metadata(garment_id)
            if not metadata:
                return False
            
            # Check required fields
            required = ['id', 'name', 'category', 'brand', 'image_filename', 'size_chart']
            if not all(field in metadata for field in required):
                return False
            
            # Check image exists
            image_filename = metadata.get('image_filename', 'image.png')
            image_path = garment_path / image_filename
            if not image_path.exists():
                return False
            
            return True
        except Exception:
            return False
    
    def list_available_garments(self) -> List[str]:
        """
        List all available garment IDs
        
        Returns:
            Sorted list of garment IDs
        """
        if not self.garment_base_path.exists():
            return []
        
        garment_ids = []
        for item in self.garment_base_path.iterdir():
            if item.is_dir() and self.validate_garment_file(item.name):
                garment_ids.append(item.name)
        
        return sorted(garment_ids)
    
    def get_size_chart(self, garment_id: str) -> Dict:
        """
        Get size chart for a garment
        
        Args:
            garment_id: Garment identifier
            
        Returns:
            Dictionary with size chart
        """
        metadata = self.load_garment_metadata(garment_id)
        return metadata.get('size_chart', {})


# Module-level convenience functions
_manager = None

def get_manager(base_path: str = "database/data/garments") -> GarmentManager:
    """Get or create global manager"""
    global _manager
    if _manager is None:
        _manager = GarmentManager(base_path)
    return _manager

def load_garment_metadata(garment_id: str) -> Dict:
    """Load garment metadata"""
    return get_manager().load_garment_metadata(garment_id)

def load_garment_image(garment_id: str) -> np.ndarray:
    """Load garment image"""
    return get_manager().load_garment_image(garment_id)

def list_available_garments() -> List[str]:
    """List available garments"""
    return get_manager().list_available_garments()

def validate_garment_file(garment_id: str) -> bool:
    """Validate garment file"""
    return get_manager().validate_garment_file(garment_id)
