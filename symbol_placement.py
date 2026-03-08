"""
Module 3: Symbol Placement using Pole of Inaccessibility.
Finds the optimal position for placing symbols in irregular shapes.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import math


class SymbolPlacer:
    """
    Places symbols at optimal positions within irregular shapes.
    Uses the Pole of Inaccessibility algorithm to find the point
    farthest from all edges.
    """
    
    def __init__(self, min_region_area: int = 50, font_size_ratio: float = 0.5):
        """
        Initialize the symbol placer.
        
        Args:
            min_region_area: Minimum area (pixels) for a region to receive a symbol
            font_size_ratio: Symbol size as ratio of region diameter (0.0 to 1.0)
        """
        self.min_region_area = min_region_area
        self.font_size_ratio = font_size_ratio
    
    def find_pole_of_inaccessibility(self, mask: np.ndarray, contour: np.ndarray) -> Tuple[int, int, float]:
        """
        Find the pole of inaccessibility - the point inside a polygon that is
        farthest from all edges.
        
        This is superior to centroid for irregular shapes (L, U, C shapes).
        
        Args:
            mask: Binary mask of the region (255 = inside, 0 = outside)
            contour: Contour points of the region (not used, kept for interface compatibility)
            
        Returns:
            tuple: (x, y, distance) where distance is the radius of the largest circle
                   that fits inside the shape at that point
        """
        # Ensure mask is binary (255 or 0)
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Use distance transform to find the point farthest from edges
        # This computes the distance from each white pixel to the nearest black pixel
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        
        # Find the maximum distance point
        _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
        
        # max_loc is (x, y) in the coordinate system of the mask
        x, y = max_loc
        radius = max_val
        
        return x, y, radius
    
    def calculate_centroid(self, contour: np.ndarray) -> Tuple[int, int]:
        """
        Calculate the centroid (center of mass) of a contour.
        Used as fallback if pole of inaccessibility fails.
        
        Args:
            contour: Contour points
            
        Returns:
            tuple: (x, y) coordinates of centroid
        """
        M = cv2.moments(contour)
        
        if M["m00"] == 0:
            # Fallback: use mean of points
            if len(contour.shape) == 3:
                contour = contour.squeeze()
            if len(contour.shape) == 2 and contour.shape[1] == 2:
                cx = int(np.mean(contour[:, 0]))
                cy = int(np.mean(contour[:, 1]))
            else:
                cx, cy = 0, 0
        else:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        
        return cx, cy
    
    def is_point_inside_contour(self, point: Tuple[int, int], contour: np.ndarray) -> bool:
        """
        Check if a point is inside a contour.
        
        Args:
            point: (x, y) coordinates
            contour: Contour points
            
        Returns:
            bool: True if point is inside
        """
        result = cv2.pointPolygonTest(contour, point, False)
        return result >= 0
    
    def calculate_symbol_size(self, area: float, radius: float) -> int:
        """
        Calculate appropriate symbol size based on region area and inscribed radius.
        
        Args:
            area: Area of the region in pixels
            radius: Radius of largest inscribed circle
            
        Returns:
            int: Font size in points
        """
        # All symbols have the same size
        return 10
    
    def find_symbol_placement(self, region_dict: dict) -> Optional[dict]:
        """
        Find optimal placement for a symbol in a region.
        
        Args:
            region_dict: Dictionary with keys:
                - 'contour': polygon points
                - 'area': area in pixels
                - 'color_id': color ID
                
        Returns:
            dict or None: Placement info with keys:
                - 'position': (x, y) tuple
                - 'font_size': size in points
                - 'color_id': color ID
                - 'radius': inscribed radius
                or None if region is too small
        """
        area = region_dict['area']
        
        # Skip if region is too small
        if area < self.min_region_area:
            return None
        
        contour = region_dict['contour']
        color_id = region_dict['color_id']
        
        # Ensure contour is in correct format
        if len(contour.shape) == 3:
            contour = contour.squeeze()
        
        # Reshape if necessary
        if len(contour.shape) == 1:
            contour = contour.reshape(-1, 2)
        
        # Create mask on-the-fly if not provided (memory optimization)
        use_local_coordinates = False
        x_offset, y_offset = 0, 0
        
        if 'mask' in region_dict:
            mask = region_dict['mask']
        else:
            # Determine mask size from contour bounds
            x_coords = contour[:, 0]
            y_coords = contour[:, 1]
            x_min, x_max = int(x_coords.min()), int(x_coords.max())
            y_min, y_max = int(y_coords.min()), int(y_coords.max())
            
            # Add padding to avoid edge issues
            padding = 2
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max += padding
            y_max += padding
            
            # Store offsets for coordinate conversion
            x_offset, y_offset = x_min, y_min
            use_local_coordinates = True
            
            # Create minimal mask (only bounding box size)
            mask_height = y_max - y_min + 1
            mask_width = x_max - x_min + 1
            mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
            
            # Adjust contour to local coordinates
            local_contour = contour.copy().astype(np.float64)
            local_contour[:, 0] -= x_min
            local_contour[:, 1] -= y_min
            
            # Fill the mask with anti-aliasing for better precision
            cv2.fillPoly(mask, [local_contour.astype(np.int32)], 255)
            
            # Apply slight erosion to ensure symbol stays away from edges
            # This creates a safety margin
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
        
        # Find pole of inaccessibility
        try:
            # Check if mask has any white pixels after erosion
            if cv2.countNonZero(mask) == 0:
                # Mask is empty after erosion, region too small
                # Use centroid without erosion
                x, y = self.calculate_centroid(contour)
                radius = math.sqrt(area / math.pi) * 0.5  # Smaller radius for safety
            else:
                x, y, radius = self.find_pole_of_inaccessibility(mask, contour)
                
                # Convert back to global coordinates if we used local coordinates
                if use_local_coordinates:
                    x += x_offset
                    y += y_offset
                
                # Verify the point is actually inside the contour (in global coordinates)
                if not self.is_point_inside_contour((x, y), contour):
                    # Fallback to centroid
                    x, y = self.calculate_centroid(contour)
                    radius = math.sqrt(area / math.pi) * 0.5  # Smaller radius for safety
        
        except Exception as e:
            # Fallback to centroid if pole calculation fails
            x, y = self.calculate_centroid(contour)
            radius = math.sqrt(area / math.pi) * 0.5  # Smaller radius for safety
        
        # Calculate appropriate font size
        font_size = self.calculate_symbol_size(area, radius)
        
        return {
            'position': (x, y),
            'font_size': font_size,
            'color_id': color_id,
            'radius': radius,
            'area': area
        }
    
    def place_symbols_in_regions(self, sub_regions: List[dict]) -> List[dict]:
        """
        Find symbol placements for all regions.
        
        Args:
            sub_regions: List of region dictionaries
            
        Returns:
            list: List of placement dictionaries (only for regions large enough)
        """
        placements = []
        
        print(f"Calculating symbol placements for {len(sub_regions)} regions...")
        
        for region in sub_regions:
            placement = self.find_symbol_placement(region)
            
            if placement is not None:
                placements.append(placement)
        
        print(f"Placed {len(placements)} symbols (skipped {len(sub_regions) - len(placements)} small regions).")
        
        return placements
    
    def get_symbol_for_color(self, color_id: int, symbol_type: str = "numbers", 
                            custom_symbols: Optional[List[str]] = None) -> str:
        """
        Get the symbol/character to use for a given color ID.
        
        Args:
            color_id: Color ID (0 to N-1)
            symbol_type: Type of symbols ('numbers', 'letters', 'custom')
            custom_symbols: List of custom symbols (if symbol_type='custom')
            
        Returns:
            str: Symbol to use
        """
        if symbol_type == "numbers":
            # 0-9, then A-Z, then AA, AB, etc.
            if color_id < 10:
                return str(color_id)
            elif color_id < 36:
                # A-Z (10-35 maps to A-Z)
                return chr(65 + color_id - 10)
            else:
                # Double letters: AA, AB, AC, ...
                adjusted = color_id - 36
                first = adjusted // 26
                second = adjusted % 26
                return chr(65 + first) + chr(65 + second)
        
        elif symbol_type == "letters":
            # A-Z, then AA, AB, etc.
            if color_id < 26:
                return chr(65 + color_id)  # A-Z
            else:
                # Double letters: AA, AB, AC, ...
                first = color_id // 26 - 1
                second = color_id % 26
                return chr(65 + first) + chr(65 + second)
        
        elif symbol_type == "custom" and custom_symbols:
            # Cycle through custom symbols
            return custom_symbols[color_id % len(custom_symbols)]
        
        else:
            # Default to numbers
            return str(color_id)
    
    def create_symbol_map(self, placements: List[dict], symbol_type: str = "numbers",
                         custom_symbols: Optional[List[str]] = None) -> List[dict]:
        """
        Create final symbol map with actual symbols assigned.
        
        Args:
            placements: List of placement dictionaries
            symbol_type: Type of symbols to use
            custom_symbols: Custom symbols list (optional)
            
        Returns:
            list: List of dictionaries with symbol info
        """
        symbol_map = []
        
        for placement in placements:
            color_id = placement['color_id']
            symbol = self.get_symbol_for_color(color_id, symbol_type, custom_symbols)
            
            symbol_map.append({
                'position': placement['position'],
                'font_size': placement['font_size'],
                'symbol': symbol,
                'color_id': color_id
            })
        
        return symbol_map
