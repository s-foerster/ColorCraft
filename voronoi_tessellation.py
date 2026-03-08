"""
Module 2: Voronoi Tessellation Engine.
Subdivides color regions into smaller shapes based on difficulty level.
Uses constrained Voronoi diagrams to create a mosaic effect.
Uses Shapely for proper polygon clipping and smoothing.
"""

import cv2
import numpy as np
from scipy.spatial import Voronoi
from typing import List, Tuple, Dict
import random
import gc
from shapely.geometry import Polygon, box, MultiPolygon, GeometryCollection
from shapely.validation import make_valid
from shapely.ops import unary_union


def extract_largest_polygon(geom) -> Polygon:
    """
    Extract the largest Polygon from any Shapely geometry.
    Handles Polygon, MultiPolygon, GeometryCollection, etc.
    
    Args:
        geom: Any Shapely geometry
        
    Returns:
        The largest Polygon, or an empty Polygon if none found
    """
    if geom is None or geom.is_empty:
        return Polygon()
    
    if isinstance(geom, Polygon):
        return geom
    
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        polygons = [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
        if not polygons:
            return Polygon()
        return max(polygons, key=lambda p: p.area)
    
    # For other geometry types (LineString, Point, etc.), return empty
    return Polygon()


def mask_to_polygon(mask: np.ndarray, simplify_tolerance: float = 0.5) -> List[Polygon]:
    """
    Convert a binary mask to Shapely polygon(s) with smoothing.
    
    Args:
        mask: Binary mask (255 = inside, 0 = outside)
        simplify_tolerance: Douglas-Peucker simplification tolerance in pixels
        
    Returns:
        List of valid Shapely Polygon objects
    """
    # Use morphological operations instead of GaussianBlur
    # This preserves hard edges while smoothing jagged pixel boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Use CHAIN_APPROX_TC89_L1 for better contour approximation
    contours, hierarchy = cv2.findContours(smoothed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    
    if not contours:
        return []
    
    polygons = []
    
    # Process contours with hierarchy for proper hole handling
    if hierarchy is None:
        return []
    
    hierarchy = hierarchy[0]
    
    for i, contour in enumerate(contours):
        # Only process outer contours (parent == -1 in RETR_CCOMP means top level)
        if hierarchy[i][3] != -1:  # This is a hole, skip (will be handled by parent)
            continue
        
        if len(contour) < 3:
            continue
        
        # Convert contour to coordinate list
        coords = contour.squeeze()
        if len(coords.shape) == 1 or len(coords) < 3:
            continue
        
        # Create exterior ring
        exterior = [(float(pt[0]), float(pt[1])) for pt in coords]
        
        # Close the ring if needed
        if exterior[0] != exterior[-1]:
            exterior.append(exterior[0])
        
        # Find holes (child contours)
        holes = []
        child_idx = hierarchy[i][2]  # First child
        while child_idx != -1:
            child_contour = contours[child_idx].squeeze()
            if len(child_contour.shape) > 1 and len(child_contour) >= 3:
                hole = [(float(pt[0]), float(pt[1])) for pt in child_contour]
                if hole[0] != hole[-1]:
                    hole.append(hole[0])
                if len(hole) >= 4:  # Need at least 3 points + closing point
                    holes.append(hole)
            child_idx = hierarchy[child_idx][0]  # Next sibling
        
        try:
            poly = Polygon(exterior, holes)
            
            # Make valid if needed
            if not poly.is_valid:
                poly = make_valid(poly)
            
            # Simplify to remove jagged edges
            if simplify_tolerance > 0:
                poly = poly.simplify(simplify_tolerance, preserve_topology=True)
            
            # Handle MultiPolygon result from make_valid
            if isinstance(poly, MultiPolygon):
                for geom in poly.geoms:
                    if geom.is_valid and not geom.is_empty and geom.area > 1:
                        polygons.append(geom)
            elif poly.is_valid and not poly.is_empty and poly.area > 1:
                polygons.append(poly)
                
        except Exception as e:
            continue
    
    return polygons


def polygon_to_contour(geom) -> np.ndarray:
    """
    Convert a Shapely geometry to OpenCV contour format.
    Handles Polygon, MultiPolygon, GeometryCollection by extracting the largest polygon.
    
    Args:
        geom: Any Shapely geometry (Polygon, MultiPolygon, GeometryCollection, etc.)
        
    Returns:
        np.ndarray: Contour in OpenCV format, or empty array if invalid
    """
    # Extract largest polygon if not already a simple Polygon
    polygon = extract_largest_polygon(geom)
    
    if polygon.is_empty or not hasattr(polygon, 'exterior') or polygon.exterior is None:
        return np.array([])
    
    # Get exterior coordinates
    coords = np.array(polygon.exterior.coords)
    
    if len(coords) < 3:
        return np.array([])
    
    # Convert to int32 for OpenCV
    contour = coords[:-1].astype(np.int32)  # Remove duplicate closing point
    
    return contour


class VoronoiTessellator:
    """
    Voronoi tessellation engine for subdividing color regions.
    The difficulty level determines the density of subdivision.
    """
    
    def __init__(self, difficulty_level: int = 5, points_per_1000px2: float = 5.0):
        """
        Initialize the Voronoi tessellator.
        
        Args:
            difficulty_level: Difficulty (1 = no subdivision, 10 = maximum subdivision)
            points_per_1000px2: Base density of Voronoi points per 1000 square pixels
        """
        self.difficulty_level = difficulty_level
        self.points_per_1000px2 = points_per_1000px2
        
    def calculate_num_points(self, area: float) -> int:
        """
        Calculate number of Voronoi points based on area and difficulty.
        Target: create sub-regions of similar size (~500-2000px² each)
        
        Args:
            area: Area in square pixels
            
        Returns:
            int: Number of points to generate
        """
        if self.difficulty_level == 1:
            return 0  # No subdivision at difficulty 1
        
        # Target sub-region size based on difficulty
        # Difficulty 2: ~2000px² per sub-region
        # Difficulty 10: ~200px² per sub-region
        target_sub_area = 2500 - (self.difficulty_level * 200)
        target_sub_area = max(200, target_sub_area)  # Minimum 200px²
        
        # Calculate number of points needed to achieve target size
        num_points = int(area / target_sub_area)
        
        # Cap maximum points to avoid very long processing times
        num_points = min(num_points, 50)  # Max 50 points per region
        
        # Minimum 2 points for subdivision to make sense
        if num_points < 2:
            return 0 if self.difficulty_level <= 2 else 2
        
        return num_points
    
    def generate_random_points_in_contour(self, mask: np.ndarray, num_points: int) -> np.ndarray:
        """
        Generate random points inside a masked region.
        
        Args:
            mask: Binary mask (255 = inside region, 0 = outside)
            num_points: Number of points to generate
            
        Returns:
            np.ndarray: Array of points [(x1, y1), (x2, y2), ...]
        """
        if num_points == 0:
            return np.array([])
        
        # Get all white pixels (inside the region)
        y_coords, x_coords = np.where(mask > 0)
        
        if len(x_coords) == 0:
            return np.array([])
        
        # Randomly sample points from inside the region
        if len(x_coords) < num_points:
            num_points = len(x_coords)
        
        indices = np.random.choice(len(x_coords), size=num_points, replace=False)
        points = np.column_stack((x_coords[indices], y_coords[indices]))
        
        return points
    
    def create_voronoi_cells(self, points: np.ndarray, image_shape: Tuple[int, int]) -> List[Polygon]:
        """
        Create Voronoi cells from points using Shapely for proper polygon clipping.
        
        Args:
            points: Array of seed points
            image_shape: (height, width) of the image
            
        Returns:
            list: List of Shapely Polygon objects (properly clipped to image bounds)
        """
        if len(points) < 2:
            return []
        
        height, width = image_shape
        
        try:
            # Add boundary points to ensure proper tessellation at edges
            boundary_points = np.array([
                [0, 0], [width-1, 0], [width-1, height-1], [0, height-1],
                [width//2, 0], [width-1, height//2], [width//2, height-1], [0, height//2]
            ])
            
            all_points = np.vstack([points, boundary_points])
            
            # Create Voronoi diagram
            vor = Voronoi(all_points)
            
            # Create bounding box for clipping
            bounds = box(0, 0, width - 1, height - 1)
            
            # Extract cell polygons using Shapely
            cells = []
            for region_index in vor.point_region[:len(points)]:  # Only process original points, not boundary
                region = vor.regions[region_index]
                
                if -1 not in region and len(region) > 0:  # Valid finite region
                    # Keep coordinates as float for precision
                    vertices = [tuple(vor.vertices[i]) for i in region]
                    
                    try:
                        # Create Shapely polygon
                        poly = Polygon(vertices)
                        
                        # Make valid if needed
                        if not poly.is_valid:
                            poly = make_valid(poly)
                        
                        # Clip to image boundaries using Shapely intersection
                        clipped = poly.intersection(bounds)
                        
                        # Handle result (could be Polygon, MultiPolygon, or empty)
                        if clipped.is_empty:
                            continue
                        
                        if isinstance(clipped, MultiPolygon):
                            for geom in clipped.geoms:
                                if geom.area > 1:
                                    cells.append(geom)
                        elif isinstance(clipped, Polygon) and clipped.area > 1:
                            cells.append(clipped)
                            
                    except Exception:
                        continue
            
            return cells
        
        except Exception as e:
            # If Voronoi fails, return empty list (will fall back to no subdivision)
            print(f"\n  Warning: Voronoi generation failed: {e}")
            return []
    
    def clip_voronoi_to_mask(self, voronoi_cells: List[Polygon], mask: np.ndarray, 
                              simplify_tolerance: float = 0.5) -> List[np.ndarray]:
        """
        Clip Voronoi cells to stay within the original color region mask using Shapely.
        Produces smooth, non-self-intersecting contours.
        
        Args:
            voronoi_cells: List of Shapely Polygon objects (Voronoi cells)
            mask: Binary mask of the original region
            simplify_tolerance: Douglas-Peucker simplification tolerance
            
        Returns:
            list: List of clipped contours (numpy arrays) with smooth edges
        """
        clipped_contours = []
        
        # Convert mask to Shapely polygon(s)
        mask_polygons = mask_to_polygon(mask, simplify_tolerance=simplify_tolerance)
        
        if not mask_polygons:
            return clipped_contours
        
        # Union all mask polygons into one geometry for intersection
        if len(mask_polygons) == 1:
            mask_geometry = mask_polygons[0]
        else:
            mask_geometry = unary_union(mask_polygons)
        
        for cell in voronoi_cells:
            try:
                # Intersect Voronoi cell with mask geometry
                clipped = cell.intersection(mask_geometry)
                
                if clipped.is_empty:
                    continue
                
                # Simplify the result
                if simplify_tolerance > 0:
                    clipped = clipped.simplify(simplify_tolerance, preserve_topology=True)
                
                # Handle different geometry types
                if isinstance(clipped, MultiPolygon):
                    for geom in clipped.geoms:
                        if geom.area > 1:
                            contour = polygon_to_contour(geom)
                            if len(contour) >= 3:
                                clipped_contours.append(contour)
                elif isinstance(clipped, GeometryCollection):
                    # Extract polygons from GeometryCollection
                    for geom in clipped.geoms:
                        if isinstance(geom, Polygon) and geom.area > 1:
                            contour = polygon_to_contour(geom)
                            if len(contour) >= 3:
                                clipped_contours.append(contour)
                elif isinstance(clipped, Polygon) and clipped.area > 1:
                    contour = polygon_to_contour(clipped)
                    if len(contour) >= 3:
                        clipped_contours.append(contour)
                        
            except Exception:
                continue
        
        return clipped_contours
    
    def tessellate_region(self, mask: np.ndarray, color_id: int) -> List[Dict]:
        """
        Tessellate a single color region into sub-regions using Voronoi.
        
        Args:
            mask: Binary mask of the region to tessellate
            color_id: ID of the color for this region
            
        Returns:
            list: List of sub-region dictionaries with keys:
                  - 'contour': numpy array of polygon points
                  - 'color_id': original color ID
                  - 'area': area in pixels
                  - 'mask': binary mask of this sub-region
        """
        # Calculate area
        area = np.sum(mask > 0)
        
        if area == 0:
            return []
        
        # Determine number of subdivision points
        num_points = self.calculate_num_points(area)
        
        # If difficulty is 1 or area is too small, don't subdivide
        if num_points == 0 or area < 100:
            # Use Shapely to get smooth contours from the mask
            mask_polygons = mask_to_polygon(mask, simplify_tolerance=0.5)
            
            sub_regions = []
            for poly in mask_polygons:
                if poly.area > 10:
                    contour = polygon_to_contour(poly)
                    if len(contour) >= 3:
                        sub_regions.append({
                            'contour': contour,
                            'color_id': color_id,
                            'area': poly.area
                        })
            
            return sub_regions
        
        # Generate random points inside the region
        points = self.generate_random_points_in_contour(mask, num_points)
        
        if len(points) < 2:
            # Not enough points to tessellate, return original using Shapely
            mask_polygons = mask_to_polygon(mask, simplify_tolerance=0.5)
            if mask_polygons:
                poly = mask_polygons[0]
                contour = polygon_to_contour(poly)
                if len(contour) >= 3:
                    return [{
                        'contour': contour,
                        'color_id': color_id,
                        'area': poly.area
                    }]
            return []
        
        # Create Voronoi cells
        voronoi_cells = self.create_voronoi_cells(points, mask.shape)
        
        # Clip cells to the original mask
        clipped_contours = self.clip_voronoi_to_mask(voronoi_cells, mask)
        
        # Create sub-region dictionaries
        sub_regions = []
        for contour in clipped_contours:
            # Create mask for this sub-region
            sub_mask = np.zeros_like(mask)
            if len(contour.shape) == 1:  # Handle 1D array case
                if len(contour) < 6:  # Need at least 3 points (x,y pairs)
                    continue
                contour = contour.reshape(-1, 2)
            
            # Validate contour has enough points
            if contour.shape[0] < 3:
                continue
            
            # No simplification - preserve exact contours
            # Directly use the contour as-is for maximum precision
            if len(contour.shape) == 1:
                contour = contour.reshape(-1, 2)
            
            if contour.shape[0] < 3:
                continue
            
            try:
                cv2.fillPoly(sub_mask, [contour.astype(np.int32)], 255)
            except:
                continue  # Skip if fillPoly fails
            
            sub_area = np.sum(sub_mask > 0)
            
            if sub_area > 10:  # Ignore tiny sub-regions
                sub_regions.append({
                    'contour': contour,
                    'color_id': color_id,
                    'area': sub_area
                })
            
            # Free sub_mask immediately
            del sub_mask
        
        return sub_regions
    
    def tessellate_all_regions(self, indexed_image: np.ndarray, color_masks: List[np.ndarray], min_component_area: int = 300) -> List[Dict]:
        """
        Tessellate all color regions in the image.
        
        Args:
            indexed_image: 2D array with color IDs
            color_masks: List of binary masks (one per color)
            min_component_area: Minimum area for a component to be processed (pixels)
            
        Returns:
            list: List of all sub-regions from all colors
        """
        all_sub_regions = []
        
        print(f"Starting Voronoi tessellation (difficulty level: {self.difficulty_level})...")
        
        # Count and filter components
        total_components = 0
        filtered_components = 0
        
        for mask in color_masks:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area >= min_component_area:
                    filtered_components += 1
                total_components += 1
        
        print(f"Found {total_components} total components, processing {filtered_components} large ones (>{min_component_area}px²)...")
        
        if total_components > 20000:
            print(f"⚠️  Very high component count ({total_components}). Filtering to manageable size.")
            print(f"  Processing only large components to preserve memory.")
        elif total_components > 10000:
            print(f"  High detail image with {total_components} components (filtering applied).")
        
        if filtered_components > 1000:
            print(f"  Expected processing time: ~{filtered_components // 8} seconds")
        
        processed = 0
        skipped = 0
        
        for color_id, mask in enumerate(color_masks):
            # Find connected components with stats
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            
            # Process each connected component separately (only large ones)
            for label_id in range(1, num_labels):  # Skip 0 (background)
                area = stats[label_id, cv2.CC_STAT_AREA]
                
                # Skip very small components
                if area < min_component_area:
                    skipped += 1
                    # Still add as single region without subdivision
                    component_mask = (labels == label_id).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) > 0:
                        all_sub_regions.append({
                            'contour': contours[0].squeeze(),
                            'color_id': color_id,
                            'area': area
                        })
                    # Free memory immediately
                    del component_mask
                    continue
                
                component_mask = (labels == label_id).astype(np.uint8) * 255
                
                # Tessellate this component
                sub_regions = self.tessellate_region(component_mask, color_id)
                all_sub_regions.extend(sub_regions)
                
                # Free component_mask immediately after use
                del component_mask
                
                processed += 1
                if processed % 25 == 0:  # More frequent updates and cleanup
                    print(f"  Progress: {processed}/{filtered_components} large components (skipped {skipped} small)", flush=True)
                    gc.collect()  # Free memory more frequently
            
            # Free memory after each color
            del mask, labels, stats
            gc.collect()
        
        print(f"✓ Tessellation complete. Created {len(all_sub_regions)} sub-regions.")
        print(f"  Processed: {processed} large components, Skipped: {skipped} small components")
        
        return all_sub_regions
    
    def merge_small_regions(self, sub_regions: List[Dict], image_shape: Tuple[int, int], 
                           min_colorable_area: int = 100) -> List[Dict]:
        """
        Merge regions that are too small to be colored by hand into their neighboring regions.
        Uses a mask-based approach to find neighbors and merge intelligently.
        
        Strategy:
        1. Identify regions smaller than min_colorable_area
        2. For each small region, find adjacent regions using dilation
        3. Prefer merging with neighbor of same color, otherwise largest neighbor
        4. Merge by expanding the neighbor's contour to include the small region
        
        Args:
            sub_regions: List of all sub-regions from tessellation
            image_shape: (height, width) of the image
            min_colorable_area: Minimum area for a region to be colorable (pixels²)
            
        Returns:
            list: Merged sub-regions with small regions absorbed into neighbors
        """
        if not sub_regions:
            return sub_regions
        
        height, width = image_shape
        
        # Separate small and large regions
        small_regions = []
        large_regions = []
        
        for i, region in enumerate(sub_regions):
            area = region.get('area', 0)
            if area < min_colorable_area and area > 0:
                small_regions.append((i, region))
            else:
                large_regions.append(region)
        
        if not small_regions:
            return sub_regions
        
        print(f"  Merging {len(small_regions)} micro-regions (< {min_colorable_area}px²) into neighbors...")
        
        # Create a label map where each pixel has the index of its region
        # -1 = no region assigned
        label_map = np.full((height, width), -1, dtype=np.int32)
        
        # Fill label map with large region indices
        for idx, region in enumerate(large_regions):
            contour = region['contour']
            if contour is None or len(contour) == 0:
                continue
            
            contour_copy = contour.copy()
            if len(contour_copy.shape) == 1:
                if len(contour_copy) < 6:
                    continue
                contour_copy = contour_copy.reshape(-1, 2)
            
            if contour_copy.shape[0] < 3:
                continue
            
            try:
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [contour_copy.astype(np.int32)], 255)
                label_map[mask > 0] = idx
            except:
                continue
        
        # Process each small region
        merged_count = 0
        merged_into_same_color = 0
        
        for _, small_region in small_regions:
            contour = small_region['contour']
            small_color_id = small_region['color_id']
            
            if contour is None or len(contour) == 0:
                continue
            
            contour_copy = contour.copy()
            if len(contour_copy.shape) == 1:
                if len(contour_copy) < 6:
                    continue
                contour_copy = contour_copy.reshape(-1, 2)
            
            if contour_copy.shape[0] < 3:
                continue
            
            try:
                # Create mask for small region
                small_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(small_mask, [contour_copy.astype(np.int32)], 255)
                
                # Dilate to find neighboring regions
                kernel = np.ones((5, 5), np.uint8)
                dilated = cv2.dilate(small_mask, kernel, iterations=2)
                
                # Find neighbor indices in the dilated area (excluding the small region itself)
                neighbor_area = dilated & ~small_mask
                neighbor_labels = label_map[neighbor_area > 0]
                neighbor_labels = neighbor_labels[neighbor_labels >= 0]  # Filter out -1 (no region)
                
                if len(neighbor_labels) == 0:
                    # No neighbors found, keep the small region as-is
                    large_regions.append(small_region)
                    continue
                
                # Count neighbors and find best candidate
                unique_neighbors, counts = np.unique(neighbor_labels, return_counts=True)
                
                # Prefer neighbor with same color_id
                best_neighbor_idx = None
                best_neighbor_count = 0
                same_color_neighbor_idx = None
                same_color_neighbor_count = 0
                
                for neighbor_idx, count in zip(unique_neighbors, counts):
                    neighbor_region = large_regions[neighbor_idx]
                    
                    if neighbor_region['color_id'] == small_color_id:
                        if count > same_color_neighbor_count:
                            same_color_neighbor_idx = neighbor_idx
                            same_color_neighbor_count = count
                    
                    if count > best_neighbor_count:
                        best_neighbor_idx = neighbor_idx
                        best_neighbor_count = count
                
                # Choose same-color neighbor if available and has significant contact
                if same_color_neighbor_idx is not None and same_color_neighbor_count >= best_neighbor_count * 0.3:
                    target_idx = same_color_neighbor_idx
                    merged_into_same_color += 1
                else:
                    target_idx = best_neighbor_idx
                
                if target_idx is None:
                    large_regions.append(small_region)
                    continue
                
                # Merge: expand the target region to include the small region
                target_region = large_regions[target_idx]
                
                # Create combined mask
                target_contour = target_region['contour']
                if target_contour is None or len(target_contour) == 0:
                    large_regions.append(small_region)
                    continue
                
                target_contour_copy = target_contour.copy()
                if len(target_contour_copy.shape) == 1:
                    if len(target_contour_copy) >= 6:
                        target_contour_copy = target_contour_copy.reshape(-1, 2)
                    else:
                        large_regions.append(small_region)
                        continue
                
                if target_contour_copy.shape[0] < 3:
                    large_regions.append(small_region)
                    continue
                
                try:
                    # Use Shapely for proper polygon union (smooth result)
                    target_coords = [(float(pt[0]), float(pt[1])) for pt in target_contour_copy]
                    small_coords = [(float(pt[0]), float(pt[1])) for pt in contour_copy]
                    
                    # Close rings
                    if target_coords[0] != target_coords[-1]:
                        target_coords.append(target_coords[0])
                    if small_coords[0] != small_coords[-1]:
                        small_coords.append(small_coords[0])
                    
                    target_poly = Polygon(target_coords)
                    small_poly = Polygon(small_coords)
                    
                    # Make valid if needed
                    if not target_poly.is_valid:
                        target_poly = make_valid(target_poly)
                    if not small_poly.is_valid:
                        small_poly = make_valid(small_poly)
                    
                    # Union the polygons
                    merged_poly = unary_union([target_poly, small_poly])
                    
                    # Simplify for smooth edges
                    merged_poly = merged_poly.simplify(1.5, preserve_topology=True)
                    
                    if merged_poly.is_empty:
                        large_regions.append(small_region)
                        continue
                    
                    # Handle MultiPolygon or GeometryCollection (take largest)
                    if isinstance(merged_poly, (MultiPolygon, GeometryCollection)):
                        polygons = [g for g in merged_poly.geoms if isinstance(g, Polygon) and not g.is_empty]
                        if polygons:
                            merged_poly = max(polygons, key=lambda g: g.area)
                        else:
                            large_regions.append(small_region)
                            continue
                    
                    if isinstance(merged_poly, Polygon) and merged_poly.area > 1:
                        new_contour = polygon_to_contour(merged_poly)
                        new_area = merged_poly.area
                        
                        # Update target region with merged contour
                        large_regions[target_idx] = {
                            'contour': new_contour,
                            'color_id': target_region['color_id'],
                            'area': new_area
                        }
                        
                        # Update label map
                        combined_mask = np.zeros((height, width), dtype=np.uint8)
                        cv2.fillPoly(combined_mask, [new_contour.astype(np.int32)], 255)
                        label_map[combined_mask > 0] = target_idx
                        
                        merged_count += 1
                    else:
                        # Merge failed, keep small region
                        large_regions.append(small_region)
                except:
                    # Merge failed, keep small region
                    large_regions.append(small_region)
                    
            except Exception as e:
                # If anything fails, keep the small region as-is
                large_regions.append(small_region)
                continue
        
        print(f"  ✓ Merged {merged_count} micro-regions ({merged_into_same_color} into same-color neighbors)")
        print(f"  Regions after merge: {len(large_regions)}")
        
        # Clean up
        del label_map
        gc.collect()
        
        return large_regions
