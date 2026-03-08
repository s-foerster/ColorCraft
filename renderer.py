"""
Module 4 & 5: Rendering Engine and Legend Generator.
Draws contours, symbols, and creates the color legend.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Optional
import os
from datetime import datetime
from scipy import ndimage


class ColoringRenderer:
    """
    Renders the final coloring page with contours and symbols.
    Also generates the color legend.
    """
    
    def __init__(self, config):
        """
        Initialize the renderer with configuration.
        
        Args:
            config: Configuration object with rendering parameters
        """
        self.config = config
        self.output_image = None
        self.legend_image = None
    
    def create_blank_canvas(self, width: int, height: int) -> np.ndarray:
        """
        Create a blank white canvas.
        
        Args:
            width: Canvas width in pixels
            height: Canvas height in pixels
            
        Returns:
            np.ndarray: Blank white image (BGR format)
        """
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        return canvas
    
    def draw_contours(self, canvas: np.ndarray, sub_regions: List[dict], 
                     main_contours: Optional[List[np.ndarray]] = None,
                     min_area: int = 0) -> np.ndarray:
        """
        Draw all contours on the canvas using boundary detection to avoid double lines.
        
        Args:
            canvas: Image to draw on
            sub_regions: List of sub-region dictionaries with 'contour' key
            main_contours: Optional list of main contours (original color boundaries)
            min_area: Minimum area (in pixels) to draw a region. Regions smaller are skipped.
            
        Returns:
            np.ndarray: Canvas with contours drawn
        """
        height, width = canvas.shape[:2]
        
        # Create a region ID map (each pixel gets a unique region ID)
        region_map = np.zeros((height, width), dtype=np.int32)
        region_id = 1
        large_region_ids = set()
        valid_count = 0
        skipped_small = 0
        invalid_count = 0
        
        for region in sub_regions:
            is_small = 'area' in region and region['area'] < min_area
            
            contour = region['contour']
            
            # Validate contour
            if contour is None or len(contour) == 0:
                invalid_count += 1
                continue
            
            # Ensure contour is in correct format
            contour_copy = contour.copy()
            if len(contour_copy.shape) == 1:
                if len(contour_copy) < 6:  # Need at least 3 points (x,y pairs)
                    invalid_count += 1
                    continue
                contour_copy = contour_copy.reshape(-1, 2)
            
            # Check we have at least 3 points for a valid polygon
            if contour_copy.shape[0] < 3:
                invalid_count += 1
                continue
            
            # Create mask for this region and assign region ID
            region_mask = np.zeros((height, width), dtype=np.uint8)
            try:
                cv2.fillPoly(region_mask, [contour_copy.astype(np.int32)], 255)
                region_map[region_mask == 255] = region_id
                if not is_small:
                    large_region_ids.add(region_id)
                    valid_count += 1
                else:
                    skipped_small += 1
                region_id += 1
            except Exception as e:
                invalid_count += 1
                continue
        
        # Detect boundaries: pixels where neighboring pixels have different region IDs
        # Only draw boundaries involving large regions
        boundary_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Check horizontal neighbors
        h_diff = region_map[:, 1:] != region_map[:, :-1]
        h_large = np.zeros_like(h_diff)
        for rid in large_region_ids:
            h_large |= (region_map[:, 1:] == rid) | (region_map[:, :-1] == rid)
        h_boundary = h_diff & h_large
        boundary_mask[:, 1:] |= h_boundary.astype(np.uint8) * 255
        boundary_mask[:, :-1] |= h_boundary.astype(np.uint8) * 255
        
        # Check vertical neighbors
        v_diff = region_map[1:, :] != region_map[:-1, :]
        v_large = np.zeros_like(v_diff)
        for rid in large_region_ids:
            v_large |= (region_map[1:, :] == rid) | (region_map[:-1, :] == rid)
        v_boundary = v_diff & v_large
        boundary_mask[1:, :] |= v_boundary.astype(np.uint8) * 255
        boundary_mask[:-1, :] |= v_boundary.astype(np.uint8) * 255
        
        # Apply line thickness by dilating the boundary mask
        if self.config.line_thickness_sub > 1:
            kernel_size = self.config.line_thickness_sub
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            boundary_mask = cv2.dilate(boundary_mask, kernel, iterations=1)
        
        # Apply boundary as black lines on canvas
        canvas[boundary_mask == 255] = self.config.line_color
        
        if invalid_count > 0:
            print(f"  Skipped {invalid_count} invalid contours ({valid_count} valid)")
        
        if skipped_small > 0:
            print(f"  Skipped {skipped_small} small regions (< {min_area}px²)")
        
        return canvas
    
    def draw_border(self, canvas: np.ndarray) -> np.ndarray:
        """
        Draw a border around the entire image.
        
        Args:
            canvas: Image to draw on
            
        Returns:
            np.ndarray: Canvas with border drawn
        """
        if not self.config.border_enabled:
            return canvas
        
        height, width = canvas.shape[:2]
        thickness = self.config.border_thickness
        color = self.config.border_color
        
        # Draw rectangle border
        cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), 
                     color, thickness, lineType=cv2.LINE_8)
        
        return canvas
    
    def binarize_image(self, image: np.ndarray, threshold: int = 250) -> np.ndarray:
        """
        Binarize image to pure black and white (remove all gray pixels).
        
        Args:
            image: Input image (BGR)
            threshold: Threshold for white (pixels above this become white)
            
        Returns:
            np.ndarray: Binarized image (only pure black and white)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold: pixels > threshold become white (255), others become black (0)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Convert back to BGR
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def is_dark_color(self, color_rgb: Tuple[int, int, int], threshold: int = 80) -> bool:
        """
        Check if a color is considered dark/black.
        
        Args:
            color_rgb: RGB color tuple
            threshold: Maximum brightness to be considered dark (0-255)
            
        Returns:
            bool: True if color is dark
        """
        # Calculate perceived brightness using luminance formula
        r, g, b = color_rgb
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        return brightness < threshold
    
    def prefill_dark_regions(self, canvas: np.ndarray, sub_regions: List[dict],
                            color_palette: List[Tuple[int, int, int]],
                            max_area: int = 500) -> np.ndarray:
        """
        Pre-fill small dark/black regions with black color.
        This avoids tedious coloring of tiny dark areas.
        
        Args:
            canvas: Image to draw on
            sub_regions: List of sub-region dictionaries
            color_palette: List of RGB colors
            max_area: Maximum area for regions to be pre-filled
            
        Returns:
            np.ndarray: Canvas with dark regions pre-filled
        """
        prefilled_count = 0
        
        for region in sub_regions:
            # Check if region is small enough
            if 'area' not in region or region['area'] >= max_area:
                continue
            
            # Check if region is dark colored
            color_id = region.get('color_id', -1)
            if color_id < 0 or color_id >= len(color_palette):
                continue
            
            rgb_color = color_palette[color_id]
            if not self.is_dark_color(rgb_color):
                continue
            
            # Pre-fill this small dark region
            contour = region['contour']
            if contour is None or len(contour) == 0:
                continue
            
            # Format contour
            contour_copy = contour.copy()
            if len(contour_copy.shape) == 1:
                if len(contour_copy) < 6:
                    continue
                contour_copy = contour_copy.reshape(-1, 2)
            
            if contour_copy.shape[0] < 3:
                continue
            
            # Fill with pure black (no gray)
            try:
                cv2.fillPoly(canvas, [contour_copy.astype(np.int32)], (0, 0, 0))
                prefilled_count += 1
            except:
                continue
        
        if prefilled_count > 0:
            print(f"  Pre-filled {prefilled_count} small dark regions")
        
        return canvas
    
    def get_font(self, font_size: int) -> ImageFont.ImageFont:
        """
        Get a font object with the specified size.
        
        Args:
            font_size: Font size in points
            
        Returns:
            ImageFont: Font object
        """
        try:
            # Try to load the specified font
            if os.path.exists(self.config.font_name):
                font = ImageFont.truetype(self.config.font_name, font_size)
            else:
                # Try common font locations
                font_paths = [
                    f"C:/Windows/Fonts/{self.config.font_name}",
                    f"C:/Windows/Fonts/arial.ttf",
                    f"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    f"/System/Library/Fonts/Helvetica.ttc"
                ]
                
                font = None
                for path in font_paths:
                    if os.path.exists(path):
                        font = ImageFont.truetype(path, font_size)
                        break
                
                if font is None:
                    # Fallback to default font
                    font = ImageFont.load_default()
        
        except Exception as e:
            # Fallback to default font
            font = ImageFont.load_default()
        
        return font
    
    def draw_symbols(self, canvas: np.ndarray, symbol_map: List[dict]) -> np.ndarray:
        """
        Draw symbols on the canvas using PIL for better text rendering.
        
        Args:
            canvas: Image to draw on (BGR format)
            symbol_map: List of symbol dictionaries with 'position', 'symbol', 'font_size'
            
        Returns:
            np.ndarray: Canvas with symbols drawn
        """
        # Convert to PIL Image for better text rendering
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(canvas_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        for symbol_info in symbol_map:
            position = symbol_info['position']
            symbol = symbol_info['symbol']
            font_size = symbol_info['font_size']
            
            # Get font
            font = self.get_font(font_size)
            
            # Get text bounding box to center it
            bbox = draw.textbbox((0, 0), symbol, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Calculate centered position
            x = position[0] - text_width // 2
            y = position[1] - text_height // 2
            
            # Draw text with same color as lines
            draw.text((x, y), symbol, fill=self.config.line_color, font=font)
        
        # Convert back to OpenCV format
        canvas_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return canvas_bgr
    
    def render_colored_preview(self, sub_regions: List[dict], color_palette: List[Tuple[int, int, int]],
                              image_shape: Tuple[int, int], min_area: int = 0) -> np.ndarray:
        """
        Create a colored preview showing what the final coloring will look like.
        Each region is filled with its corresponding color.
        
        Args:
            sub_regions: List of all sub-regions
            color_palette: List of RGB colors
            image_shape: (height, width) of original image
            min_area: Minimum area (in pixels) to draw a region. Regions smaller are skipped.
            
        Returns:
            np.ndarray: Colored preview image (BGR format)
        """
        print("Creating colored preview...")
        
        height, width = image_shape
        
        # Create white canvas
        preview = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        colored_count = 0
        skipped_count = 0
        small_filled = 0
        
        # STEP 1: Fill ALL regions with their colors first (including small ones)
        # This ensures no white gaps in the preview
        for region in sub_regions:
            contour = region['contour']
            color_id = region['color_id']
            is_small = 'area' in region and region['area'] < min_area
            
            # Validate contour
            if contour is None or len(contour) == 0:
                skipped_count += 1
                continue
            
            # Get RGB color and convert to BGR
            if color_id >= len(color_palette):
                skipped_count += 1
                continue
            
            rgb_color = color_palette[color_id]
            bgr_color = (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))  # RGB to BGR
            
            # Ensure contour format
            contour_copy = contour.copy()
            if len(contour_copy.shape) == 1:
                if len(contour_copy) < 6:
                    skipped_count += 1
                    continue
                contour_copy = contour_copy.reshape(-1, 2)
            
            if contour_copy.shape[0] < 3:
                skipped_count += 1
                continue
            
            # Fill the region with color directly (without dilation to avoid overflow)
            try:
                cv2.fillPoly(preview, [contour_copy.astype(np.int32)], bgr_color)
                if is_small:
                    small_filled += 1
                else:
                    colored_count += 1
            except Exception as e:
                skipped_count += 1
                continue
        
        print(f"  Filled {colored_count} large regions with colors")
        if small_filled > 0:
            print(f"  Filled {small_filled} small regions (< {min_area}px²) - hidden in coloring page")
        if skipped_count > 0:
            print(f"  Skipped {skipped_count} invalid regions")
        
        # STEP 1.5: Fill any remaining white pixels with nearest neighbor color
        white_mask = np.all(preview == 255, axis=2)
        white_count = np.sum(white_mask)
        if white_count > 0:
            # Create a mask of non-white pixels
            # For each channel, use distance transform to find nearest colored pixel
            indices = ndimage.distance_transform_edt(white_mask, return_distances=False, return_indices=True)
            assert indices is not None
            
            # Fill white pixels with the color of their nearest neighbor
            preview[white_mask] = preview[indices[0][white_mask], indices[1][white_mask]]
            print(f"  Filled {white_count} white gap pixels with nearest neighbor colors")
        
        # STEP 2: Draw region boundaries using edge detection
        # Instead of drawing contours of each region (which causes double lines),
        # we detect pixels at the boundary between different colored regions
        
        # Create a region ID map (each pixel gets a unique region ID)
        region_map = np.zeros((height, width), dtype=np.int32)
        region_id = 1
        large_region_ids = set()
        
        for region in sub_regions:
            contour = region['contour']
            is_small = 'area' in region and region['area'] < min_area
            
            if contour is None or len(contour) == 0:
                continue
            
            contour_copy = contour.copy()
            if len(contour_copy.shape) == 1:
                if len(contour_copy) < 6:
                    continue
                contour_copy = contour_copy.reshape(-1, 2)
            
            if contour_copy.shape[0] < 3:
                continue
            
            # Create mask for this region
            region_mask = np.zeros((height, width), dtype=np.uint8)
            try:
                cv2.fillPoly(region_mask, [contour_copy.astype(np.int32)], 255)
                # Assign region ID to all pixels in this region
                region_map[region_mask == 255] = region_id
                if not is_small:
                    large_region_ids.add(region_id)
                region_id += 1
            except:
                continue
        
        # Detect boundaries: pixels where neighboring pixels have different region IDs
        # Only draw boundaries between large regions (skip small region boundaries)
        boundary_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Check horizontal neighbors
        h_diff = region_map[:, 1:] != region_map[:, :-1]
        # Only keep boundaries where at least one side is a large region
        h_large = np.zeros_like(h_diff)
        for rid in large_region_ids:
            h_large |= (region_map[:, 1:] == rid) | (region_map[:, :-1] == rid)
        h_boundary = h_diff & h_large
        boundary_mask[:, 1:] |= h_boundary.astype(np.uint8) * 255
        boundary_mask[:, :-1] |= h_boundary.astype(np.uint8) * 255
        
        # Check vertical neighbors
        v_diff = region_map[1:, :] != region_map[:-1, :]
        v_large = np.zeros_like(v_diff)
        for rid in large_region_ids:
            v_large |= (region_map[1:, :] == rid) | (region_map[:-1, :] == rid)
        v_boundary = v_diff & v_large
        boundary_mask[1:, :] |= v_boundary.astype(np.uint8) * 255
        boundary_mask[:-1, :] |= v_boundary.astype(np.uint8) * 255
        
        # Apply boundary as black lines on preview
        preview[boundary_mask == 255] = self.config.line_color
        
        boundary_pixels = np.sum(boundary_mask == 255)
        print(f"  Drew {boundary_pixels} boundary pixels (single lines, no doubles)")
        print("✓ Colored preview created")
        
        return preview
    
    def render_colored_preview_no_lines(self, sub_regions: List[dict], color_palette: List[Tuple[int, int, int]],
                              image_shape: Tuple[int, int], min_area: int = 0) -> np.ndarray:
        """
        Create a colored preview WITHOUT boundary lines - just the filled colors.
        Each region is filled with its corresponding color.
        
        Args:
            sub_regions: List of all sub-regions
            color_palette: List of RGB colors
            image_shape: (height, width) of original image
            min_area: Minimum area (in pixels) to draw a region. Regions smaller are skipped.
            
        Returns:
            np.ndarray: Colored preview image without lines (BGR format)
        """
        print("Creating colored preview without lines...")
        
        height, width = image_shape
        
        # Create white canvas
        preview = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        colored_count = 0
        skipped_count = 0
        small_filled = 0
        
        # Fill ALL regions with their colors (including small ones)
        for region in sub_regions:
            contour = region['contour']
            color_id = region['color_id']
            is_small = 'area' in region and region['area'] < min_area
            
            # Validate contour
            if contour is None or len(contour) == 0:
                skipped_count += 1
                continue
            
            # Get RGB color and convert to BGR
            if color_id >= len(color_palette):
                skipped_count += 1
                continue
            
            rgb_color = color_palette[color_id]
            bgr_color = (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))  # RGB to BGR
            
            # Ensure contour format
            contour_copy = contour.copy()
            if len(contour_copy.shape) == 1:
                if len(contour_copy) < 6:
                    skipped_count += 1
                    continue
                contour_copy = contour_copy.reshape(-1, 2)
            
            if contour_copy.shape[0] < 3:
                skipped_count += 1
                continue
            
            # Fill the region with color
            try:
                cv2.fillPoly(preview, [contour_copy.astype(np.int32)], bgr_color)
                if is_small:
                    small_filled += 1
                else:
                    colored_count += 1
            except Exception as e:
                skipped_count += 1
                continue
        
        print(f"  Filled {colored_count} large regions with colors")
        if small_filled > 0:
            print(f"  Filled {small_filled} small regions (< {min_area}px²)")
        if skipped_count > 0:
            print(f"  Skipped {skipped_count} invalid regions")
        
        # Fill any remaining white pixels with nearest neighbor color
        white_mask = np.all(preview == 255, axis=2)
        white_count = np.sum(white_mask)
        if white_count > 0:
            indices = ndimage.distance_transform_edt(white_mask, return_distances=False, return_indices=True)
            assert indices is not None
            preview[white_mask] = preview[indices[0][white_mask], indices[1][white_mask]]
            print(f"  Filled {white_count} white gap pixels with nearest neighbor colors")
        
        print("✓ Colored preview (no lines) created")
        
        return preview
    
    def create_legend(self, color_palette: List[Tuple[int, int, int]], 
                     symbol_type: str, custom_symbols: Optional[List[str]] = None) -> np.ndarray:
        """
        Create a color legend showing color-to-symbol mapping.
        
        Args:
            color_palette: List of RGB color tuples
            symbol_type: Type of symbols used
            custom_symbols: Custom symbols list (if applicable)
            
        Returns:
            np.ndarray: Legend image (BGR format)
        """
        num_colors = len(color_palette)
        # Single line legend: all colors on one row
        columns = num_colors
        rows = 1
        
        # Cell dimensions - compact for single line
        cell_width = 80
        cell_height = 70
        
        # Legend dimensions - no extra space for title
        legend_width = cell_width * columns
        legend_height = cell_height
        
        # Create blank legend
        legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255
        
        # Convert to PIL for better text rendering
        legend_rgb = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
        pil_legend = Image.fromarray(legend_rgb)
        draw = ImageDraw.Draw(pil_legend)
        
        # Draw each color cell
        for i, color_rgb in enumerate(color_palette):
            row = i // columns
            col = i % columns
            
            x_start = col * cell_width + 5
            y_start = 5
            
            # Draw color circle
            circle_radius = 18
            circle_center = (x_start + cell_width // 2, y_start + 22)
            
            # Draw filled circle
            draw.ellipse(
                [
                    circle_center[0] - circle_radius,
                    circle_center[1] - circle_radius,
                    circle_center[0] + circle_radius,
                    circle_center[1] + circle_radius
                ],
                fill=color_rgb,
                outline=(0, 0, 0),
                width=2
            )
            
            # Get symbol for this color
            from symbol_placement import SymbolPlacer
            placer = SymbolPlacer()
            symbol = placer.get_symbol_for_color(i, symbol_type, custom_symbols)
            
            # Draw symbol below circle
            symbol_font = self.get_font(14)
            bbox = draw.textbbox((0, 0), symbol, font=symbol_font)
            text_width = bbox[2] - bbox[0]
            
            symbol_x = x_start + (cell_width - text_width) // 2
            symbol_y = y_start + 45
            
            draw.text((symbol_x, symbol_y), symbol, fill=(0, 0, 0), font=symbol_font)
        
        # Convert back to OpenCV format
        legend_bgr = cv2.cvtColor(np.array(pil_legend), cv2.COLOR_RGB2BGR)
        
        return legend_bgr
    
    def combine_with_legend(self, coloring_page: np.ndarray, legend: np.ndarray,
                           position: str = "bottom") -> np.ndarray:
        """
        Combine the coloring page with the legend.
        
        Args:
            coloring_page: Main coloring page image
            legend: Legend image
            position: 'bottom' to place below, 'separate' to keep separate
            
        Returns:
            np.ndarray: Combined image (if position='bottom') or just coloring_page
        """
        if position == "separate":
            # Keep them separate (will be saved as different files)
            return coloring_page
        
        # Combine vertically (legend at bottom)
        page_height, page_width = coloring_page.shape[:2]
        legend_height, legend_width = legend.shape[:2]
        
        # Resize legend to match page width if necessary
        if legend_width != page_width:
            aspect_ratio = legend_height / legend_width
            new_height = int(page_width * aspect_ratio)
            legend = cv2.resize(legend, (page_width, new_height))
            legend_height = new_height
        
        # Create combined image
        combined_height = page_height + legend_height
        combined = np.ones((combined_height, page_width, 3), dtype=np.uint8) * 255
        
        # Place coloring page at top
        combined[:page_height, :] = coloring_page
        
        # Place legend at bottom
        combined[page_height:, :] = legend
        
        return combined
    
    def render(self, sub_regions: List[dict], symbol_map: List[dict],
              color_palette: List[Tuple[int, int, int]], 
              image_shape: Tuple[int, int],
              main_contours: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete rendering pipeline.
        
        Args:
            sub_regions: List of all sub-regions
            symbol_map: List of symbol placements
            color_palette: List of RGB colors
            image_shape: (height, width) of original image
            main_contours: Optional main contours to emphasize
            
        Returns:
            tuple: (coloring_page, legend, colored_preview, colored_preview_no_lines)
            - all as BGR numpy arrays
        """
        print("Rendering coloring page...")
        
        height, width = image_shape
        
        # Create blank canvas
        canvas = self.create_blank_canvas(width, height)
        
        # Pre-fill small dark regions if enabled
        if self.config.prefill_dark_regions and self.config.prefill_dark_threshold > 0:
            canvas = self.prefill_dark_regions(canvas, sub_regions, color_palette,
                                              max_area=self.config.prefill_dark_threshold)
        
        # Draw contours (filter out small regions)
        # Note: main_contours are not drawn separately as sub_regions already cover all boundaries
        canvas = self.draw_contours(canvas, sub_regions, None, 
                                   min_area=self.config.min_region_area)
        
        # Draw symbols with anti-aliasing for smooth rendering
        canvas = self.draw_symbols(canvas, symbol_map)
        
        # Draw border around the final image
        canvas = self.draw_border(canvas)
        
        # Note: We don't binarize the final image to preserve symbol anti-aliasing
        # Contours use LINE_8 (no anti-aliasing) so they remain pure black
        # Symbols use PIL rendering with anti-aliasing for smooth text
        
        self.output_image = canvas
        
        print("Creating color legend...")
        
        # Create legend
        legend = self.create_legend(
            color_palette,
            self.config.symbol_type,
            self.config.custom_symbols
        )
        
        self.legend_image = legend
        
        # Create colored preview (with lines)
        colored_preview = self.render_colored_preview(
            sub_regions,
            color_palette,
            image_shape,
            min_area=self.config.min_region_area
        )
        
        # Create colored preview without lines
        colored_preview_no_lines = self.render_colored_preview_no_lines(
            sub_regions,
            color_palette,
            image_shape,
            min_area=self.config.min_region_area
        )
        
        print("Rendering complete!")
        
        return canvas, legend, colored_preview, colored_preview_no_lines
    
    def save_output(self, coloring_page: np.ndarray, legend: np.ndarray, colored_preview: np.ndarray,
                   original_image: np.ndarray, output_name: str = "mystery_coloring",
                    colored_preview_no_lines: Optional[np.ndarray] = None) -> Tuple[str, str, str]:
        """
        Save the output images to files with unique names to avoid overwriting.
        
        Args:
            coloring_page: Coloring page image
            legend: Legend image
            colored_preview: Colored preview image (with lines)
            original_image: Source image after resizing
            output_name: Base name for output files
            colored_preview_no_lines: Colored preview image without boundary lines
            
        Returns:
            tuple: (coloring_page_path, legend_path, preview_path)
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Add timestamp to make filenames unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_name = f"{output_name}_{timestamp}"
        
        # Determine output paths with timestamp
        coloring_path = os.path.join(self.config.output_dir, f"{unique_name}_page.png")
        legend_path = os.path.join(self.config.output_dir, f"{unique_name}_legend.png")
        preview_path = os.path.join(self.config.output_dir, f"{unique_name}_preview.png")
        preview_no_lines_path = os.path.join(self.config.output_dir, f"{unique_name}_preview_no_lines.png")
        source_path = os.path.join(self.config.output_dir, f"{unique_name}_source.png")
        
        # Combine if needed
        if self.config.legend_position == "bottom":
            combined = self.combine_with_legend(coloring_page, legend, "bottom")
            combined_path = os.path.join(self.config.output_dir, f"{unique_name}_combined.png")
            cv2.imwrite(combined_path, combined)
            print(f"Saved combined page: {combined_path}")
        
        # Save individual files
        cv2.imwrite(coloring_path, coloring_page)
        cv2.imwrite(legend_path, legend)
        cv2.imwrite(preview_path, colored_preview)
        cv2.imwrite(source_path, original_image)
        
        # Save preview without lines if provided
        if colored_preview_no_lines is not None:
            cv2.imwrite(preview_no_lines_path, colored_preview_no_lines)
            print(f"Saved preview (no lines): {preview_no_lines_path}")
        
        print(f"Saved coloring page: {coloring_path}")
        print(f"Saved legend: {legend_path}")
        print(f"Saved colored preview: {preview_path}")
        print(f"Saved source image: {source_path}")
        
        return coloring_path, legend_path, preview_path
