"""
Module 1: Color Quantization using K-Means Clustering.
Reduces the input image to N dominant colors and creates an indexed color map.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import generic_filter
from typing import Tuple, List


class ColorQuantizer:
    """
    Color quantization using K-Means clustering to reduce image to N dominant colors.
    Handles anti-aliasing artifacts on edges.
    """
    
    def __init__(self, num_colors: int = 16, forced_colors: List[Tuple[int, int, int]] = None):
        """
        Initialize the color quantizer.
        
        Args:
            num_colors: Number of colors to reduce the image to
            forced_colors: Optional list of RGB colors that must be in the final palette
        """
        self.num_colors = num_colors
        self.forced_colors = forced_colors if forced_colors else []
        self.color_palette: np.ndarray = None
        self.kmeans: KMeans = None
        
    def apply_bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filter to sharpen color edges while preserving flat regions.
        This helps eliminate anti-aliasing artifacts before quantization.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Filtered image with sharper color boundaries
        """
        # Bilateral filter: smooths within regions but preserves edges
        # d=9: diameter of pixel neighborhood
        # sigmaColor=75: filter sigma in the color space (larger = more colors mixed)
        # sigmaSpace=75: filter sigma in coordinate space (larger = farther pixels influence)
        filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        return filtered
    
    def apply_mode_filter(self, indexed_image: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Apply mode filter to remove gradient/transition pixels.
        Each pixel is replaced by the most common color in its neighborhood.
        This forces hard color transitions instead of gradient bands.
        
        Args:
            indexed_image: 2D array with color IDs (0 to N-1)
            window_size: Size of the neighborhood window (must be odd). Default 5.
            
        Returns:
            Cleaned indexed image with hard color boundaries
        """
        if window_size <= 1:
            return indexed_image
        
        # Ensure window size is odd
        if window_size % 2 == 0:
            window_size += 1
        
        def local_mode(values):
            """Find the most common value in the neighborhood."""
            vals, counts = np.unique(values.astype(int), return_counts=True)
            return vals[np.argmax(counts)]
        
        # Apply mode filter - each pixel becomes the majority color in its neighborhood
        cleaned = generic_filter(
            indexed_image.astype(np.float64),
            local_mode,
            size=window_size,
            mode='nearest'
        )
        
        return cleaned.astype(indexed_image.dtype)
        
    def quantize(self, image: np.ndarray, apply_bilateral: bool = True, 
                 mode_filter_size: int = 5) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int]]]:
        """
        Quantize the input image to N dominant colors using K-Means clustering.
        
        Args:
            image: Input image in BGR format (OpenCV format)
            apply_bilateral: Apply bilateral filter before quantization to sharpen edges
            mode_filter_size: Size of mode filter for post-quantization cleanup (0 to disable)
            
        Returns:
            tuple: (indexed_image, quantized_image, color_palette)
                - indexed_image: 2D array where each pixel has value 0 to N-1 (color ID)
                - quantized_image: Image with only the N dominant colors (BGR format)
                - color_palette: List of RGB tuples representing each color ID
        """
        # Apply bilateral filter to sharpen color edges before quantization
        if apply_bilateral:
            print("  - Applying bilateral filter to sharpen color edges...")
            image = self.apply_bilateral_filter(image)
        
        # Convert BGR to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Reshape image to be a list of pixels
        pixels = image_rgb.reshape(-1, 3)
        
        # Prepare initial cluster centers if forced colors are specified
        init_centers = 'k-means++'
        if self.forced_colors:
            num_forced = len(self.forced_colors)
            if num_forced >= self.num_colors:
                raise ValueError(f"Cannot force {num_forced} colors with only {self.num_colors} total colors")
            
            print(f"Forcing {num_forced} colors in the palette: {self.forced_colors}")
            # Use forced colors as initial centers, let K-means find the rest
            init_centers = np.array(self.forced_colors, dtype=np.float32)
            # Add random samples for remaining centers
            random_indices = np.random.choice(len(pixels), self.num_colors - num_forced, replace=False)
            random_centers = pixels[random_indices].astype(np.float32)
            init_centers = np.vstack([init_centers, random_centers])
        
        # Apply K-Means clustering
        print(f"Applying K-Means clustering with {self.num_colors} colors...")
        self.kmeans = KMeans(
            n_clusters=self.num_colors,
            init=init_centers,
            n_init=1 if self.forced_colors else 10,
            random_state=42,
            max_iter=300
        )
        
        # Fit and predict
        labels = self.kmeans.fit_predict(pixels)
        
        # Get the color palette (cluster centers)
        self.color_palette = self.kmeans.cluster_centers_.astype(np.uint8)
        
        # Create indexed image (2D array with color IDs)
        indexed_image = labels.reshape(image.shape[0], image.shape[1])
        
        # Apply mode filter to eliminate gradient/transition zones
        if mode_filter_size > 1:
            print(f"  - Applying mode filter (window={mode_filter_size}) to eliminate transition zones...")
            indexed_image = self.apply_mode_filter(indexed_image, mode_filter_size)
        
        # Create quantized image (image with only N colors) - using cleaned indexed_image
        quantized_image_rgb = self.color_palette[indexed_image]
        
        # Convert back to BGR for OpenCV
        quantized_image_bgr = cv2.cvtColor(quantized_image_rgb, cv2.COLOR_RGB2BGR)
        
        # Convert palette to RGB tuples for easier use
        color_palette_list = [tuple(color) for color in self.color_palette]
        
        print(f"Color quantization complete. Found {len(color_palette_list)} unique colors.")
        
        return indexed_image, quantized_image_bgr, color_palette_list
    
    def get_color_for_id(self, color_id: int) -> Tuple[int, int, int]:
        """
        Get the RGB color for a given color ID.
        
        Args:
            color_id: Color ID (0 to N-1)
            
        Returns:
            tuple: RGB color tuple
        """
        if self.color_palette is None:
            raise ValueError("Color palette not initialized. Run quantize() first.")
        
        if color_id < 0 or color_id >= self.num_colors:
            raise ValueError(f"Color ID must be between 0 and {self.num_colors - 1}")
        
        return tuple(self.color_palette[color_id])
    
    def get_color_masks(self, indexed_image: np.ndarray) -> List[np.ndarray]:
        """
        Create binary masks for each color in the indexed image.
        
        Args:
            indexed_image: 2D array with color IDs (0 to N-1)
            
        Returns:
            list: List of binary masks (one per color ID)
        """
        masks = []
        
        for color_id in range(self.num_colors):
            # Create binary mask where True = this color ID
            mask = (indexed_image == color_id).astype(np.uint8) * 255
            masks.append(mask)
        
        return masks
    
    def clean_indexed_image(self, indexed_image: np.ndarray, kernel_size: int = 3, min_area: int = 150) -> np.ndarray:
        """
        Light clean to preserve contours with high precision.
        Only removes very small isolated noise.
        
        Args:
            indexed_image: 2D array with color IDs
            kernel_size: Size of the morphological kernel (small)
            min_area: Minimum area for a region
            
        Returns:
            np.ndarray: Cleaned indexed image
        """
        print(f"  - Light cleaning for high precision (kernel={kernel_size}, min_area={min_area})...")
        
        cleaned = indexed_image.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # For each color ID, very light cleaning
        for color_id in range(self.num_colors):
            mask = (indexed_image == color_id).astype(np.uint8)
            
            if np.sum(mask) == 0:
                continue
            
            # Very light closing (just 1 iteration)
            mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # Light opening to remove single-pixel noise only
            mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Remove only small connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_cleaned, connectivity=8)
            
            # Keep components above minimum area
            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area >= min_area:
                    component_mask = (labels == label_id)
                    cleaned[component_mask] = color_id
                else:
                    # Mark small regions for removal
                    component_mask = (labels == label_id)
                    cleaned[component_mask] = -1
            
            # Free memory after each color
            del mask, mask_cleaned, labels, stats
        
        # Fill removed regions with nearest neighbor
        mask_to_fill = (cleaned == -1)
        if np.any(mask_to_fill):
            for color_id in range(self.num_colors):
                color_mask = (cleaned == color_id).astype(np.uint8)
                if np.sum(color_mask) == 0:
                    continue
                # Light dilation (just 2 iterations)
                dilated = cv2.dilate(color_mask, kernel, iterations=2)
                cleaned[mask_to_fill & (dilated > 0)] = color_id
                del color_mask, dilated
            
            cleaned[cleaned == -1] = 0
        
        print(f"  - Cleaning complete (high precision contours)")
        
        return cleaned
    
    def visualize_palette(self, width: int = 800, height: int = 100) -> np.ndarray:
        """
        Create a visual representation of the color palette.
        
        Args:
            width: Width of the output image
            height: Height of the output image
            
        Returns:
            np.ndarray: Image showing the color palette
        """
        if self.color_palette is None:
            raise ValueError("Color palette not initialized. Run quantize() first.")
        
        palette_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        color_width = width // self.num_colors
        
        for i, color in enumerate(self.color_palette):
            x_start = i * color_width
            x_end = (i + 1) * color_width if i < self.num_colors - 1 else width
            palette_img[:, x_start:x_end] = color[::-1]  # RGB to BGR
        
        return palette_img


def load_and_quantize_image(image_path: str, num_colors: int = 16) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int]], ColorQuantizer]:
    """
    Convenience function to load an image and quantize it in one step.
    
    Args:
        image_path: Path to the input image
        num_colors: Number of colors to reduce to
        
    Returns:
        tuple: (indexed_image, quantized_image, color_palette, quantizer)
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    print(f"Loaded image: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Create quantizer and process
    quantizer = ColorQuantizer(num_colors=num_colors)
    indexed_image, quantized_image, color_palette = quantizer.quantize(image)
    
    # Clean up anti-aliasing artifacts
    indexed_image = quantizer.clean_indexed_image(indexed_image)
    
    return indexed_image, quantized_image, color_palette, quantizer
