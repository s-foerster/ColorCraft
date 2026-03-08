"""
Configuration module for Mystery Coloring Generator.
Contains all configurable parameters for the generation process.
"""

from typing import List, Literal, Tuple


class Config:
    """Configuration class for Mystery Coloring Generator."""
    
    def __init__(self):
        """Initialize configuration with default values."""
        
        # Image Processing
        self.max_image_dimension: int = 2500
        """Maximum dimension (width or height) for image processing.
        Higher values = better resolution and finer lines, but more memory usage.
        Recommended values: 1400-2000 for standard, 2500-3500 for high quality."""
        
        # Module 1: Color Quantization
        self.num_colors: int = 16
        """Number of colors to reduce the image to (K-means clustering)"""
        
        self.mode_filter_size: int = 5
        """Size of the mode filter window for post-quantization cleanup.
        Removes gradient/transition pixels by replacing each pixel with the
        most common color in its neighborhood. Larger values = more aggressive
        cleanup but may lose fine details. Recommended: 3-7. Set to 0 to disable."""
        
        self.bilateral_filter_enabled: bool = True
        """Apply bilateral filter before quantization to sharpen color edges
        while preserving flat color regions. Helps eliminate anti-aliasing artifacts."""
        
        self.forced_colors: List[Tuple[int, int, int]] = []
        """List of RGB colors that must be present in the final palette.
        These colors will be used to initialize K-means cluster centers.
        Example: [(0, 0, 0), (255, 255, 255)] forces black and white.
        Maximum: num_colors - 1 (at least one color must be auto-detected)"""
        
        # Module 2: Voronoi Tessellation & Difficulty
        self.difficulty_level: int = 5
        """Difficulty level (1 = no subdivision, 10 = 1000 pieces puzzle)
        Determines the density of Voronoi cutting points:
        - Difficulty 1: No cutting (original zones preserved)
        - Difficulty 10: 50 points generated per 1000px²
        """
        
        self.points_per_1000px2: float = 5.0
        """Base number of Voronoi points per 1000px² (scales with difficulty)"""
        
        # Module 3: Symbol Placement
        self.min_region_area: int = 50
        """Minimum area (in pixels) for a region to be drawn and receive a symbol.
        Regions smaller than this will be completely ignored (not drawn, no symbol).
        Increase (100-200) to reduce visual clutter and avoid tiny regions."""
        
        self.min_colorable_area: int = 100
        """Minimum area (in pixels) for a region to remain as a separate colorable zone.
        Regions smaller than this will be merged into neighboring regions.
        This prevents tiny zones that are impractical to color by hand.
        Recommended: 80-150 for detailed images, 150-300 for simpler coloring."""
        
        self.prefill_dark_regions: bool = True
        """Pre-fill small dark/black regions on the coloring page.
        When enabled, small dark regions (< prefill_dark_threshold) are filled in black
        to avoid tedious coloring of tiny dark areas."""
        
        self.prefill_dark_threshold: int = 500
        """Maximum area (in pixels) for dark regions to be pre-filled.
        Dark regions smaller than this will be filled in black automatically.
        Set to 0 to disable pre-filling, or increase to fill larger dark areas."""
        
        self.symbol_type: Literal["numbers", "letters", "custom"] = "numbers"
        """Type of symbols to use: 'numbers', 'letters', or 'custom'"""
        
        self.custom_symbols: List[str] = ["@", "#", "€", "★", "♦", "♣", "♥", "♠"]
        """Custom symbols list (used when symbol_type='custom')"""
        
        # Module 4: Rendering
        self.line_thickness_main: int = 1
        """Thickness of main contours from the original image (in pixels)"""
        
        self.line_thickness_sub: int = 1
        """Thickness of Voronoi subdivision lines (in pixels).
        Use 1 for fine lines, increase for bolder appearance."""
        
        self.line_color: tuple = (150, 150, 150)
        """Color of lines (RGB). Default is gray."""
        
        self.border_enabled: bool = True
        """Draw a border around the final image."""
        
        self.border_thickness: int = 3
        """Thickness of the border (in pixels)."""
        
        self.border_color: tuple = (100, 100, 100)
        """Color of the border (RGB). Darker gray than lines."""
        
        self.background_color: tuple = (255, 255, 255)
        """Background color (RGB). Default is white."""
        
        self.font_name: str = "arial.ttf"
        """Font name for symbols. Should be sans-serif (e.g., Arial, Helvetica)"""
        
        self.font_size_ratio: float = 0.8
        """Symbol size as ratio of region diameter (0.0 to 1.0)"""
        
        # Module 5: Output
        self.output_format: str = "A4"
        """Output page format: 'A4', 'Letter', or custom (width, height) in mm"""
        
        self.dpi: int = 600
        """Output resolution in DPI"""
        
        self.legend_position: Literal["bottom", "separate"] = "bottom"
        """Where to place the legend: 'bottom' (same page) or 'separate' (new page)"""
        
        self.legend_columns: int = 8
        """Number of columns in the legend grid"""
        
        # File paths
        self.input_image_path: str = ""
        """Path to the input colored image"""
        
        self.output_dir: str = "output"
        """Directory for output files"""
        
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        if not (1 <= self.difficulty_level <= 10):
            raise ValueError("difficulty_level must be between 1 and 10")
        
        if self.num_colors < 2 or self.num_colors > 256:
            raise ValueError("num_colors must be between 2 and 256")
        
        if self.min_region_area < 0:
            raise ValueError("min_region_area must be positive")
        
        if not (0.1 <= self.font_size_ratio <= 1.0):
            raise ValueError("font_size_ratio must be between 0.1 and 1.0")
        
        if self.dpi < 72:
            raise ValueError("dpi must be at least 72")
        
        return True
    
    def get_a4_size_px(self) -> tuple:
        """
        Get A4 size in pixels based on DPI.
        
        Returns:
            tuple: (width, height) in pixels
        """
        # A4 size in mm: 210 x 297
        mm_to_inch = 0.0393701
        width_inch = 210 * mm_to_inch
        height_inch = 297 * mm_to_inch
        
        width_px = int(width_inch * self.dpi)
        height_px = int(height_inch * self.dpi)
        
        return (width_px, height_px)
    
    def get_points_for_area(self, area_px2: float) -> int:
        """
        Calculate number of Voronoi points for a given area based on difficulty.
        
        Args:
            area_px2: Area in square pixels
            
        Returns:
            int: Number of points to generate
        """
        # Scale points with difficulty level
        base_density = self.points_per_1000px2 * (self.difficulty_level / 5.0)
        num_points = int((area_px2 / 1000.0) * base_density)
        
        # Minimum 1 point, even for small areas at high difficulty
        return max(1, num_points) if self.difficulty_level > 1 else 0


# Default configuration instance
DEFAULT_CONFIG = Config()
