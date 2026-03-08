"""
Mystery Coloring Generator - Main Entry Point
Orchestrates all modules to generate mystery coloring pages from colored images.
"""

import cv2
import numpy as np
import os
import argparse
from typing import Dict, List, Optional, Tuple

from config import Config
from color_quantization import ColorQuantizer
from voronoi_tessellation import VoronoiTessellator
from symbol_placement import SymbolPlacer
from renderer import ColoringRenderer


class MysteryColoringGenerator:
    """
    Main class that orchestrates the entire mystery coloring generation pipeline.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the generator with configuration.
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config if config is not None else Config()
        
        # Validate configuration
        self.config.validate()
        
        # Initialize modules
        self.quantizer: Optional[ColorQuantizer] = None
        self.tessellator: Optional[VoronoiTessellator] = None
        self.symbol_placer: Optional[SymbolPlacer] = None
        self.renderer: Optional[ColoringRenderer] = None
        
        # Data storage
        self.original_image: Optional[np.ndarray] = None
        self.indexed_image: Optional[np.ndarray] = None
        self.quantized_image: Optional[np.ndarray] = None
        self.color_palette: Optional[List[Tuple[int, int, int]]] = None
        self.sub_regions: Optional[List[dict]] = None
        self.symbol_map: Optional[List[dict]] = None

    def _prepare_loaded_image(self, image: np.ndarray, source_name: str) -> np.ndarray:
        """
        Normalize and resize a loaded image before processing.

        Args:
            image: Loaded OpenCV image
            source_name: Human-readable source label for logging

        Returns:
            np.ndarray: Prepared BGR image
        """
        print(f"\n{'='*60}")
        print("MYSTERY COLORING GENERATOR")
        print(f"{'='*60}\n")

        print("Step 1/5: Loading image...")

        if image is None or image.size == 0:
            raise ValueError(f"Could not decode image: {source_name}")

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Unsupported image format for: {source_name}")

        original_size = (image.shape[1], image.shape[0])
        print(f"  - Source: {source_name}")
        print(f"✓ Image loaded: {original_size[0]}x{original_size[1]} pixels")

        max_dimension = self.config.max_image_dimension
        height, width = image.shape[:2]

        if width > max_dimension or height > max_dimension:
            scale = max_dimension / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)

            print(f"  Resizing for optimization: {new_width}x{new_height} ({scale:.1%} scale)")
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        self.original_image = image.copy()

        print()

        return image

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load the input image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            np.ndarray: Loaded image
        """
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        return self._prepare_loaded_image(image, os.path.basename(image_path))

    def load_image_bytes(self, image_bytes: bytes, source_name: str = "uploaded_image") -> np.ndarray:
        """
        Load an image from raw bytes.

        Args:
            image_bytes: Raw image bytes
            source_name: Human-readable source label for logging

        Returns:
            np.ndarray: Loaded image
        """
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ValueError(f"Could not decode uploaded image: {source_name}")

        return self._prepare_loaded_image(image, source_name)
    
    def quantize_colors(self, image: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Apply color quantization (Module 1).
        
        Args:
            image: Input image
            
        Returns:
            tuple: (indexed_image, color_palette)
        """
        print(f"Step 2/5: Color Quantization (K-Means)...")
        print(f"  - Target colors: {self.config.num_colors}")
        
        self.quantizer = ColorQuantizer(
            num_colors=self.config.num_colors,
            forced_colors=self.config.forced_colors
        )
        self.indexed_image, self.quantized_image, self.color_palette = self.quantizer.quantize(
            image,
            apply_bilateral=self.config.bilateral_filter_enabled,
            mode_filter_size=self.config.mode_filter_size
        )
        
        # Light cleaning to preserve contours with high precision
        # Adaptive parameters based on number of colors
        if self.config.num_colors <= 8:
            kernel = 3
            min_area = 150
        elif self.config.num_colors <= 12:
            kernel = 3
            min_area = 200
        else:  # 16+ colors
            kernel = 4
            min_area = 250
        
        self.indexed_image = self.quantizer.clean_indexed_image(
            self.indexed_image, 
            kernel_size=kernel,
            min_area=min_area
        )
        
        print(f"✓ Color quantization complete\n")
        
        return self.indexed_image, self.color_palette
    
    def tessellate_regions(self) -> List[dict]:
        """
        Apply Voronoi tessellation (Module 2).
        
        Returns:
            list: Sub-regions after tessellation
        """
        print(f"Step 3/5: Voronoi Tessellation...")
        print(f"  - Difficulty level: {self.config.difficulty_level}/10")

        assert self.quantizer is not None
        assert self.indexed_image is not None
        
        # Get color masks
        color_masks = self.quantizer.get_color_masks(self.indexed_image)
        
        # Initialize tessellator
        self.tessellator = VoronoiTessellator(
            difficulty_level=self.config.difficulty_level,
            points_per_1000px2=self.config.points_per_1000px2
        )
        
        # Calculate adaptive min_component_area based on image resolution
        # Higher resolution = higher threshold to avoid too many tiny components
        image_area = self.indexed_image.shape[0] * self.indexed_image.shape[1]
        # Base threshold: 300px² for 1400x1000 image (1.4M pixels)
        # Scale proportionally with image area
        base_area = 1400 * 1000  # Reference resolution
        min_component_area = int(300 * (image_area / base_area))
        min_component_area = max(200, min(min_component_area, 800))  # Clamp between 200-800
        
        print(f"  - Min component area: {min_component_area}px² (adaptive based on resolution)")
        
        # Tessellate all regions
        self.sub_regions = self.tessellator.tessellate_all_regions(
            self.indexed_image,
            color_masks,
            min_component_area=min_component_area
        )
        
        # Merge micro-regions that are too small to color by hand
        if self.config.min_colorable_area > 0:
            image_shape = (self.indexed_image.shape[0], self.indexed_image.shape[1])
            self.sub_regions = self.tessellator.merge_small_regions(
                self.sub_regions,
                image_shape,
                min_colorable_area=self.config.min_colorable_area
            )
        
        print(f"✓ Tessellation complete\n")
        
        return self.sub_regions
    
    def place_symbols(self) -> List[dict]:
        """
        Calculate optimal symbol placements (Module 3).
        
        Returns:
            list: Symbol placement information
        """
        print(f"Step 4/5: Symbol Placement (Pole of Inaccessibility)...")
        print(f"  - Minimum region area: {self.config.min_region_area}px²")
        print(f"  - Symbol type: {self.config.symbol_type}")

        assert self.sub_regions is not None
        
        # Initialize symbol placer
        self.symbol_placer = SymbolPlacer(
            min_region_area=self.config.min_region_area,
            font_size_ratio=self.config.font_size_ratio
        )
        
        # Find placements
        placements = self.symbol_placer.place_symbols_in_regions(self.sub_regions)
        
        # Create symbol map with actual symbols
        self.symbol_map = self.symbol_placer.create_symbol_map(
            placements,
            self.config.symbol_type,
            self.config.custom_symbols
        )
        
        print(f"✓ Symbol placement complete\n")
        
        return self.symbol_map
    
    def render_output(self, image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Render the final coloring page, legend, and colored preview (Modules 4 & 5).
        
        Args:
            image_shape: Original image shape (height, width)
            
        Returns:
            tuple: (coloring_page, legend, colored_preview, colored_preview_no_lines)
        """
        print(f"Step 5/5: Rendering...")

        assert self.indexed_image is not None
        assert self.sub_regions is not None
        assert self.symbol_map is not None
        assert self.color_palette is not None
        
        # Initialize renderer
        self.renderer = ColoringRenderer(self.config)
        
        # Extract main contours from indexed image for emphasis
        main_contours = []
        for color_id in range(self.config.num_colors):
            mask = (self.indexed_image == color_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            main_contours.extend(contours)
        
        # Render
        coloring_page, legend, colored_preview, colored_preview_no_lines = self.renderer.render(
            self.sub_regions,
            self.symbol_map,
            self.color_palette,
            image_shape,
            main_contours
        )
        
        print(f"✓ Rendering complete\n")
        
        return coloring_page, legend, colored_preview, colored_preview_no_lines

    def generate_images(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run the full pipeline on a pre-loaded image and keep outputs in memory.

        Args:
            image: Input image already loaded as a BGR numpy array

        Returns:
            dict: Generated images keyed by output type
        """
        image_shape = (image.shape[0], image.shape[1])

        self.quantize_colors(image)
        self.tessellate_regions()
        self.place_symbols()

        coloring_page, legend, colored_preview, colored_preview_no_lines = self.render_output(image_shape)
        assert self.renderer is not None
        assert self.original_image is not None
        combined = self.renderer.combine_with_legend(coloring_page, legend, "bottom")

        return {
            "source": self.original_image,
            "coloring_page": coloring_page,
            "legend": legend,
            "preview": colored_preview,
            "preview_no_lines": colored_preview_no_lines,
            "combined": combined,
        }
    
    def generate(self, input_image_path: str, output_name: Optional[str] = None) -> Tuple[str, str, str]:
        """
        Complete generation pipeline.
        
        Args:
            input_image_path: Path to input colored image
            output_name: Base name for output files (optional)
            
        Returns:
            tuple: (coloring_page_path, legend_path, preview_path)
        """
        # Step 1: Load image
        image = self.load_image(input_image_path)
        outputs = self.generate_images(image)
        
        # Save output
        if output_name is None:
            output_name = os.path.splitext(os.path.basename(input_image_path))[0]

        assert self.renderer is not None
        assert self.original_image is not None

        coloring_path, legend_path, preview_path = self.renderer.save_output(
            outputs["coloring_page"],
            outputs["legend"],
            outputs["preview"],
            self.original_image,
            output_name,
            outputs["preview_no_lines"]
        )
        
        print(f"{'='*60}")
        print(f"GENERATION COMPLETE!")
        print(f"{'='*60}")
        print(f"Output files:")
        print(f"  - Source image (scaled): {coloring_path.replace('_page.png', '_source.png')}")
        print(f"  - Coloring page: {coloring_path}")
        print(f"  - Combined page: {coloring_path.replace('_page.png', '_combined.png')}")
        print(f"  - Colored preview: {preview_path}")
        print(f"  - Legend: {legend_path}")
        print(f"{'='*60}\n")
        
        return coloring_path, legend_path, preview_path


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate mystery coloring pages from colored images"
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Path to input colored image"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file base name (default: same as input)"
    )
    
    parser.add_argument(
        "-c", "--colors",
        type=int,
        default=16,
        help="Number of colors to reduce to (default: 16)"
    )
    
    parser.add_argument(
        "-d", "--difficulty",
        type=int,
        default=5,
        choices=range(1, 11),
        help="Difficulty level 1-10 (default: 5)"
    )
    
    parser.add_argument(
        "-s", "--symbols",
        type=str,
        default="numbers",
        choices=["numbers", "letters", "custom"],
        help="Symbol type (default: numbers)"
    )
    
    parser.add_argument(
        "-m", "--min-area",
        type=int,
        default=50,
        help="Minimum region area in pixels - smaller regions are not drawn and receive no symbol (default: 50, increase to 100-200 to reduce clutter)"
    )
    
    parser.add_argument(
        "--legend",
        type=str,
        default="bottom",
        choices=["bottom", "separate"],
        help="Legend position (default: bottom)"
    )
    
    parser.add_argument(
        "-r", "--resolution",
        type=int,
        default=1400,
        help="Max image dimension in pixels - higher values give finer lines (default: 1400, recommended: 2000-3500 for high quality)"
    )
    
    parser.add_argument(
        "--symbol-size",
        type=float,
        default=0.5,
        help="Symbol size as ratio of region size, 0.1-1.0 (default: 0.5, smaller=0.3, larger=0.7)"
    )
    
    parser.add_argument(
        "--prefill-dark",
        type=int,
        default=500,
        help="Pre-fill dark regions smaller than this (px²) with black - set to 0 to disable (default: 500)"
    )
    
    parser.add_argument(
        "--mode-filter",
        type=int,
        default=5,
        help="Size of mode filter to eliminate gradient zones (0 to disable, default: 5). Larger = more aggressive cleanup."
    )
    
    parser.add_argument(
        "--no-bilateral",
        action="store_true",
        help="Disable bilateral filter pre-processing (not recommended)"
    )
    
    parser.add_argument(
        "--force-colors",
        type=str,
        default=None,
        help="Force specific RGB colors in the palette (format: 'R,G,B;R,G,B;...'). Example: '0,0,0;255,255,255' forces black and white"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    
    # Parse forced colors if provided
    if args.force_colors:
        try:
            forced_colors = []
            for color_str in args.force_colors.split(';'):
                rgb = tuple(map(int, color_str.strip().split(',')))
                if len(rgb) != 3 or any(c < 0 or c > 255 for c in rgb):
                    raise ValueError(f"Invalid RGB values: {rgb}")
                forced_colors.append(rgb)
            config.forced_colors = forced_colors
            print(f"Forcing {len(forced_colors)} colors: {forced_colors}")
        except Exception as e:
            print(f"Error parsing forced colors: {e}")
            print("Format should be: 'R,G,B;R,G,B;...' (e.g., '0,0,0;255,255,255')")
            return
    config.input_image_path = args.input
    config.output_dir = "output"
    config.num_colors = args.colors
    config.difficulty_level = args.difficulty
    config.symbol_type = args.symbols
    config.min_region_area = args.min_area
    config.legend_position = args.legend
    config.max_image_dimension = args.resolution
    config.font_size_ratio = args.symbol_size
    config.prefill_dark_threshold = args.prefill_dark
    config.prefill_dark_regions = args.prefill_dark > 0
    config.mode_filter_size = args.mode_filter
    config.bilateral_filter_enabled = not args.no_bilateral
    
    # Create generator
    generator = MysteryColoringGenerator(config)
    
    # Generate
    try:
        generator.generate(args.input, args.output)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
