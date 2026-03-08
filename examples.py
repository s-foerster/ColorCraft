"""
Example script showing how to use the Mystery Coloring Generator.
"""

from main import MysteryColoringGenerator
from config import Config
import os


def example_basic():
    """Basic usage example with default settings."""
    print("Example 1: Basic usage with defaults\n")
    
    generator = MysteryColoringGenerator()
    
    # Make sure you have an image in the input folder
    input_path = "input/example.jpg"
    
    if os.path.exists(input_path):
        generator.generate(input_path, "example_basic")
    else:
        print(f"⚠️ Please place an image at: {input_path}")


def example_easy_children():
    """Example optimized for young children."""
    print("\nExample 2: Easy coloring for children\n")
    
    config = Config()
    config.num_colors = 8               # Few colors
    config.difficulty_level = 2         # Low difficulty
    config.min_region_area = 100        # Larger regions only
    config.line_thickness_main = 4      # Thicker lines
    config.line_thickness_sub = 2
    
    generator = MysteryColoringGenerator(config)
    
    input_path = "input/example.jpg"
    if os.path.exists(input_path):
        generator.generate(input_path, "example_children")


def example_expert_puzzle():
    """Example for expert level (puzzle-like)."""
    print("\nExample 3: Expert level puzzle\n")
    
    config = Config()
    config.num_colors = 16
    config.difficulty_level = 9         # Very high difficulty
    config.min_region_area = 30         # Include small regions
    config.symbol_type = "letters"      # Use letters instead of numbers
    config.font_size_ratio = 0.4        # Smaller symbols
    
    generator = MysteryColoringGenerator(config)
    
    input_path = "input/example.jpg"
    if os.path.exists(input_path):
        generator.generate(input_path, "example_expert")


def example_custom_symbols():
    """Example with custom symbols."""
    print("\nExample 4: Custom symbols\n")
    
    config = Config()
    config.num_colors = 10
    config.difficulty_level = 5
    config.symbol_type = "custom"
    config.custom_symbols = ["★", "♥", "♦", "♣", "♠", "●", "■", "▲", "◆", "✿"]
    
    generator = MysteryColoringGenerator(config)
    
    input_path = "input/example.jpg"
    if os.path.exists(input_path):
        generator.generate(input_path, "example_custom")


def example_separate_legend():
    """Example with legend on separate page."""
    print("\nExample 5: Separate legend page\n")
    
    config = Config()
    config.legend_position = "separate"  # Legend on separate file
    config.difficulty_level = 6
    
    generator = MysteryColoringGenerator(config)
    
    input_path = "input/example.jpg"
    if os.path.exists(input_path):
        generator.generate(input_path, "example_separate")


if __name__ == "__main__":
    print("=" * 60)
    print("MYSTERY COLORING GENERATOR - EXAMPLES")
    print("=" * 60)
    print("\nMake sure you have an image file at: input/example.jpg")
    print("(Or modify the examples to use your own image path)\n")
    
    # Run all examples (uncomment the ones you want to try)
    
    example_basic()
    # example_easy_children()
    # example_expert_puzzle()
    # example_custom_symbols()
    # example_separate_legend()
    
    print("\n" + "=" * 60)
    print("Check the 'output' folder for generated coloring pages!")
    print("=" * 60)
