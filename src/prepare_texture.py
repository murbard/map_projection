
from PIL import Image
import sys

def resize_texture():
    input_path = "equi_sat.png"
    output_path = "equi_sat_8000x2000.png"
    
    print(f"Loading {input_path}...")
    # This might take a while for 43k image
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(input_path)
    
    print(f"Original size: {img.size}")
    
    # Resize to 8000x2000 as requested
    print(f"Resizing to 8000x2000...")
    img_resized = img.resize((8000, 2000), Image.Resampling.LANCZOS)
    
    print(f"Saving to {output_path}...")
    img_resized.save(output_path, quality=95)
    print("Done.")

if __name__ == "__main__":
    resize_texture()
