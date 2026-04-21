from PIL import Image
import numpy as np

def load(filepath):
    # 1. Load image and convert to Grayscale ('L' mode)
    img = Image.open(filepath).convert('L')
    
    # 2. Convert to a NumPy array (values will be 0 to 255)
    img_array = np.array(img)
    
    # 3. Thresholding: Convert to 0s and 1s.
    # Assuming you drew obstacles as BLACK (near 0) and free space as WHITE (near 255).
    # This says: "If the pixel is darker than 128, make it a 1 (obstacle). Otherwise, 0 (free)."
    binary_map = (img_array < 128).astype(int)
    
    return binary_map


if __name__ == "__main__":
    MAP_CPU = load_map_from_image("map.png")
    
    with np.printoptions(threshold=np.inf):
        print( MAP_CPU )
