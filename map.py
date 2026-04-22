from PIL import Image
import numpy as np

LOWER_BOUND = 0.2
UPPER_BOUND = 5

def load(filepath):
    img = Image.open(filepath).convert('L')
    arr = np.array(img)
    return (arr < 128).astype(int)[:, ::-1]

_data = load("map.png")
MAP_SIZE = _data.shape[0]

# PIXELS_PER_METER = MAP_SIZE / (UPPER_BOUND * 2)
# 56 pixels wide of the table in the map / 1.2m wide from the MuJoCo XML
PIXELS_PER_METER = 56 / 1.2

if __name__ == "__main__":
    with np.printoptions(threshold=np.inf):
        print(_data)
