import os
import json
import numpy as np
from PIL import Image

def create_dummy_data(root_dir="dummy_data"):
    val_dir = os.path.join(root_dir, "validation")
    os.makedirs(val_dir, exist_ok=True)

    # Create meta.json
    traj_id = "dummy_traj_0"
    meta = {
        traj_id: {
            "num_frames": 30,
            "text": "move the block to the right"
        }
    }
    
    with open(os.path.join(val_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    # Create dummy images
    traj_dir = os.path.join(val_dir, traj_id)
    os.makedirs(traj_dir, exist_ok=True)
    
    for i in range(30):
        # Create a random image
        img_array = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(os.path.join(traj_dir, f"{i}_static.png"))

    print(f"Dummy dataset created at {os.path.abspath(root_dir)}")

if __name__ == "__main__":
    create_dummy_data()
