import subprocess
import sys

# Function to install packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
required_packages = ["ultralytics", "opencv-python", "tk"]

# Try to import and install missing packages
for pkg in required_packages:
    try:
        __import__(pkg if pkg != "opencv-python" else "cv2")
    except ImportError:
        print(f"[INFO] '{pkg}' not found. Installing...")
        install(pkg)

# Now import everything
import os
import cv2
from ultralytics import YOLO
from tkinter import Tk, filedialog

def run_inference(image_path, model_path="model.pt", output_dir="outputs"):
    print("[INFO] Loading model...")
    model = YOLO(model_path)

    print(f"[INFO] Running inference on {image_path}...")
    results = model(image_path)

    os.makedirs(output_dir, exist_ok=True)

    for r in results:
        result_image = r.plot()
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"pred_{filename}")
        cv2.imwrite(output_path, result_image)
        print(f"[INFO] Saved prediction to {output_path}")

        cv2.imshow("Prediction", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Hide the root Tk window
    Tk().withdraw()

    print("Please select an image file for inference...")
    image_path = filedialog.askopenfilename(
        title="Select image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )

    if not image_path:
        print("No image selected, exiting.")
        exit()

    run_inference(image_path)
