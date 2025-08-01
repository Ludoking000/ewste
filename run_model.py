import argparse
from ultralytics import YOLO
import cv2
import os

def run_inference(image_path, model_path="e_waste_model_20250731_125115.pt", output_dir="outputs"):
    # Load YOLOv8 model
    print("[INFO] Loading model...")
    model = YOLO(model_path)

    # Run inference
    print(f"[INFO] Running inference on {image_path}...")
    results = model(image_path)

    # Create output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save and show the result
    for r in results:
        result_image = r.plot()  # Draw boxes and labels on image
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"pred_{filename}")
        cv2.imwrite(output_path, result_image)
        print(f"[INFO] Saved prediction to {output_path}")

        # Display image
        cv2.imshow("Prediction", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 E-Waste Detection on an image.")
    parser.add_argument("--source", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="model.pt", help="Path to .pt model file")

    args = parser.parse_args()
    run_inference(args.source, args.model)
