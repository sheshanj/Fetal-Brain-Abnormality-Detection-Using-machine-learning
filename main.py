from pathlib import Path
import argparse
import cv2


def process_image(image_path, output_dir):
    """Load an ultrasound image and create a processed copy."""

    image_path = Path(image_path)
    output_dir = Path(output_dir)

    # Check whether the input file exists.
    if not image_path.is_file():
        print(f"Error: Image not found: {image_path}")
        return

    # Load the image using OpenCV.
    image = cv2.imread(str(image_path))

    if image is None:
        print("Error: Could not read the image.")
        return

    # Resize the image while preserving its aspect ratio.
    height, width = image.shape[:2]
    max_dimension = 640
    scale = min(max_dimension / width, max_dimension / height, 1.0)

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # Convert the image to grayscale for inspection.
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Create the output folder if it does not exist.
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{image_path.stem}_processed.png"

    # Save the processed image.
    success = cv2.imwrite(str(output_path), grayscale)

    if success:
        print("Image processing completed successfully.")
        print(f"Original dimensions: {width} x {height}")
        print(f"Processed dimensions: {new_width} x {new_height}")
        print(f"Saved image: {output_path}")
        print("Note: This program does not detect or diagnose abnormalities.")
    else:
        print("Error: Could not save the processed image.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetal ultrasound image preprocessing demo"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to an ultrasound image"
    )

    parser.add_argument(
        "--output",
        default="output_images",
        help="Folder for processed images"
    )

    args = parser.parse_args()
    process_image(args.image, args.output)
