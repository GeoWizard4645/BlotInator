from PIL import Image, ImageOps
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import measure, morphology

# Function to process the image and extract edges with higher precision
def process_image(image_path):
    # Load image and convert to grayscale
    image = Image.open(image_path).convert("L")
    image = ImageOps.invert(image)  # Invert to make the background black and foreground white
    image_array = np.array(image)

    # Apply a slight blur to reduce noise
    blurred = cv2.GaussianBlur(image_array, (3, 3), 0)

    # Use Canny edge detection with lower thresholds for more sensitivity
    edges = cv2.Canny(blurred, 30, 100)

    # Optionally, thicken the edges slightly
    edges = morphology.dilation(edges, morphology.disk(1))

    # Use contours to find connected components
    contours = measure.find_contours(edges, 0.8)
    
    return contours, image_array.shape

# Function to generate the Blot code
def generate_blot_code(contours, dimensions, detail_level=0.8):
    print("Generating Blot code...")
    lines = []
    
    # Set a fixed tolerance for fine control over details
    max_tolerance = 0.5  # More detail
    min_tolerance = 0.1  # Minimal simplification
    tolerance = (1 - detail_level) * (max_tolerance - min_tolerance) + min_tolerance
    
    # Calculate bounding box of all contours
    all_points = np.concatenate(contours)
    min_y, min_x = np.min(all_points, axis=0)
    max_y, max_x = np.max(all_points, axis=0)
    
    # Calculate scale and translation to center the drawing
    scale_x = (dimensions[1] - 1) / (max_x - min_x)
    scale_y = (dimensions[0] - 1) / (max_y - min_y)
    scale = min(scale_x, scale_y)  # Maintain aspect ratio by using the smallest scale factor
    
    translate_x = (dimensions[1] - (max_x - min_x) * scale) / 2 - min_x * scale
    translate_y = (dimensions[0] - (max_y - min_y) * scale) / 2 - min_y * scale
    
    for contour in contours:
        # Smooth the contour and simplify based on the detail level
        smoothed_contour = measure.approximate_polygon(contour, tolerance=tolerance)
        if len(smoothed_contour) >= 2:  # Only consider meaningful contours
            for i in range(len(smoothed_contour) - 1):
                y1, x1 = smoothed_contour[i]
                y2, x2 = smoothed_contour[i + 1]
                
                # Scale and translate coordinates
                x1 = int(x1 * scale + translate_x)
                y1 = int(y1 * scale + translate_y)
                x2 = int(x2 * scale + translate_x)
                y2 = int(y2 * scale + translate_y)
                
                lines.append(f"finalLines.push([[{x1}, {y1}], [{x2}, {y2}]]);\n")

    blot_code = [
        "// Produced by Aditya Anand's Blotinator, not human-written\n",
        f"setDocDimensions({dimensions[0]}, {dimensions[1]});\n",
        "const finalLines = [];\n"
    ]
    blot_code.extend(lines)
    blot_code.append("drawLines(finalLines);")

    return blot_code

# Main function
if __name__ == "__main__":
    # Use the correct image path
    image_path = '/Users/vivaanshahani/Downloads/image.png'

    # Process the image
    contours, dimensions = process_image(image_path)

    # Set the detail level (1 for maximum detail, 0 for minimal detail)
    detail_level = 1  # Higher detail level for better quality

    # Generate the Blot code with the specified detail level
    blot_code = generate_blot_code(contours, dimensions, detail_level)

    # Write the Blot code to a file
    output_path = "/Users/vivaanshahani/Downloads/Blotcode.js"
    with open(output_path, "w") as file:
        file.writelines(blot_code)

    print(f"Blot code generated and saved to {output_path}")
