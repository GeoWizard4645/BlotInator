from PIL import Image, ImageOps
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import measure, transform
import math


# Set the detail level (1 for maximum detail, 0 for minimal detail)
detail_level = 0.8  # Higher detail level for better quality

# Function to process the image and extract edges with higher precision
def process_image(image_path):
    # Load image and convert to grayscale
    image = Image.open(image_path).convert("L")
    image = ImageOps.mirror(image)  # Mirror the image horizontally
    image = ImageOps.invert(image)  # Invert to make the background black and foreground white
    image = image.rotate(180)  # Rotate the image by 180 degrees
    image_array = np.array(image)

    # Calculate ksize (ensuring it's odd and positive)
    ksize_value = max(3, int(round(-3.333333333333 * detail_level + 5.666666666666666667)))
    if ksize_value % 2 == 0:
        ksize_value += 1
    ksize = (ksize_value, ksize_value)

    # Apply a slight blur to reduce noise
    blurred = cv2.GaussianBlur(image_array, ksize, 0)

    # Use Canny edge detection with lower thresholds for more sensitivity
    canny_threshold1 = int(round(-33.333333333 * detail_level + 56.6666666666667))
    canny_threshold2 = int(round(-83.33333333 * detail_level + 166.666666667))
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)

    # Optionally, thicken the edges slightly
    edges = transform.rescale(edges, 1.0, anti_aliasing=True)
    edges = cv2.dilate(edges, np.ones((2,2),np.uint8), iterations=1)

    # Use contours to find connected components
    contours = measure.find_contours(edges, 0.8)
    
    return contours, image_array.shape

# Function to generate the Blot code
def generate_blot_code(contours, dimensions, detail_level=0.8):
    maxDimension_y = 0
    maxDimension_x = 0    
    print("Generating Blot code...")
    lines = []
    
    # Set a fixed tolerance for fine control over details
    if -15 * detail_level + 13>0:
        max_tolerance =  -15 * detail_level + 13>0 # High tolerance for significant simplification
    else :
        max_tolerance = 0.1
    if -15 * detail_level + 13>0:
        min_tolerance =  -1.5 * detail_level + 1.3>0 # High tolerance for significant simplification
    else :
        min_tolerance = 0.01
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
                if x1 > maxDimension_x:
                    maxDimension_x = x1 +5
                if x2 > maxDimension_x:
                    maxDimension_x = x2 +5
                if y1 > maxDimension_x:
                    maxDimension_x = y1 +5
                if y2 > maxDimension_x:
                    maxDimension_x = y2 +5           

    blot_code = [
        "// Produced by Vivaan Shahani, based on Aditya Anand's Blotinator, not human-written\n",
        f"setDocDimensions({str(maxDimension_x)}, {str(maxDimension_y)});\n",
        "const finalLines = [];\n"
    ]
    blot_code.extend(lines)
    blot_code.append("drawLines(finalLines);")

    return blot_code

# Main function
if __name__ == "__main__":
    # Use the correct image path
    image_path = '/Users/vivaanshahani/Downloads/IMG_9654.png'

    # Process the image
    contours, dimensions = process_image(image_path)

    # Generate the Blot code with the specified detail level
    blot_code = generate_blot_code(contours, dimensions, detail_level)

    # Write the Blot code to a file
    output_path = "/Users/vivaanshahani/Downloads/Blotcode.js"
    with open(output_path, "w") as file:
        file.writelines(blot_code)

    print(f"Blot code generated and saved to {output_path}")
