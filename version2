from PIL import Image, ImageOps
import numpy as np
import cv2
from skimage import measure
import math

# Set the detail level (1 for maximum detail, 0 for minimal detail)
detail_level = 0.5  # Increased detail level for more detail

# Function to calculate Euclidean distance between two points
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# Function to calculate angle between three points (p1-p2-p3)
def angle_between(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

# Function to process the image and extract edges
def process_image(image_path, downscale_factor=1.0):
    image = Image.open(image_path).convert("L")
    image = ImageOps.mirror(image)
    image = ImageOps.invert(image)
    image = image.rotate(180)
    
    # Downscale the image (adjusted to 100% to preserve detail)
    image = image.resize((int(image.width * downscale_factor), int(image.height * downscale_factor)))
    image_array = np.array(image)

    # Apply Gaussian blur (reduced blur for preserving edges)
    ksize_value = max(3, int(round(-3.333333333333 * detail_level + 5.666666666666666667)))
    ksize_value += 1 if ksize_value % 2 == 0 else 0
    ksize = (ksize_value, ksize_value)
    blurred = cv2.GaussianBlur(image_array, ksize, 0)

    # Canny edge detection (adjusted thresholds for more edges)
    canny_threshold1 = int(round(-20.333333333 * detail_level + 66.6666666666667))
    canny_threshold2 = int(round(-50.33333333 * detail_level + 150.666666667))
    edges = cv2.Canny(blurred, max(1, canny_threshold1), max(1, canny_threshold2))

    # Use contours to find connected components
    contours = measure.find_contours(edges, 0.8)
    
    # Filter out small contours (lower threshold to keep more contours)
    filtered_contours = [contour for contour in contours if len(contour) > 10]
    return filtered_contours, image_array.shape

# Function to generate the Blot code with filtered unnecessary segments
def generate_blot_code(contours, dimensions, detail_level=0.9, min_length=0.5, bounding_box=125):
    print("Generating optimized Blot code...")
    lines = []
    tolerance = max(0.01, (1 - detail_level) * 0.1)  # Reduced tolerance for more detail

    drawn_lines = set()  # Keep track of drawn lines to avoid duplication

    all_points = np.concatenate(contours)
    min_y, min_x = np.min(all_points, axis=0)
    max_y, max_x = np.max(all_points, axis=0)

    scale_x = (bounding_box - 1) / (max_x - min_x)
    scale_y = (bounding_box - 1) / (max_y - min_y)
    scale = min(scale_x, scale_y)
    translate_x = (bounding_box - (max_x - min_x) * scale) / 2 - min_x * scale
    translate_y = (bounding_box - (max_y - min_y) * scale) / 2 - min_y * scale

    def add_line(p1, p2):
        """ Helper function to add line and ensure it's not duplicated """
        p1_rounded = tuple(map(float, p1))  # Use float for higher precision
        p2_rounded = tuple(map(float, p2))

        if p1_rounded > p2_rounded:
            p1_rounded, p2_rounded = p2_rounded, p1_rounded
        
        if (p1_rounded, p2_rounded) not in drawn_lines:
            drawn_lines.add((p1_rounded, p2_rounded))
            lines.append([p1_rounded, p2_rounded])

    for contour in contours:
        # Simplify the contour minimally to retain detail
        smoothed_contour = measure.approximate_polygon(contour, tolerance=tolerance)
        if len(smoothed_contour) >= 2:
            for i in range(len(smoothed_contour) - 1):
                y1, x1 = smoothed_contour[i]
                y2, x2 = smoothed_contour[i + 1]

                # Scale and translate coordinates
                x1 = x1 * scale + translate_x
                y1 = y1 * scale + translate_y
                x2 = x2 * scale + translate_x
                y2 = y2 * scale + translate_y

                if distance((x1, y1), (x2, y2)) < min_length:
                    continue

                add_line((x1, y1), (x2, y2))

    # Remove redundant lines
    unique_lines = []
    unique_set = set()
    for line in lines:
        p1, p2 = line
        key = (p1, p2)
        if key not in unique_set:
            unique_set.add(key)
            unique_lines.append(line)

    # Prepare Blot code
    blot_code = [
        "// Produced by Vivaan Shahani, enhanced for detail\n",
        f"setDocDimensions({bounding_box}, {bounding_box});\n",
        "const finalLines = [];\n"
    ]

    for line in unique_lines:
        x1, y1 = line[0]
        x2, y2 = line[1]
        blot_code.append(f"finalLines.push([[{x1}, {y1}], [{x2}, {y2}]]);\n")

    blot_code.append("drawLines(finalLines);")
    return blot_code

# Main function
if __name__ == "__main__":
    image_path = '/Users/vivaanshahani/Downloads/IMG_9654 3.jpg'

    contours, dimensions = process_image(image_path)

    blot_code = generate_blot_code(contours, dimensions)

    output_path = "/Users/vivaanshahani/Downloads/Blotcode.js"
    with open(output_path, "w") as file:
        file.writelines(blot_code)

    print(f"Blot code generated and saved to {output_path}")
