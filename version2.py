from PIL import Image, ImageOps
import os
os.environ['JOBLIB_MULTIPROCESSING'] = '0'
import numpy as np
import cv2
from skimage import measure, transform
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
import math

# Set the detail level (1 for maximum detail, 0 for minimal detail)
detail_level = 1  # Adjusted to balance between details and line reduction

# Function to process the image and extract edges
def process_image(image_path, downscale_factor=0.85):
    print("Processing image...")
    # Load image and convert to grayscale
    image = Image.open(image_path).convert("L")
    image = ImageOps.mirror(image)  # Mirror the image horizontally
    image = ImageOps.invert(image)  # Invert to make the background black and foreground white
    image = image.rotate(180)  # Rotate the image by 180 degrees

    # Downscale the image
    image = image.resize((int(image.width * downscale_factor), int(image.height * downscale_factor)))
    image_array = np.array(image)

    # Slight blurring to reduce noise and smooth edges
    blurred = cv2.GaussianBlur(image_array, (3, 3), 0)

    # Use Canny edge detection with thresholds for detail control
    canny_threshold1 = int(round(-15.0 * detail_level + 60))
    canny_threshold2 = int(round(-40.0 * detail_level + 140))
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)

    # Thicken the edges slightly for clearer contour detection
    edges = transform.rescale(edges, 1.0, anti_aliasing=True)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    # Use contours to find connected components
    contours = measure.find_contours(edges, 0.8)

    print(f"Found {len(contours)} contours.")
    return contours, image_array.shape

# Function to apply Ramer-Douglas-Peucker algorithm for line simplification
def simplify_contours(contours, tolerance=2.0):
    print("Simplifying contours...")
    simplified_contours = []
    for contour in contours:
        simplified = measure.approximate_polygon(contour, tolerance=tolerance)
        simplified_contours.append(simplified)
    print(f"Simplified to {len(simplified_contours)} contours.")
    return simplified_contours

# Function to optimize the drawing sequence by minimizing line travel distance
# Function to optimize the drawing sequence by minimizing line travel distance
def optimize_drawing_order(lines):
    print("Optimizing drawing order...")
    
    # List to hold the optimized lines
    optimized_lines = []
    visited_lines = set()  # Keep track of visited lines
    
    # Start from the first line and initialize current point
    current_line = lines[0]
    optimized_lines.append(current_line)
    visited_lines.add(0)
    
    current_point = current_line[1]  # Start with the end of the first line
    
    while len(visited_lines) < len(lines):
        min_distance = float('inf')
        next_line_index = None
        
        # Find the nearest unvisited line
        for i, line in enumerate(lines):
            if i in visited_lines:
                continue
            
            # Calculate the distance from the current point to both ends of the line
            dist_to_start = np.linalg.norm(np.array(current_point) - np.array(line[0]))
            dist_to_end = np.linalg.norm(np.array(current_point) - np.array(line[1]))
            
            # Choose the shortest distance
            if dist_to_start < min_distance:
                min_distance = dist_to_start
                next_line_index = i
                next_point = line[1]  # Move to the end of the next line
            if dist_to_end < min_distance:
                min_distance = dist_to_end
                next_line_index = i
                next_point = line[0]  # Move to the start of the next line
        
        # If no next line was found (shouldn't happen), break the loop
        if next_line_index is None:
            break
        
        # Add the selected line to the optimized sequence and mark it as visited
        optimized_lines.append(lines[next_line_index])
        visited_lines.add(next_line_index)
        current_point = next_point  # Update the current point
        
        # Print progress
        print(f"Visited {len(visited_lines)} lines out of {len(lines)}.")
    
    return optimized_lines


# Function to generate the Blot code with a limited number of lines and efficient drawing order
def generate_blot_code(contours, dimensions, max_lines=600, tolerance=2.0):
    print("Generating Blot code...")
    max_dimension = 125  # Set maximum dimensions for both x and y to 125
    lines = []

    # Calculate bounding box of all contours
    all_points = np.concatenate(contours)
    min_y, min_x = np.min(all_points, axis=0)
    max_y, max_x = np.max(all_points, axis=0)

    # Calculate scale to fit within the 125x125 coordinate plane
    scale_x = max_dimension / (max_x - min_x)
    scale_y = max_dimension / (max_y - min_y)
    scale = min(scale_x, scale_y)  # Maintain aspect ratio by using the smallest scale factor

    # Translation to center the drawing in the 125x125 plane
    translate_x = (max_dimension - (max_x - min_x) * scale) / 2 - min_x * scale
    translate_y = (max_dimension - (max_y - min_y) * scale) / 2 - min_y * scale

    for contour in contours:
        if len(contour) >= 2:  # Only consider meaningful contours
            for i in range(len(contour) - 1):
                y1, x1 = contour[i]
                y2, x2 = contour[i + 1]

                # Scale and translate coordinates
                x1 = int(x1 * scale + translate_x)
                y1 = int(y1 * scale + translate_y)
                x2 = int(x2 * scale + translate_x)
                y2 = int(y2 * scale + translate_y)

                lines.append([[x1, y1], [x2, y2]])

    print(f"Total number of lines before clustering: {len(lines)}")

    # Limit the number of lines using K-means clustering
    if len(lines) > max_lines:
        print(f"Clustering lines to reduce from {len(lines)} to {max_lines} lines.")
        kmeans = KMeans(n_clusters=max_lines)
        kmeans.fit(np.array(lines).reshape(-1, 4))
        cluster_centers = kmeans.cluster_centers_

        # Reshape the cluster centers into pairs of points and cast to integers
        lines = []
        for center in cluster_centers:
            x1, y1, x2, y2 = map(int, center)  # Unpack the flattened array into four values
            lines.append([[x1, y1], [x2, y2]])  # Create the line segment from the four values

    # Optimize the drawing order of lines
    optimized_lines = optimize_drawing_order(lines)

    # Generate the Blot code
    blot_code = [
        "// Produced by Vivaan Shahani, based on Aditya Anand's Blotinator, not human-written\n",
        f"setDocDimensions({max_dimension}, {max_dimension});\n",
        "const finalLines = [];\n"
    ]

    for line in optimized_lines:
        x1, y1 = line[0]
        x2, y2 = line[1]
        blot_code.append(f"finalLines.push([[{x1}, {y1}], [{x2}, {y2}]]);\n")

    blot_code.append("drawLines(finalLines);")
    return blot_code

# Main function
if __name__ == "__main__":
    print("Starting the process...")
    # Use the correct image path
    image_path = '/Users/vivaanshahani/Downloads/IMG_9654 3.jpg'

    # Process the image
    contours, dimensions = process_image(image_path)

    # Simplify the contours
    simplified_contours = simplify_contours(contours, tolerance=2.0)

    # Generate the optimized Blot code with a maximum of 600 lines
    blot_code = generate_blot_code(simplified_contours, dimensions, max_lines=600)

    # Write the Blot code to a file
    output_path = "/Users/vivaanshahani/Downloads/Blotcode.js"
    with open(output_path, "w") as file:
        file.writelines(blot_code)

    print(f"Blot code generated and saved to {output_path}")
