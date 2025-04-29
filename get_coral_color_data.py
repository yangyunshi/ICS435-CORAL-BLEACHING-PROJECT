from sklearn.cluster import KMeans
import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os
import csv
import pandas as pd

## Special thanks to Karan Bhanot for providing foundational code that I have modified to suit our purposes

#####
#      FUNCTION: RGB2HEX
#
#      ARGUMENTS: color (int[]) : RGB values to be converted to hexadecimal
#
#      RETURN: Color in hexadecimal string
#
#      DESCRIPTION: This takes a color represented by an [r,g,b] array and returns it as a hexadecimal
#
#####

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


#####
#      FUNCTION: get_image
#
#      ARGUMENTS: image_path (string) : file path or name to be read in
#
#      RETURN: The image imported
#
#      DESCRIPTION: This reads in an image and converts BGR image to RGB
#
#####

def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


#####
#      FUNCTION: get_color_frequency
#
#      ARGUMENTS: image (image file) : the image file to be checked
#                 num_colors (int) : The number of top colors to extract (default 5)
#
#      RETURN: Hexadecimal colors as RGB colors and their frequency
#
#      DESCRIPTION: This takes in the image, resizes it to 100x100, and calculates the frequency of the top `num_colors` colors.
#
#####

def get_color_frequency(image, num_colors=5):
    # Resize the input image for processing
    image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
    image = image.reshape(-1, 3)

    clf = KMeans(n_clusters=num_colors, n_init=10)
    labels = clf.fit_predict(image)

    counts = Counter(labels)
    total = sum(counts.values())
    center_colors = clf.cluster_centers_
    
    # Sort by frequency in descending order
    ordered = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    # Get top colors and their frequencies
    hex_colors = [RGB2HEX(center_colors[i]) for i, _ in ordered]
    frequencies = [round(count / total, 4) for _, count in ordered]

    return hex_colors, frequencies


# Set the base directory and output file path
base_dir = os.path.join(os.path.expanduser("~"), "Downloads", "Coral_Dataset")
output_file = os.path.join(base_dir, "coral_colors5.csv")

# Prepare CSV header
header = ["label"]
for i in range(1, 6):
    header.extend([f"color{i}", f"frequency{i}"])

# Collect data and write to CSV
with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)

    for label in ["CORAL", "CORAL_BL"]:
        folder = os.path.join(base_dir, label)
        images_processed = 0  # Counter for the number of images processed
        
        for filename in os.listdir(folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(folder, filename)
                img = get_image(img_path)
                
                # Get the top 5 colors and their frequencies
                colors, freqs = get_color_frequency(img, num_colors=5)
                
                # Write the data to the CSV
                row = [label]
                for c, f in zip(colors, freqs):
                    row.extend([c, f])
                print("Writing row:", row)
                writer.writerow(row)
                
                # Update progress every 10 images
                images_processed += 1
                if images_processed % 10 == 0:
                    print(f"Processed {images_processed} images from {label} folder.")
        
        print(f"Finished processing {images_processed} images in {label} folder.")

print(f"Data saved to {output_file}")
