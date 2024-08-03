import os
import cv2
import csv

def is_black_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Check if all pixels are black (0)
    return cv2.countNonZero(image) == 0

def process_images(directory):
    # Create a list to store the results
    results = []

    # Iterate through the files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith(".png") and not filename.endswith("_GT.png"):
            base_name = filename[:-4]
            gt_image_path = os.path.join(directory, f"{base_name}_GT.png")
            
            if os.path.exists(gt_image_path):
                label = 0 if is_black_image(gt_image_path) else 1
                results.append([os.path.join(directory, f"{base_name}.png"), label])

    # Write the results to a CSV file
    with open('dataset/test/image_labels.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["img_path", "label"])
        writer.writerows(results)

    print("CSV file 'image_labels.csv' created successfully.")

# Example usage
process_images('/home/aleks/Documents/conv_ae_ad/dataset/test')
