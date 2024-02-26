import os
from PIL import Image

def resize_image(input_image_path, output_image_path, size):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    print(f"original_image_size: {6000}x{4000} ")

    resized_image = original_image.resize(size)
    width, height = resized_image.size
    print(f"resized_image_size: {512}x{384}")
    resized_image.save(output_image_path)

def resize_images_in_folder(folder_path, output_folder_path, size):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_image_path = os.path.join(folder_path, filename)
            output_image_path = os.path.join(output_folder_path, filename)
            resize_image(input_image_path, output_image_path, size)

folder_path = "./datasets/XXX"
output_folder_path = "./datasets/XXX"
size = (1600, 1200)

resize_images_in_folder(folder_path, output_folder_path, size)
