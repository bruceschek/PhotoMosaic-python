import sys
import os
import csv
from re import match

import numpy as np
from PIL import Image
from PIL.Image import Resampling
import matplotlib.pyplot as plt

# Some naming conventions:
# There are two algorithms for matching:
# 1) average color matching
# 2) pixel-level matching (either compare every pixel, or at least a sample grid of comparisons)
#
# "target inmage" - this is the original source image, from which the photomosaic is created
# "image collection" - some large collection of photos used as tiles in the eventual photomosaic.
# "photomosaic" - the end product

# Define source and destination folders
source_folder = 'images00'
image_tiles_folder = 'images02' # 'images01'

#*****************************************************************************************************
# Functions for pre-processing the image collection (and metadata) for matching tiles
#*****************************************************************************************************

def populate_image_tiles_folder():
    # Define fixed for tile images
    tile_width_in_collection = 200
    tile_height_in_collection = 150

    min_aspect_ratio = 1.2   # so, the image is used only if landscape, with width > 1.2 * height
    count = 0
    target_number_of_tile_images = 20_000

    # Create the destination folder if it doesn't exist
    if not os.path.exists(image_tiles_folder):
        os.makedirs(image_tiles_folder)

    # Iterate through all JPEG images in the source folder
    for file in os.listdir(source_folder):
        if file.lower().endswith('.jpg'):   # or filename.lower().endswith('.jpeg'):
            image_file_path = os.path.join(source_folder, file)
            # Open the image to check its dimensions
            with Image.open(image_file_path) as img_temp:
                img = img_temp.convert( "RGB" )
                width, height = img.size
                #  eliminate portrait mode photos and leave all the landscape mode
                if width > min_aspect_ratio * height:
                    # Resize the image to the fixed size (200x150)
                    resized_img = img.resize((tile_width_in_collection, tile_height_in_collection))
                    # Save the resized image to the destination folder
                    resized_img.save(os.path.join(image_tiles_folder, file))
                    count += 1

        if count >= target_number_of_tile_images: break

    print("Image processing and resizing complete.")

def generate_metadata_file_for_tiles():
    r_sum, g_sum, b_sum = 0, 0, 0
    pixel_count_width, pixel_count_height = 0, 0
    metadata = []
    index = 0
    for file in os.listdir( image_tiles_folder ):
        path_and_file_name = f"{image_tiles_folder}/{file}"
        img = Image.open( path_and_file_name )
        img_np = np.array(img)
        r, g, b = get_ave_color_from_image_as_np_array(img_np)
        metadata.append( (index, file, r, g, b ))
        index += 1

    csv_file = 'metadata.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in metadata:
            writer.writerow(row)

#*****************************************************************************************************
# Helper functions for matching tiles from image collection to the target image
#*****************************************************************************************************

# given a full path+name of an image file, find average (R,B,G) value across all pixels
def get_ave_color_from_image_file(file) -> tuple:
    with Image.open(f'{image_tiles_folder}/{file}') as img:
        # Ensure the image is in RGB mode
        img = img.convert('RGB')
        # Convert the image to a NumPy array (faster processing)
        img_np = np.array(img)
        return get_ave_color_from_image_as_np_array(img_np)

# given a full path+name of an image file, find average (R,B,G) value across all pixels
def get_ave_color_from_image_as_np_array(img_np) -> tuple:
    # Ensure the image is RGB (3 channels)
    if img_np.shape[-1] != 3:
        raise ValueError("Expected an image with 3 channels (RGB).")
    # Calculate the mean along the width, height, and RGB channels
    r_avg, g_avg, b_avg = np.mean(img_np, axis=(0, 1))
    return (int(r_avg), int(g_avg), int(b_avg))

# given_tile is a tile of type numpy array
def find_closest_tile(given_tile, metadata) -> tuple[str, int, int, int]:
    min_distance = 10**9
    match_file_name = ''
    r0, g0, b0 = -1, -1, -1
    r, g, b = get_ave_color_from_image_as_np_array( given_tile )
    for d in metadata:
        rm, gm, bm = d[2], d[3], d[4]
        # if rm == 186 and gm == 16 and bm == 14:
        #     print()
        distance = (r-rm)**2 + (g-gm)**2 + (b-bm)**2
        if distance < min_distance:
            match_file_name = d[1]
            r0, g0, b0 = rm, gm, bm
            min_distance = distance
    # if match_file_name == '':
    #     print()
    return match_file_name, r0, g0, b0


#*****************************************************************************************************
# Functions orchestrating the overall matching process
#*****************************************************************************************************

def load_metadata_from_csv(csv_file) -> list:
    metadata = []
    with open(csv_file, mode='r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)
        # Iterate over each row in the CSV file
        for row in csv_reader:
            row[0], row[2], row[3], row[4] = int(row[0]), int(row[2]), int(row[3]), int(row[4])
            metadata.append(row)
    return metadata


# source_file_name is filename of the image to be re-created from tiles
# tile_size is tuple of width, height of the tiles
def tile_the_target_image(target_file_name, tile_size, tile_scale_factor, metadata):
    # normalize the width, height of the image, to round number of tiles
    tile_width, tile_height = tile_size[0] // tile_scale_factor, tile_size[1] // tile_scale_factor

    img_temp = Image.open(target_file_name)
    w_temp, h_temp = img_temp.size
    tile_count_w = w_temp // tile_width
    tile_count_h = h_temp // tile_height
    width, height = tile_count_w * tile_width, tile_count_h * tile_height
    img_target = img_temp.resize( ( width, height ), resample=Resampling.LANCZOS )
    # This is the target image, as numpy, normalized to proper multiple of tile size
    image_target_np = np.array(img_target)
    height, width, channels = image_target_np.shape  # Assuming it's an RGB image

    img_photomosaic = np.zeros((height, width, 3), dtype=np.uint8)
    # Break the image into tiles
    # tiles = []  # This will hold all the image tiles as numpy arrays, two dimensions
    for y in range(0, height, tile_height):
        for x in range(0, width, tile_width):
            # Extract tile from target image, to compare against all possible images in tile collection
            tile_from_target = image_target_np[y:y + tile_height, x:x + tile_width]
            r_target, g_target, b_target = get_ave_color_from_image_as_np_array(tile_from_target)
            # Get data from closest image in tile collection
            file_name, r, g, b = find_closest_tile(tile_from_target, metadata)
            file_folder_and_name = f"{image_tiles_folder}/{file_name}"

            # Beware, one or more of the collection images may be greyscale, and thus wrong "shape". So, convert to RGB.
            img_tile_from_collection_fullsize = Image.open(file_folder_and_name).convert('RGB')  # Ensure the image is RGB
            img_tile_from_collection = img_tile_from_collection_fullsize.resize(
                ( tile_width, tile_height), resample=Resampling.LANCZOS )
            img_tile_from_collection_np = np.array(img_tile_from_collection)  # Convert to NumPy array
            img_photomosaic[ y:y + tile_height, x:x + tile_width ] = img_tile_from_collection_np

            # debug
            # print("tile extracted from img_target (original) image")
            # print(f"RGB = {r_target}, {g_target}, {b_target}")
            # plt.imshow(tile_from_target)
            # plt.axis('off')  # Hide axes
            # plt.show()
            #
            # print("tile matched and selected from collection of possible tiles")
            # print(f"filename: {file_name}, RGB = ( {r}, {g}, {b} )")
            # plt.imshow(img_tile_from_collection)
            # plt.axis('off')  # Hide axes
            # plt.show()
            # print()

    # Convert the NumPy array to a PIL Image
    img_target = Image.fromarray(img_photomosaic)
    img_target.save('output_image.jpg')

if __name__ == '__main__':
    print(f"Python Version: {sys.version}")

    # pre-processing.  only needs to run when adopting a new collection fo image tile files
    # populate_image_tiles_folder()
    # generate_metadata_file_for_tiles()
    # print("pre-processing complete.")
    # exit()

    metadata = load_metadata_from_csv("metadata.csv")

    target_image_file = 'fam sf xmas.jpg' # 'maui.jpeg' #'mona_lisa.jpg' # 'bruce and emma.jpg' # 'pure rgbw2.jpg'

    tile_width_in_collection = 200
    tile_height_in_collection = 150
    tile_collection_size =  tile_width_in_collection, tile_height_in_collection
    tile_scale_factor = 3  # images from tile collection will be diveded by this scale

    tile_the_target_image( target_image_file, tile_collection_size, tile_scale_factor, metadata )

    print("The End 👍")
    exit()