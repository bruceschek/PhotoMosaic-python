import sys
import os
import csv
import time

import numpy as np
from PIL import Image, ImageEnhance
from PIL.Image import Resampling
import matplotlib.pyplot as plt

# Some naming conventions:
# There are two algorithms for matching:
# 1) average color matching
# 2) pixel-level matching (either compare every pixel, or at least a sample grid of comparisons)
#
# "target image" - this is the original source image, from which the photomosaic is created
# "image collection" - some large collection of photos used as tiles in the eventual photomosaic.
# "photomosaic" - the end product

# Define source and destination folders
source_folder =      'images00'
image_tiles_folder = 'images02'

#*****************************************************************************************************
# Functions for pre-processing the image collection (and metadata) for matching tiles
#*****************************************************************************************************

def populate_image_tiles_folder():
    # Define fixed for tile images
    tile_width_in_collection = 200
    tile_height_in_collection = 150

    min_aspect_ratio = 1.2   # so, the image is used only if landscape, with width > 1.2 * height
    count = 0
    target_number_of_tile_images = 41_000

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

                    # brighten the images, as this dataset seems very bland or dull
                    enhancer = ImageEnhance.Brightness(resized_img)
                    brightness_factor = 1.2  # Increase brightness by 50%
                    brightened_image = enhancer.enhance(brightness_factor)

                    # Save the resized image to the destination folder
                    brightened_image.save(os.path.join(image_tiles_folder, file))
                    count += 1
                    if count % 100 == 0 :
                        print(f"Preprocessing Count: {count}")

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

# def get_sum_of_sqrs_color_distance_between_tiles(tile1, tile2) -> int :

# given_tile is a tile of type numpy array. it's a single tile out of the target image.
# The given_tile is compared one by one against all images in the image collection.
def find_closest_tile_by_pixel_level_comparison(tile_from_target, image_collection_in_memory ) -> Image :
    # "distance" will be a sum of squares of rgb distance at several points across the tile
    min_distance = 10**9
    match_image = 0

    # iterate across all images in the image collection
    for img_from_collection in image_collection_in_memory :
        # generally, below, "1" and "2" refer to target image and image collection items respectively
        h1, w1, depth = tile_from_target.shape  # tile from the target image.  note that numpy arrays return shape in an odd tuple
        w2, h2 = img_from_collection.size                 # the image pulled sequentially from image collection

        # Divide the target and sample image each into m x m parts, to sample colors across all m x m points.
        m = 10  # an arbitrary choice, depending how high resolution at we can to compare the images
        # prepare to step horizontally across a row
        row_step1 = (w1 - 1) / (m - 1)  # must be float, to assure that we end at n-1.  long story.
        row_step2 = (w2 - 1) / (m - 1)

        # prepare to step vertically down a column
        col_step1 = (h1 - 1) / (m - 1)  # must be float, to assure that we end at n-1.  long story.
        col_step2 = (h2 - 1) / (m - 1)

        distance = 0  # sum of squares, across all pixels examined for this image

        for i in range( m ):
            row_index1 = int(round( i * row_step1 ))
            row_index2 = int(round( i * row_step2 ))

            for j in range( m ):
                # print(f"--------------------i, j = {i}, {j}")
                col_index1 = int(round(j * col_step1))
                col_index2 = int(round(j * col_step2))

                # print(f"----(col_step1, row_step1), (col_step2, row_step2) = " +
                #       f"{col_index1}, {row_index1}, -  {col_index2}, {row_index2} ")

                # note that numpy image pixel values are accessed by image[row, column] (NOT like x,y coordinates)
                rgb1 = tile_from_target[ col_index1, row_index1 ]
                rgb2 = img_from_collection.getpixel(( row_index2, col_index2 ))

                # Convert rgb1 and rgb2 to larger integer types to avoid overflow
                distance += (int(rgb1[0]) - int(rgb2[0])) ** 2 + (int(rgb1[1]) - int(rgb2[1])) ** 2 + (int(rgb1[2]) - int(rgb2[2])) ** 2


        if distance < min_distance:
            match_image = img_from_collection
            min_distance = distance

    return match_image


# given_tile is a tile of type numpy array
# def find_closest_tile_by_tile_comparison(given_tile) -> str :
#     # "distance" will be a sum of squares of rgb distance at several points across the tile
#     min_distance = 10**9
#     match_file_name = ''
#     for file in os.listdir(image_tiles_folder):
#         image_file_path = os.path.join(image_tiles_folder, file)
#         with Image.open(image_file_path) as img:   # grab a candidate from image collection, and check its distance
#             depth, h1, w1 = given_tile.shape  # tile from the source image.  note that numpy arrays return shape in an odd tuple
#             w2, h2 = img.size                 # the image pulled sequentially from image collection
#
#             rgb1 = given_tile[ h1//2, w1//2 ]  # note that access to rgb value in numpy image is accessed as [y, x]
#             rgb2 = img.getpixel((w2//2, h2//2))
#
#             # Convert rgb1 and rgb2 to larger integer types to avoid overflow
#             distance = (int(rgb1[0]) - int(rgb2[0])) ** 2 + (int(rgb1[1]) - int(rgb2[1])) ** 2 + (int(rgb1[2]) - int(rgb2[2])) ** 2
#
#             if distance < min_distance:
#                 match_file_name = file
#                 min_distance = distance
#     return match_file_name

# given_tile is a tile of type numpy array
def find_closest_tile_by_average_color(given_tile, metadata) -> str :
    min_distance = 10**9
    match_file_name = ''
    r, g, b = get_ave_color_from_image_as_np_array( given_tile )
    for d in metadata:
        rm, gm, bm = d[2], d[3], d[4]
        distance = (r-rm)**2 + (g-gm)**2 + (b-bm)**2
        if distance < min_distance:
            match_file_name = d[1]
            min_distance = distance

    return match_file_name


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
def tile_the_target_image(target_file_name, tile_size, tile_scale_factor, metadata, algorithm ):
    # pixel count for width and height of our riles
    tile_width, tile_height = tile_size[0] // tile_scale_factor, tile_size[1] // tile_scale_factor
    img_temp = Image.open(target_file_name)
    w_temp, h_temp = img_temp.size
    tile_count_w = w_temp // tile_width
    tile_count_h = h_temp // tile_height
    width, height = tile_count_w * tile_width, tile_count_h * tile_height
    # So, the width, height below may be varied slightly from original w, h because of rounding
    img_target = img_temp.resize( ( width, height ), resample=Resampling.LANCZOS )
    # This is the target image, as numpy, normalized to proper multiple of tile size
    image_target_np = np.array(img_target)
    # height, width, channels = image_target_np.shape  # Assuming it's an RGB image

    # this is the eventual, final resulting image, but empthy of tiles for now
    img_photomosaic = np.zeros((height, width, 3), dtype=np.uint8)
    start_time = time.time()

    # Load all images from the image collection into memory, to prevent repeated loading later
    image_collection_in_memory = []
    if algorithm == "pixel_level" :  # need to preload all the sample images into memeory for rapid access
        for file in os.listdir(image_tiles_folder):
            image_file_path = os.path.join(image_tiles_folder, file)
            with open(image_file_path, 'rb') as f:
                img_from_collection = Image.open(f).copy()
                image_collection_in_memory.append( img_from_collection )

    # im = image_collection_in_memory[0]
    # im.show()

    # Break the image into tiles
    for y in range(0, height, tile_height):
        for x in range(0, width, tile_width):
            elapsed_time = time.time() - start_time
            print(f"y, x = {y}, {x},   Elapsed time (min): {elapsed_time/60.0:.2f}")

            # Extract tile from target image, to compare against all possible images in tile collection
            tile_from_target = image_target_np[y:y + tile_height, x:x + tile_width]
            # r_target, g_target, b_target = get_ave_color_from_image_as_np_array(tile_from_target)

            # Get data from closest image in entire image collection, using one of the two algorithms
            if algorithm == "average":
                file_name = find_closest_tile_by_average_color(tile_from_target, metadata)
                file_folder_and_name = f"{image_tiles_folder}/{file_name}"
                # Beware, one or more of the collection images may be greyscale, and thus wrong "shape". So, convert to RGB.
                img_tile_from_collection_full_size = Image.open(file_folder_and_name).convert('RGB')
            elif algorithm == "pixel_level":
                img_tile_from_collection_full_size = find_closest_tile_by_pixel_level_comparison( tile_from_target, image_collection_in_memory )
            else:
                print("failed, at 496766")
                exit()

            # Ensure the image is RGB
            img_tile_from_collection = img_tile_from_collection_full_size.resize(
                ( tile_width, tile_height), resample=Resampling.LANCZOS )
            img_tile_from_collection_np = np.array(img_tile_from_collection)  # Convert to NumPy array
            img_photomosaic[ y:y + tile_height, x:x + tile_width ] = img_tile_from_collection_np

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

    target_image_file = 'mona_lisa.jpg' # 'fam sf xmas.jpg' # 'pure rgbw3.jpg' # 'pure rgbw3.jpg' #  'maui.jpeg' #  'bruce and emma.jpg' #

    # per the wikipedia article on "photomosaic", "avereage" is their first example, "pixel_level" is second.
    algorithm =  "pixel_level"  # "average" #

    tile_width_in_collection =  100 # 200
    tile_height_in_collection =  75 # 150
    tile_collection_size =  tile_width_in_collection, tile_height_in_collection
    tile_scale_factor = 2  # images from tile collection will be divided by this scale

    tile_the_target_image( target_image_file, tile_collection_size, tile_scale_factor, metadata, algorithm )

    print("The End 👍")
    exit()