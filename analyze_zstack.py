import nd2
import numpy as np
from PIL import Image
import pims_nd2
import json
import cv2
import os
import warnings
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tifffile as tiff

warnings.filterwarnings("ignore")


def get_stacks(stacks_path):
    stacks = []  # List to store the loaded image stacks

    # Iterate through the files in the specified directory
    for root, _, files in os.walk(stacks_path):
        for file in files:
            if file.endswith(".tiff") or file.endswith(".tif"):  # Adjust the file format as needed
                # Construct the full path to the .tiff image stack file
                stack_path = os.path.join(root, file)

                # Load the .tiff image stack
                stack = tiff.imread(stack_path)

                # Append the stack to the list of stacks
                stacks.append(stack)

    return stacks

def calculate_sharpness_score(frame):
    try:
        if isinstance(frame, np.ndarray):
            # Convert to 16-bit unsigned integer if not already
            if frame.dtype != np.uint16:
                frame = frame.astype(np.uint16)

            # Calculate sharpness score using Laplacian method
            sharpness_score = cv2.Laplacian(frame, cv2.CV_64F).var()
            return sharpness_score

        elif isinstance(frame, Image.Image):
            # Convert to NumPy array
            frame = np.array(frame, dtype=np.uint16)

            # Calculate sharpness score using Laplacian method
            sharpness_score = cv2.Laplacian(frame, cv2.CV_64F).var()
            return sharpness_score
        else:
            return None  # Return None for unsupported frame type

    except Exception as e:
        print(f"Error calculating sharpness score: {str(e)}")
        return None  # Return None to indicate that there was an error processing the frame

def analyze_stack(stack_path):
    print("==================================")
    print(f"Analyzing Stack: {stack_path}...")

    # Load frames from the stack directory (you may need to adjust this part)
    stack_frames = get_frames_from_directory(stack_path)

    if not stack_frames:
        print(f"No frames found in Stack at {stack_path}.")
        return

    # Initialize a list to store sharpness scores for each frame
    sharpness_scores = []

    # Calculate the optimal frame index (middle frame)
    optimal_frame = (1 + len(stack_frames)) // 2

    for frame_index, frame in enumerate(stack_frames):
        score = calculate_sharpness_score(frame)

        if score is not None:
            sharpness_scores.append(score)
            print(f"Frame {frame_index}: Sharpness Score = {score}")

        if frame_index == optimal_frame:
            print(f"Frame {frame_index}: Optimal Focus Frame, Sharpness Score = {score}")

    # Create a directory to save analysis results
    analysis_dir = os.path.join(stack_path, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    if sharpness_scores:
        # Calculate the overall sharpness score for the stack (e.g., mean or median)
        overall_sharpness = np.mean(sharpness_scores)
        median_sharpness = np.median(sharpness_scores)

        print(f"Overall Sharpness Score (Mean) for Stack: {overall_sharpness}")
        print(f"Overall Sharpness Score (Median) for Stack: {median_sharpness}")

        # Create a box plot for the Sharpness scores of this stack and save it under the analysis folder
        plt.figure(figsize=(8, 6))
        plt.boxplot(sharpness_scores)
        plt.title("Sharpness Score Variation for Stack")
        plt.ylabel("Sharpness Score")
        plt.savefig(os.path.join(analysis_dir, "fsharpnessscore_variation.png"))

        # Create a histogram for the Sharpness scores of this stack and save it under the analysis folder
        plt.figure(figsize=(8, 6))
        plt.hist(sharpness_scores, bins=10, edgecolor='black')
        plt.title("Sharpness Score Histogram for Stack")
        plt.xlabel("Sharpness Score")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(analysis_dir, "sharpness_score_histogram.png"))

    else:
        print("No frames with valid sharpness scores in Stack.")

    print("Analysis completed for Stack.")

# Function to load frames from a stack directory (you may need to adjust this)
def get_frames_from_directory(stack_dir):
    frames = []
    frames_dir = os.path.join(stack_dir, "frames") # we need this subfolder, where individual frames within a stack are found
    for root, _, files in os.walk(frames_dir):
        for file in files:
            if file.endswith(".png"):  # Adjust the file format as needed
                frame_path = os.path.join(root, file)
                frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                frames.append(frame)
    return frames


def analyze_stacks(stacks_dir):
    # Get a sorted list of subdirectories
    stack_dirs = sorted([d for d in os.listdir(stacks_dir) if os.path.isdir(os.path.join(stacks_dir, d))])

    # Iterate through the stack directories
    for stack_dir in stack_dirs:
        stack_path = os.path.join(stacks_dir, stack_dir)
        analyze_stack(stack_path)

def create_plots(stacks_dir):
    try:
        # Initialize empty lists to store dPFSOffset and position values
        dPFSOffset_values = []
        pos_x_values = []
        pos_y_values = []
        pos_z_values = []

        # Iterate through the files in the specified directory
        for root, _, files in os.walk(stacks_dir):
            for file in files:
                if file.endswith(".json"):
                    # Construct the full path to the JSON file
                    file_path = os.path.join(root, file)

                    # Load the metadata from the JSON file
                    with open(file_path, "r") as json_file:
                        metadata = json.load(json_file)

                    # Extract dPFSOffset and position values
                    dPFSOffset_values.append(metadata.get("dPFSOffset", 0))
                    pos_x_values.append(metadata.get("dPosX", 0))
                    pos_y_values.append(metadata.get("dPosY", 0))
                    pos_z_values.append(metadata.get("dPosZ", 0))

        # Create a box plot to visualize dPFSOffset variation and save as an image
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=dPFSOffset_values)
        plt.ylabel("dPFSOffset")
        plt.title("dPFSOffset Variation Across Stacks")
        plt.grid(True)
        plt.savefig(os.path.join(stacks_dir, "dPFSOffset_variation.png"))
        plt.close()

        # Create a scatter plot to visualize position variation and save as an image
        plt.figure(figsize=(10, 6))
        plt.scatter(pos_x_values, pos_y_values, c=pos_z_values, cmap='viridis', marker='o', s=100)
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Position Variation Across Stacks (Color-mapped by Z Position)")
        plt.colorbar(label="Z Position")
        plt.grid(True)
        plt.savefig(os.path.join(stacks_dir, "position_variation.png"))
        plt.close()

        # Create a histogram to visualize the distribution of dPFSOffset values
        plt.figure(figsize=(8, 6))
        plt.hist(dPFSOffset_values, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        plt.xlabel("dPFSOffset")
        plt.ylabel("Frequency")
        plt.title("dPFSOffset Distribution Across Stacks")
        plt.grid(True)
        plt.savefig(os.path.join(stacks_dir, "dPFSOffset_distribution.png"))
        plt.close()

        # Create a heatmap to visualize position variation
        plt.figure(figsize=(10, 8))
        position_data = pd.DataFrame({'X Position': pos_x_values, 'Y Position': pos_y_values, 'Z Position': pos_z_values})
        position_heatmap = position_data.corr()
        sns.heatmap(position_heatmap, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title("Position Correlation Heatmap")
        plt.savefig(os.path.join(stacks_dir, "position_heatmap.png"))
        plt.close()

        # Create a box plot to visualize position variations
        plt.figure(figsize=(10, 6))
        position_data = pd.DataFrame({'X Position': pos_x_values, 'Y Position': pos_y_values, 'Z Position': pos_z_values})
        sns.boxplot(data=position_data, palette="Set3")
        plt.xlabel("Position Dimension")
        plt.ylabel("Position Value")
        plt.title("Position Variations Across Stacks")
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.grid(True)

        # Save the box plot as an image
        plt.savefig(os.path.join(stacks_dir, "position_variations_boxplot.png"))
        plt.close()

        return "Plots saved successfully"

    except Exception as e:
        return f"Error: {str(e)}"

# Define a function to get the data from a provided URL
def get_data(url, save_path):
    try:
        # Check if the file already exists
        if os.path.exists(save_path):
            print(f"--- File {os.path.basename(save_path)} already exists. Skipping download.")
            return True

        # Send an HTTP GET request to the URL
        response = requests.get(url, stream=True)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                print(f"--- Downloading {os.path.basename(save_path)}...")
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            print(f"--- File {os.path.basename(save_path)} downloaded successfully.")
            return True
        else:
            print(f"--- Failed to download {os.path.basename(save_path)}. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"--- An error occurred while downloading {os.path.basename(save_path)}: {str(e)}")
        return False

# load the nd2 files and begin the analysis
def load_data(nd2filepath):
    file_name_without_extension = nd2filepath[:-4]
    experiment_dir = file_name_without_extension
    stacks_dir = file_name_without_extension + "/stacks"

    # Create directories if they don't exist
    print("=========================================")
    print("Creating folders for the stacks and metadata...")
    os.makedirs(stacks_dir, exist_ok=True)

    # Save the metadata of the ND2 file itself
    with pims_nd2.ND2_Reader(nd2filepath) as ndfile:
        metadata_dict = ndfile.metadata
        json.dump(metadata_dict, open(experiment_dir + "/experiment_metadata.json", 'w'), default=str)

    # Get dimensions and frame count
    with nd2.ND2File(nd2filepath) as ndfile:
        dimension1 = 1
        dimension2 = 1
        try:
            dimension1 = ndfile.shape[-4]
            dimension2 = ndfile.shape[-5]
        except:
            pass

        tiff_frame_count = ndfile.shape[-3]

        print("--- Data:", file_name_without_extension)
        print("--- #Stacks:", dimension1 * dimension2)
        print("--- #Frames per Stack:", tiff_frame_count)  
        # get the perfectly focused frame
        print("--- Optimal Focus Frame Index",  (1 + tiff_frame_count) // 2 )# Calculate the middle frame index

        image_arrays = ndfile.asarray().reshape((dimension1 * dimension2 * tiff_frame_count, ndfile.shape[-2], ndfile.shape[-1]))

        # one nd2 file is made of multiple stacks
        # one stack is made of multiple frames
        for tiff_stack_counter in range(dimension1 * dimension2):
            stack_images = []
            print("=========Stack#:" + str(tiff_stack_counter))
            # create a dir for the individual stack
            stack_dir = os.path.join(stacks_dir, f"stack_{tiff_stack_counter}")
            os.makedirs(stack_dir, exist_ok=True)
            # loop for the frames within the stack
            for frame_index in range(tiff_frame_count):
                # create the necessary folder for frames
                os.makedirs(stack_dir + "/frames", exist_ok=True)
                image_array = image_arrays[tiff_stack_counter * tiff_frame_count + frame_index]
                image_array = image_array.astype(np.uint16)
                image = Image.fromarray(image_array, 'I;16')
                stack_images.append(image)
                # Save the frame under the /frames subdirectory
                frame = os.path.join(stack_dir, "frames", f"frame_{frame_index}.png")
                image.save(frame, format="png")

            str_tiff_stack_counter = str(tiff_stack_counter % dimension2)
            stack_key = "i" + (10 - len(str_tiff_stack_counter)) * "0" + str_tiff_stack_counter
            positions_metadata = ndfile.unstructured_metadata()["ImageMetadataLV"]['SLxExperiment']["ppNextLevelEx"]["i0000000000"]["uLoopPars"]["Points"][stack_key]
            print("=Metadata:" , positions_metadata)
            json.dump(positions_metadata, open(stack_dir + "/stack_" + str(tiff_stack_counter) + "_metadata.json", 'w'))
            image1 = stack_images[0]
            image1.save(stack_dir + "/stack_" + str(tiff_stack_counter) + ".tiff", format="tiff", append_images=stack_images[1:], save_all=True, duration=500, loop=0)

    # return the path of the created analysis folder
    return stacks_dir

if __name__ == "__main__":

    # URLs for the ND2 files
    url_to_zstackonechamber_nd2_file = zstack_once_with_full_chambers.nd2"
    url_to_zstackallchambers_nd2_file = "zstackexperiment_29082023.nd2"

    # Determine the directory where the script is located
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Define paths for ND2 files
    zstack_one_chamber_nd2_path = os.path.join(script_directory, "zstack_once_with_full_chambers.nd2")
    zstack_all_chambers_nd2_path = os.path.join(script_directory, "zstackexperiment_29082023.nd2")

    # Check if the ND2 files already exist and perform download or load accordingly
    if not os.path.exists(zstack_one_chamber_nd2_path) and not os.path.exists(zstack_all_chambers_nd2_path):
        print("=========================================")
        print("Downloading ND2 files...")

        if not os.path.exists(zstack_one_chamber_nd2_path):
            isdownloaded_onechamber = get_data(url_to_zstackonechamber_nd2_file, zstack_one_chamber_nd2_path)
        else:
            print("--- ND2 file already exists: zstack_once_with_full_chambers.nd2")
            isdownloaded_onechamber = True

        if not os.path.exists(zstack_all_chambers_nd2_path):
            isdownloaded_allchambers = get_data(url_to_zstackallchambers_nd2_file, zstack_all_chambers_nd2_path)
        else:
            print("--- ND2 file already exists: zstackexperiment_29082023.nd2")
            isdownloaded_allchambers = True

        if isdownloaded_onechamber and isdownloaded_allchambers:
            print("=========================================")
            print("Loading data...")
            #metadata_dir_one_chamber = load_data(zstack_one_chamber_nd2_path)
            #metadata_dir_all_chambers = load_data(zstack_all_chambers_nd2_path)

    else:
        print("=========================================")
        print("ND2 files already exist. Skipping download and begining the analysis.")
        # do the first experiment
        stacks_dir = load_data(zstack_one_chamber_nd2_path)
        analyze_stacks(stacks_dir)
        create_plots(stacks_dir)
        # do the second experiment
        stacks_dir = load_data(zstack_all_chambers_nd2_path)
        create_plots(stacks_dir) 
        analyze_stacks(stacks_dir)

    print("=========================================")
    print("Data processing completed.")
