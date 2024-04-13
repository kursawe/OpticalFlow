import os

# Directory containing the files
directory = os.path.join(os.path.dirname(__file__),'data','12_grayscale') 

# Loop through files in the directory
for filename in os.listdir(directory):
    # Check if the file starts with 'file_'
    if filename.startswith("file_"):
        # Extract the numeric part of the filename
        # number_str = filename.split("_")[1]
        number_str = filename.split("_")[2].split(".")[0]
        number = int(number_str)
        # Construct the new filename with zero-padding
        new_filename = "12_grayscale_{:03d}.tif".format(number)
        # Rename the file
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))