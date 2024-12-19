import os
import shutil

def remove_files(folder):
    """Function to remove all the previous outputs from the output folder

    Args:
        folder (str): output folder path
    """
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.is_file():
                    os.unlink(entry.path)
                else:
                    shutil.rmtree(entry.path)
    except OSError:
        print("Error occurred while deleting files and subdirectories.")
        
def get_folder_name(base_path):
    """Function to create a new directory

    Args:
        base_path (string): Path to the output folder
    """
    
    # Get a list of subdirectories in the base_path
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if subdirs:
        # Convert directory names to integers
        dir_number = sorted([int(d) for d in subdirs])
        # Find the highest number and add 1 for the new directory
        next_dir_number = dir_number[-1] + 1
    else:
        # If there are no numbered directories, start with 0
        next_dir_number = 0
    # Create the new directory name
    new_dir_name = str(next_dir_number)
    # Create the new directory path
    new_dir_path = os.path.join(base_path, new_dir_name)
    # Create the new directory
    os.makedirs(new_dir_path)
    return next_dir_number