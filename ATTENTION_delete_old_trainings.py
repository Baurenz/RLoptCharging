import pathlib
import shutil

# Define the directories to clean up
directories = [
    pathlib.Path("solvers/models/PPO"),
    pathlib.Path("solvers/results/logs/PPO"),
    pathlib.Path("solvers/models/DDPG"),
    pathlib.Path("solvers/results/logs/DDPG"),
    pathlib.Path("solvers/models/A2C"),
    pathlib.Path("solvers/results/logs/A2C"),
    pathlib.Path("rl_OptV2GEnv/files/Results"),
    pathlib.Path("network_env/results")
]

# Iterate over each directory
for directory in directories:
    # Check if directory exists
    if directory.exists() and directory.is_dir():
        # Get a list of all subdirectories
        subdirectories = [d for d in directory.iterdir() if d.is_dir()]

        # Check if there are more than one subdirectory
        if len(subdirectories) > 1:
            # Sort the subdirectories by modification time
            subdirectories.sort(key=lambda dir: dir.stat().st_mtime)

            # Delete all subdirectories except the last one
            for subdir in subdirectories[:-1]:
                # Use shutil's rmtree function to remove the directory and all its contents
                shutil.rmtree(subdir)
                print(f"Deleted: {subdir}")

        else:
            print(f"No directories to delete in: {directory}")
    else:
        print(f"Directory does not exist: {directory}")
