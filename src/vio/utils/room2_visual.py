import os

# Put the path you are trying to access here
path_to_check = r"C:\Users\selva\Downloads\TUM_VI\dataset-room2_512_16"

# Check if the folder exists at all
if os.path.exists(path_to_check):
    print(f"Folder exists! Contents: {os.listdir(path_to_check)}")
    
    # Check if there is a subfolder inside (like 'mav0')
    for root, dirs, files in os.walk(path_to_check):
        if "imu0" in dirs:
            print(f"Found 'imu0' inside: {root}")
            print(f"Contents of imu0: {os.listdir(os.path.join(root, 'imu0'))}")
            break
else:
    print("The main folder path is wrong. Please double-check the folder name in File Explorer.")