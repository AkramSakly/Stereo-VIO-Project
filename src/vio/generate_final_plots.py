import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_comparison(dataset_name):
    # Update these names to match your actual files in C:\Users\selva\vio_project\
    # If you don't have the old Week 6 files anymore, we can just plot the new ones.
    file_vio = f"St_{dataset_name}_Akram.txt" 
    
    data_vio = pd.read_csv(file_vio, sep=" ", header=None, 
                           names=['ts', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

    # ... rest of the plotting code ...
if __name__ == "__main__":
    datasets = ["dataset-outdoors5_512_16", "dataset-room2_512_16", "dataset-corridor3_512_16"]
    for ds in datasets:
        try:
            plot_comparison(ds)
        except Exception as e:
            print(f"Skipping {ds}: {e}")