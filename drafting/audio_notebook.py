# %% [markdown]
# # Audio Data Analysis Notebook
# 
# This notebook processes CSV data to analyze and generate audio representations of the data patterns. 
# It includes data loading, transformation, visualization, and sound generation to help identify patterns 
# through both visual and auditory means.


# %%
import os
from utils import read_csv_data, extract_columns

csv_file = os.path.join("..", "data", "fig3b_multilevel.csv")

# Create a platform-independent path to the output directory
output_directory = os.path.join("..", "output", "sound")

# Ensure the directory exists
os.makedirs(output_directory, exist_ok=True)

df = read_csv_data(csv_file)
df.head()

# %%
import polars as pl
import numpy as np

# Calculate the Euclidean distance between consecutive points in 2D space
# For the first value, we'll use 0 as there's no previous point to compare with
df_with_dist = df.with_columns([
    (
        (pl.col("microseconds,").diff() ** 2 + 
         pl.col("optical_contrast").diff() ** 2).sqrt()
    ).fill_null(0).alias("_euclidean_distance")
])

# Display the dataframe with the new column
print("DataFrame with Euclidean distance between consecutive points:")
df_with_dist.head(10)


# %%
import matplotlib.pyplot as plt
from visualize import plot_data

# Create a figure with subplots in a grid layout
plt.figure(figsize=(15, 10))

# Get the data from the CSV file
data = extract_columns(read_csv_data(csv_file))
if data:
    x_vals, y_vals = zip(*data)
    
    # Time Series Plot
    plt.subplot(2, 2, 1)
    time_index = list(range(len(x_vals)))
    plt.plot(time_index, x_vals, marker='o', label='Microseconds')
    plt.plot(time_index, y_vals, marker='o', label='Optical Contrast')
    plt.xlabel('Time (index)')
    plt.ylabel('Value')
    plt.title('Time Series Plot of Data')
    plt.legend()
    plt.grid(True)

    # Line plot
    plt.subplot(2, 2, 2)
    plt.plot(x_vals, y_vals, marker='o', color='red')
    plt.xlabel('Microseconds')
    plt.ylabel('Optical Contrast')
    plt.title('Line Plot of CSV Data')
    plt.grid(True)
    
    # Histogram for Y (optical contrast)
    plt.subplot(2, 2, 3)
    plt.hist(y_vals, bins=20, color='purple', alpha=0.7)
    plt.xlabel('Optical Contrast')
    plt.ylabel('Frequency')
    plt.title('Histogram of Optical Contrast')
    plt.grid(True)

    # Bar plot of the difference between consecutive optical contrast values
    plt.subplot(2, 2, 4)
    plt.bar(df_with_dist.get_column("microseconds,").to_numpy(), df_with_dist.get_column("_euclidean_distance").to_numpy(), color='blue', width=0.001)
    plt.xlabel('Microseconds')
    plt.ylabel('Euclidean Distance between Consecutive Points in 2D Space')
    plt.title('Bar Plot of Euclidean Distance between Consecutive Points in 2D Space')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
else:
    print("No valid data found in CSV file.")


# %%
from strategies import STRATEGY_MAP, generate_audio_file

for strategy_name in STRATEGY_MAP.keys():
    output_file = os.path.join(output_directory, f"{strategy_name}.wav")
    generate_audio_file(csv_file, strategy_name=strategy_name, output_file=output_file)
    print(f"Generated audio using {strategy_name} strategy")

# %%
import IPython.display as ipd
import wave
import numpy as np

# Display audio information and player for each strategy
for strategy_name in STRATEGY_MAP.keys():
    wave_path = os.path.join(output_directory, f"{strategy_name}.wav")
    
    # Open and display basic info about the wave file
    with wave.open(wave_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        
        # Print audio file information
        print(f"\n--- {strategy_name.upper()} STRATEGY ---")
        print(f"Channels: {n_channels}")
        print(f"Sample Width: {sample_width} bytes")
        print(f"Frame Rate: {framerate} Hz")
        print(f"Number of Frames: {n_frames}")
        print(f"Duration: {n_frames / framerate:.2f} seconds")
    
    # Display audio player in the notebook
    print(f"Audio player for {strategy_name} strategy:")
    display(ipd.Audio(wave_path))

# %%
from visualize import plot_spectogram_of_generated_audio
import matplotlib.pyplot as plt
from IPython.display import display

# Display spectrogram for each strategy audio file
for strategy_name in STRATEGY_MAP.keys():
    print(f"\n--- {strategy_name.upper()} STRATEGY SPECTROGRAM ---")
    wave_path = os.path.join(output_directory, f"{strategy_name}.wav")
    # Generate the spectrogram
    plot_spectogram_of_generated_audio(wave_path, output_directory)
    
    # Display the spectrogram in the notebook
    spectogram_path = os.path.join(output_directory, 'spectogram.png')
    plt.figure(figsize=(10, 6))
    img = plt.imread(spectogram_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Spectrogram of {strategy_name} Strategy Audio')
    plt.tight_layout()
    display(plt.gcf())
    plt.close()






# %%
