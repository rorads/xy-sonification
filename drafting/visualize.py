#!/usr/bin/env python3

import argparse
import os
import matplotlib.pyplot as plt
import polars as pl
import soundfile as sf

from utils import read_csv_data


def plot_data(csv_file, output_folder):
    data = read_csv_data(csv_file)
    if not data:
        print("No valid data found in CSV file.")
        return
    x_vals, y_vals = zip(*data)
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Scatter plot
    plt.figure(figsize=(8,6))
    plt.scatter(x_vals, y_vals, c='blue')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot of CSV Data')
    plt.grid(True)
    scatter_file = os.path.join(output_folder, 'scatter_plot.png')
    plt.savefig(scatter_file)
    print(f"Scatter plot saved to {scatter_file}")
    plt.close()
    
    # Line plot
    plt.figure(figsize=(8,6))
    plt.plot(x_vals, y_vals, marker='o', color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Line Plot of CSV Data')
    plt.grid(True)
    line_file = os.path.join(output_folder, 'line_plot.png')
    plt.savefig(line_file)
    print(f"Line plot saved to {line_file}")
    plt.close()
    
    # Histograms for X and Y
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.hist(x_vals, bins=20, color='green', alpha=0.7)
    plt.xlabel('X')
    plt.ylabel('Frequency')
    plt.title('Histogram of X')
    
    plt.subplot(1,2,2)
    plt.hist(y_vals, bins=20, color='purple', alpha=0.7)
    plt.xlabel('Y')
    plt.ylabel('Frequency')
    plt.title('Histogram of Y')
    
    hist_file = os.path.join(output_folder, 'histogram.png')
    plt.savefig(hist_file)
    print(f"Histogram saved to {hist_file}")
    plt.close()
    
    # Time Series Plot of X and Y
    plt.figure(figsize=(8,6))
    time_index = list(range(len(x_vals)))
    plt.plot(time_index, x_vals, marker='o', label='X')
    plt.plot(time_index, y_vals, marker='o', label='Y')
    plt.xlabel('Time (index)')
    plt.ylabel('Value')
    plt.title('Time Series Plot of X and Y')
    plt.legend()
    plt.grid(True)
    timeseries_file = os.path.join(output_folder, 'timeseries_plot.png')
    plt.savefig(timeseries_file)
    print(f"Time series plot saved to {timeseries_file}")
    plt.close()


def plot_spectogram_of_generated_audio(audio_file, output_folder):
    """
    Plot the spectogram of the generated audio file.
    """
    audio_data, sample_rate = sf.read(audio_file)
    plt.figure(figsize=(10, 6))
    plt.specgram(audio_data, Fs=sample_rate)
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Spectogram of Generated Audio')
    
    os.makedirs(output_folder, exist_ok=True)
    spec_file = os.path.join(output_folder, 'spectogram.png')
    plt.savefig(spec_file)
    print(f"Spectogram saved to {spec_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate graphs for a given CSV file or audio file.")
    parser.add_argument("input_file", help="Path to the input csv or audio (.wav) file")
    parser.add_argument("output_folder", help="Folder to save the visualisations")
    args = parser.parse_args()

    if args.input_file.endswith(".csv"):
        plot_data(args.input_file, args.output_folder)
    elif args.input_file.endswith(".wav"):
        plot_spectogram_of_generated_audio(args.input_file, args.output_folder)
    else:
        print("Invalid file type. Please provide a CSV or WAV file.")

if __name__ == '__main__':
    main()
