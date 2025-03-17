#!/usr/bin/env python3

import argparse
import csv
import os
import matplotlib.pyplot as plt


def read_csv_data(csv_file):
    data = []
    with open(csv_file, newline='') as f:
        sample = f.read(1024)
        f.seek(0)
        if ',' in sample:
            reader = csv.reader(f, delimiter=',')
            rows = list(reader)
        else:
            rows = [line.strip().split() for line in f if line.strip()]
        header = False
        if rows:
            try:
                float(rows[0][0])
                float(rows[0][1])
            except Exception:
                header = True
        for row in (rows[1:] if header else rows):
            if len(row) < 2:
                continue
            try:
                x = float(row[0])
                y = float(row[1])
                data.append((x, y))
            except Exception:
                continue
    return data


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


def main():
    parser = argparse.ArgumentParser(description="Generate graphs for a given CSV file.")
    parser.add_argument("csv_file", help="Path to the input CSV file")
    parser.add_argument("output_folder", help="Folder to save the graphs")
    args = parser.parse_args()
    
    plot_data(args.csv_file, args.output_folder)
    

if __name__ == '__main__':
    main()
