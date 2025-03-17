#!/usr/bin/env python3

import argparse
import os
from strategies import generate_audio_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sound Generator CLI that generates sound from a two-column CSV file using algorithmic strategies."
    )
    parser.add_argument("csv_file", help="Path to the input CSV file containing two columns")
    parser.add_argument(
        "--strategy",
        type=str,
        default="sine",
        choices=["sine", "phase_mod", "chord", "all"],
        help="Sound generation strategy to use"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/output.wav",
        help="Output WAV file path"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    if args.strategy == "all":
        # Define all available strategies
        all_strategies = ["sine", "phase_mod", "chord"]
        
        # Loop through each strategy and generate audio files
        for strategy in all_strategies:
            # Create strategy-specific output filename
            output_filename = os.path.basename(args.output)
            output_name, output_ext = os.path.splitext(output_filename)
            strategy_output = os.path.join(output_dir, f"{output_name}_{strategy}{output_ext}.wav")
            
            # Generate audio file for current strategy
            generate_audio_file(args.csv_file, strategy, strategy_output)
            print(f"Generated {strategy} audio: {strategy_output}")
    else:
        # Generate single audio file with specified strategy
        generate_audio_file(args.csv_file, args.strategy, args.output)


if __name__ == '__main__':
    main()
