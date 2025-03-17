import numpy as np
import wave


from utils import read_csv_data, extract_columns


def sine_strategy(data, sample_rate=44100, tone_duration=0.5):
    """
    Basic sine wave sound generation strategy.
    
    Creates a sequence of pure tones (sine waves) where:
    - Each data point becomes a single tone segment
    - The x-value controls frequency (pitch)
    - The y-value controls amplitude (volume)
    
    Parameters:
    - data: List of (x,y) coordinates from CSV
    - sample_rate: Number of samples per second (44.1kHz CD quality)
    - tone_duration: Length of each tone segment in seconds
    """
    samples = []
    max_amplitude = 32767 * 0.7  # 70% of maximum 16-bit amplitude to avoid distortion
    min_amplitude = max_amplitude * 0.2  # Minimum amplitude to ensure audibility
    
    # Process each data point as a discrete tone
    for y, x in data:
        # Map x to frequency (Hz): higher x = higher pitch
        # Base frequency of 200Hz + scaling factor creates audible range
        frequency = 200 + x * 50  # arbitrary mapping
        
        # Map y to amplitude (volume) with a guaranteed minimum
        # This ensures sound is always produced
        amplitude_factor = 0.5 * (1 + np.tanh(y))  # Maps to [0-1] range
        amplitude = min_amplitude + (max_amplitude - min_amplitude) * amplitude_factor
        
        # Create time axis for this tone segment
        t = np.linspace(
            0, tone_duration, int(sample_rate * tone_duration), endpoint=False
        )
        
        # Generate sine wave for this tone
        tone = amplitude * np.sin(2 * np.pi * frequency * t)
        samples.append(tone)
    
    # Join all tone segments into continuous audio stream
    return np.concatenate(samples) if samples else np.array([])


def phase_mod_strategy(data, sample_rate=44100, tone_duration=0.5):
    """
    Phase modulation sound generation strategy.
    
    Creates more complex tones using phase modulation (PM):
    - A carrier wave (base frequency) is modulated by a secondary wave
    - Creates richer, more dynamic sounds than pure sine waves
    - Similar to FM synthesis used in many synthesizers
    
    Parameters:
    - data: List of (x,y) coordinates from CSV
    - sample_rate: Number of samples per second (44.1kHz CD quality)
    - tone_duration: Length of each tone segment in seconds
    """
    samples = []
    max_amplitude = 32767 * 0.7  # 70% of maximum 16-bit amplitude
    
    # Process each data point as a modulated tone
    for x, y in data:
        # Map x to carrier/base frequency (Hz)
        base_freq = 200 + x * 5
        
        # Fixed modulation frequency creates consistent sound character
        modulation_freq = 5  # modulation frequency in Hz
        
        # y controls modulation depth/intensity
        # Higher y values create more dramatic modulation
        modulation_index = y * 2  # arbitrary modulation depth scaling
        
        # Create time axis for this tone segment
        t = np.linspace(
            0, tone_duration, int(sample_rate * tone_duration), endpoint=False
        )
        
        # Generate phase-modulated wave:
        # 1. Inner sin(): creates modulator wave that varies over time
        # 2. Multiplied by modulation_index: controls modulation intensity
        # 3. Added to carrier phase (2π * base_freq * t): shifts carrier phase
        # 4. Outer sin(): final modulated carrier wave
        # 5. Scaled by amplitude: controls overall volume
        tone = max_amplitude * np.sin(
            2 * np.pi * base_freq * t
            + modulation_index * np.sin(2 * np.pi * modulation_freq * t)
        )
        samples.append(tone)
    
    # Join all tone segments into continuous audio stream
    return np.concatenate(samples) if samples else np.array([])


def chord_strategy(data, sample_rate=44100, tone_duration=0.5):
    """
    Chord-based sound generation strategy.
    
    Creates harmonically rich sounds by combining multiple sine waves:
    - Each data point produces three simultaneous tones (a triad)
    - Base frequency plus two additional frequencies (±20 Hz)
    - Creates fuller, more harmonic sound than single sine waves
    
    Parameters:
    - data: List of (x,y) coordinates from CSV
    - sample_rate: Number of samples per second (44.1kHz CD quality)
    - tone_duration: Length of each tone segment in seconds
    """
    samples = []
    max_amplitude = 32767 * 0.7  # 70% of maximum 16-bit amplitude
    min_amplitude = max_amplitude * 0.2  # Minimum amplitude to ensure audibility
    
    # Process each data point as a three-tone chord
    for x, y in data:
        # Map x to the central frequency (Hz)
        frequency = 200 + x * 5
        
        # Map y to amplitude (volume) with a guaranteed minimum
        amplitude_factor = 0.5 * (1 + np.tanh(y))  # Maps to [0-1] range
        amplitude = min_amplitude + (max_amplitude - min_amplitude) * amplitude_factor
        
        # Create time axis for this chord segment
        t = np.linspace(
            0, tone_duration, int(sample_rate * tone_duration), endpoint=False
        )
        
        # Generate three separate tones with related frequencies
        tone1 = amplitude * np.sin(2 * np.pi * frequency * t)
        tone2 = amplitude * np.sin(2 * np.pi * (frequency + 20) * t)
        tone3 = amplitude * np.sin(2 * np.pi * (frequency - 20) * t)
        
        # Mix the three tones together
        tone = (tone1 + tone2 + tone3) / 3
        samples.append(tone)
    
    # Join all chord segments into continuous audio stream
    return np.concatenate(samples) if samples else np.array([])


STRATEGY_MAP = {
    "sine": sine_strategy,
    "phase_mod": phase_mod_strategy,
    "chord": chord_strategy,
}


def generate_audio_file(
    csv_file, strategy_name=None, strategy_func=None, output_file="output.wav"
):
    data = extract_columns(read_csv_data(csv_file))
    sample_rate = 44100
    tone_duration = 0.5  # duration for each tone segment in seconds

    if strategy_func is None and strategy_name is None:
        raise ValueError("Either strategy_name or strategy_func must be provided")

    if strategy_func is not None and strategy_name is not None:
        raise ValueError("Only one of strategy_name or strategy_func can be provided")

    if strategy_func is None:
        strategy_func = STRATEGY_MAP.get(strategy_name, sine_strategy)
    else:
        strategy_func = strategy_func

    samples = strategy_func(data, sample_rate=sample_rate, tone_duration=tone_duration)

    if samples.size == 0:
        print("No valid data found in CSV file.")
        return

    with wave.open(output_file, "wb") as wf:
        wf.setnchannels(1)  # mono
        wf.setsampwidth(2)  # 16-bit samples
        wf.setframerate(sample_rate)
        wf.writeframes(samples.astype(np.int16).tobytes())

    print(f"Audio file generated and saved as {output_file}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate an audio file from CSV data using algorithmic strategies."
    )
    parser.add_argument(
        "csv_file", help="Path to the input CSV file containing two columns"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--strategy_name",
        default="sine",
        choices=["sine", "phase_mod", "chord"],
        help="Strategy to use for generating sound",
    )
    group.add_argument(
        "--strategy_func",
        default=None,
        help="Strategy function to use for generating sound",
    )
    parser.add_argument("--output", default="output.wav", help="Output WAV file name")
    args = parser.parse_args()
    
    # Check if both strategy options were provided
    if args.strategy_name != "sine" and args.strategy_func is not None:
        parser.error("Only one of --strategy_name or --strategy_func can be provided")
        
    generate_audio_file(
        args.csv_file,
        strategy_name=args.strategy_name if args.strategy_func is None else None,
        strategy_func=args.strategy_func,
        output_file=args.output,
    )
