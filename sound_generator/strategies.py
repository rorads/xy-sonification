import csv
import numpy as np
import wave


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


def sine_strategy(data, sample_rate=44100, tone_duration=0.5):
    samples = []
    max_amplitude = 32767 * 0.7
    # Map the first column to frequency and second to amplitude (normalized between 0 and 1)
    for x, y in data:
        frequency = 200 + x * 5  # arbitrary mapping
        amplitude = max_amplitude * max(0, min(y, 1))
        t = np.linspace(0, tone_duration, int(sample_rate * tone_duration), endpoint=False)
        tone = amplitude * np.sin(2 * np.pi * frequency * t)
        samples.append(tone)
    return np.concatenate(samples) if samples else np.array([])


def phase_mod_strategy(data, sample_rate=44100, tone_duration=0.5):
    samples = []
    max_amplitude = 32767 * 0.7
    # Use phase modulation to vary the tone
    for x, y in data:
        base_freq = 200 + x * 5
        modulation_freq = 5  # modulation frequency in Hz
        modulation_index = y * 2  # arbitrary modulation depth scaling
        t = np.linspace(0, tone_duration, int(sample_rate * tone_duration), endpoint=False)
        tone = max_amplitude * np.sin(2 * np.pi * base_freq * t + modulation_index * np.sin(2 * np.pi * modulation_freq * t))
        samples.append(tone)
    return np.concatenate(samples) if samples else np.array([])


def chord_strategy(data, sample_rate=44100, tone_duration=0.5):
    samples = []
    max_amplitude = 32767 * 0.7
    # Create a chord by summing three sine waves around the base frequency
    for x, y in data:
        frequency = 200 + x * 5
        amplitude = max_amplitude * max(0, min(y, 1))
        t = np.linspace(0, tone_duration, int(sample_rate * tone_duration), endpoint=False)
        tone1 = amplitude * np.sin(2 * np.pi * frequency * t)
        tone2 = amplitude * np.sin(2 * np.pi * (frequency + 20) * t)
        tone3 = amplitude * np.sin(2 * np.pi * (frequency - 20) * t)
        tone = (tone1 + tone2 + tone3) / 3
        samples.append(tone)
    return np.concatenate(samples) if samples else np.array([])


def generate_audio_file(csv_file, strategy, output_file="output.wav"):
    data = read_csv_data(csv_file)
    sample_rate = 44100
    tone_duration = 0.5  # duration for each tone segment in seconds
    strategy_map = {
        "sine": sine_strategy,
        "phase_mod": phase_mod_strategy,
        "chord": chord_strategy
    }
    strategy_func = strategy_map.get(strategy, sine_strategy)
    samples = strategy_func(data, sample_rate=sample_rate, tone_duration=tone_duration)

    if samples.size == 0:
        print("No valid data found in CSV file.")
        return

    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(1)          # mono
        wf.setsampwidth(2)          # 16-bit samples
        wf.setframerate(sample_rate)
        wf.writeframes(samples.astype(np.int16).tobytes())

    print(f"Audio file generated and saved as {output_file}.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate an audio file from CSV data using algorithmic strategies.")
    parser.add_argument("csv_file", help="Path to the input CSV file containing two columns")
    parser.add_argument("--strategy", default="sine", choices=["sine", "phase_mod", "chord"], help="Strategy to use for generating sound")
    parser.add_argument("--output", default="output.wav", help="Output WAV file name")
    args = parser.parse_args()
    generate_audio_file(args.csv_file, args.strategy, args.output)
