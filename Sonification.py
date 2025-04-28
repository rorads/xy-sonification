import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from scipy.io import wavfile
import librosa
import librosa.display
import os

# Set page config
st.set_page_config(
    page_title="Data Sonification Playground", 
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add custom CSS for audio player styling
st.markdown("""
<style>
audio::-webkit-media-controls-panel,
audio::-webkit-media-controls-enclosure {
    background-color: #3f8f54;  /* Pastel green background */
}

audio::-webkit-media-controls-time-remaining-display,
audio::-webkit-media-controls-current-time-display {
    color: #000000;  /* Black text */
    text-shadow: none;
}

audio::-webkit-media-controls-timeline {
    background-color: #80ba90;  /* Lighter pastel green */
    border-radius: 25px;
    margin-left: 10px;
    margin-right: 10px;
}
</style>
""", unsafe_allow_html=True)

# Utility functions
def read_csv_data(csv_file):
    """Read CSV data and return as pandas DataFrame"""
    try:
        df = pd.read_csv(csv_file, sep=None, engine="python")
        return df
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None


def extract_columns(df):
    """Extract x and y columns from DataFrame"""
    if df is None or df.empty:
        return [], []

    columns = df.columns
    if len(columns) < 2:
        st.error("CSV file must have at least two columns")
        return [], []

    x_col, y_col = columns[0], columns[1]

    # Convert to numeric values, skipping any non-numeric values
    x_vals = pd.to_numeric(df[x_col], errors="coerce")
    y_vals = pd.to_numeric(df[y_col], errors="coerce")

    # Drop NaN values
    mask = ~(x_vals.isna() | y_vals.isna())
    return x_vals[mask].values, y_vals[mask].values


def normalize_data(data, min_val=0, max_val=1):
    """Normalize data to range [min_val, max_val]"""
    data_min, data_max = np.min(data), np.max(data)
    if data_min == data_max:
        return np.full_like(data, (min_val + max_val) / 2)
    return min_val + (data - data_min) * (max_val - min_val) / (data_max - data_min)


def create_spectrogram(audio_data, sample_rate):
    """Create spectrogram from audio data"""
    fig, ax = plt.subplots(figsize=(10, 4))
    S = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    img = librosa.display.specshow(
        S, y_axis="log", x_axis="time", ax=ax, sr=sample_rate
    )
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.tight_layout()
    return fig


def get_download_link(audio_data, sample_rate, filename="sonification.wav"):
    """Generate a download link for audio data"""
    virtual_file = io.BytesIO()
    wavfile.write(virtual_file, sample_rate, audio_data.astype(np.int16))
    virtual_file.seek(0)
    b64 = base64.b64encode(virtual_file.read()).decode()
    return f'<a href="data:audio/wav;base64,{b64}" download="{filename}">Download {filename}</a>'


# Sonification strategies
def sine_wave_strategy(
    x_vals,
    y_vals,
    sample_rate=22050,
    tone_duration=0.05,
    min_freq=220,
    max_freq=880,
    amplitude_scale=0.8,
):
    """
    Basic sine wave sonification:
    - Maps x values to time
    - Maps y values to frequency
    """
    # Normalize y values to frequency range
    norm_y = normalize_data(y_vals, min_val=min_freq, max_val=max_freq)

    # Calculate total duration based on number of points and tone_duration
    total_samples = int(len(x_vals) * tone_duration * sample_rate)
    audio_data = np.zeros(total_samples)

    # Generate audio
    for i, freq in enumerate(norm_y):
        # Calculate start and end sample for this tone
        start_sample = int(i * tone_duration * sample_rate)
        end_sample = int((i + 1) * tone_duration * sample_rate)
        if end_sample > total_samples:
            end_sample = total_samples

        # Generate time array for this segment
        t = np.linspace(0, tone_duration, end_sample - start_sample, endpoint=False)

        # Apply envelope to avoid clicks (simple linear fade in/out)
        envelope = np.ones_like(t)
        fade_samples = int(0.1 * len(t))  # 10% fade in/out
        if fade_samples > 0:
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

        # Generate sine wave and apply envelope
        segment = amplitude_scale * 32767 * np.sin(2 * np.pi * freq * t) * envelope

        # Add to audio data
        audio_data[start_sample:end_sample] = segment[
            : len(audio_data[start_sample:end_sample])
        ]

    return audio_data


def fm_synthesis_strategy(
    x_vals,
    y_vals,
    sample_rate=22050,
    tone_duration=0.05,
    carrier_freq=440,
    mod_index_min=1,
    mod_index_max=5,
    amplitude_scale=0.8,
):
    """
    FM synthesis sonification:
    - Maps x values to time
    - Maps y values to modulation index
    - Creates more complex, dynamic sounds
    """
    # Normalize y values to modulation index range
    mod_indices = normalize_data(y_vals, min_val=mod_index_min, max_val=mod_index_max)

    # Calculate total duration based on number of points and tone_duration
    total_samples = int(len(x_vals) * tone_duration * sample_rate)
    audio_data = np.zeros(total_samples)

    # Generate audio
    for i, mod_index in enumerate(mod_indices):
        # Calculate start and end sample for this tone
        start_sample = int(i * tone_duration * sample_rate)
        end_sample = int((i + 1) * tone_duration * sample_rate)
        if end_sample > total_samples:
            end_sample = total_samples

        # Generate time array for this segment
        t = np.linspace(0, tone_duration, end_sample - start_sample, endpoint=False)

        # Apply envelope to avoid clicks
        envelope = np.ones_like(t)
        fade_samples = int(0.1 * len(t))  # 10% fade in/out
        if fade_samples > 0:
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

        # Calculate FM-modulated wave
        # Modulator frequency is a fraction of carrier frequency
        mod_freq = carrier_freq / 2
        # Phase modulation: carrier + modulation * sin(mod_freq * t)
        phase = 2 * np.pi * carrier_freq * t + mod_index * np.sin(
            2 * np.pi * mod_freq * t
        )
        segment = amplitude_scale * 32767 * np.sin(phase) * envelope

        # Add to audio data
        audio_data[start_sample:end_sample] = segment[
            : len(audio_data[start_sample:end_sample])
        ]

    return audio_data


def granular_synthesis_strategy(
    x_vals, y_vals, sample_rate=22050, grain_size=0.02, density=0.5, amplitude_scale=0.8
):
    """
    Granular synthesis sonification:
    - Creates small sound grains from the data
    - Maps x values to time positioning
    - Maps y values to frequency content
    - Creates textural, evolving sounds
    """
    # Normalize x and y values
    norm_x = normalize_data(x_vals)
    norm_y = normalize_data(y_vals)

    # Calculate total duration - slightly longer to account for grain overlap
    total_duration = len(x_vals) * grain_size * (1 + density)
    total_samples = int(total_duration * sample_rate)
    audio_data = np.zeros(total_samples)

    # Calculate grain size in samples
    grain_samples = int(grain_size * sample_rate)

    # Create window function for smoothing grains (Hann window)
    window = np.hanning(grain_samples)

    # Generate grains
    for i, (x, y) in enumerate(zip(norm_x, norm_y)):
        # Map y to frequency between 110Hz and 880Hz
        freq = 110 + 770 * y

        # Position is determined by normalized x value, scaled by total duration
        position = int(x * (total_samples - grain_samples))

        # Generate grain
        t = np.linspace(0, grain_size, grain_samples, endpoint=False)
        grain = amplitude_scale * 32767 * np.sin(2 * np.pi * freq * t) * window

        # Add grain to audio data
        end_pos = position + grain_samples
        if end_pos > total_samples:
            end_pos = total_samples
        audio_data[position:end_pos] += grain[: end_pos - position]

    # Normalize to avoid clipping
    if np.max(np.abs(audio_data)) > 32767:
        audio_data = 32767 * audio_data / np.max(np.abs(audio_data))

    return audio_data


def harmonic_mapping_strategy(
    x_vals,
    y_vals,
    sample_rate=22050,
    tone_duration=0.05,
    base_freq=110,
    num_harmonics=8,
    amplitude_scale=0.8,
):
    """
    Harmonic mapping sonification:
    - Maps x values to time
    - Maps y values to harmonic content
    - Creates richer timbral experience
    """
    # Normalize y values
    norm_y = normalize_data(y_vals)

    # Calculate total duration
    total_samples = int(len(x_vals) * tone_duration * sample_rate)
    audio_data = np.zeros(total_samples)

    # Generate audio
    for i, y in enumerate(norm_y):
        # Calculate start and end sample for this tone
        start_sample = int(i * tone_duration * sample_rate)
        end_sample = int((i + 1) * tone_duration * sample_rate)
        if end_sample > total_samples:
            end_sample = total_samples

        # Generate time array for this segment
        t = np.linspace(0, tone_duration, end_sample - start_sample, endpoint=False)

        # Apply envelope to avoid clicks
        envelope = np.ones_like(t)
        fade_samples = int(0.1 * len(t))  # 10% fade in/out
        if fade_samples > 0:
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

        # Generate harmonics
        segment = np.zeros_like(t)
        for h in range(1, num_harmonics + 1):
            # Harmonic frequency
            harmonic_freq = base_freq * h

            # Harmonic amplitude decreases with higher harmonics
            # Use y value to control harmonic content
            harmonic_amp = (1.0 / h) * (1.0 - (1.0 - y) * (h / num_harmonics))

            # Add harmonic to segment
            segment += harmonic_amp * np.sin(2 * np.pi * harmonic_freq * t)

        # Normalize segment, apply envelope and scale
        segment = amplitude_scale * 32767 * segment * envelope / num_harmonics

        # Add to audio data
        audio_data[start_sample:end_sample] = segment[
            : len(audio_data[start_sample:end_sample])
        ]

    return audio_data


def euclidean_distance_strategy(
    x_vals,
    y_vals,
    sample_rate=22050,
    tone_duration=0.05,
    min_freq=110,
    max_freq=1760,
    amplitude_scale=0.8,
):
    """
    Euclidean distance sonification:
    - Maps the distance between consecutive points to frequency
    - Creates sounds that highlight changes in the data
    """
    # Calculate Euclidean distances between consecutive points
    dx = np.diff(x_vals)
    dy = np.diff(y_vals)
    distances = np.sqrt(dx**2 + dy**2)
    # Add a first distance to match length of original data
    distances = np.insert(distances, 0, 0)

    # Normalize distances to frequency range
    frequencies = normalize_data(distances, min_val=min_freq, max_val=max_freq)

    # Calculate total duration
    total_samples = int(len(x_vals) * tone_duration * sample_rate)
    audio_data = np.zeros(total_samples)

    # Generate audio
    for i, freq in enumerate(frequencies):
        # Calculate start and end sample for this tone
        start_sample = int(i * tone_duration * sample_rate)
        end_sample = int((i + 1) * tone_duration * sample_rate)
        if end_sample > total_samples:
            end_sample = total_samples

        # Generate time array for this segment
        t = np.linspace(0, tone_duration, end_sample - start_sample, endpoint=False)

        # Apply envelope to avoid clicks
        envelope = np.ones_like(t)
        fade_samples = int(0.1 * len(t))  # 10% fade in/out
        if fade_samples > 0:
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

        # Generate sine wave with frequency based on distance
        segment = amplitude_scale * 32767 * np.sin(2 * np.pi * freq * t) * envelope

        # Add to audio data
        audio_data[start_sample:end_sample] = segment[
            : len(audio_data[start_sample:end_sample])
        ]

    return audio_data


# Main app
def main():
    st.title("Data Sonification Playground")
    
    with st.expander("About this Exhibition", expanded=False):
        st.write("There is no way for a human to ever hear quantum oscillations directly, but through sonification, we can make them audible. Moving from Sine Wave through to Euclidean Distance, we can increase the complexity and abstractness of the sound we hear.")

        st.write("The Data Sonification Playground is an interactive web application that transforms quantum data into sound. Built using Python and Streamlit, it allows users to explore different sonification techniques and parameters in real-time, creating a bridge between quantum physics and auditory experience.")

        st.write("These sounds trace Rabi oscillations in the pentacene triplet level, following microsecond-scale swings in optical contrast as the molecule's spin state coherently flips between ground and excited triplet configurations. Through bespoke sonification techniques, those quantum pulses become immersive soundscapes—audible echoes of invisible oscillations—that invite listeners to feel the hidden rhythm of light–matter interaction in this exhibition.")

    # Sidebar for file upload and configuration
    st.sidebar.header("Configuration")

    # Default data path
    default_data_path = "data/fig3b_multilevel.csv"

    # Check if default file exists, otherwise prompt for upload
    if os.path.exists(default_data_path):
        use_default = st.sidebar.checkbox(
            "Use default data (fig3b_multilevel.csv)", value=True
        )
        if use_default:
            df = read_csv_data(default_data_path)
            csv_file = default_data_path
        else:
            uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
            if uploaded_file:
                df = read_csv_data(uploaded_file)
                csv_file = uploaded_file.name
            else:
                st.sidebar.warning("Please upload a CSV file or use the default data.")
                df = None
                csv_file = None
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file:
            df = read_csv_data(uploaded_file)
            csv_file = uploaded_file.name
        else:
            st.sidebar.warning("Default data not found. Please upload a CSV file.")
            df = None
            csv_file = None

    # Navigation links in sidebar
    st.sidebar.header("Navigation")

    # Proceed only if data is available
    if df is not None:
        x_vals, y_vals = extract_columns(df)

        if len(x_vals) == 0 or len(y_vals) == 0:
            st.error("Could not extract valid numerical data from the CSV file.")
            return

        # Default to sonification
        st.subheader("Data Sonification")
        
        # Create tabs for each sonification method at the top of the screen
        sonification_tabs = st.tabs([
            "Sine Wave", 
            "FM Synthesis",
            "Granular Synthesis",
            "Harmonic Mapping",
            "Euclidean Distance"
        ])

        # Sine Wave tab
        with sonification_tabs[0]:
            st.markdown("### Sine Wave Sonification")
            st.write(
                "Maps the data to simple sine wave tones. Y values control frequency."
            )

            # Parameters
            col1, col2 = st.columns(2)
            with col1:
                sine_duration = st.slider(
                    "Tone Duration (seconds)",
                    0.01,
                    0.2,
                    0.05,
                    0.01,
                    key="sine_duration",
                    help="Controls how long each data point sounds. Shorter durations create more staccato sounds, while longer durations create smoother transitions."
                )
                sine_min_freq = st.slider(
                    "Minimum Frequency (Hz)", 
                    50, 
                    500, 
                    220, 
                    10, 
                    key="sine_min_freq",
                    help="The lowest frequency that will be used in the sonification. 220 Hz is approximately A3 on a piano."
                )
            with col2:
                sine_max_freq = st.slider(
                    "Maximum Frequency (Hz)",
                    200,
                    2000,
                    880,
                    50,
                    key="sine_max_freq",
                    help="The highest frequency that will be used in the sonification. 880 Hz is approximately A5 on a piano."
                )
                sine_amplitude = st.slider(
                    "Amplitude Scale", 
                    0.1, 
                    1.0, 
                    0.8, 
                    0.1, 
                    key="sine_amplitude",
                    help="Controls the volume of the sound. Higher values create louder sounds."
                )

            # Generate audio automatically
            with st.spinner("Generating audio..."):
                sine_audio = sine_wave_strategy(
                    x_vals,
                    y_vals,
                    tone_duration=sine_duration,
                    min_freq=sine_min_freq,
                    max_freq=sine_max_freq,
                    amplitude_scale=sine_amplitude,
                )

                # Display audio player
                st.audio(sine_audio.astype(np.int16), sample_rate=22050)

                # Display spectrogram
                st.write("#### Spectrogram")
                spectrogram_fig = create_spectrogram(sine_audio, 22050)
                st.pyplot(spectrogram_fig)

                # Download link
                st.markdown(
                    get_download_link(
                        sine_audio, 22050, "sine_wave_sonification.wav"
                    ),
                    unsafe_allow_html=True,
                )

        # FM Synthesis tab
        with sonification_tabs[1]:
            st.markdown("### FM Synthesis Sonification")
            st.write(
                "Uses frequency modulation to create more complex sounds. Y values control modulation index."
            )

            # Parameters
            col1, col2 = st.columns(2)
            with col1:
                fm_duration = st.slider(
                    "Tone Duration (seconds)",
                    0.01,
                    0.2,
                    0.05,
                    0.01,
                    key="fm_duration",
                    help="Controls how long each data point sounds. Shorter durations create more percussive sounds, while longer durations allow hearing the modulation effects more clearly."
                )
                fm_carrier = st.slider(
                    "Carrier Frequency (Hz)", 
                    20, 
                    1000, 
                    440, 
                    20, 
                    key="fm_carrier",
                    help="The base frequency that gets modulated. This is the primary pitch you hear. 440 Hz is A4 (concert A)."
                )
            with col2:
                fm_mod_min = st.slider(
                    "Min Modulation Index", 
                    0.1, 
                    5.0, 
                    1.0, 
                    0.1, 
                    key="fm_mod_min",
                    help="The minimum amount of frequency modulation. Lower values create more subtle timbral variations."
                )
                fm_mod_max = st.slider(
                    "Max Modulation Index", 
                    1.0, 
                    10.0, 
                    5.0, 
                    0.5, 
                    key="fm_mod_max",
                    help="The maximum amount of frequency modulation. Higher values create more dramatic timbral changes and complex sounds."
                )

            # Generate audio automatically
            with st.spinner("Generating audio..."):
                fm_audio = fm_synthesis_strategy(
                    x_vals,
                    y_vals,
                    tone_duration=fm_duration,
                    carrier_freq=fm_carrier,
                    mod_index_min=fm_mod_min,
                    mod_index_max=fm_mod_max,
                )

                # Display audio player
                st.audio(fm_audio.astype(np.int16), sample_rate=22050)

                # Display spectrogram
                st.write("Spectrogram:")
                spectrogram_fig = create_spectrogram(fm_audio, 22050)
                st.pyplot(spectrogram_fig)

                # Download link
                st.markdown(
                    get_download_link(
                        fm_audio, 22050, "fm_synthesis_sonification.wav"
                    ),
                    unsafe_allow_html=True,
                )

        # Granular Synthesis tab
        with sonification_tabs[2]:
            st.markdown("### Granular Synthesis Sonification")
            st.write(
                "Creates small sound 'grains' from the data, creating textural sounds. Y values control frequency of grains."
            )

            # Parameters
            col1, col2 = st.columns(2)
            with col1:
                grain_size = st.slider(
                    "Grain Size (seconds)",
                    0.005,
                    0.1,
                    0.02,
                    0.005,
                    key="grain_size",
                    help="Controls the duration of each sound 'grain'. Smaller grains create more textural, cloud-like sounds. Larger grains create more distinct, recognizable tones."
                )
            with col2:
                grain_density = st.slider(
                    "Grain Density", 
                    0.1, 
                    1.0, 
                    0.5, 
                    0.1, 
                    key="grain_density",
                    help="Controls how much the grains overlap. Higher values create denser, more continuous textures. Lower values create more sparse, distinct grains."
                )

            # Generate audio automatically
            with st.spinner("Generating audio..."):
                granular_audio = granular_synthesis_strategy(
                    x_vals, y_vals, grain_size=grain_size, density=grain_density
                )

                # Display audio player
                st.audio(granular_audio.astype(np.int16), sample_rate=22050)

                # Display spectrogram
                st.write("Spectrogram:")
                spectrogram_fig = create_spectrogram(granular_audio, 22050)
                st.pyplot(spectrogram_fig)

                # Download link
                st.markdown(
                    get_download_link(
                        granular_audio,
                        22050,
                        "granular_synthesis_sonification.wav",
                    ),
                    unsafe_allow_html=True,
                )

        # Harmonic Mapping tab
        with sonification_tabs[3]:
            st.markdown("### Harmonic Mapping Sonification")
            st.write(
                "Maps data to harmonic content, creating rich timbral variations. Y values control harmonic distribution."
            )

            # Parameters
            col1, col2 = st.columns(2)
            with col1:
                harmonic_duration = st.slider(
                    "Tone Duration (seconds)",
                    0.01,
                    0.2,
                    0.05,
                    0.01,
                    key="harmonic_duration",
                    help="Controls how long each data point sounds. Longer durations allow better perception of the harmonic content."
                )
                harmonic_base = st.slider(
                    "Base Frequency (Hz)", 
                    50, 
                    440, 
                    110, 
                    10, 
                    key="harmonic_base",
                    help="The fundamental frequency upon which harmonics are built. 110 Hz is approximately A2 on a piano."
                )
            with col2:
                harmonic_count = st.slider(
                    "Number of Harmonics", 
                    2, 
                    16, 
                    8, 
                    1, 
                    key="harmonic_count",
                    help="Controls how many harmonic overtones are included. More harmonics create richer, more complex timbres. Fewer harmonics create simpler, purer sounds."
                )

            # Generate audio automatically
            with st.spinner("Generating audio..."):
                harmonic_audio = harmonic_mapping_strategy(
                    x_vals,
                    y_vals,
                    tone_duration=harmonic_duration,
                    base_freq=harmonic_base,
                    num_harmonics=harmonic_count,
                )

                # Display audio player
                st.audio(harmonic_audio.astype(np.int16), sample_rate=22050)

                # Display spectrogram
                st.write("Spectrogram:")
                spectrogram_fig = create_spectrogram(harmonic_audio, 22050)
                st.pyplot(spectrogram_fig)

                # Download link
                st.markdown(
                    get_download_link(
                        harmonic_audio,
                        22050,
                        "harmonic_mapping_sonification.wav",
                    ),
                    unsafe_allow_html=True,
                )

        # Euclidean Distance tab
        with sonification_tabs[4]:
            st.markdown("### Euclidean Distance Sonification")
            st.write(
                "Sonifies the distance between consecutive data points. Larger jumps create higher frequencies."
            )

            # Parameters
            col1, col2 = st.columns(2)
            with col1:
                distance_duration = st.slider(
                    "Tone Duration (seconds)",
                    0.01,
                    0.2,
                    0.05,
                    0.01,
                    key="distance_duration",
                    help="Controls how long each data point sounds. Shorter durations emphasize rapid changes in the data."
                )
                distance_min_freq = st.slider(
                    "Minimum Frequency (Hz)",
                    20,
                    500,
                    110,
                    10,
                    key="distance_min_freq",
                    help="The frequency used for the smallest distances between data points. 110 Hz is approximately A2 on a piano."
                )
            with col2:
                distance_max_freq = st.slider(
                    "Maximum Frequency (Hz)",
                    50,
                    2000,
                    1760,
                    50,
                    key="distance_max_freq",
                    help="The frequency used for the largest distances between data points. 1760 Hz is approximately A6 on a piano."
                )

            # Generate audio automatically
            with st.spinner("Generating audio..."):
                distance_audio = euclidean_distance_strategy(
                    x_vals,
                    y_vals,
                    tone_duration=distance_duration,
                    min_freq=distance_min_freq,
                    max_freq=distance_max_freq,
                )

                # Display audio player
                st.audio(distance_audio.astype(np.int16), sample_rate=22050)

                # Display spectrogram
                st.write("Spectrogram:")
                spectrogram_fig = create_spectrogram(distance_audio, 22050)
                st.pyplot(spectrogram_fig)

                # Download link
                st.markdown(
                    get_download_link(
                        distance_audio,
                        22050,
                        "euclidean_distance_sonification.wav",
                    ),
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
