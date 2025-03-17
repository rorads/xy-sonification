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
    page_title="Data Sonification Dashboard", page_icon="🔊", layout="wide"
)


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
    st.title("Data Sonification Dashboard")
    st.write(
        "This dashboard allows you to explore different ways of converting data to sound."
    )

    st.markdown("""
    > **Note:** The default data has been set to `fig3b_multilevel.csv`. Uncheck to upload your own data.
    > the data is of the following format:
    >
    > ```csv
    > microseconds, optical_contrast
    > 1.500000000000000118e-02, -4.925499858765510774e-01
    > 1.750000000000000167e-02, -4.933005565083004584e-01
    > 2.000000000000000042e-02, -6.617802563678457650e-01
    > ```
    """)

    # Sidebar for file upload and strategy selection
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

    # Proceed only if data is available
    if df is not None:
        x_vals, y_vals = extract_columns(df)

        if len(x_vals) == 0 or len(y_vals) == 0:
            st.error("Could not extract valid numerical data from the CSV file.")
            return

        # Replace radio button with selectbox for more traditional navigation
        st.sidebar.header("Navigation")
        page = st.sidebar.selectbox("Select page", ["Data Exploration", "Sonification"])

        # Display basic data stats
        if page == "Data Exploration":
            st.subheader("Data Statistics and preview")
            # Display data statistics as bullet points
            st.markdown(f"""
            * **Number of data points:** {len(x_vals)}
            * **X range:** [{np.min(x_vals):.4f}, {np.max(x_vals):.4f}]
            * **Y range:** [{np.min(y_vals):.4f}, {np.max(y_vals):.4f}]
            * **Data source:** {csv_file}
            """)

            st.write(df.head())

            # Visualization of the data
            st.subheader("Data Visualization")
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(
                ["Time Series", "Scatter Plot", "Histogram"]
            )

            with viz_tab1:
                fig, ax = plt.subplots(figsize=(10, 6))
                time_index = np.arange(len(x_vals))
                ax.plot(time_index, x_vals, "b-", label=df.columns[0])
                ax.plot(time_index, y_vals, "r-", label=df.columns[1])
                ax.set_xlabel("Time (index)")
                ax.set_ylabel("Value")
                ax.set_title("Time Series Plot of Data")
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)

            with viz_tab2:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(x_vals, y_vals, color="purple", alpha=0.6)
                ax.set_xlabel(df.columns[0])
                ax.set_ylabel(df.columns[1])
                ax.set_title("Scatter Plot of Data")
                ax.grid(True)
                st.pyplot(fig)

            with viz_tab3:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(y_vals, bins=20, color="green", alpha=0.7)
                ax.set_xlabel(df.columns[1])
                ax.set_ylabel("Frequency")
                ax.set_title(f"Histogram of {df.columns[1]}")
                ax.grid(True)
                st.pyplot(fig)

        # Sonification page
        elif page == "Sonification":
            st.subheader("Sonification")

            # Add sonification method selection to sidebar when on sonification page
            st.sidebar.subheader("Sonification Method")
            sonification_method = st.sidebar.selectbox(
                "Select method",
                [
                    "Sine Wave",
                    "FM Synthesis",
                    "Granular Synthesis",
                    "Harmonic Mapping",
                    "Euclidean Distance",
                ],
            )

            # Display the selected sonification method
            if sonification_method == "Sine Wave":
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
                    )
                    sine_min_freq = st.slider(
                        "Minimum Frequency (Hz)", 50, 500, 220, 10, key="sine_min_freq"
                    )
                with col2:
                    sine_max_freq = st.slider(
                        "Maximum Frequency (Hz)",
                        500,
                        2000,
                        880,
                        50,
                        key="sine_max_freq",
                    )
                    sine_amplitude = st.slider(
                        "Amplitude Scale", 0.1, 1.0, 0.8, 0.1, key="sine_amplitude"
                    )

                # Generate audio
                if st.button("Generate Sine Wave Audio", key="sine_generate"):
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
                        st.write("Spectrogram:")
                        spectrogram_fig = create_spectrogram(sine_audio, 22050)
                        st.pyplot(spectrogram_fig)

                        # Download link
                        st.markdown(
                            get_download_link(
                                sine_audio, 22050, "sine_wave_sonification.wav"
                            ),
                            unsafe_allow_html=True,
                        )

            elif sonification_method == "FM Synthesis":
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
                    )
                    fm_carrier = st.slider(
                        "Carrier Frequency (Hz)", 100, 1000, 440, 20, key="fm_carrier"
                    )
                with col2:
                    fm_mod_min = st.slider(
                        "Min Modulation Index", 0.1, 5.0, 1.0, 0.1, key="fm_mod_min"
                    )
                    fm_mod_max = st.slider(
                        "Max Modulation Index", 1.0, 10.0, 5.0, 0.5, key="fm_mod_max"
                    )

                # Generate audio
                if st.button("Generate FM Synthesis Audio", key="fm_generate"):
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

            elif sonification_method == "Granular Synthesis":
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
                    )
                with col2:
                    grain_density = st.slider(
                        "Grain Density", 0.1, 1.0, 0.5, 0.1, key="grain_density"
                    )

                # Generate audio
                if st.button(
                    "Generate Granular Synthesis Audio", key="granular_generate"
                ):
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

            elif sonification_method == "Harmonic Mapping":
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
                    )
                    harmonic_base = st.slider(
                        "Base Frequency (Hz)", 50, 440, 110, 10, key="harmonic_base"
                    )
                with col2:
                    harmonic_count = st.slider(
                        "Number of Harmonics", 2, 16, 8, 1, key="harmonic_count"
                    )

                # Generate audio
                if st.button(
                    "Generate Harmonic Mapping Audio", key="harmonic_generate"
                ):
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

            elif sonification_method == "Euclidean Distance":
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
                    )
                    distance_min_freq = st.slider(
                        "Minimum Frequency (Hz)",
                        50,
                        500,
                        110,
                        10,
                        key="distance_min_freq",
                    )
                with col2:
                    distance_max_freq = st.slider(
                        "Maximum Frequency (Hz)",
                        500,
                        2000,
                        1760,
                        50,
                        key="distance_max_freq",
                    )

                # Generate audio
                if st.button(
                    "Generate Euclidean Distance Audio", key="distance_generate"
                ):
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

    # Footer
    st.markdown("---")
    st.markdown("### About")
    st.write(
        "This dashboard allows you to explore different methods of sonifying scientific data, "
        "specifically focusing on Rabi oscillation patterns. Each sonification strategy highlights "
        "different aspects of the data through sound."
    )


if __name__ == "__main__":
    main()
