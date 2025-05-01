import streamlit as st

# Set page config
st.set_page_config(
    page_title="How to Use - Sonification Dashboard", 
    page_icon="❓", 
    layout="wide"
)

# Main app
def main():
    st.title("How to Use This Dashboard")
    
    # Navigation links in sidebar
    st.sidebar.header("Navigation")

    # How to use instructions
    st.markdown("""
    ## How to Use This Dashboard

    This dashboard allows you to explore different ways of converting data to sound through sonification techniques.

    ### 1. Data Input
    - By default, the dashboard uses the `fig3b_multilevel.csv` dataset located in the `data` folder.
    - To use your own data, uncheck the "Use default data" option in the sidebar and upload a CSV file. Ensure your file adheres to the format requirements below.

    ### 2. Navigation
    - Use the links in the sidebar to navigate between the available pages.
    - The main **Data Sonification Playground** page is where you can generate and listen to sounds based on your data.
    - The **Data Exploration** page (if available) shows statistics and visualisations of your data.
    - This **How to...** page provides these instructions.

    ### 3. Sonification Process
    - On the **Data Sonification Playground** page, you will find several tabs, each representing a different sonification strategy.
    - Click on a tab (e.g., "Sine Wave", "FM Synthesis") to select that strategy.
    - Within each tab, you'll find controls (sliders) to adjust the parameters specific to that sonification method. Hover over the `?` icon next to each slider for a detailed explanation of its effect.
    - Below the controls, there is a **Spectrogram Preview** which updates in real-time as you adjust the parameters, giving you a visual representation of the sound's frequency content over time.
    - When you are happy with the parameters, click the **"▶️ Load [Strategy Name] Audio"** button within the tab. This generates the audio based on your settings.
    - The generated audio will appear in the **Current Audio** section at the top of the page. Here you can:
        - See the **Spectrogram** of the generated audio.
        - Use the **Audio Player** to listen to the sound.
        - **Download** the audio as a `.wav` file using the provided link.

    ### Data Format Requirements
    Your CSV data should contain at least two numerical columns. The application will use the **first column as the X-axis (time)** and the **second column as the Y-axis (value to be sonified)**.

    Example format:
    ```csv
    time_microseconds, optical_contrast
    0.015, -0.4925
    0.0175, -0.4933
    0.020, -0.6618
    ...more data...
    ```
    Non-numeric values or rows with missing data in the first two columns will be skipped.

    ### Sonification Methods Explained

    Each tab offers a different way to map your data (specifically, the Y-axis values) to sound:

    #### Sine Wave
    *Analogy:* Reading data points like musical notes on a simple scale.
    
    *Detail:* This is the most direct method. Each data point from your Y-axis is mapped to the **frequency (pitch)** of a pure sine wave tone. Higher data values produce higher pitched notes. The X-axis determines the sequence in time. You control the **duration** of each tone and the **minimum/maximum frequency** range.
    
    #### FM Synthesis
    *Analogy:* A base musical tone gets "wobbled" or warped in a controlled way.
    
    *Detail:* Frequency Modulation (FM) synthesis creates complex timbres. Here, the Y-axis data controls the **modulation index** – essentially, *how much* the base **carrier frequency** is modulated by another (modulator) frequency. Higher data values lead to a larger modulation index, resulting in richer, more complex, and sometimes dissonant sounds. You control the **tone duration**, the base **carrier frequency**, and the **range of the modulation index**.
    
    #### Granular Synthesis
    *Analogy:* Breaking the data into tiny sound fragments ("grains") and scattering them to form a texture.
    
    *Detail:* This technique creates evolving soundscapes. Each data point influences the **frequency (pitch)** of a very short sound segment called a grain. The X-axis value determines the placement of the grain in the overall audio timeline. You control the **grain size** (duration of each fragment) and the **density** (how much the grains overlap), shaping the resulting texture from sparse clicks to dense clouds.
    
    #### Harmonic Mapping
    *Analogy:* Adjusting the overtones of an instrument to change its sound colour (timbre).
    
    *Detail:* This method focuses on timbre rather than just pitch. A **base frequency** is chosen, and the Y-axis data determines the **number and intensity of harmonics** (overtones) added to that base frequency. Higher data values introduce more or stronger harmonics, making the sound richer or brighter. You control the **tone duration**, the **base frequency**, and the **maximum number of harmonics** considered.
    
    #### Euclidean Distance
    *Analogy:* Listening to the *change* or *jump* between consecutive data points, not their absolute level.
    
    *Detail:* This strategy highlights the rate of change in your data. It calculates the **Euclidean distance** (a measure of the straight-line separation) between consecutive (X, Y) points. This distance value is then mapped to **frequency (pitch)**. Larger jumps or differences between successive data points result in higher-pitched sounds, making it effective for identifying volatility or sudden shifts. You control the **tone duration** and the **minimum/maximum frequency** range corresponding to the smallest/largest calculated distances.
    """)


if __name__ == "__main__":
    main() 