import streamlit as st

# Set page config
st.set_page_config(
    page_title="How to Use - Sonification Dashboard", page_icon="❓", layout="wide"
)

# Main app
def main():
    st.title("How to Use This Dashboard")
    
    # Navigation links in sidebar
    st.sidebar.header("Navigation")
    st.sidebar.markdown("[Sonification](/) | [Data Exploration](/Data_Exploration) | [How to...](/How_to)")

    # How to use instructions
    st.markdown("""
    ## How to Use This Dashboard

    This dashboard allows you to explore different ways of converting data to sound through sonification techniques.

    ### 1. Data Input
    - By default, the dashboard uses the `fig3b_multilevel.csv` dataset
    - To use your own data, uncheck the "Use default data" option in the sidebar and upload a CSV file

    ### 2. Navigation
    - Use the links in the sidebar to navigate between pages
    - The **Sonification** page is where you can convert your data to sound using different methods
    - The **Data Exploration** page shows statistics and visualizations of your data
    - This **How to...** page provides instructions on how to use the dashboard

    ### 3. Sonification
    - Select a sonification method from the tabs at the top of the Sonification page
    - Adjust parameters to customize the sound
    - Click "Generate" to create and play the audio
    - Download the generated audio file using the download button

    ### Data Format Requirements
    Your CSV data should have at least two columns in this format:

    ```csv
    microseconds, optical_contrast
    1.500000000000000118e-02, -4.925499858765510774e-01
    1.750000000000000167e-02, -4.933005565083004584e-01
    2.000000000000000042e-02, -6.617802563678457650e-01
    ```

    ### Sonification Methods

    #### Sine Wave
    Maps the data to simple sine wave tones. Y values control frequency.
    
    #### FM Synthesis
    Uses frequency modulation to create more complex sounds. Y values control modulation index.
    
    #### Granular Synthesis
    Creates small sound 'grains' from the data, creating textural sounds. Y values control frequency of grains.
    
    #### Harmonic Mapping
    Maps data to harmonic content, creating rich timbral variations. Y values control harmonic distribution.
    
    #### Euclidean Distance
    Sonifies the distance between consecutive data points. Larger jumps create higher frequencies.
    """)


if __name__ == "__main__":
    main() 