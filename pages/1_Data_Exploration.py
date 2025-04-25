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
    page_title="Data Exploration - Sonification Dashboard", 
    page_icon="📊", 
    layout="wide"
)

# Utility functions (copied from main dashboard)
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

# Main app
def main():
    st.title("Data Exploration")
    st.write(
        "Explore and analyse your data before sonification."
    )

    # Sidebar for file upload configuration
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

        # Display data statistics
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


if __name__ == "__main__":
    main() 