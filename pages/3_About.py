import streamlit as st

# Set page config
st.set_page_config(
    page_title="About - Sonification Dashboard", 
    page_icon="ℹ️", 
    layout="wide"
)

# Main app
def main():
    st.title("About")
    
    # Navigation links in sidebar
    st.sidebar.header("Navigation")

    # About content
    st.markdown("""
    ## About this Dashboard
    
    This dashboard is a tool to help you explore different ways of converting data to sound. 
    It allows you to explore various methods of sonifying scientific data,
    specifically focusing on Rabi oscillation patterns. Each sonification strategy highlights
    different aspects of the data through sound.
    
    ### What is Data Sonification?
    
    Data sonification is the process of converting data into sound. While data visualization 
    uses visual elements to represent data, sonification uses auditory elements. 
    This approach can provide new insights into data patterns, especially for:
    
    - Time-series data where patterns evolve over time
    - Multi-dimensional data that might be difficult to visualize
    - Accessibility purposes, offering data interpretation for visually impaired users
    - Identifying subtle patterns that might be missed in visual representations
    
    ### Developed by
    
    This tool was created by [Rory Scott](https://github.com/rorads).
    
    ### Contact and Feedback
    
    If you have suggestions, questions, or would like to contribute to this project, 
    please reach out via GitHub.
    """)


if __name__ == "__main__":
    main() 