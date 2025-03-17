import polars as pl
from typing import List, Tuple, Optional

def read_csv_data(csv_file) -> Optional[pl.DataFrame]:
    """
    Reads data from a CSV file and returns a polars DataFrame.
    
    Args:
        csv_file: Path to the CSV file
        
    Returns:
        A polars DataFrame or None if there was an error
    """
    try:
        # Read CSV with polars, automatically handles headers
        df = pl.read_csv(csv_file, separator=" ")

        # Check if the dataframe is empty or has parsing issues with space separator
        if df.height == 0 or df.width < 2:
            # Try with comma separator instead
            print("Trying with comma separator instead of space...")
            df = pl.read_csv(csv_file, separator=",")
            
            # If still empty, try with auto-detection
            if df.height == 0 or df.width < 2:
                print("Trying with automatic separator detection...")
                df = pl.read_csv(csv_file)
        
        if df.height == 0:
            print("Could not read any data from the CSV file")
            return None
            
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
def extract_columns(df: pl.DataFrame, column_names: Optional[List[str]] = None) -> List[Tuple[float, float]]:
    """
    Extracts two columns from a DataFrame and returns them as a list of tuples.
    
    Args:
        df: The polars DataFrame to extract columns from
        column_names: Optional list of column names to extract. If None, uses the first two columns.
        
    Returns:
        A list of tuples containing the data from the two columns
    """
    if df is None or df.height == 0:
        return []
        
    # Get the columns to extract
    columns = df.columns
    if len(columns) < 2:
        print("CSV file must have at least two columns")
        return []
        
    # Use specified columns or default to first two
    if column_names and len(column_names) >= 2 and all(col in columns for col in column_names[:2]):
        x_col, y_col = column_names[0], column_names[1]
    else:
        x_col, y_col = columns[0], columns[1]
    
    # Convert to list of tuples, skipping any non-numeric values
    data = []
    for x, y in zip(df[x_col], df[y_col]):
        try:
            x_float = float(x)
            y_float = float(y)
            data.append((x_float, y_float))
        except (ValueError, TypeError):
            # Skip this row if values can't be converted to float
            continue
            
    return data