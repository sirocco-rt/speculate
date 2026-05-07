import pandas as pd

def convert_log_to_parquet(input_file, output_file):
    """
    Reads a grid runs logfile and converts it to a parquet file.
    """
    try:
        # Use pandas to read the CSV directly
        df = pd.read_csv(input_file)
        
        # If Run Number is missing, add it using the index
        if 'Run Number' not in df.columns:
            print("Run Number column missing, adding it based on index.")
            df.insert(0, 'Run Number', df.index)

        # Clean the dataframe
        # Strip whitespace from column names just in case
        df.columns = df.columns.astype(str).str.strip()
        
        # Let's ensure numeric conversion where possible
        for col in df.columns:
            if df[col].dtype == object:
                 df[col] = pd.to_numeric(df[col], errors='coerce')

        # Run Number should likely be integer if it exists
        if 'Run Number' in df.columns:
            df['Run Number'] = df['Run Number'].astype(int)

        print(f"DataFrame loaded successfully. Shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        print(df.head())

        # Save to parquet
        df.to_parquet(output_file, index=False)
        print(f"Successfully converted {input_file} to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_filename = "Grid_runs_logfile.txt"
    output_filename = "grid_run_lookup_table.parquet"
    
    convert_log_to_parquet(input_filename, output_filename)
