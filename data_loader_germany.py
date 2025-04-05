import pandas as pd

# Path to the large original CSV file
file_path = "Epidemiology Data.csv"  # Make sure it's in your working directory

# Output path for the smaller, filtered file
output_path = "germany_epidemiology_data.csv"

# Filter and write only Germany rows to a new file
with open(output_path, 'w') as output_file:
    write_header = True
    for chunk in pd.read_csv(file_path, chunksize=500_000):
        germany_chunk = chunk[chunk["location_key"] == "DE"]
        if not germany_chunk.empty:
            germany_chunk.to_csv(output_file, index=False, header=write_header)
            write_header = False

print(f"Filtered data saved to: {output_path}")