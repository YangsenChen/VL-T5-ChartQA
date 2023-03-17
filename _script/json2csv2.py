import json
import pandas as pd
from tqdm import tqdm


# Define a function to extract and flatten data from a CSV file
def get_additional_data(image_id):
    with open(f'/mnt/c/Users/Yangsen/Desktop/intern/ChartQA/ChartQA Dataset/train/tables/{image_id}.csv', 'r') as f:
        df = pd.read_csv(f)
    col_strings = []
    for col in df.columns:
        col_series = df[col]
        col_string = ' | '.join([str(val) for val in col_series])
        col_strings.append(f'{col}: {col_string}')
    table_data = ', '.join(col_strings)
    return f' Table: {table_data}'


# Load the JSON data
with open('data.json', 'r') as f:
    data = json.load(f)

# Create an empty DataFrame to store the output
output_df = pd.DataFrame(columns=['input', 'output', 'imageId', 'Question ID'])

# Process each item in the JSON data
for item in tqdm(data):
    # Extract the image ID from the file name
    image_id = item['imgname'].split('.')[0]

    # Extract the additional data from the CSV file
    additional_data = get_additional_data(image_id)

    # Create a new row for the output DataFrame
    output_df = output_df.append({
        'input': f"question: {item['query']}{additional_data}",
        'output': item['label'],
        'imageId': image_id,
        'Question ID': 'auto_increase'
    }, ignore_index=True)

# Save the output DataFrame to a CSV file
output_df.to_csv('output0.csv', index=False)
