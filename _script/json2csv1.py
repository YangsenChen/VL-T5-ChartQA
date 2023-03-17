import json
import pandas as pd

# Define a function to extract additional data from a JSON file
# def get_additional_data(image_id):
#     with open(f'/mnt/c/Users/Yangsen/Desktop/intern/ChartQA/ChartQA Dataset/train/tables/{image_id}.csv', 'r') as f:
#         data = json.load(f)
#     additional_info = data['additional_info'] # Replace with the key for your data
#     return f' (Additional info: {additional_info})'
# def get_additional_data(image_id):
#     additional_data = pd.read_csv(f'/mnt/c/Users/Yangsen/Desktop/intern/ChartQA/ChartQA Dataset/train/tables/{image_id}.csv') # Replace with the appropriate file name and location
#     additional_info = additional_data['additional_info'].iloc[0] # Replace 'additional_info' with the appropriate column name
#     return f' (Additional info: {additional_info})'

# def get_additional_data(image_id):
#     print(image_id)
#     with open(f'/mnt/c/Users/Yangsen/Desktop/intern/ChartQA/ChartQA Dataset/train/tables/{image_id}.csv', 'r') as f:
#         df = pd.read_csv(f)
#     # Replace the following line with code to extract and flatten your table data
#     table_data = df.to_string(index=False)
#     table_data = df.to_csv(index=False, line_terminator=' | ',sep='|')
#
#     table_data = table_data.replace('\n', ', ')
#     return f' Table: {table_data})'

# Define a function to extract and flatten data from a CSV file
def get_additional_data(image_id):
    with open(f'/mnt/c/Users/Yangsen/Desktop/intern/ChartQA/ChartQA Dataset/train/tables/{image_id}.csv', 'r') as f:
        df = pd.read_csv(f)
    # Replace the following line with code to extract and flatten your table data
    col_strings = []
    for col in df.columns:
        col_series = df[col]
        col_string = ' | '.join([str(val) for val in col_series])
        col_strings.append(f'{col}: {col_string}')
    table_data = ' & '.join(col_strings)
    return f' Table: {table_data}'

# Load the JSON data into a DataFrame
# df = pd.read_json('train_human.json')
# data_files = ['data.json', 'data1.json']

data_files = ['train_human.json', 'train_augmented.json']
data = []
for file in data_files:
    with open(file, 'r') as f:
        data.extend(json.load(f))
df = pd.DataFrame(data)

# Extract the required information into a new DataFrame
new_df = pd.DataFrame({
    'Input': df['query'].apply(lambda x: f'Question: {x}'),
    'Output': df['label'],
    'Image Index': df['imgname'].apply(lambda x: x.split('.')[0]),
    'Question ID': range(1, len(df) + 1) # auto-incremented IDs
})

# Add additional data to the input column
new_df['Input'] = new_df.apply(lambda row: row['Input'] + get_additional_data(row['Image Index']), axis=1)

# Save the new DataFrame to a CSV file
new_df.to_csv('output.csv', index=False)
