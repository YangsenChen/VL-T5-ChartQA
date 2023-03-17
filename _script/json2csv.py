import pandas as pd

# Load the JSON data into a DataFrame
df = pd.read_json('train_augmented.json')

# Extract the required information into a new DataFrame
new_df = pd.DataFrame({
    'input': 'Question: ' + df['query'],
    'output': df['label'],
    'imageId': df['imgname'].apply(lambda x: x.split('.')[0]),
    'Question ID': range(1, len(df) + 1) # auto-incremented IDs
})

# Save the new DataFrame to a CSV file
new_df.to_csv('output.csv', index=False)
