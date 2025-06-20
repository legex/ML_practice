import pandas as pd
from random import choice, randint
from utils.datapreprocessing import DataProcessing
import json
import requests

# Step 1: Generate synthetic input row
example_data = {
    'Time_spent_Alone': randint(0, 10),
    'Stage_fear': choice(['Yes', 'No']),
    'Social_event_attendance': randint(0, 10),
    'Going_outside': randint(0, 9),
    'Drained_after_socializing': choice(['Yes', 'No']),
    'Friends_circle_size': randint(0, 25),
    'Post_frequency': randint(0, 20),
    'Personality': choice(['Introvert', 'Extrovert'])  # dummy for structure
}

# Step 2: Convert to DataFrame and preprocess
df = pd.DataFrame([example_data])
dp = DataProcessing(df, target_col='Personality', is_inference=True)
print("Original DataFrame:\n", df)
processed_df = dp.split()

# Step 3: Convert DataFrame row to list of floats
#features = processed_df.iloc[0].tolist()  # 1D list of features
print("Features for prediction:", processed_df)
# Step 4: Prepare JSON payload
# payload = {
#     "features": features
# }

# # Optional: Print or inspect before sending
# print("Sending JSON payload:", json.dumps(payload, indent=2))

# # Step 5: Send POST request
# url = 'http://localhost:8223/predict'
# headers = {'Content-Type': 'application/json'}

# response = requests.post(url, data=json.dumps(payload), headers=headers)

# # Step 6: Output the prediction
# print("Prediction:", response.json())
