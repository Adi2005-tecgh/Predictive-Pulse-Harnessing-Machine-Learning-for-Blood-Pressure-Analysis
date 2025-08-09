import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv('dataset/patient_data.csv')

# These are the 13 input features used in your form and model
input_columns = [
    'Gender', 'Age', 'History', 'Patient', 'TakeMedication',
    'Severity', 'BreathShortness', 'VisualChanges', 'NoseBleeding',
    'Whendiagnoused', 'Systolic', 'Diastolic', 'ControlledDiet'
]

# Initialize dictionary to hold encoders
input_encoders = {}

# Create LabelEncoders for each column
for col in input_columns:
    le = LabelEncoder()
    df[col] = df[col].astype(str)  # Ensure all values are strings
    df[col] = le.fit_transform(df[col])
    input_encoders[col] = le

# Save the encoders to a pickle file
with open('encoders.pkl', 'wb') as f:
    pickle.dump(input_encoders, f)

print("✅ Successfully saved 'encoders.pkl' with 13 label encoders.")
