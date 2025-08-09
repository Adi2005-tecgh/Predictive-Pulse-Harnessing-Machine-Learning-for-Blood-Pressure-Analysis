
import pickle

# Load the encoders.pkl file
with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

print("✅ Keys found in encoders.pkl:")
for key in encoders.keys():
    print("-", key)
