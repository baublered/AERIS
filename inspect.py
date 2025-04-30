import pickle

# Load the used features from the pickle file
with open('used_features.pkl', 'rb') as f:
    used_features = pickle.load(f)

# Print the list of features to verify
print(used_features)
