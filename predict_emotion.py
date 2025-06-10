# predict_emotion.py
import opensmile
import pandas as pd
import joblib

def extract_features_from_file(file_path):
    smile_model = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    try:
        features = smile_model.process_file(file_path)
        return features.reset_index(drop=True)
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None

# Load pre-trained model
rf_model = joblib.load("/Users/amangolani/Desktop/Speech_Recog/rf_emotion_model.pkl")

def predict_emotion(file_path):
    features = extract_features_from_file(file_path)
    if features is not None:
        prediction = rf_model.predict(features)
        return prediction[0]
    else:
        return "Error: Feature extraction failed"

# Example use
if __name__ == "__main__":
    test_audio_path = "/Users/amangolani/Desktop/Speech_Recog/VideoResume_Aman.wav"
    print("Predicted emotion:", predict_emotion(test_audio_path))