# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_and_save_model():
    df = pd.read_csv("/Users/amangolani/Desktop/Speech_Recog/tess_opensmile_dataset.csv", index_col=0)
    X = df.drop(columns=["emotion", "filename"])
    y = df["emotion"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=43)

    rf_model = RandomForestClassifier(
        n_estimators=100,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features='log2',
        max_depth=20
    )
    rf_model.fit(X_train, y_train)

    joblib.dump(rf_model, "rf_emotion_model.pkl")
    print("Model trained and saved to rf_emotion_model.pkl")

if __name__ == "__main__":
    train_and_save_model()