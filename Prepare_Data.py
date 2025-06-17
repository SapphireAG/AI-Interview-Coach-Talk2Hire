import kagglehub
import opensmile
import pandas as pd
import os
from pydub import AudioSegment
# Download latest version
path = kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess")

print("Path to dataset files:", path)



def convert_to_wav(input_path, output_path):
    # Load audio file
    audio = AudioSegment.from_file(input_path)

    # Set parameters: mono, 16kHz sample rate, 16-bit PCM
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_sample_width(2)  # 2 bytes = 16-bit

    # Export as .wav with PCM encoding
    audio.export(output_path, format="wav", codec="pcm_s16le")
    print(f"Converted and saved to: {output_path}")

# Example 
input_file = "/Users/amangolani/.cache/kagglehub/datasets/ejlok1/toronto-emotional-speech-set-tess/versions/1/TESS Toronto emotional speech set data"

def convert(root_dir,output_dir ):
                os.makedirs(output_dir,exist_ok=True)

                #root_dir="/Users/amangolani/.cache/kagglehub/datasets/ejlok1/toronto-emotional-speech-set-tess/versions/1/TESS Toronto emotional speech set data"
             # Walk through each subfolder
                for i,(subdir, dirs, files) in enumerate(os.walk(root_dir)):
                    for file in files:
                        if file.lower().endswith(".wav"):
                            input_path = os.path.join(subdir, file)
                            rel_path=os.path.relpath(subdir,root_dir)
                            target_dir=os.path.join(output_dir,rel_path)
                            os.makedirs(target_dir, exist_ok=True)
                            # output_path=os.path.join(subdir,f"{file}{i}.wav")
                            output_path = os.path.join(target_dir, file)
                            try:
                                # Extract features
                                convert_to_wav(input_path,output_path)

                            except Exception as e:
                                print(f"Error with file {file}: {e}")

convert(input_file,"/Users/amangolani/Downloads/converted_tess")


# Initialize openSMILE extractor
def OpenSmileCSV():
                smile = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.eGeMAPSv02,
                    feature_level=opensmile.FeatureLevel.Functionals,
                )

                # Root directory of your TESS dataset
                root_dir = "/Users/amangolani/Downloads/converted_tess"
                results = []

                # Walk through each subfolder
                for subdir, dirs, files in os.walk(root_dir):
                    for file in files:
                        if file.lower().endswith(".wav"):
                            filepath = os.path.join(subdir, file)
                            try:
                                # Extract features
                                features = smile.process_file(filepath)

                                # Derive label from folder name
                                emotion = os.path.basename(subdir).split("_")[-1].lower()  # e.g., OAF_angry → angry

                                features["filename"] = file
                                features["emotion"] = emotion
                                results.append(features)

                            except Exception as e:
                                print(f"Error with file {file}: {e}")

                # Combine all into one DataFrame
                all_features = pd.concat(results).reset_index(drop=True)
                all_features.drop_duplicates(inplace=True)
                # Save to CSV
                all_features.to_csv("/Users/amangolani/Downloads/tess_opensmile_features.csv", index=False)

                print("✅ Feature extraction and CSV export complete.")


OpenSmileCSV()