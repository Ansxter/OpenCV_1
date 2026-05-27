import os
from deepface import DeepFace
import shutil


input_dir = "face_data"
output_dir = "face_data_ok"


os.makedirs(output_dir, exist_ok=True)


for emotion_folder in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, emotion_folder)
    if not os.path.isdir(folder_path):
        continue

    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)

        try:
            result = DeepFace.analyze(img_path, actions=['emotion'], enforce_detection=False)
            detected_emotion = result[0]['dominant_emotion']

            target_folder = os.path.join(output_dir, detected_emotion)
            os.makedirs(target_folder, exist_ok=True)

            shutil.copy(img_path, os.path.join(target_folder, img_file))

            print(f"{img_file} -> {detected_emotion}")
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
