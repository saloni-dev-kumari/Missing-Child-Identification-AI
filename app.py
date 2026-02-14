from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import face_recognition

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
KNOWN_FOLDER = "data/known"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def load_known_faces():
    known_encodings = []
    known_names = []

    for filename in os.listdir(KNOWN_FOLDER):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(KNOWN_FOLDER, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(filename)

    return known_encodings, known_names

known_encodings, known_names = load_known_faces()

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    matched_name = ""

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            unknown_image = face_recognition.load_image_file(filepath)
            unknown_encodings = face_recognition.face_encodings(unknown_image)

            if len(unknown_encodings) == 0:
                result = "No face detected in the uploaded image."
            else:
                unknown_encoding = unknown_encodings[0]
                matches = face_recognition.compare_faces(known_encodings, unknown_encoding)

                if True in matches:
                    match_index = matches.index(True)
                    matched_name = known_names[match_index]
                    result = "Match Found!"
                else:
                    result = "No Match Found."

    return render_template("index.html", result=result, matched_name=matched_name)

if __name__ == "__main__":
    app.run(debug=True)
