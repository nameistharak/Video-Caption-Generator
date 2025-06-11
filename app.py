import os
import shutil
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load BLIP2 model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
model.to(device)

# Video frame sampling function
def extract_keyframes(video_path, frame_skip=30):
    frames = []
    cap = cv2.VideoCapture(video_path)
    index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if index % frame_skip == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            frames.append(pil_img)
        index += 1
    cap.release()
    return frames

# Generate caption for a single frame
def describe_image(pil_img):
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    caption = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return caption

# Route for homepage
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "video" not in request.files:
            return "No video part"
        file = request.files["video"]
        if file.filename == "":
            return "No selected video"
        if file:
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(video_path)

            # Extract frames and generate captions
            frames = extract_keyframes(video_path)
            descriptions = []
            seen = set()
            for frame in frames:
                desc = describe_image(frame)
                if desc not in seen:
                    descriptions.append(desc)
                    seen.add(desc)

            full_description = ". ".join(descriptions)
            shutil.rmtree(UPLOAD_FOLDER)
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            return render_template("result.html", description=full_description)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
