import cv2
import face_recognition
import os

# Load the reference image of Jatin
jatin_img = face_recognition.load_image_file('image.jpg')
jatin_encoding = face_recognition.face_encodings(jatin_img)[0]

# Read the input image
img = cv2.imread('image.jpg')

# Convert the input image to RGB (face_recognition uses RGB)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Find all face locations in the input image 
face_locations = face_recognition.face_locations(rgb_img)
face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

# Loop through the reco_img directory
reco_img_dir = 'reco_img'
for filename in os.listdir(reco_img_dir):
    if filename.endswith('.jpg'):
        # Read the image from the reco_img directory
        reco_img = face_recognition.load_image_file(os.path.join(reco_img_dir, filename))
        reco_encoding = face_recognition.face_encodings(reco_img)[0]

        # Compare face encodings
        results = face_recognition.compare_faces([jatin_encoding], reco_encoding)

        if results[0]:
            print(f"Jatin found in {filename}")
        else:
            print(f"Jatin not found in {filename}")
