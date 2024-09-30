import helper
from PIL import Image
face_detector = helper.get_dlib_face_detector()

def main(input_image, output_path):
    try:
        img = Image.open(input_image).convert("RGB")
        landmarks = face_detector(img)
        for i,landmark in enumerate(landmarks):
            face = helper.crop_face(img, landmark)
            face.save(output_path)
    except:
        print("Failed to generete profile photo!")