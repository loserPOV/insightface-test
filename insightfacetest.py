import cv2
import insightface
from insightface.app import FaceAnalysis
import numpy as np
import os

def extract_embeddings_from_folders(root_folder, app):
    embeddings = []
    labels = []

    for subdir, _, files in os.walk(root_folder):
        if subdir == root_folder:
            continue  # Skip the root folder itself
        label = os.path.basename(subdir)
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(subdir, filename)
                img = cv2.imread(filepath)

                # Detect faces in the image
                faces = app.get(img)
                if len(faces) > 0:
                    face = faces[0]  # Assume the first detected face is the target
                    embedding = face.normed_embedding  # Extract the embedding
                    embeddings.append(embedding)
                    labels.append(label)

    return embeddings, labels

def main():
    # Initialize the FaceAnalysis app with the detection and recognition model
    app = FaceAnalysis(name='buffalo_l', root='~/.insightface/models', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Load known face embeddings and labels from the folder
    root_folder = 'face_db'
    known_embeddings, known_labels = extract_embeddings_from_folders(root_folder, app)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    cv2.namedWindow('Face Detection and Recognition', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Face Detection and Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def recognize_face(embedding):
        if not known_embeddings:
            return "Unknown"

        # Compare the embedding to the known embeddings
        similarities = []
        for known_embedding in known_embeddings:
            similarity = np.dot(embedding, known_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(known_embedding))
            similarities.append(similarity)

        best_match_idx = np.argmax(similarities)
        if similarities[best_match_idx] > 0.5:  # Threshold for recognition
            return known_labels[best_match_idx]
        else:
            return "Unknown"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Resize the frame to new dimensions
            frame_resized = cv2.resize(frame, (640, 480))

            # Use InsightFace to detect faces
            faces = app.get(frame_resized)

            for face in faces:
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame_resized, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

                # Extract face embedding
                embedding = face.normed_embedding

                # Recognize face
                label = recognize_face(embedding)

                # Display label
                cv2.putText(frame_resized, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('Face Detection and Recognition', frame_resized)
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
