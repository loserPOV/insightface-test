import cv2
from mtcnn import MTCNN

def main():
    # Initialize the MTCNN face detector
    detector = MTCNN()

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Detect faces in the frame
            faces = detector.detect_faces(frame)
            
            # Draw rectangles around detected faces
            for face in faces:
                x, y, width, height = face['box']
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                
                # You can also draw circles on the keypoints (eyes, nose, mouth)
                for key, value in face['keypoints'].items():
                    cv2.circle(frame, (value[0], value[1]), 2, (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('Face Detection', frame)
            if cv2.waitKey(1) == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
