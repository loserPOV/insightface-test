import cv2
import os

def main():
    face_detector = cv2.FaceDetectorYN.create(
        "face_detection_yunet_2023mar.onnx", "", (640, 480)
    )
    
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

            frame_resized = cv2.resize(frame, (640, 480))
            _, faces = face_detector.detect(frame_resized)

            if faces is not None:  # Check if faces is not None
                for face in faces:
                    if len(face) >= 4:
                        x, y, w, h = face[:4]
                        cv2.rectangle(frame_resized, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)

            cv2.imshow('Face Detection', frame_resized)
            if cv2.waitKey(1) == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
