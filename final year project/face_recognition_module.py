# face_recognition_module.py
import face_recognition
import cv2

def test_face_detection():
    video = cv2.VideoCapture(0)

    while True:
        ret, frame = video.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow("Face Detection Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# Run test only when this file is directly executed
if __name__ == "__main__":
    test_face_detection()
