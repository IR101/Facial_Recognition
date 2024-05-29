import threading
import cv2
from deepface import DeepFace

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global variables
counter = 0
face_match = 0
img_list = []
index = 0

# Load reference images
img_path = [
    'ifront.jpg','hfront.jpg','afront.jpg'
]

for i in img_path:
    img = cv2.imread(i)
    if img is not None:
        img_list.append(img)
    else:
        print(f"Error loading image {i}")

def check_face(frame):
    global face_match
    global index 
    try:
        for i, img in enumerate(img_list):
            if DeepFace.verify(frame, img)['verified']:
                face_match = 1
                index = i
                return
        face_match = -1
    except ValueError:
        face_match = 0

while True:
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 2)

        if len(faces) == 0:
            face_match = 0
        else:
            if counter % 30 == 0:
                threading.Thread(target=check_face, args=(frame,)).start()
            counter += 1

            if face_match == 1 and index == 0:
                cv2.putText(frame, "INAM", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            elif face_match == 1 and index == 1:
                cv2.putText(frame, "HADIA", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            elif face_match == 1 and index == 2:
                cv2.putText(frame, "ANOOSH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            elif face_match == -1:
                cv2.putText(frame, "NO-MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            else:
                cv2.putText(frame, "Waiting...", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

        cv2.imshow("video", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
