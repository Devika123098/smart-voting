import cv2
import numpy as np
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier
import pyttsx3

facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def register_face(aadhar_number):
    if not os.path.exists('data/'):
        os.makedirs('data/')

    video = cv2.VideoCapture(0)
    faces_data = []
    i = 0
    frames_total = 50
    capture_after_frame = 2

    while True:
        ret, frame = video.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (50,50))

            if len(faces_data) <= frames_total and i % capture_after_frame == 0:
                faces_data.append(resized_img)

            i += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50,50,255), 1)

        cv2.imshow('Register Face', frame)
        k = cv2.waitKey(1)
        if k == ord('q') or len(faces_data) >= frames_total:
            break

    video.release()
    cv2.destroyAllWindows()

    faces_data = np.asarray(faces_data).reshape((frames_total, -1))

    with open(f'data/{aadhar_number}.pkl', 'wb') as f:
        pickle.dump(faces_data, f)

    return True

def recognize_face():
    video = cv2.VideoCapture(0)
    face_data_list = []
    labels = []

    for file in os.listdir('data/'):
        if file.endswith('.pkl'):
            aadhar_number = file.split('.')[0]
            with open(f'data/{file}', 'rb') as f:
                faces = pickle.load(f)
                for face in faces:
                    face_data_list.append(face)
                    labels.append(aadhar_number)

    if not face_data_list:
        print("No registered faces found.")
        return None

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(face_data_list, labels)

    while True:
        ret, frame = video.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        # Check if no faces were detected
        if len(faces) == 0:
            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

            result = knn.predict(resized_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50,50,255), 2)
            cv2.putText(frame, result[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow('Face Recognition', frame)
            
            # Check if face is recognized
            if result[0] in labels:
                video.release()
                cv2.destroyAllWindows()
                return result[0]
            else:
                # If not recognized, generate audio feedback and show error page
                engine = pyttsx3.init()
                engine.say("You are not registered.")
                engine.runAndWait()
                
                # Here you would redirect or change view to show the error message
                print("User not registered. Redirecting to error page.")
                return None

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    return None