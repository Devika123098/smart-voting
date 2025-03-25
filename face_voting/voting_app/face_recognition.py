# import cv2
# import numpy as np
# import pickle
# import os
# from sklearn.neighbors import KNeighborsClassifier
# import pyttsx3

# facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# def register_face(aadhar_number):
#     if not os.path.exists('data/'):
#         os.makedirs('data/')

#     video = cv2.VideoCapture(0)
#     faces_data = []
#     i = 0
#     frames_total = 50
#     capture_after_frame = 2

#     while True:
#         ret, frame = video.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = facedetect.detectMultiScale(gray, 1.3, 5)

#         for (x, y, w, h) in faces:
#             crop_img = frame[y:y+h, x:x+w]
#             resized_img = cv2.resize(crop_img, (50,50))

#             if len(faces_data) <= frames_total and i % capture_after_frame == 0:
#                 faces_data.append(resized_img)

#             i += 1
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (50,50,255), 1)

#         cv2.imshow('Register Face', frame)
#         k = cv2.waitKey(1)
#         if k == ord('q') or len(faces_data) >= frames_total:
#             break

#     video.release()
#     cv2.destroyAllWindows()

#     faces_data = np.asarray(faces_data).reshape((frames_total, -1))

#     with open(f'data/{aadhar_number}.pkl', 'wb') as f:
#         pickle.dump(faces_data, f)

#     return True

# def recognize_face():
#     video = cv2.VideoCapture(0)
#     face_data_list = []
#     labels = []

#     for file in os.listdir('data/'):
#         if file.endswith('.pkl'):
#             aadhar_number = file.split('.')[0]
#             with open(f'data/{file}', 'rb') as f:
#                 faces = pickle.load(f)
#                 for face in faces:
#                     face_data_list.append(face)
#                     labels.append(aadhar_number)

#     if not face_data_list:
#         print("No registered faces found.")
#         return None

#     knn = KNeighborsClassifier(n_neighbors=5)
#     knn.fit(face_data_list, labels)

#     while True:
#         ret, frame = video.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = facedetect.detectMultiScale(gray, 1.3, 5)

#         # Check if no faces were detected
#         if len(faces) == 0:
#             cv2.imshow('Face Recognition', frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#             continue

#         for (x, y, w, h) in faces:
#             crop_img = frame[y:y+h, x:x+w]
#             resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

#             result = knn.predict(resized_img)
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (50,50,255), 2)
#             cv2.putText(frame, result[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#             cv2.imshow('Face Recognition', frame)
            
#             # Check if face is recognized
#             if result[0] in labels:
#                 video.release()
#                 cv2.destroyAllWindows()
#                 return result[0]
#             else:
#                 # If not recognized, generate audio feedback and show error page
#                 engine = pyttsx3.init()
#                 engine.say("You are not registered.")
#                 engine.runAndWait()
                
#                 # Here you would redirect or change view to show the error message
#                 print("User not registered. Redirecting to error page.")
#                 return None

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     video.release()
#     cv2.destroyAllWindows()
#     return None
import cv2
import numpy as np
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pyttsx3

# Load the face detection model
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def register_face(aadhar_number):
    # Create the data directory if it doesn't exist
    if not os.path.exists('data/'):
        os.makedirs('data/')

    # Initialize the video capture object
    video = cv2.VideoCapture(0)

    # Initialize the face data list
    faces_data = []
    labels = []

    # Set the number of frames to capture
    frames_total = 50
    capture_after_frame = 2

    # Initialize the frame counter
    i = 0

    while True:
        # Read a frame from the video
        ret, frame = video.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        # Loop through the detected faces
        for (x, y, w, h) in faces:
            # Crop the face from the frame
            crop_img = frame[y:y+h, x:x+w]

            # Resize the face to 50x50
            resized_img = cv2.resize(crop_img, (50, 50))

            # Add the face to the face data list
            if len(faces_data) <= frames_total and i % capture_after_frame == 0:
                faces_data.append(resized_img.flatten())
                labels.append(aadhar_number)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50,50,255), 1)

        # Display the frame
        cv2.imshow('Register Face', frame)

        # Check for the 'q' key press
        k = cv2.waitKey(1)
        if k == ord('q') or len(faces_data) >= frames_total:
            break

        # Increment the frame counter
        i += 1

    # Release the video capture object
    video.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    # Save the face data to a pickle file
    with open(f'data/{aadhar_number}.pkl', 'wb') as f:
        pickle.dump((faces_data, labels), f)

    return True

def recognize_face():
    # Initialize the video capture object
    video = cv2.VideoCapture(0)

    # Initialize the face data list and labels
    face_data_list = []
    labels = []

    # Load the face data from the pickle files
    for file in os.listdir('data/'):
        if file.endswith('.pkl'):
            with open(f'data/{file}', 'rb') as f:
                faces, face_labels = pickle.load(f)
                face_data_list.extend(faces)
                labels.extend(face_labels)

    # Check if there are any registered faces
    if not face_data_list:
        print("No registered faces found.")
        return None

    # Split the face data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(face_data_list, labels, test_size=0.2, random_state=42)

    # Initialize the KNN model
    knn = KNeighborsClassifier(n_neighbors=5)

    # Train the KNN model
    knn.fit(X_train, y_train)

    # Evaluate the KNN model
    y_pred = knn.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    while True:
        # Read a frame from the video
        ret, frame = video.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        # Loop through the detected faces
        for (x, y, w, h) in faces:
            # Crop the face from the frame
            crop_img = frame[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

            # Predict the label of the face
            result = knn.predict(resized_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            cv2.putText(frame, result[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow('Face Recognition', frame)

            # Check if the face is recognized
            if result[0] in labels:
                video.release()
                cv2.destroyAllWindows()
                return result[0]
            else:
                # Generate audio feedback for unrecognized faces
                engine = pyttsx3.init()
                engine.say("You are not registered.")
                engine.runAndWait()
                
                print("User  not registered. Redirecting to error page.")
                return None

        # Check for the 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    return None
