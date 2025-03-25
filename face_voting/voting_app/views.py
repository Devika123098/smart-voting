
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import Voter, Vote
from .face_recognition import register_face, recognize_face
import datetime
import csv
import os
import pyttsx3

def speak(message):
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()

def check_if_voted(aadhar_number):
    if os.path.exists("Votes.csv"):
        with open("Votes.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[0] == aadhar_number:
                    return True
    return False

def home(request):
    return render(request, 'home.html')

def register(request):
    if request.method == "POST":
        aadhar_number = request.POST['aadhar_number']
        if register_face(aadhar_number):
            try:
                Voter.objects.create(aadhar_number=aadhar_number)
                return redirect('home')
            except Exception as e:
                print(e)
                return render(request, 'register.html', {'error': 'Error creating voter'})
        else:
            return render(request, 'register.html', {'error': 'Face registration failed'})
    return render(request, 'register.html')
    
    return render(request, 'register.html')
# def vote(request):
#     if request.method == "POST":
#         aadhar_number = recognize_face()  # Authenticate voter using face recognition

#         if aadhar_number is None:
#             speak("You are not registered")
#             return render(request, 'not_registered.html', {'message': 'You are not registered.'})

#         if check_if_voted(aadhar_number):
#             speak("You have already voted")
#             return render(request, 'already_voted.html')  # Show an error page


#         candidate = request.POST['candidate']
#         date = datetime.date.today()
#         time = datetime.datetime.now().strftime("%H:%M:%S")

#         voter = Voter.objects.get(aadhar_number=aadhar_number)
#         Vote.objects.create(voter=voter, candidate=candidate)

#         with open("Votes.csv", "a") as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow([aadhar_number, candidate, date, time])

#         speak("Thank you for voting")

#         return render(request, 'success.html')  # Redirect to a success page

#     return render(request, 'vote.html')
def vote(request):
    if request.method == "POST":
        aadhar_number = recognize_face()  # Authenticate voter using face recognition

        if aadhar_number is None:
            speak("You are not registered")
            return render(request, 'not_registered.html', {'message': 'You are not registered.'})

        # Check if the face data exists in the "data" folder
        data_folder = 'data'
        face_data_file = os.path.join(data_folder, aadhar_number + '.jpg')  # Assuming face data is stored as a JPEG file

        if not os.path.exists(face_data_file):
            speak("You are not registered")
            return render(request, 'not_registered.html', {'message': 'You are not registered.'})

        if check_if_voted(aadhar_number):
            speak("You have already voted")
            return render(request, 'already_voted.html')  # Show an error page

        candidate = request.POST['candidate']
        date = datetime.date.today()
        time = datetime.datetime.now().strftime("%H:%M:%S")

        voter = Voter.objects.get(aadhar_number=aadhar_number)
        Vote.objects.create(voter=voter, candidate=candidate)

        with open("Votes.csv", "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([aadhar_number, candidate, date, time])

        speak("Thank you for voting")

        return render(request, 'success.html')  # Redirect to a success page

    return render(request, 'vote.html')