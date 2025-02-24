
# Create your models here.
from django.db import models

class Voter(models.Model):
    aadhar_number = models.CharField(max_length=12, unique=True)
    face_data = models.BinaryField()   # Store face data as binary

    def __str__(self):
        return self.aadhar_number

class Vote(models.Model):
    voter = models.ForeignKey(Voter, on_delete=models.CASCADE)
    candidate = models.CharField(max_length=50)
    date = models.DateField(auto_now_add=True)
    time = models.TimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.voter.aadhar_number} voted for {self.candidate}"
