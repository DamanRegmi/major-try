# disease_detection/models.py

from django.db import models

class Image(models.Model):
    image = models.ImageField(upload_to='images/')
    prediction = models.CharField(max_length=100)
