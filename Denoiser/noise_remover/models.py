from django.db import models

class Image(models.Model):
    
    image = models.ImageField(upload_to='')

def __str__(self):
     return "input.jpg"