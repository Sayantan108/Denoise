from django.apps import AppConfig
import tensorflow as tf
from tensorflow import keras
import os

class NoiseRemoverConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'noise_remover'
    dir=os.path.dirname(__file__)
    print(dir)
    dncnn=tf.keras.models.load_model(os.path.join(dir,'model\\dncnn.h5'))
