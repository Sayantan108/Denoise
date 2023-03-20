from django.shortcuts import render,redirect
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .apps import NoiseRemoverConfig
from rest_framework.decorators import api_view
from .forms import *
import cv2
from pathlib import Path
import os
import numpy as np
import tensorflow as tf

def extract_patches(filepath, patch_sz,crop_szs,save_dir=None):
    print("EXTRACT")
    print(filepath)
    image=cv2.imread(filepath)
    filename=filepath.split('\\')[-1].split('.')[0] #extracting imgname.jpg
    height,width,channels=image.shape
    patches=[]
    for crop_sz in crop_szs:
        crop_ht,crop_wd=int(height*crop_sz),int(width*crop_sz)
        image_scaled=cv2.resize(image,(crop_wd,crop_ht), interpolation=cv2.INTER_CUBIC)
        for i in range(0,crop_ht-patch_sz+1,patch_sz):
            for j in range(0,crop_wd-patch_sz+1,patch_sz):
                x=image_scaled[i:i+patch_sz,j:j+patch_sz]

                if save_dir is not None:
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    patch_filepath = save_dir+'/'+filename+'_'+str(crop_ht)+'_'+str(i)+'_'+str(j)+'.jpg'
                    cv2.imwrite(patch_filepath,x)

                patches.append(x)
    return patches

def join_patches(patches,img_shp):
    image=np.zeros(img_shp)
    patch_sz=patches.shape[1]
    p=0
    for i in range(0,image.shape[0]-patch_sz+1,patch_sz):
        for j in range(0,image.shape[1]-patch_sz+1,patch_sz):
            image[i:i+patch_sz,j:j+patch_sz]=patches[p]
            p+=1
    return np.array(image)

def predict_fun(model,image_path,noise_level=30):
    #Creating patches for test image
    patches=extract_patches(image_path,40,[1])
    test_image=cv2.imread(image_path)
    test_image=cv2.cvtColor(test_image,cv2.COLOR_BGR2RGB)
    patches=np.array(patches)
    ground_truth=join_patches(patches,test_image.shape)

    #predicting the output on the patches of test image
    patches = patches.astype('float32') /255.
    patches_noisy = patches+ tf.random.normal(shape=patches.shape,mean=0,stddev=noise_level/255) 
    patches_noisy = tf.clip_by_value(patches_noisy, clip_value_min=0., clip_value_max=1.)
    noisy_image=join_patches(patches_noisy,test_image.shape)

    denoised_patches=model.predict(patches_noisy)
    denoised_patches=tf.clip_by_value(denoised_patches, clip_value_min=0., clip_value_max=1.)

    #Creating entire denoised image from denoised patches
    denoised_image=join_patches(denoised_patches,test_image.shape)
    denoised_image=denoised_image*255
    return denoised_image


def post_noisy_image(request):
  
    if request.method == 'POST':
        form = ImageForm(request.POST,request.FILES)

        if form.is_valid():
            form.save()
            BASE_DIR=Path(__file__).resolve().parent.parent
            print(BASE_DIR)
            filename=os.listdir(BASE_DIR/'media')
            print(filename[0])
            string=os.path.join(BASE_DIR/'media')
            filename=os.path.join(Path(string).resolve()/filename[0])
            
            img=predict_fun(NoiseRemoverConfig.dncnn, filename)

            cv2.imwrite('output.jpg',img)
            img=cv2.imread('output.jpg')
            response = HttpResponse(img, headers={'Content-Type': 'image/jpeg','Content- Disposition': 'attachment; filename="output.jpg"'})
            return redirect('success')
    else:
        form = ImageForm()
    return render(request, 'imageForm.html', {'form' : form})
  
  
def uploadok(request):
    return HttpResponse(' upload successful')