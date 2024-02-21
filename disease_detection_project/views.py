# disease_detection/views.py

from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .models import Image
from .resnet_model import load_resnet18, preprocess_image, predict_image, get_disease_name

def predict_disease(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        # Load the pre-trained ResNet-18 model
        model = load_resnet18()
        # Preprocess the image
        input_image = preprocess_image(uploaded_image)
        # Make prediction
        prediction = predict_image(input_image, model)
        # Map prediction to disease name
        disease_name = get_disease_name(prediction)
        # Save image and prediction to database
        Image.objects.create(image=uploaded_image, prediction=disease_name)
        return render(request, 'disease_detection/result.html', {
            'uploaded_image': uploaded_image,
            'disease_name': disease_name
        })
    return render(request, 'disease_detection/predict.html')
