import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('model/cardiomegaly_model.h5')

def predict_xray(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        return "Cardiomegaly (TRUE)"
    else:
        return "Normal (FALSE)"