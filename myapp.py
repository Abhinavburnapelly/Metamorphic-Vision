import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from facenet_pytorch.models.inception_resnet_v1 import get_torch_home
torch_home = get_torch_home()
import os
import torch
import cv2
import time
import glob
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
from tqdm import tqdm
from flask import Flask,render_template,redirect,url_for

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Model

    # Height and width refer to the size of the image
    # Channels refers to the amount of color channels (red, green, blue)

    image_dimensions = {'height':256, 'width':256, 'channels':3}
    # Create a Classifier class

    class Classifier:
        def __init__(self):
            self.model = 0
        
        def predict(self, x):
            return self.model.predict(x)
        
        def fit(self, x, y):
            return self.model.train_on_batch(x, y)
        
        def get_accuracy(self, x, y):
            return self.model.test_on_batch(x, y)
        
        def load(self, path):
            self.model.load_weights(path)
    # Create a MesoNet class using the Classifier

    class Meso4(Classifier):
        def __init__(self, learning_rate = 0.001):
            self.model = self.init_model()
            optimizer = Adam(lr = learning_rate)
            self.model.compile(optimizer = optimizer,
                            loss = 'mean_squared_error',
                            metrics = ['accuracy'])
        
        def init_model(self): 
            x = Input(shape = (image_dimensions['height'],
                            image_dimensions['width'],
                            image_dimensions['channels']))
            
            x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)
            x1 = BatchNormalization()(x1)
            x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
            
            x2 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)
            x2 = BatchNormalization()(x2)
            x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
            
            x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
            x3 = BatchNormalization()(x3)
            x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
            
            x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
            x4 = BatchNormalization()(x4)
            x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
            
            y = Flatten()(x4)
            y = Dropout(0.5)(y)
            y = Dense(16)(y)
            y = LeakyReLU(alpha=0.1)(y)
            y = Dropout(0.5)(y)
            y = Dense(1, activation = 'sigmoid')(y)

            return Model(inputs = x, outputs = y)
        
        # Instantiate a MesoNet model with pretrained weights
    meso = Meso4()
    meso.load('./weights/Meso4_DF')
    # Prepare image data

    # Rescaling pixel values (between 1 and 255) to a range between 0 and 1
    dataGenerator = ImageDataGenerator(rescale=1./255)

    # Instantiating generator to feed images through the network
    # generator = dataGenerator.flow_from_directory(
    #     './data/',
    #     target_size=(256, 256),
    #     batch_size=1,
    #     class_mode='binary')
    # # Recreating generator after removing '.ipynb_checkpoints'
    # dataGenerator = ImageDataGenerator(rescale=1./255)
    
    generator = dataGenerator.flow_from_directory(
        './test/',
        target_size=(256, 256),
        batch_size=1,
        class_mode='binary')

    # Re-checking class assignment after removing it
    generator.class_indices
    # Rendering image X with label y for MesoNet
    X, y = generator.next()

# Evaluating prediction
    a=(meso.predict(X)[0][0]-0.1)
    import pandas as pd

    # Step 1: Read the CSV file into a DataFrame
    csv_file_path = 'submission.csv'
    df = pd.read_csv(csv_file_path)

    # Step 2: Replace the existing row with new data
    new_row_data = {"filename":"Image",'label':a}
    df.iloc[0] = new_row_data # Assuming there's only one row in the DataFrame

    # Step 3: Write the updated DataFrame back to the CSV file
    print(df)
    df.to_csv(csv_file_path, index=False)

    return render_template("index.html")

if __name__ == '__main__':
	app.run(debug=True)
