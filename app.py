import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from random import randint
import matplotlib.pyplot as plt

image = Image.open('medmnist.jpg')
st.image(image, use_column_width=True)


title = st.title('Medical Image Classification and Segmentation')

header1 = st.sidebar.header('Classification')
classification = st.sidebar.selectbox(
    'Select the dataset',
    ('Choose a dataset', 'MedMNIST'))

#'PathMNIST', 'ChestMNIST', 'DermaMNIST','OCTMNIST', 'PneumoniaMNIST','RetainaMNIST', 'BreastMNIST', 'OrganMNIST_Axial','OrganMNIST_Coronal','OrganMNIST_Sagittal'

header2 = st.sidebar.header('Segmentation')
segmentation = st.sidebar.selectbox(
    'Select the dataset',
    ('Choose a dataset', 'MICCAI BraTS 2017', 'Sunny Cardiac Data'))


if classification == 'Choose a dataset' and segmentation == 'Choose a dataset':
    st.write('This demo collects different medical image analysis tasks and their machine learning solutions from different researches')
    
    st.header('Introduction')
    
    st.write('Most medical image analysis tasks face the problem of scarcity of the labeled data. Therefore, data augmemtation is very important')
    
    st.write('A good tool can save a lot of time. MONAI is open sourced by Nvidia for medical image processing')
    st.subheader('Data Augmentation')
    
    st.subheader('Model Selection')
    
    st.subheader('Model Exploration')
#st.sidebar.write('Select dataset and visualization method')

b = np.load('breastmnist.npz')

train_images=b['train_images']
val_images=b['val_images']
test_images=b['test_images']
train_labels=b['train_labels']
val_labels=b['val_labels'] 


test_img_num =[train_images.shape[0]]





# if classification != 'Choose a dataset':
    
#     #st.sidebar.text('2. Make a prediction')
#     image_number = st.sidebar.number_input('2. Select image number from testset and make prediction', max_value=test_img_num[0], step=1, value=1)
   
#     #clasb = st.sidebar.button('Predict ')
    
#     cvis = st.sidebar.radio('3. Select visualization method ',
#         ('LIME', 'Saliency Map','Grad-CAM'))



# if segmentation != 'Choose a dataset':
#     st.sidebar.text('2. Make a prediction')
#     segb = st.sidebar.button('Predict  ')

#     svis = st.sidebar.radio('3. Select visualization method',
#         ('LIME', 'Saliency Map','Grad-CAM'))

    



st.sidebar.text('')

st.sidebar.markdown( '[More resources and tools](https://github.com/epadam/machine-learning-overview/blob/master/applications/healthcare.md)')









if classification == 'BreastMNIST':
            
    #fig, ax = plt.subplots(figsize=(1,1))
    st.subheader('Breast cancer classification') 
    fig=plt.figure(figsize=(1,1))
    plt.title('test image')
    plt.imshow(test_images[1], cmap='gray')
    st.pyplot(fig, width=10)
    st.button('Predict')
    st.selectbox('Select visualization method',('LIME', 'Saliency Map','Grad-CAM'))


# st.write(train_imagas[1])
