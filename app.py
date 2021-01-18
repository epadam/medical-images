import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from random import randint
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

image = Image.open('medmnist.jpg')
st.image(image, use_column_width=True)


b = np.load('breastmnist.npz')

train_images=b['train_images']
val_images=b['val_images']
test_images=b['test_images']
train_labels=b['train_labels']
val_labels=b['val_labels'] 


test_img_num =[train_images.shape[0]]

title = st.title('Medical Image Classification and Segmentation')


header1 = st.sidebar.header('Medical Image Tasks')
tasks = st.sidebar.selectbox('', ('Choose a task','Classification', 'Segmentation'))



#'PathMNIST', 'ChestMNIST', 'DermaMNIST','OCTMNIST', 'PneumoniaMNIST','RetainaMNIST', 'BreastMNIST', 'OrganMNIST_Axial','OrganMNIST_Coronal','OrganMNIST_Sagittal'



if tasks == 'Choose a task':
    st.write('This demo collects different medical image analysis tasks and their machine learning solutions from different researches')
    
    st.header('Introduction')
    
    st.write('Most medical image analysis tasks face the problem of scarcity of the labeled data. Therefore, data augmemtation is very important')
    
    st.write('A good tool can save a lot of time. MONAI is open sourced by Nvidia for medical image processing')
    st.subheader('Data Augmentation')
    st.markdown('#### GAN')
    
    st.subheader('Model Selection')
    
    st.markdown('#### Loss Function')


    st.subheader('Model Exploration')

    st.subheader('Other Techniques')

    st.markdown('#### Transfer Learning')
    st.markdown('#### Few Shot Learning')
    st.markdown('#### Active Learning')
    st.markdown('#### Weakly Supervision')
#st.sidebar.write('Select dataset and visualization method')


if tasks == 'Classification':

    classification = st.selectbox('1. Select the dataset',
        ('Choose a dataset', 'MedMNIST', 'Kaggle Diabetic Retinopathy Detection', 'Intel & MobileODT Cervical Cancer Screening', 'MLSP 2014 Schizophrenia Classification Challenge', 'Kaggle Recursion Cellular Image Classification' ))
    
    if classification == 'MedMNIST':
            
        #fig, ax = plt.subplots(figsize=(1,1))
        
        med =st.button('show images')
        if med:
            fig = plt.figure(figsize=(8., 8.))
            grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(4, 4),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )
            plt.title('test image')
            for ax, im in zip(grid, [test_images[x] for x in range(16)]):
                # Iterating over the grid returns the Axes.
                ax.imshow(im)
            plt.imshow(test_images[1], cmap='gray')
            st.pyplot(fig,use_colunm_width=True)
        st.text('2. Make a prediction')
        clab = st.button('Predict  ')
        st.selectbox('3. Select visualization method',('LIME', 'Saliency Map','Grad-CAM'))

    
    #st.sidebar.text('2. Make a prediction')
   
    #clasb = st.sidebar.button('Predict ')
    
    # cvis = st.radio('3. Select visualization method ',
    #     ('LIME', 'Saliency Map','Grad-CAM'))



if tasks == 'Segmentation':
    
    header2 = st.header('Segmentation')
    segmentation = st.selectbox('1. Select the dataset', ('Choose a dataset', ' BraTS2017 ', 'Skin Lesion Analysis towards melanoma detection', 'Kaggle Ultrasound Nerve Segmentation', 'Kaggle 2018 Data Science Bowl'  ))

    st.text('2. Make a prediction')
    segb = st.button('Predict  ')

    svis = st.radio('3. Select visualization method',('LIME', 'Saliency Map','Grad-CAM'))

    



st.sidebar.text('')

st.sidebar.markdown( '[More resources and tools](https://github.com/epadam/machine-learning-overview/blob/master/applications/healthcare.md)')











# st.write(train_imagas[1])
