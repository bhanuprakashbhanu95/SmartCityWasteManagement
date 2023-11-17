#!/usr/bin/env python
# coding: utf-8

"""
Description
This is a streamlit app to showcase the results of training 4 ML models to help triage waste/trash in a given smart city.
The project aims to fullfiull the following action items:
- Rapid emergence of smart cities worldwide
- Integration of Machine Learning for data analytics
- Contribute to a cleaner, healthier urban environment

Purpose
- first: Inefficient waste management in smart cities
- Second: Leads to environmental issues and higher operational costs
- Third: Hinders optimization of waste management
- Fourth: Solution: Utilize Machine Learning for data analytics

"""
# Core Pkgs
import streamlit as st
#from streamlit import components
import streamlit.components.v1 as components
import os
from PIL import Image 
import warnings
warnings.filterwarnings("ignore")

#Visualization
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

#from dataset_milestone1 import datasets: Add all the different diseases 
import pandas as pd
## Required Libraries 
# 1- Navigateing through the operating system:
import os
from pathlib import Path

# 2- Unzip compressed set of folders - for the dataset
import zipfile

# 3- Visualization 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import seaborn as sns

# 4- PRedictive analysis libraries 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

from sklearn import svm 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 5- Computer vision library
import cv2

# 6- Linear ALgebra library
import numpy as np
import pandas as pd

# 7- Progress bar library
from tqdm import tqdm

# 8- Exporting the best model as a pickle file 
import pickle

# 9- Streamlit App
import streamlit as st

#=================================
## Data Acquisition
dataset_dir = '../dataset'
dataset_contents = os.listdir(dataset_dir)
dataset_contents[:10]

placeholder = '''
In an attempt to build an AI-ready workforce, Microsoft announced Intelligent Cloud Hub which has been launched to empower the next generation of students with AI-ready skills. Envisioned as a three-year collaborative program, Intelligent Cloud Hub will support around 100 institutions with AI infrastructure, course content and curriculum, developer support, development tools and give students access to cloud and AI services. As part of the program, the Redmond giant which wants to expand its reach and is planning to build a strong developer ecosystem in India with the program will set up the core AI infrastructure and IoT Hub for the selected campuses. The company will provide AI development tools and Azure AI services such as Microsoft Cognitive Services, Bot Services and Azure Machine Learning.According to Manish Prakash, Country General Manager-PS, Health and Education, Microsoft India, said, "With AI being the defining technology of our time, it is transforming lives and industry and the jobs of tomorrow will require a different skillset. This will require more collaborations and training and working with AI. That’s why it has become more critical than ever for educational institutions to integrate new cloud and AI technologies. The program is an attempt to ramp up the institutional set-up and build capabilities among the educators to educate the workforce of tomorrow." The program aims to build up the cognitive skills and in-depth understanding of developing intelligent cloud connected solutions for applications across industry. Earlier in April this year, the company announced Microsoft Professional Program In AI as a learning track open to the public. The program was developed to provide job ready skills to programmers who wanted to hone their skills in AI and data science with a series of online courses which featured hands-on labs and expert instructors as well. This program also included developer-focused AI school that provided a bunch of assets to help build AI skills.

'''

#==================================
# Prepare a dictionary to hold the categories and their file counts
category_file_counts = {}

# Iterate over each category directory and count the number of files
for category in dataset_contents:
    # Skip system files like .DS_Store
    if not category.startswith('.'):
        category_path = os.path.join(dataset_dir, category)
        # Count the number of image files, assuming they do not start with '.'
        image_files = [f for f in os.listdir(category_path) if not f.startswith('.')]
        category_file_counts[category] = len(image_files)

#category_file_counts

#==================================
## Creating a pandas dataframe to showcase the stats
descriptive_analysis = pd.DataFrame(list(category_file_counts.items()), columns=['WasteCategory','Count'])
categories = list(category_file_counts.keys())
file_counts = list(category_file_counts.values())
## Showcase the stats
#descriptive_analysis
#==================================
def Waste_Cat():
    ## Showcasing the description above into a piechart
    plt.figure(figsize=(8,6))
    plt.pie(descriptive_analysis['Count'], labels=descriptive_analysis['WasteCategory'], autopct='%1.1f%%', startangle=140)
    plt.title('Overall Waste Categories')
    plt.show()
#==================================
## Creating a dictionary of categories and their respective dimensions 
dimensions = {'cardboard':[], 'glass':[], 'Hazardous':[], 'Medical waste':[], 'metal':[], 'paper':[], 'plastic':[], 'trash':[]}

for category in categories:
    for image_file in os.listdir(os.path.join(dataset_dir, category)):
        with Image.open(os.path.join(dataset_dir, category, image_file)) as img:
            dimensions[category].append(img.size)
            
# defining numpy seed
np.random.seed(0)

## Converting the dimension dictionary into a pandas dataframe
dimensions_df = pd.DataFrame(list(dimensions.items()), columns=['WasteCategory','Dimensionality'])
#dimensions_df


#==================================
## Converting all the dimensions into a suitable format for plotting
widths= []
heights = []
categories_dimensionality = []

## populate the lists above
for category, dims in dimensions.items():
    for dim in dims:
        widths.append(dim[0])
        heights.append(dim[1])
        categories_dimensionality.append(category)

## Reviewed dataframe of dimentionality
dimensions_df_2 =  pd.DataFrame({'WasteCategory':categories_dimensionality, 'Width': widths, 'Height':heights})


            
#==================================            
def Distr_Of_Img():
    plt.figure(figsize=(10, 6))
    plt.bar(categories, file_counts, color='skyblue')
    plt.title('Distribution of Images across Trash Types')
    plt.xlabel('Trash Type')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    #==================================
def Img_Width_Distr():
    ## Plotting Width Distribution
    plt.figure(figsize=(12,6))
    sns.boxplot(x='WasteCategory', y='Width', data= dimensions_df_2)
    plt.title("Image Width Distribution by Waste Category")
    plt.show()
#==================================
def Img_Height_Distr():
    ## Plotting Height Distribution
    plt.figure(figsize=(12,6))
    sns.boxplot(x='WasteCategory', y='Height', data= dimensions_df_2)
    plt.title("Image Height Distribution by Waste Category")
    plt.show()
#==================================
## Creating a disctionary of colors distributions 
average_colors = {'cardboard':np.zeros(3), 'glass':np.zeros(3), 'Hazardous':np.zeros(3), 'Medical waste':np.zeros(3), 'metal':np.zeros(3), 'paper':np.zeros(3), 'plastic':np.zeros(3), 'trash':np.zeros(3)}

## navigating through categories to extract the the average RGB data:
for category in categories:
    total_images = 0
    for image_file in os.listdir(os.path.join(dataset_dir, category)):
        with Image.open(os.path.join(dataset_dir, category, image_file)) as img:
            ## Convert to RGB if image is RGBA
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            average_colors[category] += np.mean(np.array(img), axis=(0,1))
            total_images +=1
    if total_images >0:
        average_colors[category] /=total_images

## Visualize the AVG (RGB)
# Converting the RGB format into a suitable format for plotting:
average_colors_df = pd.DataFrame(average_colors, index=['Red','Green','Blue']).T
average_colors_df
#==================================
def Avg_Clr_Dist():
    ##Plotting the Average Colors Dataframe 
    average_colors_df.plot(kind='bar', figsize=(12,6))
    plt.title('Average Color Distribution by Waste Category')
    plt.ylabel('Average RGB Value')
    plt.show()

# Predictive Analysis
#==================================
# Function to display images
def display_images(images, titles):
    plt.figure(figsize=(15, 10))
    for i, image_path in enumerate(images):
        img = mpimg.imread(image_path)
        plt.subplot(1, len(images), i+1)
        plt.title(titles[i])
        plt.imshow(img)
        plt.axis('off')
    plt.show()

# Display the images with their labels
#display_images(sample_images_paths, ['paper']*len(sample_images_paths))

#==================================
from PIL import Image
def display_images_on_streamlit(images, titles):
    for i, image_path in enumerate(images):
        image=Image.open(image_path)
        st.image(image, caption=titles[i], use_column_width=True)

#==================================


#==================================

#Starting from the top
st.markdown("# Smart City Waste Management App™ v1.0")
st.markdown("By Bhanu Prakash & Nandan Ankireddy & Maha Laxmi")
original_title = '<p style="color:Orange; font-size: 30px;">Smart Cities Waste Manmagement App - AI Powered!</p>'
st.markdown(original_title, unsafe_allow_html=True)

img=Image.open('../img/logo.png')
st.image(img,width=200)
st.markdown('''
- **Business Needs**: 
- Rapid emergence of smart cities worldwide
- Integration of Machine Learning for data  analytics
- Contribute to a cleaner, healthier urban  environment

- **Methodology**: 
In this section, we present the methodology followed for garbage classification using four  learning models: StandardScaler, Logistic Regression, Random Forest, and KNN. The objective is to develop an efficient system for classifying different types of garbage, such as paper, cardboard, plastic, metal, trash, and glass.
- Data Preparation
- Exploratory Data Analysis
- Splitting Dataset
- Model Evaluation
- Accuracy Comparisonbhanu=Image.open('img/kalyani.png')


- **Problem Statement**: 
- first: Inefficient waste  management in  smart cities
- Second: Leads to  environmental  issues and higher  operational costs
- Third: Hinders  optimization of  waste management
- Fourth: Solution: Utilize  Machine Learning  for data analytics
''')
st.markdown("The data presented is of 8 different Waste categories - 'cardboard', 'glass','Hazardous', 'Medical waste', 'metal','paper','plastic','trash'** collected from Kaggle **https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset/data**")

if st.button("Learn more about  Bhanu Prakash & Nandan Ankireddy & Maha Laxmi"):
    maha=Image.open('../img/maha.png')
    nandan=Image.open('../img/nandan.png')
    bhanu=Image.open('../img/bhanu.png')
    st.markdown('''**Maha Laxmi ** Maha Lxxmi is Security Data Scientist with a passion for teaching and coaching. | Data Analytics | Machine Learning | Predictive Modeling | Data Visualization | NLP | Network Analytics | Network Security | Ethical Hacking |
He is knowledgeable and technically certified engineer with 7 years of continued hands-on experience in the implementation, administration and troubleshooting..''')
    st.image(maha,width=200, caption="Maha Laxmi 🤵‍")
    
    st.markdown('''<br>**Nandan Ankireddy **Nandan Ankireddy is Security Data Scientist with a passion for teaching and coaching. | Data Analytics | Machine Learning | Predictive Modeling | Data Visualization | NLP | Network Analytics | Network Security | Ethical Hacking |
He is knowledgeable and technically certified engineer with 7 years of continued hands-on experience in the implementation, administration and troubleshooting..''')
    st.image(nandan,width=200, caption="Nandan Ankireddy 👩‍💼‍")
    
    st.markdown('''<br>**Bhanu Prakash **hanu Prakash is Security Data Scientist with a passion for teaching and coaching. | Data Analytics | Machine Learning | Predictive Modeling | Data Visualization | NLP | Network Analytics | Network Security | Ethical Hacking |
He is knowledgeable and technically certified engineer with 7 years of continued hands-on experience in the implementation, administration and troubleshooting..''')
    st.image(bhanu,width=200, caption="hanu Prakash 👩‍💼‍")
    
    st.markdown("The data was collected and made available by **[Reda Mastouri](https://www.linkedin.com/in/reda-mastouri/**.")
    st.markdown("and **[Kalyani Pavuluri](https://www.linkedin.com/in/kalyani-pavuluri-30416519**.")
    images=Image.open('../img/presentation.png')
    st.image(images,width=700)
    #Ballons
    st.balloons()

# ============================================================
def main():
	""" NLP Based App with Streamlit """

	# Title
	st.title("Let's get started ..")
	st.subheader("Project Presentation:")
	st.markdown('''
    	+ Our project, "SmartWaste," is a machine learning approach that revolutionizes garbage image classification in the face of growing urban waste. Our technology employs sophisticated algorithms and neural networks to precisely classify waste photos into recyclables, non-recyclables, organics, and hazardous items, thereby mitigating the inefficiencies associated with manual sorting. Our goals in automating this process are to decrease the environmental effect, increase sustainability, and improve the efficiency of trash management. As a technologically advanced solution to optimize resource allocation, reduce landfill waste, and support worldwide environmental conservation efforts, SmartWaste represents an important step towards smart city initiatives.
    	''')
	# DatSet:
	st.subheader("A quick look at the dataset:")
	st.markdown('''
    To preview the datset, please check below.
    ''')
	st.sidebar.markdown("## Side Panel")
	st.sidebar.markdown("Use this panel to explore the dataset and create own viz.")
	st.header("Now, Explore Yourself the Imagery Dataset")
	# Create a text element and let the reader know the data is loading.
	data_load_state = st.text('Loading waste categories...')

	# Notify the reader that the data was successfully loaded.
	data_load_state.text('Loading images from various waste categories datasets...Completed!')
	bot=Image.open('../img/bot.png')
	st.image(bot,width=150)   	
    # Showing the original raw data
	if st.checkbox("Show Raw Data", False):
		st.subheader('Raw data')
		st.write(dataset_contents)
        
        
	st.title('Quick  Explore')
	st.sidebar.subheader(' Quick  Explore')
	st.markdown("Tick the box on the side panel to explore the dataset.")


	if st.sidebar.checkbox('Basic info'):
		if st.sidebar.checkbox('Quick Look'):
			st.subheader('Here is a quick look at the number of items per Waste Category:')
			st.write(category_file_counts)
		if st.sidebar.checkbox("Show Trash Counts Dataframe"):
			st.subheader('Show Trash Counts Dataframe')
			st.write(descriptive_analysis)
       
		if st.sidebar.checkbox('Statistical Description'):
			st.subheader('Detecting and Diving deeper into the Imagery Dimention Analysis')
			st.write(dimensions_df)
		if st.sidebar.checkbox('Image Dimentionality?'):
			st.subheader('Dimesnions by Widths an Heights')
			st.write(dimensions_df_2)
            
		if st.sidebar.checkbox('Average Color Distribution'):
			st.subheader('The following is the Average Color Distribution..')
			st.write(average_colors_df)


	# Visualization:   
	st.subheader("I - 📊 Visualization:")
	st.markdown('''
    For visualization, click any of the checkboxes to get started.
    ''')   
	if st.checkbox("Preview the descriptive analysis of the waste categories"):
		st.subheader("Pick one visualization at the time to preview the outcome of our analysis ..")

		summary_options = st.selectbox("Choose a visualization:",['Waste_Cat','Distr_Of_Img', 'Img_Width_Distr', 'Img_Height_Distr', 'Avg_Clr_Dist'])
		if st.button("Preview"):
			if summary_options == 'Waste_Cat':
				summary_result = Waste_Cat()
				st.set_option('deprecation.showPyplotGlobalUse', False)
				summary_result
				plt.show()
				st.pyplot()
			elif summary_options == 'Distr_Of_Img':
				summary_result = Distr_Of_Img()
				st.set_option('deprecation.showPyplotGlobalUse', False)
				summary_result
				plt.show()
				st.pyplot()
			elif summary_options == 'Img_Width_Distr':
				summary_result = Img_Width_Distr()
				st.set_option('deprecation.showPyplotGlobalUse', False)
				summary_result
				plt.show()
				st.pyplot()
			elif summary_options == 'Img_Height_Distr':
				summary_result = Img_Height_Distr()
				st.set_option('deprecation.showPyplotGlobalUse', False)
				summary_result
				plt.show()
				st.pyplot()
			elif summary_options == 'Avg_Clr_Dist':
				summary_result = Avg_Clr_Dist()
				st.set_option('deprecation.showPyplotGlobalUse', False)
				summary_result
				plt.show()
				st.pyplot()
			st.success(summary_result)
    

	st.subheader("II - 🧪 Prescriptive Analysis:")
	st.markdown('''
    In this step, we are exploring the imagery if it is compliant to the earliest classification.
    ''')   
	# Summarization
	if st.checkbox("Get a look at the ingested images"):
		st.subheader("This will display samples of images per waste category")

#		message = st.text_area("Enter Text",placeholder)
		summary_options = st.selectbox("Choose the waste category",['paper','metal', 'plastic', 'cardboard', 'glass', 'Hazardous', 'Medical waste', 'Organic Waste'])
		if st.button("Displaying Images .."):
			if summary_options == 'paper':
				#st.text(placeholder)
				# Path to one of the trash type directories
				sample_dir_path = Path(dataset_dir) / 'paper'
				# Get paths of the first few images in the directory
				sample_images_paths = list(sample_dir_path.glob('*.jpg'))[:3]
				summary_result = display_images(sample_images_paths, ['paper']*len(sample_images_paths))
				sample_titles = ['paper']*len(sample_images_paths)
				display_images_on_streamlit(sample_images_paths, sample_titles)    
			elif summary_options == 'metal':
				#st.text(placeholder)
				# Path to one of the trash type directories
				sample_dir_path = Path(dataset_dir) / 'metal'
				# Get paths of the first few images in the directory
				sample_images_paths = list(sample_dir_path.glob('*.jpg'))[:3]
				summary_result = display_images(sample_images_paths, ['metal']*len(sample_images_paths))
				sample_titles = ['metal']*len(sample_images_paths)
				display_images_on_streamlit(sample_images_paths, sample_titles)                
			elif summary_options == 'plastic':
				#st.text(placeholder)
				# Path to one of the trash type directories
				sample_dir_path = Path(dataset_dir) / 'plastic'
				# Get paths of the first few images in the directory
				sample_images_paths = list(sample_dir_path.glob('*.jpg'))[:3]
				summary_result = display_images(sample_images_paths, ['plastic']*len(sample_images_paths))
				sample_titles = ['plastic']*len(sample_images_paths)
				display_images_on_streamlit(sample_images_paths, sample_titles)
			elif summary_options == 'cardboard':
				#st.text(placeholder)
				 # Path to one of the trash type directories
				sample_dir_path = Path(dataset_dir) / 'cardboard'
				# Get paths of the first few images in the directory
				sample_images_paths = list(sample_dir_path.glob('*.jpg'))[:3]
				summary_result = display_images(sample_images_paths, ['cardboard']*len(sample_images_paths))
				sample_titles = ['cardboard']*len(sample_images_paths)
				display_images_on_streamlit(sample_images_paths, sample_titles)
			elif summary_options == 'glass':
				#st.text(placeholder)
				# Path to one of the trash type directories
				sample_dir_path = Path(dataset_dir) / 'glass'
				# Get paths of the first few images in the directory
				sample_images_paths = list(sample_dir_path.glob('*.jpg'))[:3]
				summary_result = display_images(sample_images_paths, ['glass']*len(sample_images_paths))
				sample_titles = ['glass']*len(sample_images_paths)
				display_images_on_streamlit(sample_images_paths, sample_titles)
			elif summary_options == 'Hazardous':
				#st.text(placeholder)
				# Path to one of the trash type directories
				sample_dir_path = Path(dataset_dir) / 'Hazardous'
				# Get paths of the first few images in the directory
				sample_images_paths = list(sample_dir_path.glob('*.jpg'))[:3]
				summary_result = display_images(sample_images_paths, ['Hazardous']*len(sample_images_paths))
				sample_titles = ['Hazardous']*len(sample_images_paths)
				display_images_on_streamlit(sample_images_paths, sample_titles)
			elif summary_options == 'Medical waste':
				#st.text(placeholder)
				# Path to one of the trash type directories
				sample_dir_path = Path(dataset_dir) / 'Medical waste'
				# Get paths of the first few images in the directory
				sample_images_paths = list(sample_dir_path.glob('*.jpg'))[:3]
				summary_result = display_images(sample_images_paths, ['Medical waste']*len(sample_images_paths))
				sample_titles = ['Medical waste']*len(sample_images_paths)
				display_images_on_streamlit(sample_images_paths, sample_titles)
			elif summary_options == 'Organic Waste':
				#st.text(placeholder)
				# Path to one of the trash type directories
				sample_dir_path = Path(dataset_dir) / 'trash'
				# Get paths of the first few images in the directory
				sample_images_paths = list(sample_dir_path.glob('*.jpg'))[:3]
				summary_result = display_images(sample_images_paths, ['trash']*len(sample_images_paths))
				sample_titles = ['trash']*len(sample_images_paths)
				display_images_on_streamlit(sample_images_paths, sample_titles)
			st.success(summary_result)

        
	# Sidebar
	st.sidebar.subheader("About the App")
	logobottom=Image.open('../img/logo.png')
	st.sidebar.image(logobottom,width=150)
	st.sidebar.text("mart Cities Waste Management 🤖")
	st.sidebar.info("Ai driven approach for Waste triaging withing a Smart City")   
	st.sidebar.markdown("[Data Source Respository](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset/data/")
	st.sidebar.info("Linkedin [Nandan Ankireddy](https://www.linkedin.com/in/reda-mastouri/) ")
	st.sidebar.info("Linkedin [Maha Laxmi](https://www.linkedin.com/in/kalyani-pavuluri-30416519) ")
	st.sidebar.info("Linkedin Bhanu Prakash](https://www.linkedin.com/in/kalyani-pavuluri-30416519) ")
	st.sidebar.text("SmartWasteManagement™ - Copyright © 2023")




if __name__ == '__main__':
	main()
