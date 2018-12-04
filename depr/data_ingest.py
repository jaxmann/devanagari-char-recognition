import os

import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import cross_validate, GridSearchCV, learning_curve
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

#%matplotlib inline


root_dir = os.getcwd()
img_dir = os.path.join(root_dir, 'data/train')

pixels = np.array(['pixel_{:04d}'.format(x) for x in range(1024)])
flag = True

for char_name in sorted(os.listdir(img_dir)):
    char_dir = os.path.join(img_dir, char_name)
    img_df = pd.DataFrame(columns=pixels)
    
    for img_file in sorted(os.listdir(char_dir)):
        image = pd.Series(imread(os.path.join(char_dir, img_file)).flatten(), index=pixels)
        img_df = img_df.append(image.T, ignore_index=True)
        
    img_df = img_df.astype(np.uint8)
    img_df['character'] = char_name
    
    img_df.to_csv('data.csv', index=False, mode='a', header=flag)
    flag=False
    
    print('=') 
    #print('=', end='')
    break    
    
df = pd.read_csv('data.csv')

df['character_class'] = LabelEncoder().fit_transform(df.character)
df.drop('character', axis=1, inplace=True)
df = df.astype(np.uint8)
print(df)
