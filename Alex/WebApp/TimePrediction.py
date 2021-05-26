# Import the relevant packages

import streamlit as st

import pandas as pd
import numpy as np

# Import the prediction function

from Joseph_alg import ult_pred
from Joseph_alg import get_values


# Import the dataframe

data = pd.read_csv('data18m.csv', index_col=0)
smooth = pd.read_csv('Smooth_data18m_pure.csv', index_col=0)
nonsmooth = pd.read_csv('nonsmooth_data18m_pure.csv', index_col=0)

smooth_data = get_values(smooth)
nonsmooth_data = get_values(nonsmooth)


# Write the headline

st.write("""

# Joseph's magic algorithm

""")

## Set up the sidebar

# Function for the inputs

def user_input_features(data):

    No_Of_Months = st.sidebar.slider('Select No Of Months To Predict',1,12,1)

    Game_Num = int(st.sidebar.text_input('Select A Game\'s Number From The List Below', 0))

    Game_Name = data.iloc[Game_Num].Name
    
    data = {'Game Number': Game_Num,
            'Game': Game_Name,
            'Number of Months': No_Of_Months}
    features = pd.DataFrame(data, index=[0])
    return features

# Header

st.sidebar.header('User Input Values')

df = user_input_features(data)

st.sidebar.write(data['Name'])

## Set up the output

# Print the choice made

st.subheader('User\'s choice:')

st.write(df)

# ### Imaginary Examples:

# In[188]:

name = df['Game'].values[0]
months = df['Number of Months'].values[0]

game = nonsmooth_data[data.loc[data['Name']==name].index[0]][:12]
real_data = nonsmooth_data[data.loc[data['Name']==name].index[0]][11:12+months]

subdata = nonsmooth_data.copy()
del subdata[data.loc[data['Name']==name].index[0]]
smooth_subdata = smooth_data.copy()
del smooth_subdata[data.loc[data['Name']==name].index[0]]

[pred, close_index,close_games] = ult_pred(game, train = subdata, smooth_train = smooth_subdata, real_data = real_data, horizon = months)

# Print the closest three games

st.subheader('Closest three games:')

st.write(close_games)

