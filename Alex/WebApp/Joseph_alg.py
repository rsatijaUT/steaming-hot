#!/usr/bin/env python
# coding: utf-8

# # Ultimate Model App
# This notebook is almost the same as the TimeSeries_Avg_UltModel notebook, but most of the base data computation is skipped by importing them.

# In[3]:
import streamlit as st

import pandas as pd
import numpy as np

#For plotting
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import datetime

# Import the SimpleExpSmoothing object
from statsmodels.tsa.api import SimpleExpSmoothing

# ### Load in all data from the DataPreparation notebook

# In[4]:

data = pd.read_csv('data18m.csv', index_col=0)
smooth = pd.read_csv('Smooth_data18m_pure.csv', index_col=0)
nonsmooth = pd.read_csv('nonsmooth_data18m_pure.csv', index_col=0)


# In[10]:


def get_values(data):
    answer = []
    for i in range(len(data)):
        game = data.iloc[i]
        temp = game[game.first_valid_index():game.last_valid_index()].values
        answer.append(temp)
    return answer


# In[11]:


smooth_data = get_values(smooth)
nonsmooth_data = get_values(nonsmooth)


# ### Metric
# 
# Define the metric and weight using l2 norm and Gaussian kernel.

# In[6]:


def rootmse(a, b):
    return np.sqrt(np.sum((a-b)**2))

def Gauss_weight(a, b, epsilon = 20):
    return np.exp(-epsilon*rootmse(a,b)**2)


# To compare the shape of 2 curves, we want to compute the scalar that makes 2 curves close together by scaling only.

# In[7]:


def mini_scaler(a,b):
    return np.sum(a*b)/np.sum(a**2)


# ### Smoothing
# Exponential Smoothing with smoothing level = .6

# In[8]:


# Getting the smoothed curve for each game
# Ordered in increasing time order.

def smooth_values(data, smoothing_level=.6):
    smooth_timedata = []
    for i in range(len(data)):
        game_data = data.iloc[i]
        months = pd.to_datetime(game_data.index)
        game = pd.DataFrame({'Month':months,'Data': game_data.astype(float)}).sort_values(by=['Month'])
        temp = game['Data'][game['Data'].first_valid_index():game['Data'].last_valid_index()]
    
        # Fit exponential smoothing
        ses = SimpleExpSmoothing(temp.values)
        fit = ses.fit(smoothing_level=smoothing_level, optimized=False)

        smooth_timedata.append(fit.fittedvalues)
        
    return smooth_timedata


# ### Main Algorithm

# In[11]:


def wt_avg(game, smooth_game, data, smooth_data, metric = Gauss_weight, epsilon = 20, threshold=0.8, horizon = 6):
    length = len(game)
    smooth_length = len(smooth_game)
    if np.max(np.abs(smooth_game))!=0:
        smooth_game_scaled = smooth_game / rootmse(smooth_game,0)
    else:
        smooth_game_scaled = smooth_game
    pred = np.zeros(length+horizon)
    close_index = np.zeros((len(data),2))
    j=0
    for i in range(len(data)):
        temp = data[i]
        smooth_temp = smooth_data[i]
        if len(temp)>=length+horizon:
            if np.max(np.abs(temp[:length]))!=0:
                temp_scaled = temp * mini_scaler(temp[:length],game)
            else:
                temp_scaled = temp
            if np.max(np.abs(smooth_temp[:smooth_length]))!=0:
                smooth_temp_scaled = smooth_temp * mini_scaler(smooth_temp[:smooth_length],smooth_game_scaled)
            else:
                smooth_temp_scaled = smooth_temp
            weight=metric(smooth_game_scaled,smooth_temp_scaled[:smooth_length], epsilon = epsilon)
            if weight >= threshold:
                pred = pred + weight * temp_scaled[:length+horizon]
                close_index[j]=[i,weight]
                j=j+1
    if np.max(np.abs(pred[:length])) !=0:
        pred = pred * mini_scaler(pred[:length],game)
    else:
        pred = np.ones(length + horizon) * smooth_game[-1]
    close_index = close_index[:j]
    close_index = close_index[np.argsort(close_index[:, 1])][::-1]
    return pred, close_index


# # Ultimate Prediction Function
# 
# The model is default for predicting 6-month data given the first 12-month data and the parameters are optimized for such choice. However the user can enter data of any length and change the horizon for prediction.
# 
# Input: 
# - average number of players in first n months of a game
# - format: nonnegative array of length n.
# 
# 
# Output: 
# - prediction of the average number of players of m months.
# - a list of close games arranged in decreasing weight.
# - plot of prediction and some close games.

# In[16]:


def ult_pred(game, train = nonsmooth_data, smooth_train = smooth_data, real_data = [], plot = True, number=3, threshold = 0.8, horizon = 6):
    ses = SimpleExpSmoothing(game)
    fit = ses.fit(smoothing_level=.6, optimized=False)
    smooth_game = fit.fittedvalues
    [pred, close_index] = wt_avg(game, smooth_game, train, smooth_train, threshold = threshold, horizon = horizon)
    while (len(close_index)<=2 and threshold >= 0.5):
        threshold = threshold - 0.1
        [pred, close_index] = wt_avg(game, smooth_game, train, smooth_train, threshold = threshold, horizon = horizon)
    if plot:
        fig = plt.figure(figsize=(12,12))
        plt.plot(range(1,len(game)+horizon + 1), pred, 'r--', label = 'Prediction')
        plt.plot(range(1,len(game)+1), game, 'b', label = 'Selected Game Data: Past')
        if len(real_data) !=0:
            plt.plot(range(len(game),len(game)+ horizon+1), real_data, 'b--', label = 'Selected Game Data: Future')
        if len(close_index)!=0:
            plot_range = range(min(number,len(close_index)))
            info = ['']*min(number,len(close_index))
            for i in plot_range:
                close_game = train[int(close_index[i][0])][:len(game)+horizon]
                scaled_close_game = close_game * mini_scaler(close_game[:len(game)],game)
                plt.plot(range(1,len(game)+horizon +1), scaled_close_game, label = str(i+1)+'-th closest scaled Game')
                info[i] = [close_index[i][0], data.iloc[int(close_index[i][0])]['Name'], close_index[i][1]]
            close_games = pd.DataFrame(info, columns = ['Game Number', 'Game', 'Weight'])
            
        plt.legend(fontsize=12)
        plt.xlabel('Age (Months)')
        plt.ylabel('Average Players scaled according to Test Game')
        plt.title(str(horizon)+'-month Prediction given '+str(len(game))+'-month data')
        st.pyplot(fig)
    return pred, close_index,close_games
