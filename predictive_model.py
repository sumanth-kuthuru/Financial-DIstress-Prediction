import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import webbrowser
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso,ElasticNet,SGDRegressor
import streamlit.components.v1 as components
from pivottablejs import pivot_ui 
from functions import *
from streamlit import caching
import time
import seaborn as sns
from time import sleep

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")
st.title('Financial Distress Predictive Model')
# LOADING DATA

Home = st.button('Home')
if Home:
    
    """
    In this project, we attempt to create a Regression model to predict when a company may go bankrupt or in financial distress.
    
    Being able to predict companies close to financial distress will help investors make decisions to protect themselves, or invest more and help these companies prevent bankruptcy in advance because the collective number of failing companies can be regarded as an important indicator of the financial health and robustness of a country’s economy.
    Dataset:
    
    Kaggle link:https://www.kaggle.com/shebrahimi/financial-distress (all numerical data, except x80)
    
    1st column: Company represents sample companies.
    
    2nd column: Time shows different time periods that data belongs to. Time series length varies between 1 to 14 for each company.
    
    Target:3rd column: The target variable is denoted by "Financial Distress": 1) If it will be greater than -0.50 the company should be considered healthy (0). 2) Otherwise, it would be regarded as financially distressed (1).
    
    Features: 4th column to the 86th column: The features denoted by untitled x1 to x83, are some financial and non-financial characteristics of the sampled companies. These features should be used to predict whether the company will be financially distressed or not (classification).
    
    Note: Feature x80 is categorical variable(so OneHotEncoder may need to be used).
    
    """

raw = df_data()
processed = processed_df(raw)
      
#original button
original = st.sidebar.button("Raw Data")  
if original:
    st.spinner()
    with st.spinner(text='Loading Data...'):
        st.subheader('Raw data')
        raw
        descriptive_stats(raw)
    with st.spinner(text='Loading Histogram...'):
        hist_plot(raw)
    with st.spinner(text='Loading Box Plots...'):
        box_plot(raw)
        
        
        
        
    
#scaled button 
scaled = st.sidebar.button("Processed Data Analysis")  
if scaled:
    st.spinner()
    with st.spinner(text='Loading Data...'):
        st.subheader('processed data')
        processed
        descriptive_stats(processed)
    with st.spinner(text='Loading Histogram...'):
        hist_plot(processed)
    with st.spinner(text='Loading Box Plots...'):
        box_plot(processed)
    with st.spinner(text='Loading Time Series Plot...'):
        for_time(raw)
    with st.spinner(text='Loading Correlation values Heatmap...'):
        get_corr(processed)
    with st.spinner(text='Loading High Correlation Values Heatmap...'):
        high_corr(processed)
    with st.spinner(text='Loading Pairplot...'):
        pair_plot(processed)
    with st.spinner(text='Loading Financial Distress Correlation plot ...'):
        fd_corr_plot(processed)
    
    


#Train button
train = st.sidebar.button('Train model')
if train:
    with st.spinner(text='Training Model...'):
        X_train, X_val, t_train, t_val, X_test, t_test = get_train_val_test_data(processed, val = True)
        model_nn = MLPRegressor(hidden_layer_sizes=(45,), activation= 'identity', solver = 'adam', alpha= 0.00001, learning_rate= 'adaptive', 
                         learning_rate_init=0.00005, max_iter= 100000, warm_start=True, early_stopping= True)
    
        model_rf = RandomForestRegressor(n_jobs=-1, n_estimators=50, min_samples_leaf=3, max_features=10)
    
        model_linear_elastic = ElasticNet(alpha=0.05,l1_ratio= 0.5, max_iter=1000, tol=0.00001, fit_intercept= True, warm_start=False)  
        estimators = [
                ('Elastic', model_linear_elastic),
                ('Random', model_rf),
                ('MLP', model_nn)
            ]
            
        stacked_model = StackingRegressor(
                estimators=estimators,
                final_estimator= ElasticNet(alpha=0.05,l1_ratio= 0.5, max_iter=1000, tol=0.00001, fit_intercept= True, warm_start=False)
            )  
        train_model = train_model( stacked_model, X_train, t_train )
        st.subheader('Validation Results')
        evaluation_report(train_model, X_train, t_train, X_val, t_val, val = True)
            




# predict button 
predict = st.sidebar.button('Predict')
if predict:
    X_train, X_test,t_train, t_test = get_train_val_test_data(processed, val = False)
    model_nn = MLPRegressor(hidden_layer_sizes=(45,), activation= 'identity', solver = 'adam', alpha= 0.00001, learning_rate= 'adaptive', 
                     learning_rate_init=0.00005, max_iter= 100000, warm_start=True, early_stopping= True)

    model_rf = RandomForestRegressor(n_jobs=-1, n_estimators=50, min_samples_leaf=3, max_features=10)

    model_linear_elastic = ElasticNet(alpha=0.05,l1_ratio= 0.5, max_iter=1000, tol=0.00001, fit_intercept= True, warm_start=False)  
    estimators = [
            ('Elastic', model_linear_elastic),
            ('Random', model_rf),
            ('MLP', model_nn)
        ]
        
    stacked_model = StackingRegressor(
            estimators=estimators,
            final_estimator= ElasticNet(alpha=0.05,l1_ratio= 0.5, max_iter=1000, tol=0.00001, fit_intercept= True, warm_start=False)
        )  
    train_model = train_model( stacked_model, X_train, t_train )
    st.subheader('Test Results')
    evaluation_report(train_model, X_train, t_train, X_test, t_test, val = False)
        

    

    


