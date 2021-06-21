# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 01:25:50 2021

@author: suman
"""
import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import webbrowser
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso,ElasticNet,SGDRegressor
import streamlit.components.v1 as components
from pivottablejs import pivot_ui 
import functions
from sklearn.metrics import plot_confusion_matrix
from streamlit import caching
import time
import seaborn as sns

from sklearn.feature_selection import RFECV
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression,Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
# from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import copy
from sklearn.metrics import accuracy_score , max_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import scipy.stats as stats
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import  RFECV, RFE
import resreg
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor

"Load data"
@st.cache
def df_data():
    Dataset = pd.read_csv('Financial Distress.csv')
    return Dataset

" original button graphs "
def plot_feature(df):
          '''Function for generating plots for a given Feature given df and column_name'''
          
          fig = plt.figure(figsize=(40 , 90))
          for k in range(1, df.shape[1]):
            ax = fig.add_subplot(20 , 5 , k)
            plt.hist(df.iloc[:,k-1], bins = 87)
            df.columns
          st.subheader('Hist plot for all features')
          st.pyplot(fig)
          
def hist_plot(df):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    df = pd.DataFrame(df)
    fig = df.hist( grid = False,layout=(23,4) , figsize=(25 , 70) , bins = 87)
    plt.show()
    st.header('Histogram for all features')
    st.pyplot()    


    
    
    
def box_plot(df):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    df = pd.DataFrame(df)
    fig = df.plot(kind='box', subplots=True, layout=(23,4) , figsize=(25 , 70))
    plt.show()
    st.header('Box plot for all features')
    st.pyplot()
    

def fd_corr_plot(df):
    corr_df = df.corr()
    corr_df = pd.DataFrame(corr_df)           
    corr_fd = corr_df.iloc[0, :]
  
    plt.figure(figsize=(40,18))
    fig = plt.bar(corr_fd.sort_values().iloc[:-1].index, corr_fd.sort_values().iloc[:-1] , width=0.4, align='center', color = 'g')
    plt.ylabel(ylabel='Finacial Distress', size=35)
    plt.xticks(size = 10)
    plt.yticks(size = 10)
    st.header('Financial Distress Correlation values')
    
    plt.show() 
    st.pyplot()
    
    
          
@st.cache
def processed_df(df):
    '''
    Function for getting cleaned and transformed data)
    '''
    #df = pd.read_csv('Financial Distress.csv')
    df1 = df.iloc[:, 3:]
    df1 = df1.drop('x80', axis =1)
    
    Q1=df1.quantile(q=0.25)                      
    Q3=df1.quantile(q=0.75)
    IQR= Q3-Q1
    
    fd = df[['Financial Distress']] #
    Q1fd=fd.quantile(q=0.25)                      
    Q3fd=fd.quantile(q=0.75)
    IQRfd= Q3fd-Q1fd

    df_fd = fd[(fd >=(Q1fd-3*IQRfd))& (fd <= Q3fd+3*IQRfd)]  


    df_inp = df1[(df1 >=(Q1-1.5*IQR))& (df1 <= Q3+1.5*IQR)]
    df_inp = pd.concat([df_fd, df_inp], axis =1)

    imp_mean= SimpleImputer(missing_values=np.nan,strategy='mean')
    imp_mean.fit(df_inp)
    df_imp = pd.DataFrame(imp_mean.transform(df_inp))
    col = df1.columns.insert(0, 'Financial Distress')
    df_imp.columns = col
    
    scaler=StandardScaler()
    df_scal = copy.deepcopy(df_imp)
    df_scal = df_scal.iloc[:, 1:]
    
    model = scaler.fit(df_scal)
    scaled_data= pd.DataFrame(model.transform(df_scal))   
    scaled_data.columns= df_scal.columns

    df_findis = df_imp[['Financial Distress']]
    df_final =  pd.concat([df_findis, scaled_data], axis =1)
    #   scaled_data = scaled_data.iloc[:, :]
    #scaled_data = pd.concat([scaled_fd, scaled_data], axis =1)
    #final = st.dataframe(df_final)
    
    return df_final 
 
   
"pairplot"
def pair_plot(df):
    
    df_scatter = pd.DataFrame(df, columns=['x10','x48','x81','Financial Distress'])
    df_scatter['Financial Distress'] = df_scatter['Financial Distress'].apply(lambda X : 1 if X < -0.5 else (0 if X > -0.5 else X))
    fig = plt.figure(figsize=(3,3))
    fig = sns.pairplot(df_scatter ,hue= 'Financial Distress' , size=5,aspect=1, markers=["o", "D"])
    st.header('Features with high correlation w.r.t Financial Distress')
    st.pyplot(fig)
    
# for time graphs
def for_time(df):
    df_time = pd.DataFrame(df, columns= ['Time', 'Financial Distress'])
    df_time['Financial Distress'] = df_time['Financial Distress'].apply(lambda X : 1 if X < -0.5 else 0 )
    df_new = df_time.loc[df_time['Financial Distress'] == 1]
    freq = df_new['Time'].value_counts()
    for_hist = pd.DataFrame(freq , columns = ['Time'])
    
    st.header('Plot for time series')
    
    st.bar_chart(for_hist)
    
    
def descriptive_stats(df):  
      '''Function for descriptive statisctics'''        
      data = df.describe()
      temp  = pd.concat([df.skew(), df.kurtosis()], axis = 1)
      temp.columns = ['skewness', 'kurtosis']
      temp = temp.T
      stats = pd.concat([data, temp])
      st.subheader('Descriptive statistics')
      st.dataframe(stats)
# correlation matrix
def get_corr(df):
  corr_df = df.corr()
  fig3 = plt.figure(figsize = (20,20))
  st.header('Correlation values heatmap')
  sns.heatmap(corr_df)
  
  st.pyplot(fig3)
  
def high_corr(df):
    corr_df = df.corr()
    
    corr = pd.DataFrame(corr_df[(corr_df>=0.5)|(corr_df<=-0.5)]) # Correlation dictionary for highly correlated values.
    corr_dict = {}
    for f in corr.index:
        corr_dict[f] = pd.DataFrame(corr.sort_values(f)[f].dropna())
    fig,ax = plt.subplots(figsize = (50,50))
    mask = corr.isnull()
    st.header('High correlation features')
    ax = sns.heatmap(corr, annot=True, linewidths=0.5, mask = mask, cmap = "crest") 
    st.pyplot(fig)
  
    
def get_train_val_test_data(df,val = False, crit = -0.5, rel_thresh = 0.8, ovr = 3.0, rare_percentile = 4):
  '''Function that splits data into Train, Test and Validation set with oversampling rare domain'''
  T = df.iloc[:, 0]
  X = df.iloc[:, 1:]
  y = copy.deepcopy(T)
  y[(y>=crit)] = 0; y[(y<crit)] = 1
  X_train, X_test, t_train, t_test = train_test_split(X, T, test_size=0.2, random_state=15, stratify = y)
  relevance = resreg.sigmoid_relevance(t_train, cl=np.percentile(T, rare_percentile), ch=None)
  X_train, t_train = resreg.random_oversample(X_train, t_train, relevance, relevance_threshold=rel_thresh, over= ovr, random_state=34)
  if val:         #to return validation set val = True when calling the function
    y1 = copy.deepcopy(t_train)
    y1[(y1>=crit)] = 0; y1[(y1<crit)] = 1
    X_train, X_val, t_train, t_val = train_test_split(X_train, t_train, test_size=0.15, random_state = 15, stratify = y1)
    return  X_train, X_val, t_train, t_val, X_test, t_test
  else:
    return X_train, X_test, t_train, t_test


def train_model( model, X_train, t_train ):
  '''Function that return the trained model'''
  model.fit(X_train, t_train)
  return model 

def evaluation_report(train_model, X_train, t_train, X_test, t_test, val = False, crit = -0.5):
        '''Function that results evaluation report'''
        pred_val = train_model.predict(X_test)
        true_val = t_test
        #y = copy.deepcopy(t_train)
        #y[(y>=crit)] = 0; y[(y<crit)] = 1
        ytr = copy.deepcopy(t_train)
        ytr[(ytr>=crit)] = 0; ytr[(ytr<crit)] = 1
        ytst = copy.deepcopy(t_test)
        ytst[(ytst>=crit)] = 0; ytst[(ytst<crit)] = 1
        
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(221)
        #plt.subplot(131)
        axi = sns.distplot(true_val, hist = False, color = 'g', label = "Actual Value", bins = 20)
        sns.distplot(pred_val, hist = False, color = 'b', label = "Predicted Value", bins = 20)
        ax.legend(labels=['Actual Values','Predicted Values'])
        #fig2 = plt.figure(figsize = (2,2))
        ax = fig.add_subplot(222)
        sns.scatterplot(true_val, pred_val)
        plt.plot([crit]*8, range(-4, 4, 1), 'r')
        plt.plot(range(-5, 20, 1), [crit]*25, 'r')
        ax.set_xlabel('Actual Value')
        ax.set_ylabel('Predicted Value')
        
        #fig3 = plt.figure(figsize = (3,3))
        ax = fig.add_subplot(223)
        true_class = copy.deepcopy(true_val); true_class[(true_class>=crit)] = 0; true_class[(true_class<crit)] = 1
        pred_class = copy.deepcopy(pred_val); pred_class[(pred_class>=crit)] = 0; pred_class[(pred_class<crit)] = 1
        
        cf_matrix = confusion_matrix(true_class, pred_class)
        sns.heatmap(cf_matrix,annot = True, square = True, fmt= "d" , cmap='Blues') 
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        st.pyplot(fig)
        
        st.write('True Values set range',true_val.min(),',',true_val.max())
        st.write('Predicted Values set range',pred_val.min(),',',pred_val.max())
        st.write('Accuracy', (accuracy_score(true_class, pred_class)*100) , '%')
        st.write('R2 score', r2_score(true_val, pred_val))
        st.write('Mean Squared error', mean_squared_error(true_val, pred_val))
        st.write('Max Error',max_error(true_val, pred_val))
        st.write('Total Train Samples',X_train.shape[0])
        st.write('Total ratio of 0s and 1s in Train', np.count_nonzero(ytr == 0)/np.count_nonzero(ytr == 1))
        if val:
            st.write('Total validation Samples', X_test.shape[0])
            st.write('Total ratio of 0s and 1s in Validation',np.count_nonzero(ytst == 0)/np.count_nonzero(ytst == 1))
        else:
            st.write('Total Test Samples',X_test.shape[0])
            st.write('Total ratio of 0s and 1s in Test',np.count_nonzero(ytst == 0)/np.count_nonzero(ytst == 1))
        