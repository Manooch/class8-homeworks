#!/usr/bin/env python

import os
import os.path as op
import pandas as pd
import numpy as np
import math
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from argparse import ArgumentParser
from sklearn.datasets import load_breast_cancer
from sklearn import datasets, model_selection, metrics, neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

breast = load_breast_cancer()
#print(breast.data[:,:1]) #Display column 1
#print(breast.target) #Diagnosis Value
#print(breast.target_names) #meaning of Diagnosis Value
#print(breast.feature_names) #meaning of the features
#print(breast.data[:,:11])
#print(breast.feature_names[:11])
#print(breast)

df = pd.DataFrame(breast.data, columns = breast.feature_names)
X= breast.data
y = breast.target
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state = 0)


def ArgParser():
    parser = ArgumentParser(description = 'A CSV reader + stats maker')
    parser.add_argument('csvfile', type = str, help = 'path to the input csv file.')

    parsed_args = parser.parse_args()
    my_csv_file = parsed_args.csvfile

    assert op.isfile(my_csv_file), "Please give us a real file, thx"
    print('woot, the file exists')

    #*********************************************************
    # Load data, Organize dataset and Add header to the dataframe
    #*********************************************************

    data = pd.read_csv(my_csv_file, sep='\s+|,', header=None, engine='python', 
                       names = ['ID number', 'Diagnosis','Radius_M', 'Texture_M', 'Perimeter_M', 'Area_M','Smoothness_M', 'Compactness_M', 'Concavity_M', 'ConcavePoints_M', 'Symmetry_M', 'FractalDimension_M',
                                'Radius_SE', 'Texture_SE', 'Perimeter_SE', 'Area_SE','Smoothness_SE', 'Compactness_SE', 'Concavity_SE', 'ConcavePoints_SE', 'Symmetry_SE', 'FractalDimension_SE',
                                'Radius_W', 'Texture_W', 'Perimeter_W', 'Area_W','Smoothness_W', 'Compactness_W', 'Concavity_W', 'ConcavePoints_W', 'Symmetry_W', 'FractalDimension_W'])
    data.drop(['ID number'], axis=1, inplace=True)
    return data

#******** Ploting Figures ********

def plot2DHistogram(data, folder):
      i = 1
      figIndex = 1
      #columncount = len(data.columns)
      columncount = 12  # To avoid having so many scatter     
      while i < columncount - 1:
            j = i + 1
            while j < columncount:
                  iv = data.iloc[:, i]
                  jv = data.iloc[:, j]
                  plt.figure(data.columns[i])
                  plt.hist2d(iv, jv, bins = 30, cmap = 'Blues')
                  cb = plt.colorbar()
                  cb.set_label('Counts in bin')
                  plt.title('BSWisconsin DataSet')
                  plt.xlabel(data.columns[i])
                  plt.ylabel(data.columns[j])
                  plt.savefig('./{0}/Hist2d_{1}.png'.format(folder, data.columns[i] + ' ' + data.columns[j]))
                  #plt.show()
                  plt.close("all")
                  j = j + 1
                  figIndex = figIndex + 1
            i = i + 1      

def plotGroupedHistogram(data, columns, gr_feature, folder):
	l = len(columns)
	n_cols = math.ceil(math.sqrt(l))		
	n_rows = math.ceil(l / n_cols)
	
	fig = plt.figure(figsize = (11, 6), dpi = 100)
	for i, col_name in enumerate(columns):		
		if col_name != gr_feature:				
			ax = fig.add_subplot(n_rows, n_cols, i)
			ax.set_title(col_name)
			grouped = data.pivot(columns = gr_feature, values = col_name)
			for j, gr_feature_name in enumerate(grouped.columns):							
				grouped[gr_feature_name].hist(alpha = 0.5, label = gr_feature_name)
			plt.legend(loc = 'upper right')
	fig.tight_layout()
	plt.savefig('./{0}/HistGroupBy{1}.png'.format(folder,gr_feature))
	#plt.show()            

def plot2DHistogramsklearn(data, folder):
      i = 1
      figIndex = 1
      #columncount = len(data.columns)
      columncount = 12  # To avoid having so many scatter     
      while i < columncount - 1:
            j = i + 1
            while j < columncount:
                  iv = data.data[:, i]
                  jv = data.data[:, j]
                  plt.figure(data.feature_names[i])
                  plt.hist2d(iv, jv, bins = 30, cmap = 'Blues')
                  cb = plt.colorbar()
                  cb.set_label('Counts in bin')
                  plt.title('BSWisconsin DataSet')
                  plt.xlabel(data.feature_names[i])
                  plt.ylabel(data.feature_names[j])
                  plt.savefig('./{0}/Hist2d_{1}.png'.format(folder, data.columns[i] + ' ' + data.columns[j]))
                  #plt.show()
                  plt.close("all")
                  j = j + 1
                  figIndex = figIndex + 1
            i = i + 1      

def plotGroupedHistogramsklearn(data, columns, gr_feature, folder):
	
	#data1 = pd.DataFrame(data = np.c_[data.data, data.target],columns = data.feature_names + 'target')
	#data1 = pd.DataFrame(data= np.c_[data['data'], data['target']], columns= data['feature_names'] + ['target'])
	#data1 = pd.DataFrame(data.data, columns=data.feature_names)

	data1 = pd.DataFrame(np.c_[data.data[:,:3], data.target], columns = np.append(data.feature_names[:3], ["target"]))	

	l = len(data1.columns)
	n_cols = math.ceil(math.sqrt(l))		
	n_rows = math.ceil(l / n_cols)

	#data1[data1.columns].hist(by = data1['target'])

	fig = plt.figure(figsize = (10, 6), dpi = 100)
	for i, col_name in enumerate(data1.columns):	
		ax = fig.add_subplot(n_rows, n_cols, i + 1)
		ax.set_title(col_name)
		grouped = pd.pivot_table(data1, index = ["target"], values = col_name)
		#grouped.hist(alpha = 0.5, label = data1['target'])
		for j, gr_feature_name in enumerate(grouped.columns):		
			grouped[gr_feature_name].hist(alpha = 0.5, label = data1['target'])
		plt.legend(loc = 'upper right')
	fig.tight_layout()
	#plt.savefig('./{0}/HistGroupBy{1}.png'.format(folder,gr_feature))
	plt.show()   

def plotScattersklearn(data, folder):
    df = pd.DataFrame(np.c_[data.data[:,:3], data.target], columns = np.append(data.feature_names[:3], ["target"]))	
    for i, column1 in enumerate(df.columns):
        for j, column2 in enumerate (df.columns[i+1:]):
            data1 = df[column1]
            data2 = df[column2]
            plt.figure()
            sns.scatterplot(data1, data2)
            plt.title("Scatter Plot Between {} and {}".format(column1,column2))
            #plt.savefig('./{0}/Scatter{1}.png'.format(folder, column1 + '_' + column2))
    plt.show()

#******** End of Ploting Figures ********

#******** Prediction ********
def KNNPredict_BreastCancer():
    clf = KNeighborsClassifier(n_neighbors = 1).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print("\nK Nearest Neighbors Classification:")
    print("\n1. Confusion_matrix")
    print(metrics.confusion_matrix(y_test, y_pred))
    print("\n2. Classification_report")
    print(metrics.classification_report(y_test, y_pred))
    print("\n3. F1_score")
    print(metrics.f1_score(y_test, y_pred, average = "macro"))
    print("\n4. Accuracy")
    print(clf.score(X_test, y_test))

def GaussianPredict_BreastCancer():
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\nGaussian Naive Bayes:")
    print("\n1. Confusion_matrix")
    print(metrics.confusion_matrix(y_test, y_pred))
    print("\n2. Classification_report")
    print(metrics.classification_report(y_test, y_pred))
    print("\n3. F1_score")
    print(metrics.f1_score(y_test, y_pred, average = "macro"))
    print("\n4. Accuracy")
    print(clf.score(X_test, y_test))

def DecisionTree_BreastCancer():
    clf = DecisionTreeClassifier(max_depth = 10, random_state = 101, max_features = None, min_samples_leaf = 15)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print("\nDecision Tree:")
    print("\n1. Confusion_matrix")
    print(metrics.confusion_matrix(y_test, y_pred))
    print("\n2. Classification_report")
    print(metrics.classification_report(y_test, y_pred))
    print("\n3. F1_score")
    print(metrics.f1_score(y_test, y_pred, average = "macro"))
    print("\n4. Accuracy")
    print(clf.score(X_test, y_test))

#******** End of Prediction ********

def BreastCancer_Prediction():
    KNNPredict_BreastCancer()
    GaussianPredict_BreastCancer()
    DecisionTree_BreastCancer()

def BreastCancer_PlotFigures():
    plot2DHistogram(data,'Figures')
    plotGroupedHistogram(data.iloc[:,:11], data.iloc[:,:11].columns, 'Diagnosis', 'Figures') 

    plot2DHistogramsklearn(breast, 'Figures')
    #plotGroupedHistogramsklearn(breast, breast.feature_names, 'Diagnosis', 'Figures') # Not working correctly
    plotScattersklearn(breast, 'Figures')

data = ArgParser()
BreastCancer_PlotFigures()
BreastCancer_Prediction()