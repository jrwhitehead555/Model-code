#this code is located in a google colab https://colab.research.google.com/drive/1zO3lRr8YV7iw8QCfO_jyA9ZRgfMtwz84#scrollTo=wpR8ndRMmk8n
#when running for first time, run function build_training(num_twins) to create the trainging data set and save it
!pip install similaritymeasures
!pip install pyyaml h5py  # Required to save models in HDF5 format
!pip install --upgrade tensorflow
!pip install --upgrade numpy
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow import keras 
from tensorflow.keras import models
from tensorflow.keras import backend
from tensorflow.keras import metrics as metrics_module
import os
import pandas as pd
import numpy as np
import math
import similaritymeasures
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from datetime import datetime
#from mpl_toolkits.mplot3d import Axes3D
from google.colab import files
from google.colab import drive
drive.mount('/content/gdrive')


# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


data = build_training_file()
train_data_norm = data[0]
yields = data[1]
indexes = data[2]
yield_data_formatted = yield_data_format(yields,15) #the second value lets you round to the nearest multiple of that value
out = maximum(yield_data_formatted) + 1 #ensures the final layer is the correct size
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(680),
  tf.keras.layers.Normalization(mean=43., variance=1544.),
  tf.keras.layers.Dense(180, activation='relu'),
  tf.keras.layers.Dense(180, activation='relu'),
  tf.keras.layers.Dense(120, activation='relu'),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Dense(out)
])



loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


x_train = train_data_norm[0:18642] #80% test data
x_test = train_data_norm[18642:] #20% test data


y_train = yield_data_formatted[0:18642]#80% test data
y_test = yield_data_formatted[18642:] #20% test data

batch_size = 1000

model.fit(x_train, y_train,epochs=100, 
          batch_size=batch_size, 
          verbose=0)

model.evaluate(x_test,  y_test,verbose = 2)
filepath = "/content/gdrive/MyDrive/colab/MLmodel_weights/ML_CC_EXG"  #saves the ML model for exporting and ease of training
save_model(model, filepath)


def Stand_Deviation(data):
  n = len(data)
  mean = sum(data) / n
  deviations = [(x - mean) ** 2 for x in data]
  variance = sum(deviations) / n
  dev = np.sqrt(variance)
  return dev, mean

def maximum(data):
  temp = data[0]
  for i in data:
    if i > temp:
      temp = i
  return temp

def yield_data_format(yields,distance): #formats the yield data for the ML model 
  sd, mean = Stand_Deviation(yields)
  yield_data_formatted = []
  for i in yields:
    yield_data_formatted.append(int(i-8))
  output = []
  for i in yield_data_formatted:
    if i >= 90:
      ans = 9
    elif i >= 80:
      ans = 8
    elif i >= 70:
      ans = 7
    elif i >= 60:
      ans = 6
    elif i >= 50:
      ans = 5
    elif i >= 40:
      ans = 4
    elif i >=30:
      ans = 3
    elif i >= 20:
      ans = 2
    elif i >= 10:
      ans = 1
    elif i >= 0:
      ans = 0
    output.append(ans)
  return output

def predict_data(data,yield_base): # predicts the yield and harvest date with helper functions
  filepath = "/content/gdrive/MyDrive/colab/MLmodel_weights/ML_CC_EXG_special"
  model = load_model(filepath, compile = True)
  predictions = model.predict(data)
  predictions = np.argmax(predictions, axis = 1)
  dates = []
  for i in data:
    dates.append(predict_harvest_date(i))
  output = []
  for i in predictions:
    temp = "between " + str(i*10) + " and "+ str(i*10 + 10)
    output.append((temp,dates[i]))
  return output

def predict_harvest_date(data): #prediction helper function
  data = x_train[0]
  EXG_data = data[170:339]
  max_val = 0
  index = 0
  count = 0
  for i in EXG_data:
    if i >= max_val:
      max_val = i
      index = count
    count += 1
  index = index + 4
  start_day = '0301' 
  start_day = datetime.strptime(start_day, "%m%d")
  harvest_day = start_day + pd.DateOffset(days=index)
  output = str(harvest_day.month) + "/" + str(harvest_day.day)
  return output



def fin_get_twins(num_twins): #pulls num_twins twins for each data point, then matches them to the raw data
  data_2020 = get_data(2020)
  data_2021 = get_data(2021)

  twins = get_twins(num_twins)

  CC_data_20 = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/CC_raw_data_20")
  CC_data_21 = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/CC_raw_data_21")
  CH_data_20 = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/CH_raw_data_20")
  CH_data_21 = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/CH_raw_data_21")
  CV_data_20 = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/CV_raw_data_20")
  CV_data_21 = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/CV_raw_data_21")
  EXG_data_20 = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/EXG_raw_data_20")
  EXG_data_21 = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/EXG_raw_data_21")

  CC_twins = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/CC_twins")
  CH_twins = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/CH_twins")
  CV_twins = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/CV_twins")
  EXG_twins = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/EXG_twins")

  CC_sel_twins = get_twins_helper(CC_twins,num_twins)
  CH_sel_twins = get_twins_helper(CH_twins,num_twins)
  CV_sel_twins = get_twins_helper(CV_twins,num_twins)
  EXG_sel_twins = get_twins_helper(EXG_twins,num_twins)

  CC_matched = compare_twins(CC_data_20,data_2020[8],CC_sel_twins) + compare_twins(CC_data_21,data_2021[9],CC_sel_twins)
  destination = "/content/gdrive/MyDrive/colab/twins/Matched_twins/CC_matched"
  pd.DataFrame(CC_matched).to_pickle(destination)
  print("files saved")

  CH_matched = compare_twins(CH_data_20,data_2020[8],CH_sel_twins) + compare_twins(CH_data_21,data_2021[9],CH_sel_twins)
  destination = "/content/gdrive/MyDrive/colab/twins/Matched_twins/CH_matched"
  pd.DataFrame(CH_matched).to_pickle(destination)
  print("files saved")

  CV_matched = compare_twins(CV_data_20,data_2020[8],CV_sel_twins) + compare_twins(CV_data_21,data_2021[9],CV_sel_twins)
  destination = "/content/gdrive/MyDrive/colab/twins/Matched_twins/CV_matched"
  pd.DataFrame(CV_matched).to_pickle(destination)
  print("files saved")

  EXG_matched = compare_twins(EXG_data_20,data_2020[10],EXG_sel_twins) + compare_twins(EXG_data_21,data_2021[11],EXG_sel_twins)
  destination = "/content/gdrive/MyDrive/colab/twins/Matched_twins/EXG_matched"
  pd.DataFrame(EXG_matched).to_pickle(destination)
  print("files saved")

  return 0

def get_twins(num_twins): #pulls all twins from drive

  CC_twins = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/CC_twins")
  CH_twins = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/CH_twins")
  CV_twins = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/CV_twins")
  EXG_twins = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/EXG_twins")

  CC_sel_twins = get_twins_helper(CC_twins,num_twins)
  CH_sel_twins = get_twins_helper(CH_twins,num_twins)
  CV_sel_twins = get_twins_helper(CV_twins,num_twins)
  EXG_sel_twins = get_twins_helper(EXG_twins,num_twins)

  return [CC_sel_twins,CH_sel_twins,CV_sel_twins,EXG_sel_twins]

def get_twins_helper(data,num_twins):
  ind = np.linspace(0,len(data)-1,num_twins)
  ind = ind.astype(int)
  output = []
  for i in ind:
    output.append((data[0][i],data[1][i]))
  return output

def compare_twins(data,dates,twins):  #compares a graph to all twins the select the best match
  output = []
  for i in range(0,len(data[0])):
    output.append(compare_graph_to_twins(data[0][i],dates,twins))
  output2 = []
  for i in range(0,len(data[0])):
    output2.append(data[1][i])
  fin_output = []
  for i in range(0,len(data[0])):
    fin_output.append((output[i],output2[i]))
  return fin_output



def fin_get_twins_file(): #pulls the matched twins from the drive to save on time
  CC_twins = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/Matched_twins/CC_matched")
  CH_twins = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/Matched_twins/CH_matched")
  CV_twins = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/Matched_twins/CV_matched")
  EXG_twins = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/Matched_twins/EXG_matched")
  return [CC_twins,CH_twins,CV_twins,EXG_twins]

def get_yields(): # pulls the yield data from the drive
  data_20 = pd.read_csv("/content/gdrive/MyDrive/colab/Driscoll Data/Yield/2020_cc_driscoll_cotton_yield_10m_grid.csv")
  data_21 = pd.read_csv("/content/gdrive/MyDrive/colab/Driscoll Data/Yield/2021_cc_driscoll_cotton_yield_10m_grid.csv")
  data_20 = data_20.astype(int)
  data_21 = data_21.astype(int)
  data_20 = data_20.values.tolist()
  data_21 = data_21.values.tolist()

  return[data_20,data_21]

def build_training(num_twins): #builds the training sets by matching the raw data to the twin, then filling in the known data points, then saves the data
  yields = get_yields()
  yields = yields[0] + yields[1]
  data = fin_get_twins_file()
  yields = yield_OLR(data,yields)
  twins = get_twins(num_twins)
  CC_twins = twins[0]
  CH_twins = twins[1]
  CV_twins = twins[2]
  EXG_twins = twins[3]
  CC_data_20 = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/CC_raw_data_20")
  CC_data_21 = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/CC_raw_data_21")
  CH_data_20 = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/CH_raw_data_20")
  CH_data_21 = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/CH_raw_data_21")
  CV_data_20 = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/CV_raw_data_20")
  CV_data_21 = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/CV_raw_data_21")
  EXG_data_20 = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/EXG_raw_data_20")
  EXG_data_21 = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/EXG_raw_data_21")

  CC_data_graphs_20 = only_graph(CC_data_20)
  CC_data_graphs_21 = only_graph(CC_data_21)

  CH_data_graphs_20 = only_graph(CH_data_20)
  CH_data_graphs_21 = only_graph(CH_data_21)

  CV_data_graphs_20 = only_graph(CV_data_20)
  CV_data_graphs_21 = only_graph(CV_data_21)

  EXG_data_graphs_20 = only_graph(EXG_data_20)
  EXG_data_graphs_21 = only_graph(EXG_data_21)

  known_days = [5,10,15]
  CC_data_train = twins_for_known_days(CC_data_graphs_20,CC_twins,7,2020)
  CC_data_train = np.concatenate((CC_data_train,twins_for_known_days(CC_data_graphs_21,CC_twins,7,2021)),axis = 0)
  CH_data_train = twins_for_known_days(CH_data_graphs_20,CH_twins,7,2020)
  CH_data_train = np.concatenate((CH_data_train,twins_for_known_days(CH_data_graphs_21,CH_twins,7,2021)),axis = 0)
  CV_data_train = twins_for_known_days(CV_data_graphs_20,CV_twins,7,2020)
  CV_data_train = np.concatenate((CV_data_train,twins_for_known_days(CV_data_graphs_21,CV_twins,7,2021)),axis = 0)
  EXG_data_train = twins_for_known_days(EXG_data_graphs_20,EXG_twins,7,2020)
  EXG_data_train = np.concatenate((EXG_data_train,twins_for_known_days(EXG_data_graphs_21,EXG_twins,7,2021)),axis = 0)

  for i in known_days:
    CC_data_train = np.concatenate((CC_data_train,twins_for_known_days(CC_data_graphs_20,CC_twins,i,2020)),axis = 0)
    CC_data_train = np.concatenate((CC_data_train,twins_for_known_days(CC_data_graphs_21,CC_twins,i,2021)),axis = 0)

    CH_data_train = np.concatenate((CH_data_train,twins_for_known_days(CH_data_graphs_20,CH_twins,i,2020)),axis = 0)
    CH_data_train = np.concatenate((CH_data_train,twins_for_known_days(CH_data_graphs_21,CH_twins,i,2021)),axis = 0)

    CV_data_train = np.concatenate((CV_data_train,twins_for_known_days(CV_data_graphs_20,CV_twins,i,2020)),axis = 0)
    CV_data_train = np.concatenate((CV_data_train,twins_for_known_days(CV_data_graphs_21,CV_twins,i,2021)),axis = 0)

    EXG_data_train = np.concatenate((EXG_data_train,twins_for_known_days(EXG_data_graphs_20,EXG_twins,i,2020)),axis = 0)
    EXG_data_train = np.concatenate((EXG_data_train,twins_for_known_days(EXG_data_graphs_21,EXG_twins,i,2021)),axis = 0)

    print("finished with " + str(i) + " known days")

  destination = "/content/gdrive/MyDrive/colab/Training_sets/fin_train/CC_training_data"
  pd.DataFrame(CC_data_train).to_pickle(destination)

  destination = "/content/gdrive/MyDrive/colab/Training_sets/fin_train/CH_training_data"
  pd.DataFrame(CV_data_train).to_pickle(destination)

  destination = "/content/gdrive/MyDrive/colab/Training_sets/fin_train/CV_training_data"
  pd.DataFrame(CH_data_train).to_pickle(destination)

  destination = "/content/gdrive/MyDrive/colab/Training_sets/fin_train/EXG_training_data"
  pd.DataFrame(EXG_data_train).to_pickle(destination)
  
  return 0


def only_graph(data): #pulls the only the graphs from a tuple
  output = []
  for stuff in data[0]:
    output.append(stuff)
  return output

def only_indexes(data):#pulls the only the indexes from a tuple
  output = []
  for stuff in data[1]:
    output.append(stuff)
  return output

def yield_OLR(data,yields): #removes outliers from yield data
  indexes = []
  for i in data[0][1]:
    indexes.append(i)
  fin_yields = []
  for i in yields:
    if i[0] in indexes:
      fin_yields.append(i[1])
  return fin_yields

def twins_for_known_days(data,twins,known_days,year): #helper function, matches twin then fills in known days
  days_2020 = [0,8,19,31,37,46,53,60,67,74,80,88,95,102,109,115,123,128]
  days_2021 = [0,25,37,49,59,65,73,80,87,93,101,108,115,121,130,135,140,148,156,158,165]
  normal_dates = np.linspace(0,169,170)
  if year == 2020:
    data_known = [0]
    dates_known = days_2020[0:known_days+1]
    output = []
    for i in data:
      data_known = np.concatenate(([0],i[0:known_days]), axis=0)
      twin = compare_graph_to_twins(data_known,dates_known,twins)
      train_data = connect_data(data_known,dates_known,twin)
      output.append(train_data)

    return output
  elif year == 2021:
    data_known = [0]
    dates_known = days_2021[0:known_days+1]
    output = []
    for i in data:
      data_known =np.concatenate(([0],i[0:known_days]), axis=0)
      twin = compare_graph_to_twins(data_known,dates_known,twins)
      train_data = connect_data(data_known,dates_known,twin)
      output.append(train_data)
    return output

  else:
    return 0
  
def connect_data(data,dates,twin): #helper function, connects the known days data with the twin to ensure that the size is consistent
  fin_date = int(dates[len(dates)-1])
  y_all = np.linspace(data[0],data[1],(dates[1] - dates[0])+1)
  dates_all = np.linspace(0,int(dates[len(dates)-1]),int(dates[len(dates)-1]+1))
  for i in range(1,len(data)-1):
    dif = dates[i + 1] - dates[i] + 1
    y = np.linspace(data[i],data[i+1],int((dif)))
    y = y[1:]
    y_all = np.concatenate((y_all,y), axis = 0)

  temp = twin[fin_date + 1:]
  output = np.concatenate((y_all,temp),axis = 0)
  return output

def compare_graph_to_twins(graph,dates,twins): #compares graphs to all twins
  graph_1 = np.zeros((len(graph), 2))
  graph_1[:, 0] = dates
  graph_1[:, 1] = graph
  graph_2 = np.zeros((len(twins[0][0]), 2))
  twins_dates = np.linspace(0, 169,170 )
  graph_2[:, 0]  = twins_dates
  graph_2[:, 1] = twins[0][0]
  best_diff,y = similaritymeasures.dtw(graph_1,graph_2)
  index = 0
  for i in range(1,len(twins)):
    graph_2[:, 0]  = twins_dates
    graph_2[:, 1] = twins[i][0]
    diff,y = similaritymeasures.dtw(graph_1,graph_2)
    if (diff < best_diff):
      best_diff = diff
      index = i
  return twins[index][0]

def build_training_file(): #gets training data from file saved to drive
  CC_train = pd.read_pickle("/content/gdrive/MyDrive/colab/Training_sets/fin_train/CC_training_data")
  CH_train = pd.read_pickle("/content/gdrive/MyDrive/colab/Training_sets/fin_train/CH_training_data")
  CV_train = pd.read_pickle("/content/gdrive/MyDrive/colab/Training_sets/fin_train/CV_training_data")
  EXG_train = pd.read_pickle("/content/gdrive/MyDrive/colab/Training_sets/fin_train/EXG_training_data")
  CC_train = CC_train.values.tolist()
  CH_train = CH_train.values.tolist()
  CV_train = CV_train.values.tolist()
  EXG_train = EXG_train.values.tolist()
  CC_data_20 = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/CC_raw_data_20")
  CC_data_21 = pd.read_pickle("/content/gdrive/MyDrive/colab/twins/OLR_twins/CC_raw_data_21")
  train_data = []
  for i in range(0,len(CC_train)):
    temp = CC_train[i] + CH_train[i] + CV_train[i] + EXG_train[i]
    #temp = CC_train[i] + EXG_train[i]
    train_data.append(temp)
  yields = get_yields()

  yields = yields[0] + yields[1]

  ind_20 = only_indexes(CC_data_20)
  ind_21 = only_indexes(CC_data_21)
  ind = ind_20 + ind_21
  yields = yield_OLR_TR(ind,yields)
  ind = ind + ind +ind + ind

  yields = np.concatenate((yields,yields,yields,yields),axis=0)
  
  
  
  return [train_data,yields,ind]


def yield_OLR_TR(data,yields): #removes outliers from yield data for build_training_file function
  indexes = []
  for i in data:
    indexes.append(i)
  fin_yields = []
  for i in yields:
    if i[0] in indexes:
      fin_yields.append(i[1])
  return fin_yields


def normalize_date(date,year): #normalizes the dates based on march first start date (normally mid-march is time of first scan)
  start_day = '0301'
  start_day = datetime.strptime(start_day, "%m%d")
  temparr = []
  ans = np.array([])
  for string in date:
    temp = datetime.strptime(string, "%m%d")
    temparr.append(temp)
  for i in range(0,len(temparr)):
    ans = np.append(ans, (temparr[i] - start_day).days)

  return ans



def get_data(year): #pulls original driscoll data from drive to analyze
  DData = [('2019',['2019_cc_commercial_cotton_cc_grid_csv.csv','2019_cc_commercial_cotton_exg_grid_csv.csv']),('2020',['2020_cc_driscoll_10m_grid_canopy_cover_cotton_csv.csv',
 '2020_cc_driscoll_10m_grid_ch_cv_cotton_csv.csv',
 '2020_cc_driscoll_10m_grid_exg_cotton_csv.csv']),('2021',['2021_cc_driscoll_10m_grid_canopy_cover_cotton_csv.csv',
 '2021_cc_driscoll_10m_grid_ch_cv_cotton_csv.csv',
 '2021_cc_driscoll_10m_grid_exg_cotton_csv.csv'])]

  if year == 2019: #checks year of driscoll data to pull, same for elif statements later
    DData_read_int = "/content/gdrive/MyDrive/colab/Driscoll Data/Driscoll Data csv/" + '2019' +"/"+ DData[0][1][0]
    DData_read = pd.read_csv(DData_read_int)

    DData_id = []
    DData_read_id = DData_read["id"]
    for i in range(0,len(DData_read)):
      data_temp = [DData_read_id[i]]
      DData_id.append(data_temp)
    temp = []
    for i in range(0,len(DData_id)):
      temp.append(DData_id[i][0])
    DData_id = temp
    DData_date_19 = []

    for col in DData_read.columns:
        DData_date_19.append(col.replace('CC19', ''))
    del DData_date_19[0:5]

    DData_read = [DData_read["id"],DData_read["CC190412"],DData_read["CC190509"],DData_read["CC190604"],DData_read["CC190614"],DData_read["CC190628"],DData_read["CC190711"],DData_read["CC190718"]]
    DData_fin = []
    for i in range(0,len(DData_read[0])):
      data_temp = [DData_read[1][i],DData_read[2][i],DData_read[3][i],DData_read[4][i],DData_read[5][i],DData_read[6][i],DData_read[7][i]]
      DData_fin.append(data_temp)
    
    DData_read_int = "/content/gdrive/MyDrive/colab/Driscoll Data/Driscoll Data csv/" + '2019' +"/"+ DData[0][1][1]
    DData_read = pd.read_csv(DData_read_int)
    DData_date = []

    Dcol = DData_read.columns
    DData_read = [DData_read[Dcol[5]],DData_read[Dcol[6]],DData_read[Dcol[7]],DData_read[Dcol[8]],DData_read[Dcol[9]],DData_read[Dcol[10]],DData_read[Dcol[11]]]
    DData_fin_av = []
    for i in range(0,len(DData_read[0])):
      data_temp = [DData_read[0][i],DData_read[1][i],DData_read[2][i],DData_read[3][i],DData_read[4][i],DData_read[5][i],DData_read[6][i]]
      DData_fin_av.append(data_temp)

    DData_read = pd.read_csv(DData_read_int)
    DData_read = [DData_read[Dcol[12]],DData_read[Dcol[13]],DData_read[Dcol[14]],DData_read[Dcol[15]],DData_read[Dcol[16]],DData_read[Dcol[17]],DData_read[Dcol[18]]]
    DData_fin_sd = []
    for i in range(0,len(DData_read[0])):
      data_temp = [DData_read[0][i],DData_read[1][i],DData_read[2][i],DData_read[3][i],DData_read[4][i],DData_read[5][i],DData_read[6][i]]
      DData_fin_sd.append(data_temp)

    DData_date_fin = normalize_date(DData_date_19,2019)
    #DData_date_fin = normalize_date(DData_date_fin,2019)
    print(["DData_fin_CC [0] ", "DData_fin_av_EXG [1] ", "DData_fin_sd_EXG [2] ", "DData_date_fin [3] ","DData_id [4] "])
    return [DData_fin, DData_fin_av, DData_fin_sd, DData_date_fin,DData_id]

  elif year == 2020:
    DData_read_int = "/content/gdrive/MyDrive/colab/Driscoll Data/Driscoll Data csv/" + '2020' +"/"+ DData[1][1][0]
    DData_read = pd.read_csv(DData_read_int)

    DData_id = []
    DData_read_id = DData_read["FID"]
    for i in range(0,len(DData_read)):
      data_temp = [DData_read_id[i]]
      DData_id.append(data_temp)
    temp = []
    for i in range(0,len(DData_id)):
      temp.append(DData_id[i][0])
    DData_id = temp

    DData_date = []

    for col in DData_read.columns:
        DData_date.append(col.replace('CC20', ''))
    del DData_date[0]
    DData_fin_CC = []
    for i in range(0,len(DData_read)):
      DData_temp = []
      for j in range(1,len(DData_read.iloc[1])):
        DData_temp.append(DData_read.iloc[i][j])
      DData_fin_CC.append(DData_temp)

    DData_date_fin_CC = normalize_date(DData_date,2020)

    DData_read_int = "/content/gdrive/MyDrive/colab/Driscoll Data/Driscoll Data csv/" + '2020' +"/"+ DData[1][1][1]
    DData_read = pd.read_csv(DData_read_int)

    DData_date = []


    for col in DData_read.columns:
      DData_date.append(col.replace('avCH20', ''))
    del DData_date[0]
    del DData_date[16:]

    DData_date_CHandCV = normalize_date(DData_date,2020)

    DData_fin_avCH = []
    temp = 1
    spacing = 16
    for i in range(0,len(DData_read)):
      DData_temp = []
      for j in range(temp,temp + spacing):
        DData_temp.append(DData_read.iloc[i][j])
      DData_fin_avCH.append(DData_temp)
    temp = temp + spacing

    DData_fin_mxCH = []
    for i in range(0,len(DData_read)):
      DData_temp = []
      for j in range(temp,temp + spacing):
        DData_temp.append(DData_read.iloc[i][j])
      DData_fin_mxCH.append(DData_temp)
    temp = temp + spacing

    DData_fin_95CH = []
    for i in range(0,len(DData_read)):
      DData_temp = []
      for j in range(temp,temp + spacing):
        DData_temp.append(DData_read.iloc[i][j])
      DData_fin_95CH.append(DData_temp)
    temp = temp + spacing

    DData_fin_sdCH = []
    for i in range(0,len(DData_read)):
      DData_temp = []
      for j in range(temp,temp + spacing):
        DData_temp.append(DData_read.iloc[i][j])
      DData_fin_sdCH.append(DData_temp)
    temp = temp + spacing

    DData_fin_CV = []
    for i in range(0,len(DData_read)):
      DData_temp = []
      for j in range(temp,temp + spacing):
        DData_temp.append(DData_read.iloc[i][j])
      DData_fin_CV.append(DData_temp)

    DData_read_int = "/content/gdrive/MyDrive/colab/Driscoll Data/Driscoll Data csv/" + '2020' +"/"+ DData[1][1][2]
    DData_read = pd.read_csv(DData_read_int)

    DData_date = []

    for col in DData_read.columns:
      DData_date.append(col.replace('avEG20', ''))
    del DData_date[0]
    del DData_date[17:]

    DData_date_EG = normalize_date(DData_date,2020)

    DData_fin_avEG = []

    for i in range(0,len(DData_read)):
      DData_temp = []
      for j in range(1,18):
        DData_temp.append(DData_read.iloc[i][j])
      DData_fin_avEG.append(DData_temp)

    DData_fin_sdEG = []
    for i in range(0,len(DData_read)):
      DData_temp = []
      for j in range(18,34):
        DData_temp.append(DData_read.iloc[i][j])
      DData_fin_sdEG.append(DData_temp)
    print(["DData_fin_CC [0] ","DData_fin_avCH [1] ","DData_fin_mxCH [2] ","DData_fin_95CH [3] ","DData_fin_sdCH [4]","DData_fin_CV [5] ","DData_fin_avEG [6] ","DData_fin_sdEG [7] ","DData_date_fin_CC [8] ","DData_date_CHandCV [9] ","DData_date_EG [10], DData_id [11]"])
    return [DData_fin_CC,DData_fin_avCH,DData_fin_mxCH,DData_fin_95CH,DData_fin_sdCH,DData_fin_CV,DData_fin_avEG,DData_fin_sdEG,DData_date_fin_CC,DData_date_CHandCV,DData_date_EG, DData_id]

  elif year == 2021:
    DData_read_int = "/content/gdrive/MyDrive/colab/Driscoll Data/Driscoll Data csv/" + '2021' +"/"+ DData[2][1][0]
    DData_read = pd.read_csv(DData_read_int)

    DData_id = []
    DData_read_id = DData_read["FID"]
    for i in range(0,len(DData_read)):
      data_temp = [DData_read_id[i]]
      DData_id.append(data_temp)
    temp = []
    for i in range(0,len(DData_id)):
      temp.append(DData_id[i][0])
    DData_id = temp
    
    DData_date = []


    for col in DData_read.columns:
      DData_date.append(col.replace('CC21', ''))
    del DData_date[0]
    DData_date_CC = normalize_date(DData_date,2021)

    DData_fin_CC = []
    for i in range(0,len(DData_read)):
      DData_temp = []
      for j in range(1,len(DData_read.iloc[1])):
        DData_temp.append(DData_read.iloc[i][j])
      DData_fin_CC.append(DData_temp)

    DData_read_int = "/content/gdrive/MyDrive/colab/Driscoll Data/Driscoll Data csv/" + '2021' +"/"+ DData[2][1][1]
    DData_read = pd.read_csv(DData_read_int)

    DData_date = []


    for col in DData_read.columns:
      DData_date.append(col.replace('avCH21', ''))
    del DData_date[0]
    del DData_date[19:]

    DData_date_CHandCV = normalize_date(DData_date,2020)

    DData_fin_avCH = []
    temp = 1
    spacing = 19
    for i in range(0,len(DData_read)):
      DData_temp = []
      for j in range(temp,temp + spacing):
        DData_temp.append(DData_read.iloc[i][j])
      DData_fin_avCH.append(DData_temp)
    temp = temp + spacing

    DData_fin_mxCH = []
    for i in range(0,len(DData_read)):
      DData_temp = []
      for j in range(temp,temp + spacing):
        DData_temp.append(DData_read.iloc[i][j])
      DData_fin_mxCH.append(DData_temp)
    temp = temp + spacing

    DData_fin_99CH = []
    for i in range(0,len(DData_read)):
      DData_temp = []
      for j in range(temp,temp + spacing):
        DData_temp.append(DData_read.iloc[i][j])
      DData_fin_99CH.append(DData_temp)
    temp = temp + spacing

    DData_fin_95CH = []
    for i in range(0,len(DData_read)):
      DData_temp = []
      for j in range(temp,temp + spacing):
        DData_temp.append(DData_read.iloc[i][j])
      DData_fin_95CH.append(DData_temp)
    temp = temp + spacing

    DData_fin_sdCH = []
    for i in range(0,len(DData_read)):
      DData_temp = []
      for j in range(temp,temp + spacing):
        DData_temp.append(DData_read.iloc[i][j])
      DData_fin_sdCH.append(DData_temp)
    temp = temp + spacing

    DData_fin_CV = []
    for i in range(0,len(DData_read)):
      DData_temp = []
      for j in range(temp,temp + spacing):
        DData_temp.append(DData_read.iloc[i][j])
      DData_fin_CV.append(DData_temp)
    temp = temp + spacing


    DData_read_int = "/content/gdrive/MyDrive/colab/Driscoll Data/Driscoll Data csv/" + '2021' +"/"+ DData[2][1][2]
    DData_read = pd.read_csv(DData_read_int)

    DData_date = []

    for col in DData_read.columns:
      DData_date.append(col.replace('avEG21', ''))
    del DData_date[0]
    del DData_date[20:]

    DData_date_EG = normalize_date(DData_date,2021)

    DData_fin_avEG = []

    for i in range(0,len(DData_read)):
      DData_temp = []
      for j in range(1,21):
        DData_temp.append(DData_read.iloc[i][j])
      DData_fin_avEG.append(DData_temp)

    DData_fin_sdEG = []
    for i in range(0,len(DData_read)):
      DData_temp = []
      for j in range(21,41):
        DData_temp.append(DData_read.iloc[i][j])
      DData_fin_sdEG.append(DData_temp)
    print(["DData_fin_CC [0]","DData_fin_avCH [1] ","DData_fin_mxCH [2] ","DData_fin_99CH [3] ","DData_fin_95CH [4] ","DData_fin_sdCH [5] ","DData_fin_CV [6] ","DData_fin_avEG [7] ","DData_fin_sdEG [8] ","DData_date_CC [9] ","DData_date_CHandCV [10] ","DData_date_EG [11]","DData_id [12]"])
    return [DData_fin_CC,DData_fin_avCH,DData_fin_mxCH,DData_fin_99CH,DData_fin_95CH,DData_fin_sdCH,DData_fin_CV,DData_fin_avEG,DData_fin_sdEG,DData_date_CC,DData_date_CHandCV,DData_date_EG,DData_id]
  



