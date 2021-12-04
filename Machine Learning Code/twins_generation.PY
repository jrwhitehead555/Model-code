#this code is located in a google colab https://colab.research.google.com/drive/1x04MhqAYDTNFSZwLpo4N4jUzRo43tOef
#only need to run function generate_all_possible_twins()
!pip install similaritymeasures
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

def generate_all_possible_twins(): #combines all following funtions into a single one for simplictiy
  #Data_2019 = get_data(2019)
  Data_2020 = get_data(2020)
  Data_2021 = get_data(2021)

  CC_data_20 = [] #put the data in a form to remove outliers
  for i in range(0,len(Data_2020[0])):
    CC_data_20.append((Data_2020[0][i],Data_2020[11][i]))
  CC_data_21 = []
  for i in range(0,len(Data_2021[0])):
    CC_data_21.append((Data_2021[0][i],Data_2021[12][i]))

  CH_data_20 = []
  for i in range(0,len(Data_2020[1])):
    CH_data_20.append((Data_2020[1][i],Data_2020[11][i]))
  CH_data_21 = []
  for i in range(0,len(Data_2021[1])):
    CH_data_21.append((Data_2021[1][i],Data_2021[12][i]))

  CV_data_20 = []
  for i in range(0,len(Data_2020[5])):
    CV_data_20.append((Data_2020[5][i],Data_2020[11][i]))
  CV_data_21 = []
  for i in range(0,len(Data_2021[6])):
    CV_data_21.append((Data_2021[6][i],Data_2021[12][i]))

  EXG_data_20 = []
  for i in range(0,len(Data_2020[6])):
    EXG_data_20.append((Data_2020[6][i],Data_2020[11][i]))
  EXG_data_21 = []
  for i in range(0,len(Data_2021[7])):
    EXG_data_21.append((Data_2021[7][i],Data_2021[12][i]))

  CC_data_20_tup = outlier_remove(CC_data_20)  #remove outliers
  CC_data_21_tup = outlier_remove(CC_data_21)
  CH_data_20_tup = outlier_remove(CC_data_20)
  CH_data_21_tup = outlier_remove(CC_data_21)
  CV_data_20_tup = outlier_remove(CC_data_20)
  CV_data_21_tup = outlier_remove(CC_data_21)
  EXG_data_20_tup = outlier_remove(CC_data_20)
  EXG_data_21_tup = outlier_remove(CC_data_21)
  
  data_OLR_2020 = [CC_data_20_tup,CH_data_20_tup,CV_data_20_tup,EXG_data_20_tup]
  data_OLR_2021 = [CC_data_21_tup,CH_data_21_tup,CV_data_21_tup,EXG_data_21_tup]
  data_OLR_2020_str = ["CC_raw_data_20","CH_raw_data_20","CV_raw_data_20","EXG_raw_data_20"]
  data_OLR_2021_str = ["CC_raw_data_21","CH_raw_data_21","CV_raw_data_21","EXG_raw_data_21"]

  for i in range(0,len(data_OLR_2020)):
    destination = "/content/gdrive/MyDrive/colab/twins/OLR_twins/" + data_OLR_2020_str[i]
    pd.DataFrame(data_OLR_2020[i]).to_pickle(destination)
  print("files saved")

  for i in range(0,len(data_OLR_2021_str)):
    destination = "/content/gdrive/MyDrive/colab/twins/OLR_twins/" + data_OLR_2021_str[i]
    pd.DataFrame(data_OLR_2021[i]).to_pickle(destination)
  print("files saved")

  print("raw data saved")
  CC_data = interpolate_data(CC_data_20_tup,Data_2020[8]) + interpolate_data(CC_data_21_tup,Data_2021[9])
  CH_data = interpolate_data(CH_data_20_tup,Data_2020[8]) + interpolate_data(CH_data_21_tup,Data_2021[9])
  CV_data = interpolate_data(CV_data_20_tup,Data_2020[8]) + interpolate_data(CV_data_21_tup,Data_2021[9])
  EXG_data = interpolate_data(EXG_data_20_tup,Data_2020[8]) + interpolate_data(EXG_data_21_tup,Data_2021[9])
  
  CC_data = difference(CC_data)
  CH_data = difference(CH_data)
  CV_data = difference(CV_data)
  EXG_data = difference(EXG_data)

  CC_diss = sorted(CC_data, key=lambda tup: (tup[2]) )
  CH_diss = sorted(CH_data, key=lambda tup: (tup[2]) )
  CV_diss = sorted(CV_data, key=lambda tup: (tup[2]) )
  EXG_diss = sorted(EXG_data, key=lambda tup: (tup[2]) )

  saves = [CC_diss,EXG_diss,CV_diss,CH_diss]
  saves_string = ["CC_twins.csv","EXG_twins.csv","CV_twins.csv","CH_twins.csv"]

  for i in range(0,len(saves)):
    destination = "/content/gdrive/MyDrive/colab/twins/OLR_twins/" + saves_string[i]
    pd.DataFrame(saves[i]).to_csv(destination)
  print("files saved")



def difference(data):     #finds the difference between each graph in an array and the first graph, This is used to sort by differences for selecting twins
  ans = [(data[0][0],data[0][1],0)]
  normal_dates = np.linspace(0, 169, 170)
  graph_1 = np.zeros((170, 2))
  graph_1[:, 0] = normal_dates
  graph_1[:, 1] = data[0][0]
  for i in range(1,len(data)):
    graph_2 = np.zeros((170, 2))
    graph_2[:, 0]  = normal_dates
    graph_2[:, 1] = data[i][0]
    diff, d = similaritymeasures.dtw(graph_1, graph_2)
    ans.append((data[i][0],data[i][1],diff))
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

def outlier_remove(data_tuple_orig):  #checks each datapoint in an array and removes any outliers
  data_tuple = data_tuple_orig.copy()
  for j in range(0,len(data_tuple_orig[0][0])):
    temp = []
    for i in data_tuple:
      temp.append(i[0][j])
    info = outlier_remove_assist(temp)
    output = []
    low_value = info[1] - 3*info[0]
    high_value = info[1] + 3*info[0]
    for i in data_tuple:
      if (i[0][j] > low_value and i[0][j] < high_value):
        output.append(i)
    data_tuple = output
  return data_tuple

def outlier_remove_assist(data_list):
  output1 = np.std(data_list)
  output2 = np.mean(data_list)
  return [output1,output2]

def interpolate_data(data,dates): #interpolates the data from values 0 to 170. 
  output = []
  fin_date = dates[len(dates)-1]
  xi  = np.linspace(0, 169, 170)
  x = np.insert(dates,0,0).copy()
  x = np.append(x,169)
  for tup in data:
    y = tup[0]
    y = np.insert(y,0,0)
    y = np.append(y,y[len(y)-1])
    rbfi = Rbf(x, y,kernal = 'gaussian',smooth = 0.2)  # radial basis function interpolator instance
    di = rbfi(xi)   # interpolated values
    output.append((di,tup[1]))
  return output