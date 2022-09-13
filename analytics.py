from tkinter.filedialog import asksaveasfile
import pyvista as pv
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from numpy.linalg import norm
import sys
import os
from matplotlib import pyplot as plt
#from sklearn.metrics.mean_squared_error


def main(argv):

    faces = []
    prova = "Data"
    with open(prova + "/FaceCNN/CUSTOM_processed/dataset_test.txt") as f:
        for line in f:
            faces.append(prova + '/FaceCNN/CUSTOM/' + line.strip())
    f.close()
    
    dict = []
    faces_list = {}


    
    for face in faces: 
        lmks_true = []
        lmks_pred = []
        with open(face+'_landmarks.txt') as file:
            for lmk_pred in file:
                x,y,z = lmk_pred.strip().split(' ')
                lmks_pred.append([float(x),float(y),float(z)])
        file.close()
        with open(face+'.txt') as file:
            for lmk_true in file:
                x,y,z = lmk_true.strip().split(' ')
                lmks_true.append([float(x),float(y),float(z)])
        file.close()
        faces_list[str(face).split('/')[-1]] = {'lmks_pred':lmks_pred, 'lmks_true':lmks_true}
    #print(faces_list.keys())
    
    # RMSE medio totale
    RMSE_dict = {}
    for key in faces_list:
        face = faces_list[key]
        
        #print(np.asarray(face['lmks_pred']) - np.asarray(face['lmks_true']))
        lmks_pred_face = np.asarray(face['lmks_pred'])
        lmks_true_face = np.asarray(face['lmks_true'])
        sum = 0 
        for i in range(0,50):
            error = 0
            for coor in range(0, 3):
                error += (lmks_pred_face[i][coor] - lmks_true_face[i][coor])**2
            distance = np.sqrt(error)
            sum+=distance**2
        MSE = sum/50
        RMSE = np.sqrt(MSE)
        RMSE_dict[key] = RMSE
    
    print(RMSE_dict)
    mean = np.mean([RMSE_dict[i] for i in RMSE_dict.keys()])
    print(f"Total RMSE: {mean}")
    
    RMSE_dict_landmarks = {}
    for i in range(0,50):
        sum_distances = 0
        for key in faces_list:
            face = faces_list[key]
            lmks_pred_face = np.asarray(face['lmks_pred'][i])
            lmks_true_face = np.asarray(face['lmks_true'][i])
            error = 0
            for coor in range(0, 3):
                error += (lmks_pred_face[coor] - lmks_true_face[coor])**2
            distance = np.sqrt(error)
            sum_distances+=distance**2
        MSE = sum_distances/len(faces_list)
        RMSE = np.sqrt(MSE)
        RMSE_dict_landmarks[i+1] = RMSE

    print(f'Landmark più critico: {max(RMSE_dict_landmarks, key=RMSE_dict_landmarks.get)}')
    for key in RMSE_dict_landmarks.keys():
        print(str(key) + ' - ' + str(RMSE_dict_landmarks[key]))
    plt.grid(True, axis='y')
    plt.bar([str(i) for i in RMSE_dict_landmarks.keys()], RMSE_dict_landmarks.values())
    plt.xlim((-1,50))
    plt.ylim((0, 65))
    plt.xlabel("ID landmark")
    plt.ylabel("RMSE")
    plt.show()
    
if "__main__":
    main(sys.argv)