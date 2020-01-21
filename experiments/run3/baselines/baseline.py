

import pandas as pd
import numpy as np
from utils import *


def createBaselines():
    
    # Generate and save data
    df_control , df_trajectory = stateData(15000, saveData=True)
    
    # Control baseline
    control_dataset = df_control.values
    x_train , y_train = control_dataset[0:10000,0:2], control_dataset[0:10000, 2:]
    print("CONTROL TRAIN ",x_train.shape,",", y_train.shape)
    x_test , y_test = control_dataset[10000:,0:2], control_dataset[10000:, 2:]
    print("CONTROL TEST ", x_test.shape,",", y_test.shape)
    controlNet = kerasBaselineNet(x_train,
                     y_train,
                     x_test,
                     y_test,
                     saveModel=True,
                     name="ControlNet"      
    )
    del x_train, x_test, y_train, y_test, controlNet

    trajectory_dataset = df_trajectory.values
    x_train , y_train = trajectory_dataset[0:10000,0:2], trajectory_dataset[0:10000, 2:]
    print("STATE TRAIN ",x_train.shape,",", y_train.shape)

    x_test , y_test = trajectory_dataset[10000:,0:2], trajectory_dataset[10000:, 2:]
    print("STATE TEST ", x_test.shape,",", y_test.shape)

    stateNet = kerasBaselineNet(x_train,
                     y_train,
                     x_test,
                     y_test,
                     saveModel=True,
                     name= "StateNet"      
    )
    print("Done")
    del x_train, x_test, y_train, y_test, stateNet
    
if __name__=="__main__":
    createBaselines()
    
