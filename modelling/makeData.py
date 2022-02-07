import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time, datetime
import os.path

import modelling.tree as tr
import modelling.model_response as mr
import modelling.synthetic_data as sd


from multiprocessing import Pool, cpu_count
from multiprocessing.pool import TimeoutError

import warnings
warnings.filterwarnings('ignore')

pd.set_option("display.max_columns",200)

# Set up logger
logfile = "logfile.txt"

def writelog(theString=None):
    with open(logfile,'a') as f:
        f.write(theString + '\n')
        f.close()
    print(theString)
    
    
def makeTwoClassData(total_num_events=4000,fracClassA=0.5,probFeat0zero=0.5, t = (),intervention_times=(),randRange=2,savePath=".",showPlots=True):
    numClassA = int( total_num_events *fracClassA)
    numClassB = total_num_events - numClassA

    spikeFunc = mr.choose_model('general', params=(0.995, 9, 0.0,0.0,0))  #0.995/(1 +exp9t)                                              
    noResponse = mr.choose_model('skewed',params=(1,2,3,0,0,0))

    classA_Pattern = sd.generate_curve(t, intervention_times, (spikeFunc,spikeFunc,noResponse,noResponse), noise = 0.000)
    classB_Pattern = sd.generate_curve(t,intervention_times, (noResponse,spikeFunc,spikeFunc,noResponse),noise=0.000)
    
    theName= f"{total_num_events}_events_pclassA_{fracClassA}_fracZerofirstFeat_{probFeat0zero}"
    theFile= f"{savePath}/{theName}.csv"
    
    # if file exists, we can read it, otherwise have to create from scratch
    if os.path.exists(theFile):
        writelog(f"reading data from file: {theFile}")
        theData = pd.read_csv(theFile)

    else: #have to create data        
        groupA = sd.make_dataframe(classA_Pattern,
                           (0,numClassA),
                           [
                            [(1-probFeat0zero), [1, 0, 0]],
                            [ probFeat0zero   , [0, 1, 0]]
                           ], 
                           proportional=False, 
                           random_feature=True ,
                           num_rand=randRange
                          )

        groupB = sd.make_dataframe(classB_Pattern, 
                           (numClassA, numClassA +numClassB),
                           [
                            [ (1.0-probFeat0zero), [1, 1, 0]],
                            [  probFeat0zero,      [0, 0, 0]]
                           ], 
                           proportional=False, 
                           random_feature=True, 
                           num_rand=randRange
                            )

        theData = sd.append_groups((groupA,groupB))
        #drop episodes with no response
        #theData.drop(theData[theData.dayOfEp < 0].index, inplace=True)
        # only accept data on certain dsays
        theData  = theData[(theData.dayOfEp==0)|(theData.dayOfEp==5)|(theData.dayOfEp==10)]
        
        #shuffle the rows
        theData = theData.sample(frac=1).reset_index(drop=True)
        #save to file
        theData.to_csv(theFile)


        
        
    #can get summmary by passing theData to a tree function
    summary = tr.get_response(theData, t)
    
    
    if(showPlots):
        fig,ax = plt.subplots(nrows=1,ncols=3,sharey=True,sharex=True,figsize=(9,4))
        ax[0].bar(classA_Pattern.dayOfEp, classA_Pattern.response)
        ax[0].set_title("class A response pattern")
        ax[1].bar(classB_Pattern.dayOfEp, classB_Pattern.response)
        ax[1].set_title("classB response pattern")
        ax[2].bar(summary.t, summary.response)
        ax[2].set_title("Combined Response")
        fig.suptitle(theName)
        plt.show()
    
    maxScore = 0.5
    
    return theData, maxScore




def makeNativeBinomialData(total_num_events=4000,fracClassA=0.5,probFeat0zero=0.5, t = (),intervention_times=(),randRange=2,savePath=".",showPlots=True):
    numClassA = int( total_num_events *fracClassA)
    numClassB = total_num_events - numClassA


   
    noResponse = mr.choose_model('skewed',params=(1,2,3,0,0,0))
    n = len(t)
    classAparam = 0.1
    classBparam = 0.6
    binomialA = mr.choose_model('binom', params=(classAparam,n))
    binomialB = mr.choose_model('binom', params=(classBparam,n))
    print (binomialA.predict(t[:]))
    classA_Pattern = sd.generate_curve(t, intervention_times, (binomialA,noResponse,noResponse,noResponse), noise = 0.000)
    classB_Pattern = sd.generate_curve(t,intervention_times, (binomialB,noResponse,noResponse,noResponse),noise=0.000)
    
  
    
    theName= f"{total_num_events}_events_pclassA_{fracClassA}_fracZerofirstFeat_{probFeat0zero}"
    theFile= f"{savePath}/{theName}.csv"
    
    # if file exists, we can read it, otherwise have to create from scratch
    if os.path.exists(theFile):
        writelog(f"reading data from file: {theFile}")
        theData = pd.read_csv(theFile)

    else: #have to create data        
        groupA = sd.make_dataframe(classA_Pattern,
                           (0,numClassA),
                           [
                            [(1-probFeat0zero), [1, 0, 0]],
                            [ probFeat0zero   , [0, 1, 0]]
                           ], 
                           proportional=False, 
                           random_feature=True ,
                           num_rand=randRange
                          )

        groupB = sd.make_dataframe(classB_Pattern, 
                           (numClassA, numClassA +numClassB),
                           [
                            [ (1.0-probFeat0zero), [1, 1, 0]],
                            [  probFeat0zero,      [0, 0, 0]]
                           ], 
                           proportional=False, 
                           random_feature=True, 
                           num_rand=randRange
                            )

        theData = sd.append_groups((groupA,groupB))
        #drop episodes with no response
        #theData.drop(theData[theData.dayOfEp < 0].index, inplace=True)
        # only accept data on certain dsays
        theData  = theData[(theData.dayOfEp==0)|(theData.dayOfEp==5)|(theData.dayOfEp==10)]
        
        #shuffle the rows
        theData = theData.sample(frac=1).reset_index(drop=True)
        #save to file
        theData.to_csv(theFile)


        
        
    #can get summmary by passing theData to a tree function
    summary = tr.get_response(theData, t)
    
    
    if(showPlots):
        fig,ax = plt.subplots(nrows=1,ncols=3,sharey=True,sharex=True,figsize=(9,4))
        ax[0].bar(classA_Pattern.dayOfEp, classA_Pattern.response)
        ax[0].set_title("class A response pattern")
        ax[1].bar(classB_Pattern.dayOfEp, classB_Pattern.response)
        ax[1].set_title("classB response pattern")
        ax[2].bar(summary.t, summary.response)
        ax[2].set_title("Combined Response")
        fig.suptitle(theName)
        plt.show()
    
    return theData


def makeBinomialData(total_num_events=4000,fracClassA=0.5,probFeat0zero=0.5, t = (),intervention_times=(),randRange=2,savePath=".", n=10, parameters=[0.1, 0.9], showPlots=True):

    numClassA = int( total_num_events *fracClassA)
    numClassB = total_num_events - numClassA

    
    theName= f"{total_num_events}_events_pclassA_{fracClassA}_fracZerofirstFeat_{probFeat0zero}"
    theFile= f"{savePath}/{theName}.csv"
    
    # if file exists, we can read it, otherwise have to create from scratch
    if os.path.exists(theFile):
        writelog(f'reading data from file: {theFile}')
        theData = pd.read_csv(theFile)

    else: #have to create data
        groupA = sd.make_component_df([n, parameters[0]],
                                      (0,numClassA),
                                      [
                                          [(1-probFeat0zero), [1, 0, 0]],
                                          [ probFeat0zero   , [0, 1, 0]]
                                      ], 
                                      random_feature=True,
                                      num_rand=randRange,
                                      component_id=0,
                                      noise=0.0
                                     )
        groupB = sd.make_component_df([n, parameters[1]],
                                      (numClassA, numClassA +numClassB),
                                      [
                                          [ (1.0-probFeat0zero), [1, 1, 0]],
                                          [  probFeat0zero,      [0, 0, 0]]
                                      ], 
                                      random_feature=True, 
                                      num_rand=randRange,
                                      component_id=1,
                                      noise=0.0
                                     )
        
        

        theData = sd.create_dataframe((groupA,groupB))
        #drop episodes with no response
        #theData.drop(theData[theData.dayOfEp < 0].index, inplace=True)

        #shuffle the rows
        theData = theData.sample(frac=1).reset_index(drop=True)
        #save to file
        theData.to_csv(theFile)

    # automatically find the max possible score for the dataset
    feature_val_combos = [[1, 0, 0], [0, 1, 0],[1, 1, 0], [0, 0, 0]]
    feature_rand = True
    cw = [[1,0],[1,0],[0,1],[0,1]]
    params = [[n,parameters[0]],[n,parameters[1]]]
    maxPossibleScore = sd.target_score(synthetic_data=theData, feature_val_combos=feature_val_combos, 
                                                       params=params, cw=cw, feature_rand=True, n=n)
        
  
    
    if(showPlots):
          #can get summmary by passing theData to a tree function
        summary = tr.get_response(theData, t)
        classA_Pattern = tr.get_response(theData.loc[theData['component_id']==0], t)
        classB_Pattern = tr.get_response(theData.loc[theData['component_id']==1], t)
        fig,ax = plt.subplots(nrows=1,ncols=3,sharey=True,sharex=True,figsize=(9,4))
        ax[0].bar(classA_Pattern.t, classA_Pattern.response)
        ax[0].set_title("class A response pattern")
        ax[1].bar(classB_Pattern.t, classB_Pattern.response)
        ax[1].set_title("classB response pattern")
        ax[2].bar(summary.t, summary.response)
        ax[2].set_title("Combined Response")
        fig.suptitle(theName)
        plt.show()
    
    return theData, maxPossibleScore

