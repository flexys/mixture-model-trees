import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time, datetime
import os.path

import modelling.tree as tr
import modelling.synthetic_data as sd
import modelling.genetic_algorithm as gp
import modelling.rFlexMixTree as rflexmix

import modelling.makeData as md

from multiprocessing import Pool, cpu_count
from multiprocessing.pool import TimeoutError

import warnings
warnings.filterwarnings('ignore')

pd.set_option("display.max_columns",200)

logfile = "logfile.txt"


def writelog(theString=None):
    with open(logfile,'a') as f:
        f.write(theString + '\n')
        f.close()
    print(theString)



#### Variables describing the data

#t should be a numpy array of the days where the response level is recorded. This can be every two or more days if the response is binned in that way.

#intervention_times should be a list of the days on which an intervention happens, including the first one.

#total_num_events is self explanatory

#features is the set of features used to describe each event(need to tell the tree building algorithms)


##### Variables controlling the distribution of the data

#problem is the base name e.g. "twoSpikeTwoClass"

#useNoise denotes whether a noise variable is added to the features

#fracVals is a list of possible fractions of ther total data set that are of class A (as defined in the functin above)

#pzeroVals is a list of possible fractions of the data that have a zero for feature 0.
#makeTwoClassData() generates episodes using classA =XOR(feature0,feature1)

#so value of pzeroVale= zero means that feature 1 becomes the sole discriminant between the classes
#a value of 0.5 creates an XOR
#maxPossibleScore, if set, can be passed to the GA as a termination criteria to save runtime



class experiment:
    def  __init__(self):
        self.fit_models=False
        self.verbose=False
        pass
  

    def setUpAlgorithms(self,fit_models=False,verbose=False):
        #choice of algorithms and parameters
        self.algorithms = ('flexmix','recursive','LocalSearch', 'EA')
        self.fit_models=fit_models
        self.verbose = verbose
        self.max_depth = 6
        self.min_leaf_node_size = (self.total_num_events*self.sampleFrac*0.005)
        self.sensitivity = 0.005
        self.algorithms= ('flexmix','recursive','LocalSearch', 'EA')

        self.eaParams= { 
            'features': self.features,
            'days': self.days,
            'intervention_times': self.intervention_times,
            'generations':50,   
            'max_possible_fitness':  self.maxPossibleScore, 
            'min_improvement':0.004, 
            'gens_no_improvement':30,
            'population_size':100, 
            'tournament_size':5,  
            'mutation_rate':0.25, 
            'crossover_rate':0.8,
            'max_depth':self.max_depth, 
            'min_size':self.min_leaf_node_size, 
            'prune_tol':self.sensitivity, 
            'epsilon':1,
            'file_name':'data/tree_results', 
            'test_set':None, 
            'fit_models':self.fit_models,
            'verbose':self.verbose,
            'internal_parallelism': False
          }

        self.lsParams =  { 
            'features': self.features,
            'days': self.days,
            'intervention_times': self.intervention_times,
            'generations':5000,#to be comparable witgh 50 generations for popsize 100   
            'max_possible_fitness': self.maxPossibleScore, 
            'min_improvement':0.004, 
            'gens_no_improvement':3000,  # to be comparable 
            'population_size':1, 
            'tournament_size':1,  
            'mutation_rate':0.25, 
            'crossover_rate':0.0,
            'max_depth':self.max_depth, 
            'min_size':self.min_leaf_node_size, 
            'prune_tol':self.sensitivity, 
            'epsilon':1,
            'file_name':'data/tree_results', 
            'test_set':None, 
            'fit_models':self.fit_models,
            'verbose':self.verbose,
            'internal_parallelism': False
          }

        self.recursiveParams = {
            'features': self.features,
            'days': self.days,
            'intervention_times': self.intervention_times,
            #'sensitivity': self.sensitivity,
            #'max_depth': self.max_depth,
            #'min_leaf_node_size': self.min_leaf_node_size,
            'split_criteria': 'fit_improvement',
            #'fit_models' : self.fit_models,
            #'verbose':self.verbose
                  }

        self.flexmixParams = {
            'feature_names': self.features,
            'n': len(self.days),
            'maxComponents': 2*len(self.intervention_times),
            'modelFamily': "binomial",
            'verbosity':0 # 0 no output to screen, 1 basic, 2 lots 
                }
    
        self.cmdNames = {'EA':gp.gp, 'LocalSearch': gp.gp,
                         'recursive':self.MakeRecursiveTree,
                         'flexmix':rflexmix.getFittedFlexMixTree }
        self.paramSets = {'EA': self.eaParams, 'LocalSearch':self.lsParams,
                      'recursive':self.recursiveParams,'flexmix':self.flexmixParams}
    

    def MakeRecursiveTree(self,runData,features=None,days=None,intervention_times=None,split_criteria=None):
        theTree = tr.Tree(runData,intervention_times,days,sensitivity=self.sensitivity,verbose=self.verbose)
        theTree.build_tree(features, self.max_depth, self.min_leaf_node_size, split_criteria, self.fit_models)
        theTree.setNodeCount()
        return theTree


    ## Function to call from cell/script to set up data set attributes of experiment
    def setUpExp(self,problem="",testing=False):
        #number of events and the rqnge (time span) of their dayOfEpisode feature
        self.days= np.arange(0,20)
        self.intervention_times = [0,5,10,15]
        self.total_num_events = 10000
        self.sampleFrac = 0.4 #proportion of data used for each repetition
        self.sampleReplacement = False
        self.maxPossibleScore = 0.0 # gets overwritten when data is created
        self.timeout=1000
        self.testing=testing
        self.numNoise=0
        
 
        #look for noise features
        # strip off number and set variable to record it
        #use basename to generte original problem then extend with appropriate values:
        #.  self.features, and theData
        self.longProblem=problem
        try:
            self.numNoise = int(problem[-2:])
            self.problem = problem[:-2]
        except    ValueError:
            self.problem = problem
            self.numNoise=0
 
        
    
        #other features specified for each event
        self.features = ['feature_0',
                'feature_1',
                'feature_2',
                'feature_rand'
               ]
        for extraFeature in range(self.numNoise):
            featName= "feature_rand" + str(extraFeature +1)
            self.features.extend ( [featName] )
        
        """
        fiveRand= ['feature_rand2','feature_rand3','feature_rand4','feature_rand5']
        tenRand= ['feature_rand6','feature_rand7','feature_rand8','feature_rand9','feature_rand10']


        if (self.fiveNoise==True):
            self.features.extend(fiveRand) 
        if(self.tenNoise==True):
            self.features.extend(fiveRand)
            self.features.extend(tenRand)
         """
        
        if(self.problem=="CaseStudy"):
            self.features = ['feature_0','feature_1','feature_2','feature_3','feature_4','feature_5',
                            'feature_6','feature_7','feature_8','feature_9',
                            'feature_10','feature_11','feature_12','feature_13']
            self.days=np.arange(0,32)
    

        # file paths and names 
        if(self.numNoise==0):
            self.thepath= f"data/{self.problem}/Noiseless"
        else :
            self.thepath= f"data/{self.problem}/BinaryNoise"

        if not os.path.exists(self.thepath): 
            os.makedirs(self.thepath)

        #Distribution and hence number of experiments
        if (testing):
            self.fracVals = (0.5,0.1)
            self.pzeroVals =   (0.0, 0.5)
            self.repetitions = 1
        elif(self.problem != self.longProblem): 
            #don't need all combinations for scaleability experiments
            self.fracVals = (0.5,0.1)
            self.pzeroVals =   (0.0, 0.5)
            self.repetitions = 25
        else:
            #proportions of class A in the dataset
            self.fracVals= (0.1, 0.2, 0.3, 0.4, 0.5 )    
            #probability that feature 0 has the value 
            self.pzeroVals = (0, 0.1, 0.2, 0.3, 0.4, 0.425,0.45,0.475, 0.5 )
            #number of rums for each combination
            self.repetitions = 25



    ## Function to call from cell/script to run experiment
    def runExp( self,runParallel=False,methods=[],testing=False):
        #dataframe to store results
        results = pd.DataFrame(columns=("Problem","Noise","Algorithm","fitmodels",
                                        "fracClassA","pFeatZeroZero","Run","Runtime","Gens","Accuracy"))
        trees = pd.DataFrame (columns=("Problem","Noise","Algorithm","fitmodels",
                                        "fracClassA","pFeatZeroZero","Run","Accuracy","Tree"))
    
        resfilename = 'results/results-' +str(datetime.date.today()) +'.csv'
        treefilename = 'results/trees-' +str(datetime.date.today()) +'.csv'

        #should data contains noise of fixed varibles?
        if(self.numNoise>0):
            randRange= 2
        else:
            randRange = 1
    
        for fracClassA in self.fracVals:
            for probFeat0zero in self.pzeroVals:
                if(self.problem =="twoSpikeTwoClass"):
                    theData,self.maxPossibleScore = md.makeTwoClassData(self.total_num_events,fracClassA, 
                                                  probFeat0zero,self.days, self.intervention_times,
                                                  randRange=randRange,savePath=self.thepath,showPlots=False)
                elif(self.problem =="twoClassBinomial"):
                    n=self.days[-1]+1
                    p=[0.1, 0.6]
                    theData, self.maxPossibleScore = md.makeBinomialData(self.total_num_events,fracClassA, probFeat0zero,
                                                                         self.days, self.intervention_times,
                                                                         randRange=randRange,savePath=self.thepath, 
                                                                         n=n, parameters=p, showPlots=False)

                elif(problem =="twoClassNativeBinomial"):
                    n=self.days[-1]+1
                    p=[0.1, 0.6]
                    theData, self.maxPossibleScore = md.makeBinomialData(self.total_num_events,fracClassA, probFeat0zero,
                                                                         self.days, self.intervention_times ,
                                                                         randRange=randRange,savePath=self.thepath, 
                                                                         n=n, parameters=p, showPlots=False)
                    if('failures' in list(theData.columns.values)):
                        theData = theData.drop(columns='failures')
          
                else:
                    writelog(f"problem {problem} not configured")
                    exit(0)
                    
                for i in range(self.numNoise):
                    feat = "feature_rand"+str(i+1)
                    theData[feat] = np.random.randint(0, 2, theData.shape[0])
                
                writelog(f"data made for frac {fracClassA},  pval {probFeat0zero}, numNoise {self.numNoise} to be sampled at rate {self.sampleFrac}")
                with open('data/maxScores.csv','a') as f:
                    f.write(f"{self.problem},{fracClassA},{probFeat0zero},{self.maxPossibleScore}\n")
                    f.close()
                    
                if(self.verbose):
                    print(f'data has these fields: {theData.columns.values}')
            
                #make data
                runData = []
                for run in range (self.repetitions):
                    thisRunData = theData.sample(frac= self.sampleFrac, replace=self.sampleReplacement).reset_index(drop=True)
                    runData.append (thisRunData)
                
                #make dummy 'failed Tree to use when there is a timeout
                failedTree = tr.Tree(thisRunData,self.intervention_times,self.days)
                failedTree.evals = -1
                failedTree.accuracy = 0.0
                failedTree.nodeCount = 1

        
                #set up and run algorithm
                for method in methods: 
                    runResults = [()] * self.repetitions
                    alg = self.algorithms[method]
                    thecmd = self.cmdNames[alg]
                    paramset = self.paramSets[alg]
                    if( (alg=='EA') or alg=='LocalSearch'):
                        paramset['max_possible_fitness'] = self.maxPossibleScore       
                    writelog(f"running method {alg} for data {fracClassA},{probFeat0zero}")
                    tic = time.perf_counter() 
                    if(runParallel==False):
                        for run in range (self.repetitions):
                            runResults[run] = thecmd(runData[run],**paramset)
                        runTrees = runResults
                    else:
                        pool = Pool(cpu_count() -1)
                        for run in range(self.repetitions):
                            runResults[run] = pool.apply_async(thecmd,args=(runData[run],) , kwds= paramset)
                        #runTrees = [res.get(timeout=1000) for res in runResults]
                        runTrees = []
                        for res in runResults:
                            try:
                                runTrees.append(res.get(self.timeout))
                            except TimeoutError:
                                writelog('adding failedTree for timed outrun')
                                runTrees.append(failedTree)

               
                    toc = time.perf_counter()
                    meantime = (toc -tic)/self.repetitions
                    writelog(f"   took {toc-tic:.4f} seconds for {self.repetitions} runs.    Now summarising those results")            
                
                    #get results from this list
                    for run in range (self.repetitions):
                        nodeCount = runTrees[run].nodeCount
                        trees = trees.append({'Problem':self.longProblem,'Noise':self.numNoise,'Algorithm' : alg,
                                              'fitmodels':self.fit_models,'fracClassA' : fracClassA, 
                                              'pFeatZeroZero' : probFeat0zero,'Run':run, 'Accuracy': runTrees[run].accuracy,
                                              'NodeCount':nodeCount, 'Tree': runTrees[run].end_nodes}, ignore_index = True)    
                        results = results.append({'Problem':self.longProblem,'Noise':self.numNoise,'Algorithm' : alg,
                                                  'fitmodels':self.fit_models,'fracClassA':fracClassA, 
                                                  'pFeatZeroZero': probFeat0zero, 
                                                  'Run':run,'Runtime':meantime,'Gens':runTrees[run].evals, 
                                                  'Accuracy': runTrees[run].accuracy, 'NodeCount':nodeCount},
                                                 ignore_index = True)
        
                    if (self.testing):
                        print(results)
                    else:
                        results.to_csv(resfilename,mode='w')
                        trees.to_csv(treefilename,mode='w')

                       
                       
        writelog(f"Summary for {self.longProblem} with   {self.numNoise}  noise variables.")
        writelog(results.to_string())

           
## Function to call from cell/script to run experiment
    def runCaseStudy( self,runParallel=False,methods=[],testing=False):
        #dataframe to store results
        results = pd.DataFrame(columns=("Problem","Noise","Algorithm","fitmodels",
                                        "fracClassA","pFeatZeroZero","Run","Runtime","Gens","Accuracy"))
        trees = pd.DataFrame (columns=("Problem","Noise","Algorithm","fitmodels",
                                        "fracClassA","pFeatZeroZero","Run","Accuracy","Tree"))
    
        resfilename = 'results/results-' +str(datetime.date.today()) +'.csv'
        treefilename = 'results/trees-' +str(datetime.date.today()) +'.csv'

        theData = pd.read_csv('data/CaseStudy/Noiseless/case_study_data.csv')
        self.maxPossibleScore = 1.0    
        if(self.testing==False):
            self.repetitions=25
        else:
            self.repetitions=1
        self.timeout=3600
        
        writelog(f"case study data read, setting repetions = {self.repetitions} and ,maxscore={self.maxPossibleScore}")
                   
        #make data
        runData = []
        #consider sampling later
        self.sampleFrac=1.0
        for run in range (self.repetitions):
            thisRunData = theData.sample(frac= self.sampleFrac, replace=self.sampleReplacement).reset_index(drop=True)
            runData.append (thisRunData)
                
        #make dummy 'failed Tree to use when there is a timeout
        failedTree = tr.Tree(thisRunData,self.intervention_times,self.days)
        failedTree.evals = -1
        failedTree.accuracy = 0.0
        failedTree.nodeCount = 1

        #set up and run algorithm
        for method in methods: 
            runResults = [()] * self.repetitions
            alg = self.algorithms[method]
            thecmd = self.cmdNames[alg]
            paramset = self.paramSets[alg]
            if( (alg=='EA') or alg=='LocalSearch'):
                paramset['max_possible_fitness'] = self.maxPossibleScore       
            writelog(f"running method {alg} for case study")
            tic = time.perf_counter() 
            if(runParallel==False):
                for run in range (self.repetitions):
                    runResults[run] = thecmd(runData[run],**paramset)
                runTrees = runResults
            else:
                pool = Pool(cpu_count() -1)
                for run in range(self.repetitions):
                    runResults[run] = pool.apply_async(thecmd,args=(runData[run],) , kwds= paramset)
                #runTrees = [res.get(timeout=1000) for res in runResults]
                runTrees = []
                for res in runResults:
                    try:
                        runTrees.append(res.get(self.timeout))
                    except TimeoutError:
                        writelog('adding failedTree for timed outrun')
                        runTrees.append(failedTree)

               
            toc = time.perf_counter()
            meantime = (toc -tic)/self.repetitions
            writelog(f"   took {toc-tic:.4f} seconds for {self.repetitions} runs.    Now summarising those results")            
                
            #create null values to keep resylts format consistent
            probFeat0zero, fracClassA = 0.0, 0.0
           
            #get results from this list
            for run in range (self.repetitions):
                nodeCount = runTrees[run].nodeCount
           
                trees = trees.append({'Problem':self.longProblem,'Noise':self.numNoise,'Algorithm' : alg,
                                        'fitmodels':self.fit_models,'fracClassA' : fracClassA, 
                                        'pFeatZeroZero' : probFeat0zero,'Run':run, 'Accuracy': runTrees[run].accuracy,
                                        'NodeCount':nodeCount, 'Tree': runTrees[run].end_nodes}, ignore_index = True)    
                results = results.append({'Problem':self.longProblem,'Noise':self.numNoise,'Algorithm' : alg,
                                                  'fitmodels':self.fit_models,'fracClassA':fracClassA, 
                                                  'pFeatZeroZero': probFeat0zero, 
                                                  'Run':run,'Runtime':meantime,'Gens':runTrees[run].evals, 
                                                  'Accuracy': runTrees[run].accuracy, 'NodeCount':nodeCount},
                                                 ignore_index = True)
        
                if (self.testing):
                    print(results)
                else:
                    results.to_csv(resfilename,mode='w')
                    trees.to_csv(treefilename,mode='w')

                       
                       
        writelog(f"Summary for {self.longProblem} with  {self.numNoise} noise variables")
        writelog(results.to_string())

