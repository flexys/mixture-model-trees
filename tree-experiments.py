

import modelling.makeData as md
import modelling.experiment as experiment

# Set up logger
logfile = "logfile.txt"

def writelog(theString=None):
    with open(logfile,'a') as f:
        f.write(theString + '\n')
        f.close()
    print(theString)


import importlib
importlib.reload(experiment)
importlib.reload(md)


#importlib.reload(genetic_algorithm)


# ## This one runs the actual experiments

#algorithms = ('flexmix','recursive','LocalSearch', 'EA')
methods = [1,3]#[0,1,2,3]

#problems = ['twoSpikeTwoClass','twoClassBinomial']#,'twoClassNativeBinomial']
#problems = ['twoClassNativeBinomial']
#problems= ['twoSpikeTwoClass05','twoClassBinomial05','twoSpikeTwoClass10','twoClassBinomial10']
#problems= ['twoSpikeTwoClass00','twoClassBinomial00','twoSpikeTwoClass08','twoClassBinomial10']
#problems=['CaseStudy']

problems= ["twoSpikeTwoClass00", "twoClassBinomial00" ]
basenames= ['twoSpikeTwoClass','twoClassBinomial']
extensions= ["01","02","03","04","05","06","08","10","15","20"]

for base in basenames:
    for extension in extensions:
        newprob= base+extension
        problems.append(newprob )

runParallel = True
testing = False
verbose= False
fit_models = False
thepath = ""


for problem in problems:
    for fit_models in ['False']:#(True, False):
        print(f"prob {problem} fit {fit_models} ")
        thisExperiment = experiment.experiment()
        thisExperiment.setUpExp(problem,testing)
        thisExperiment.setUpAlgorithms(fit_models,verbose)
        if(problem=='CaseStudy'):
            thisExperiment.runCaseStudy(runParallel, methods)
        else:
            thisExperiment.runExp(runParallel,methods)


