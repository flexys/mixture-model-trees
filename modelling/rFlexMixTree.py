# load python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import modelling.tree as tr

# make sure it can find R
import os, platform
host = platform.system()
if (host=="Windows"):
    print('Set R_HOME environment variable')
elif (host=="Darwin"):
    os.environ['R_HOME']= '/Library/Frameworks/R.framework/Resources'
elif(host=="Linux"):
    os.environ['R_HOME']= '/usr/lib/R' #ubuntu
else:
    print(f"unknown operating system{host}")
    exit(0)

# load rpy2
import rpy2.robjects as robjects
from functools import partial
from rpy2.ipython import html
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()
html.html_rdataframe=partial(html.html_rdataframe, table_class="docutils")
#html.init_printing()

# load R libraries
utils = importr("utils")
flex = importr("flexmix")

# turn relevant R functions into python functions
parameters = robjects.r['parameters']
summary = robjects.r['summary']
refit = robjects.r['refit']
getModel = robjects.r['getModel']
datafr = robjects.r['data.frame'] 
predict = robjects.r['predict']
fitted = robjects.r['fitted']
Rcoeffs = robjects.r['coefficients']


##### flexmix gives us the parameters of a multinomial logistic model
def component_weights(w, a, k, categorical=True): # w = concomittant var, a params of the models
    
    s = 1
    
    for i in range(k-1):
        ac = 0
        for j, aa in enumerate(a[i]):
            if categorical:
                if j==0:
                    ac += aa
                elif j==w:
                    ac += aa
            else:
                if j==0:
                    ac += aa
                else:
                    ac += w[j-1]*aa
        s += np.exp(ac)
    
    proportions = np.zeros(k)
    for i in range(k-1):
        ac = 0
        for j, aa in enumerate(a[i]):
            if categorical:
                if j==0:
                    ac += aa
                elif j==w:
                    ac += aa
            else:
                if j==0:
                    ac += aa
                else:
                    ac += w[j-1]*aa
        proportions[i+1] = np.exp(ac) / s
    proportions[0] = 1-np.sum(proportions)
    return proportions



### main function to fit models
def fit_mixture_model(df, n ,feature_names, k_range, feat_val_combos,mfamily="binomial",verbosity=0):
        
    if 'feature_rand' in feature_names:
        feature_rand = True
    
    form = ''
    for i, feat in enumerate(feature_names):
        if i == 0:
            form = f'~ {feat}'
            tmp = f'{feat} = 0'
        else:
            form = f'{form} + {feat}'
            tmp = f'{tmp}, {feat} = 0'
    tmp = f'datafr({tmp})'
    
    if(verbosity>1):
        print(f'formula is {form}')
    conc = flex.FLXPmultinom(robjects.Formula(form))   
    
 
    #construct model and formual depnending on the distribution family chosen
    m1 = flex.FLXMRglm( family=mfamily)# could use FLEXrobglm(family=mfamily)
    if(mfamily=="binomial"):
        theFormula = robjects.Formula('cbind(dayOfEp, failures) ~ 1')
    elif mfamily=="gaussian":
        theFormula =     robjects.Formula('dayOfEp ~ 1')
    else:
        print("unrecognised model family")
        return 0
    
 
    #make and fit the models
    if len(k_range) > 1 :
        # fit models and get best value of k 
        ff = flex.stepFlexmix(theFormula, data = df, k = robjects.IntVector(k_range), concomitant = conc, model = m1, verbose=False)
        ff = getModel(ff, "BIC")
    else:
        ff = flex.flexmix(theFormula, data = df, k = k_range[0], concomitant = conc, model = m1, verbose=False)
    
    #extract the fitted parameters
    params = np.asarray(predict(ff, newdata = eval(tmp)))
    params = params.flatten()
    #print(f'params are {params}')
    conc_params = parameters(ff, which='concomitant')
    conco_params = np.delete(conc_params, 0, 1).transpose().tolist()
    #print(f'conco paras are {conco_params}')
    k = len(params)
    if(verbosity>0):
        print(f"{k} components chosen")
    
    # need meand and std deviation for gaussians
    if(mfamily=="gaussian"):
        componentParams = np.empty((k,2))
        for component in range (1,k+1):
            cparams = parameters(ff,component=component)
            mean,sdev = float(cparams[[0]]), float(cparams[[1]])
            #print( f'parameters for component {component} are {cparams}')
            componentParams[component-1][0] = mean
            componentParams[component-1][1] = sdev
        if(verbosity>0):
            print(f'Fitted model has {k} components with params {componentParams}')
            
            
    end_nodes = []
    for j, f in enumerate(feat_val_combos):
        mask = ''
        for i, v in enumerate(f):
            if feature_rand and i == len(f)-1:
                mask = f'{mask}&(dataset.feature_rand=={v})'
            else:
                mask = f'{mask}&(dataset.feature_{i}=={v})'
        end_nodes.append([mask[1:],[]])
        
        cw = component_weights(f, conco_params, k, categorical=False)
        if(verbosity>1):
            print(f'for combo {f}, component weights are = {cw}')
        exp = []

        c = np.zeros(n+1)

        if(mfamily=="binomial"):
            for u, p in enumerate(params):
                c += cw[u]*stats.binom.pmf(np.arange(n+1), n, p)
        elif (mfamily=="gaussian"):
            rng = np.random.default_rng()
            for component in range (k):
                samples = np.zeros(n+1)
                #number of points to draw
                numDraws = int(cw[component]*1000)
                # normal parameters
                mean = componentParams[component][0]
                std = componentParams[component][1]
                for draw in range(numDraws):
                    val = int(rng.normal(mean, std,1).round(0))
                    if ( val>=0 and val <  n+1):
                        samples[val] +=1
                    
                #print(samples)
                c += samples/1000    #c[val] += 0.001
        exp.append(c.tolist())

        end_nodes[j][1] = exp[0]
    return end_nodes,k

##### do the actua lfitting




def getFittedFlexMixTree(data, n,maxComponents=3, feature_names=None,modelFamily="None", verbosity=0 ):

    # range of numbers of components to try, if using single k still make it a list i.e.: [k]
    k_range = (np.arange(maxComponents) + 1).tolist()  

    if not (modelFamily == "gaussian" or modelFamily=="binomial" ):
        print(f"unrecognised model Family {modelFamily} not yet supported")
        return(-1)
    
    #### get the data and transdform to R dataframe
    # add one to each dayOfEp to help avoid NaNs when solving for parameters
    data['dayOfEp'] += 1
    t = []
    for d in data.dayOfEp.to_numpy():
        t.append(n-d)
    data['failures'] = t
    if(verbosity>1):
        print(data.head())
        print(data.groupby('dayOfEp').count())
    with localconverter(robjects.default_converter + robjects.pandas2ri.converter):
        df = robjects.conversion.py2rpy(data) 
    
    # list of feature names, feature_rand must be last
    if len(feature_names)==0:
        print('error: empty feature name list passed to getFittedFlexMixTree()')
        return -1
    else:
        # get all possible combinations of features
        feat_val_combos = data.drop_duplicates(feature_names)[feature_names].to_numpy(dtype='int16').tolist()
        if(verbosity>1):
            print(f"feat val combos = {feat_val_combos}")

    if(verbosity>1):
        print('about to fit data')    
    ## now do the fitting    
    end_nodes,numComponents = fit_mixture_model(df,n, feature_names, k_range, feat_val_combos, mfamily = modelFamily,verbosity=verbosity)


    # score it
    accuracy = tr.score_tree(data, end_nodes, np.arange(n+1))

    # and report
    k = len(end_nodes)
    if(verbosity>0):
        print(f'fitted tree has {numComponents} components and accuracy {accuracy}')
        for i in range (k):
            print("partition " + str(end_nodes[i][0]) + '\n' + "distribution " + str(end_nodes[i][1]) + '\n')
    
    if(verbosity>1):
        fig2,ax2 = plt.subplots(ncols=k,figsize=(12,4))
        for i in range (k):
            ax2[i].bar(np.arange(n+1),end_nodes[i][1])
            ax2[i].set_title(str(feat_val_combos[i]))
        plt.show

    theTree = tr.Tree(dataset=None,intervention_times=None, days=None)
    theTree.accuracy = accuracy
    theTree.end_nodes = end_nodes
    theTree.nodeCount = numComponents
    
    return(theTree)



