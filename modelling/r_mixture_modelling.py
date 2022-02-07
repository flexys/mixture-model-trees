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
# html.init_printing()

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


def fit_mixture_model(pd_df, k_range, n,verbose=False):
    with localconverter(robjects.default_converter + robjects.pandas2ri.converter):
        df = robjects.conversion.py2rpy(pd_df)

    m1 = flex.FLXMRglm(family='binomial')
    if len(k_range) > 1:
        # fit models and get best value of k
        ff = flex.stepFlexmix(robjects.Formula('cbind(dayOfEp, failures) ~ 1'),
                              data=df, k=robjects.IntVector(k_range), model=m1,verbose=verbose)
        ff = getModel(ff, "BIC")
    else:
        ff = flex.flexmix(robjects.Formula('cbind(dayOfEp, failures) ~ 1'),
                          data=df, k=k_range[0], model=m1,verbose=verbose)

    weights = np.array(parameters(ff, which='concomitant'))[0]
    params = predict(ff)
    param = []
    for col in params:
        param.append(col[0])

    exp = []
    model_info = {'model_type': 'binomial',
                  'parameters': [],
                  'component_weights': []}

    c = np.zeros(n + 1)

    for u, p in enumerate(param):
        c += weights[u] * stats.binom.pmf(np.arange(n + 1), n, p)
        model_info['parameters'].append(p)
        model_info['component_weights'].append(weights[u])

    exp.append(c.tolist())

    return exp[0], model_info
