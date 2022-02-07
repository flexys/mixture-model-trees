import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.optimize import minimize
import modelling.r_mixture_modelling as rmm
from scipy.special import comb


class Model:
    """Parent class for models."""

    def __init__(self, bounds=None, params=None, p0=None):
        self.params = params
        self.gof = None
        self.bounds = bounds
        self.p0 = p0
        self.runerror = False
        self.valerror = False

    def func(self, *kwargs):
        raise NameError('You have not defined a function for your model')

    def _gof(self, x, y):
        fit = self.predict(x)
        gof = np.sqrt(np.sum((y - fit) ** 2) / len(y)) #rmse
        return gof

    def fit(self, x, y):
        try:
            popt, pcov = curve_fit(self.func, x, y, p0=self.p0, bounds=self.bounds, ftol=1e-9, xtol=1e-9)
        except RuntimeError:
            self.runerror = True
            self.params = x, y
            self.gof = 99
        except ValueError:
            self.valerror = True
            self.params = x, y
            self.gof = 99
        else: #fix this
            self.params = popt.tolist()
            self.gof = self._gof(x, y)

    def predict(self, t):
        return self.func(t, *self.params)


class CustomModel(Model):
    """"1/(1+e^kt)"""

    def __init__(self, bounds=(0, 5), params=None, p0=[0, 0, 0, 0]):
        super().__init__(bounds, params, p0)

    def func(self, t, a, b, c, d):
        return (a * (1 / (1 + np.exp(b * t)))) + c * np.exp(-d * t)


class CustomModel2(Model):
    """"1/(1+e^kt)"""

    def __init__(self, bounds=(0, 5), params=None, p0=[0, 0, 0]):
        super().__init__(bounds, params, p0)

    def func(self, t, a, b, c):
        return (a * (1 / (1 + np.exp(b * t)))) + c


class General(Model):
    """"1/(1+e^kt) with offset"""

    def __init__(self, bounds=(0, 10), params=None, p0=[0, 0, 0, 0, 0]):
        super().__init__(bounds, params, p0)

    def func(self, t, a, b, c, d, e):
        z = list(filter(lambda x: x < (e-1e-9), t))
        z = [x*0 for x in z]
        x = np.array(list(filter(lambda x: x >= (e-1e-9), t)))
        y = (a * (1 / (1 + np.exp(b * (x - e))))) + c * np.exp(-d * (x - e))
        y = np.append(z, y)
        return y


class CustomModel4(Model):
    """"1/(1+e^kt) """

    def __init__(self, bounds=(0, 10), params=None, p0=[0, 0, 0, 0, 0]):
        super().__init__(bounds, params, p0)

    def func(self, t, a, b, c, d, e):
        y = (a * (1 / (1 + np.exp(b * (t - e))))) + c * np.exp(-d * (t - e))
        return y


class LogNorm(Model):
    """Log normal"""

    def __init__(self, bounds=(0, 100), params=None, p0=[0, 0, 0]):
        super().__init__(bounds, params, p0)

    def func(self, t, a, b, c):
        return a * (1 / (np.sqrt(2 * np.pi) * b * t)) * np.exp(-(np.log(t) ** 2) / (2 * (b ** 2))) + c


class InvGauss(Model):
    """Inverse gaussian"""

    def __init__(self, bounds=(0, 100), params=None, p0=[0, 0, 0]):
        super().__init__(bounds, params, p0)

    def func(self, t, a, b, c):
        return a * ((1 / np.sqrt(2 * np.pi * (t ** 3))) * np.exp(-((t - b) ** 2) / (2 * t * (b ** 2)))) + c


    
    

def pdf(x):
    return 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)


def cdf(x):
    return (1 + erf(x/np.sqrt(2))) / 2


class Skewed(Model):
    """skewed normal"""

    def __init__(self, bounds=([0, 0.01, 0, 0, 0, 0.03], [10., 10., 200., 10., 0.1, 0.1]), params=None, p0=[0.5, 0.5, 0.5, 0, 0, 0.03]):
        super().__init__(bounds, params, p0)

    def func(self, x, e, w, a, b, c, d):
        t = (x - e) / w
        skewed_norm = 2 / w * pdf(t) * cdf(a * t)
        return b * skewed_norm + c * np.exp(-d * x)


class Binom(Model):
    """binomial"""

    def __init__(self, bounds=([0.001], [0.999]), params=None, p0=[0.05], n=None):
        super().__init__(bounds, params, p0)
        self.n = n

    def func(self, x, p, n):
        q = 1-p
        y = []
        for i in x:
            ncr = comb(n, i)
            y.append(ncr*(p**i)*(q**(n-i)))
        return y

    def fit(self, x, y):
        # loop over range of ns, fit model for each, find best n, store this along with p
        rmse_best = 99
        params_best = []
        for n in range(5, 25):
            def f(x, p):
                return self.func(x, p, n)
            try:
                popt, pcov = curve_fit(f, x, y, p0=self.p0, bounds=self.bounds, ftol=1e-9, xtol=1e-9)
                self.params = popt.tolist()
                self.params.append(n)
                rmse = self._gof(x, y)
                if rmse < rmse_best:
                    self.gof = rmse
                    params_best = self.params.copy()
                    rmse_best = rmse
            except RuntimeError:
                pass

        if params_best:
            self.params = params_best
        else:
            self.runerror = True
            self.params = x, y
            self.gof = 99


class Line(Model):
    """Straight line with gradient a, intercept c"""

    def __init__(self, bounds=(0, 100), params=None):
        super().__init__(bounds, params)

    def fit(self, x, y):
        try:
            popt, pcov = curve_fit(self.func, x, y, p0=[0, 0], bounds=self.bounds)
        except RuntimeError:
            self.runerror = True
            self.params = x, y
            self.gof = 99
        except ValueError:
            self.valerror = True
            self.params = x, y
            self.gof = 99
        else: #fix this
            self.params = popt.tolist()
            self.gof = self._gof(x, y)

    def func(self, t, a, c):
        return a*t + c


def response_rate(I, params):
    q = params[0]
    r = params[1]
    pi = params[2]

    P = np.zeros(len(I))

    for c, i in enumerate(I):

        s = 0
        for j in range(1, i + 1):
            s += (q ** (j - 1)) * (r ** (i - j))

        P[c] = pi * (1 - q) * s * (1 - r)

    #         P[c] = pi*(1-q)*(q**(i-1)) # instant delivery

    return P


def response_model(y):
    y = np.append(np.array(0), y)

    def ll_func(x):
        q = x[0]
        r = x[1]

        k = len(y)
        s_k = np.sum(y)

        A = 0
        B = 0

        for i in range(1, k):

            s = 0
            for j in range(1, i + 1):
                s += (q ** (j - 1)) * (r ** (i - j))

            P_i = (1 - q) * s * (1 - r)

            A += y[i] * np.log(P_i)
            B += P_i

        return -(A - s_k * np.log(B))

    res = minimize(ll_func, np.array([0.01, 0.01]), method='BFGS')

    def get_pi(y, q, r):
        k = len(y)
        s_k = np.sum(y)
        P = 0
        for i in range(1, k):
            s = 0
            for j in range(1, i + 1):
                s += (q ** (j - 1)) * (r ** (i - j))

            P_i = (1 - q) * s * (1 - r)
            P += P_i
        return s_k / P

    pi = get_pi(y, res.x[0], res.x[1])
    params = [res.x[0], res.x[1], pi]
    return params


def choose_model(model_type, params=None, max_offset=None, n=None):
    if model_type == 'custom':
        model = CustomModel(params=params)
    elif model_type == 'custom2':
        model = CustomModel2(params=params)
    elif model_type == 'general':  # in use
        model = General(params=params)
    elif model_type == 'custom4':
        model = CustomModel4(params=params)
    elif model_type == 'lognormal':
        model = LogNorm(params=params)
    elif model_type == 'invgauss':
        model = InvGauss(params=params)
    elif model_type == 'skewed':  # in use
        model = Skewed(bounds=([0, 0.01, 0, 0, 0, 0.03], [max_offset, 10., 200., 10., 0.1, 0.1]), params=params)
    elif model_type == 'binom':  # in use
        model = Binom(params=params, n=None)
    elif model_type == 'line':
        model = Line(params=params)
    else:
        raise NameError("Invalid model type")
    return model


class ModelResponse:
    """Fit models to response over time data"""

    def __init__(self):
        self.model_types = ['general', 'binom', 'skewed']  # 'lognormal', #'invgauss', # 'custom4',#'line', 'custom', 'custom2',
        self.best_models = {}
        self.expected = None
        self.total_rmse = 0

    def fit(self, df, intervention_times, days=None,verbose=False):
        data = df.copy()

        if 'failures' in df:
            #print('...r version')
            n = int(data.iloc[0]['dayOfEp'] + data.iloc[0]['failures'])
            k_range = [1, 2, 3]
            exp, self.best_models = rmm.fit_mixture_model(data, k_range, n, verbose)
            # making things more robust
            if (len(exp) != len(days)):
                print('warning: ModelResponse.fit() changing length of exp to match days')
                #print(f'problem in ModelResponse.fit()\n exp is {exp} with len {len(exp)}\n but days is {days} with len {len(days)}\n')
                exp = exp[:len(days)] + [0]*(len(days) - len(exp))
                #print(f"new exp is {exp}")
            #end robustness fix
            self.expected = pd.DataFrame({'t': days, 'exp': exp})
        else:
            
            #print('....our version')
            for i in range(len(intervention_times)):

                # get the section of data from intervention_times[i] to [i+1]
                if i != len(intervention_times) - 1:
                    data_i = data.loc[(data.t >= intervention_times[i])
                                   & (data.t < intervention_times[i + 1])]
                else:
                    data_i = data.loc[(data.t >= intervention_times[i])]

                # initialise best "goodness of fit"
                best_gof = 100
                size = len(data_i.index)

                # get the days we need to work with for this section
                t_i = data.iloc[:data_i.shape[0]].t.values.astype('float64')
                max_offset = max(t_i)

                # if this section has less than 5 data points, & this isn't the last section, & mn is less than the minimum
                # response in this section, add mn as a data point on the end
                mn = data.loc[data.t > max(data_i.t)].response.min()
                if (len(t_i) < 5) & (i != len(intervention_times) - 1) & \
                        ((data_i.response.iloc[-1] > mn) or (data_i.response.iloc[-2] > mn)):
                    xt = data.loc[(data.t > max(data_i.t)) & (data.response == mn), 't'].values[0] - intervention_times[i]
                    t_i = np.append(t_i, xt)
                    y = np.append(data_i.response.values, mn)
                else:
                    y = data_i.response.values

                # for each curve type we are considering, fit this curve to the data for this section, choose the best one
                # line = False
                if len(t_i) > 2:
                    model_types_i = self.model_types
                else:
                    model_types_i = self.model_types.copy()
                    model_types_i.remove('skewed')

                # n for binomial
                # only valid if days start from 0 and go up in increments of 1
                n = data.t.shape[0] - 1

                for model_type in model_types_i:

                    # instantiate & fit the model we're fitting this iteration
                    model = choose_model(model_type, max_offset=max_offset, n=n)
                    model.fit(t_i, y)

                    # if error, move on to try the next model type
                    if model.runerror:
                        # print('runtime error for', model_type)
                        pass
                        # print('Warning: Could not fit ' + model_type + ' for curve ' + str(i) + ' (runtime)')
                    elif model.valerror:
                        # print('value error for', model_type)
                        # print('Warning: Could not fit ' + model_type + ' for curve ' + str(i) + ' (value)')
                        # print('t:', t_i.values)
                        # print('r:', y)
                        # best_model = choose_model('line', params=(0, 0))
                        # best_gof = 0.000001
                        # best_model_type = 'line'
                        pass

                    # # to stop it using custom3 instead of line if it is a straight line
                    # if model_type == 'line' and model.gof < 2e-3:
                    #     line = True

                    # if the goodness of fit is better than any previous gof, this is new best gof
                    if model.gof < best_gof: #and (not line or model_type == 'line'):
                        best_model = model
                        best_gof = model.gof
                        best_model_type = model_type

                # save model info
                self.total_rmse = self.total_rmse + ((size / len(data.index)) * best_gof)
                self.best_models['curve' + str(i)] = {'best_model': best_model_type,
                                                      'parameters': best_model.params,
                                                      'rmse': best_gof
                                                      }

                # take away this curve from the rest of the response
                if i != len(intervention_times) - 1:
                    next_section_response = data.loc[(data.t >= intervention_times[i]), 'response'].to_numpy()
                    next_sect_model_predict = best_model.predict(data.loc[(data.t >= intervention_times[i]), 't'
                                                                       ].values.astype('float64') - intervention_times[i])

                    data.loc[(data.t >= intervention_times[i]), 'response'] = next_section_response - next_sect_model_predict

                    # don't let the response be negative
                    if any(x <= 0 for x in data.response.values):
                        data.loc[data.response <= 0, 'response'] = 1e-11

    def predict(self, data, intervention_times):
        expected = pd.DataFrame(data)
        expected['exp'] = 0

        for i in range(len(intervention_times)):
            model_type = self.best_models['curve' + str(i)]['best_model']
            params = self.best_models['curve' + str(i)]['parameters']
            if model_type == 'binom':
                model = choose_model(model_type, params, n=data.t.shape[0]-1)
            else:
                model = choose_model(model_type, params)

            s = len(expected.loc[(expected.t >= intervention_times[i]), 'exp'].index)
            expected.loc[(expected.t >= intervention_times[i]), 'exp'] = (
                    expected.loc[(expected.t >= intervention_times[i]), 'exp'].values
                    + model.predict(data.iloc[:s].t.values.astype('float64')))

        self.expected = expected

    def plot(self, data, intervention_times):
        plt.figure(figsize=(10, 7))
        plt.scatter(data.t, data.response, color='Orange', label='Observed data')

        expected = data.copy()
        expected['exp'] = 0

        for i in range(len(intervention_times)):

            model_type = self.best_models['curve' + str(i)]['best_model']
            params = self.best_models['curve' + str(i)]['parameters']

            model = choose_model(model_type, params)

            s = expected.loc[(expected.t >= intervention_times[i]), 'exp'].shape[0]
            expected.loc[(expected.t >= intervention_times[i]), 'exp'] = (
                    expected.loc[(expected.t >= intervention_times[i]), 'exp']
                    + model.predict(data.iloc[:s].t.values))

            if i != len(intervention_times) - 1:
                data_i = data.loc[(data.t >= intervention_times[i])
                                  & (data.t < intervention_times[i + 1])]
            else:
                data_i = data.loc[(data.t >= intervention_times[i])]

            plt.plot(data_i.t, expected.loc[expected.t.isin(data_i.t), 'exp'], lw=1, label=('Fitted curve' + str(i)))
            plt.legend()
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show(block=False)

        self.expected = expected

        plt.figure(figsize=(10, 7))
        residuals = expected.exp - expected.response
        plt.scatter(expected.t, residuals)
        plt.plot(expected.t, np.zeros(expected.shape[0]))
        plt.title('Residuals')
        plt.show(block=False)
