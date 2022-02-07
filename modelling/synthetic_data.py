import numpy as np
import pandas as pd
import modelling.tree as tr
import scipy.stats as stats


def generate_curve(t, intervention_times, functions, noise=0.001):
    for j, i in enumerate(intervention_times):
        if j == 0:
            y = functions[j].predict(t)
        else:
            shape = len(t[np.where(t >= i)[0][:]])
            y[np.where(t >= i)[0][:]] = y[np.where(t >= i)[0][:]] + functions[j].predict(t[:shape])

    # add normally distributed noise
    y = y + np.random.normal(0, noise, len(y))
    for i in range(len(y)):
        if y[i] < 0:
            y[i] = 0

    data_a = pd.DataFrame(y, columns=['response'])

    data_a['dayOfEp'] = t

    return data_a


def make_dataframe(data_a, sei_range, feature_vals, proportional=True, random_feature=False, num_rand=4):
    a = pd.DataFrame(columns=['dayOfEp', 'loggedIn', 'subjectEntityId', 'episodeDuration'])

    i = 0

    sei = np.arange(sei_range[0], sei_range[1])
    n_tot = n = sei_range[1] - sei_range[0]

    for d in data_a.dayOfEp:

        if proportional:
            m = data_a.loc[data_a.dayOfEp == d, 'response'].values[0]
        else:
            r = data_a.loc[data_a.dayOfEp == d, 'response'].values[0]
            m = r * n_tot / n

        nl = [0] * int(n - int(m * n))
        l = [1] * int(m * n)
        ls = l + nl
        ds = [d] * n
        dur = [data_a.dayOfEp.max()] * n

        if i == 0:
            a = a.append(pd.DataFrame(np.array([ds, ls, sei, dur]).T,
                                      columns=['dayOfEp', 'loggedIn', 'subjectEntityId', 'episodeDuration']))

        else:
            sei = a.loc[(a.dayOfEp == (d - 1)) & (a.loggedIn == 0), 'subjectEntityId'].values
            a = a.append(pd.DataFrame(np.array([ds, ls, sei, dur]).T,
                                      columns=['dayOfEp', 'loggedIn', 'subjectEntityId', 'episodeDuration']))

        a.loc[(a.dayOfEp == d) & (a.loggedIn == 1), 'episodeDuration'] = d

        n = int(n - int(n * m))

        i += 1
    a['episodeDuration'] = a['subjectEntityId'].map(
        a.drop_duplicates('subjectEntityId', keep='last').set_index('subjectEntityId')['episodeDuration'])

    sids = a.subjectEntityId.drop_duplicates().to_numpy()
    np.random.shuffle(sids)
    g_old = 0
    for j, f in enumerate(feature_vals):

        g = int(n_tot * f[0])+g_old
        if j != len(feature_vals)-1:
            sids_j = sids[g_old:g]
        else:
            sids_j = sids[g_old:]

        for i, v in enumerate(f[1]):
            a.loc[a.subjectEntityId.isin(sids_j), 'feature_' + str(i)] = v
        g_old = g

    # for i in range(len(feature_vals)):
    #     f = 1 / len(feature_vals[i])
    #     #g = int(n_tot * f)
    #     proportion_old = 0
    #     proportions = []
    #     for j in range(len(feature_vals[i])):
    #         if j > 0:
    #             proportion = np.random.normal(f, rand_prop)
    #             if proportion > 0.6:
    #                 proportion = 0.6
    #             elif proportion < 0.1:
    #                 proportion = 0.1
    #         else:
    #             proportion = 0
    #         proportions.append(proportion + proportion_old)
    #         proportion_old = proportion
    #
    #     for j in range(len(feature_vals[i])):
    #         g = int(n_tot * proportions[j])
    #
    #         if j != len(feature_vals[i]) - 1:
    #             sids_j = sids[g:int(n_tot * proportions[j+1])]
    #         else:
    #             sids_j = sids[g:]
    #         a.loc[a.subjectEntityId.isin(sids_j), 'feature_' + str(i)] = feature_vals[i][j]

    if random_feature:
        for i in sids:
            a.loc[a.subjectEntityId == i, 'feature_rand'] = np.random.randint(num_rand)

    a.reset_index(inplace=True)

    a.loc[a.loggedIn == 1, 'eventType'] = 'UserLoggedIn'
    a.loc[a.loggedIn == 0, 'eventType'] = 'None'

    return a


def append_groups(groups):
    c = 0
    for group in groups:
        if c == 0:
            synthetic_data = group
        else:
            synthetic_data = synthetic_data.append(group, ignore_index=True)
        c += 1

    ds = pd.DataFrame(synthetic_data.sort_values(by='eventType').drop_duplicates(['subjectEntityId'], keep='last'))
    ds.loc[ds.eventType == 'None', 'dayOfEp'] = -1
    ds.drop(columns=['eventType', 'index', 'loggedIn', 'episodeDuration'], inplace=True)
    ds.reset_index(drop=True, inplace=True)
    return ds


# binomial
def make_component_df(params, sei_range, feature_vals, random_feature=False, num_rand=4, component_id=0, noise=0.0):
    component_ids = []
    value = []
    N = sei_range[1] - sei_range[0]

    for i in range(N):
        theDay = np.random.binomial(params[0], params[1])+np.random.normal(0, noise)
        if(theDay>= params[0]):
            theDay=params[0] -1
        value.append(theDay)
        component_ids.append(component_id)

    a = pd.DataFrame(
        {'dayOfEp': value, 'component_id': component_ids, 'subjectEntityId': np.arange(sei_range[0], sei_range[1])})

    sids = a.subjectEntityId.to_numpy()

    g_old = 0

    for j, f in enumerate(feature_vals):

        g = int(N * f[0]) + g_old

        if j != len(feature_vals) - 1:
            sids_j = sids[g_old:g]
        else:
            sids_j = sids[g_old:]

        for i, v in enumerate(f[1]):
            a.loc[a.subjectEntityId.isin(sids_j), 'feature_' + str(i)] = v
        g_old = g

    if random_feature:
        for i in sids:
            a.loc[a.subjectEntityId == i, 'feature_rand'] = np.random.randint(num_rand)
    return a


def create_dataframe(groups):

    c = 0
    for group in groups:
        if c == 0:
            synthetic_data = group
        else:
            synthetic_data = synthetic_data.append(group, ignore_index=True)
        c += 1

    n = int(synthetic_data.dayOfEp.max())
    t = []
    for d in synthetic_data.dayOfEp.to_numpy():
        t.append(n - d)

    synthetic_data['failures'] = t

    return synthetic_data


def target_score(synthetic_data, feature_val_combos, params, cw, feature_rand, n):
    end_nodes = []
    for j, f in enumerate(feature_val_combos):
        mask = ''
        for i, v in enumerate(f):
            if feature_rand and i == len(feature_val_combos)-1:
                pass
                #mask = f'{mask}&(dataset.feature_rand=={v})'
            else:
                mask = f'{mask}&(dataset.feature_{i}=={v})'
        end_nodes.append([mask[1:],[]])

        exp = []

        c = np.zeros(n+1)

        for u, p in enumerate(params):
            c += cw[j][u]*stats.binom.pmf(np.arange(n+1), n, p[1])
        exp.append(c.tolist())

        end_nodes[j][1] = exp[0]

    return tr.score_tree(synthetic_data, end_nodes, np.arange(n+1))