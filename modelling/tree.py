import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from scipy.stats import chisquare
import json
from sklearn.cluster import DBSCAN
import modelling.model_response as mr
import copy

class Tree:
    def __init__(self, dataset, intervention_times, days, sensitivity=0.005,verbose=False):
        self.dataset = dataset
        self.tree = {}
        self.intervention_times = intervention_times
        self.sensitivity = sensitivity
        self.days = days
        self.end_nodes = []
        self.accuracy = 0
        self.nodeCount=0
        self.evals = 1 
        self.verbose = verbose

    # Build a decision tree
    def build_tree(self, features, max_depth, min_size, split_criteria='fit_improvement', 
                   epsilon=1, gen_tree=None, fit_models=False):
        # get the root split
        mask = None
        root = self.get_split(mask, features, 0, split_criteria, epsilon, gen_tree, min_size, fit_models)

        # recursively split down to max depth / min_size
        self.split(root, features, max_depth, min_size, split_criteria, 1, epsilon, gen_tree, fit_models=fit_models)

        # if no splits, get model info for whole dataset
        try:
            if root['left']:
                pass
        except KeyError:
            root = self.to_terminal('[True]*dataset.shape[0]', fit_models)
            self.nodeCount =1

        # store the resulting tree, in form of nested dictionary
        self.tree = root

        # score the tree
        self.score_tree(self.tree)

    def get_split(self, mask, features, depth, split_criteria, epsilon, gen_tree, min_size, fit_models):
        dataset = self.dataset
        if split_criteria == 'fit_improvement':
            best_feature, best_value, best_score, best_groups = self.choose_split_fit_improvement(mask, features, min_size, fit_models)

        elif split_criteria == 'chisquare':
            best_feature, best_value, best_score, best_groups = self.choose_split_chisquare(mask, features, min_size)

        elif split_criteria == 'random':
            best_feature, best_value, best_score, best_groups = self.choose_split_random(mask, features, epsilon, depth,
                                                                                         min_size)

        elif (split_criteria == 'from tree') or (split_criteria == 'from tree test'):
            try:
                best_feature = gen_tree['feature']
                best_value = gen_tree['value']
                best_groups = self.test_split(best_feature, best_value, mask)

                if split_criteria == 'from tree':
                    if len(dataset[eval(best_groups[0])].index) < min_size or len(
                            dataset[eval(best_groups[1])].index) < min_size:
                        best_feature, best_value, best_groups = 999, 999, None
            except:
                best_feature, best_value, best_groups = 999, 999, None

        return {'feature': best_feature, 'value': best_value, 'groups': best_groups}

    def choose_split_random(self, mask, features, epsilon, depth, min_size):
        dataset = self.dataset

        rand = np.random.random()
        # if rand is less than epsilon, randomly split, otherwise, go by accuracy
        if rand < epsilon:
            rand = np.random.random()

            if (rand > 0.8) & (depth > 0):
                # randomly terminate 20% of the time, but not at node 0
                best_feature, best_value, best_groups = 999, 999, None
            else:
                if depth == 0:
                    # at node zero randomly choose feature - value pair, keep trying if not valid
                    best_value = None
                    while best_value is None:
                        try:
                            best_feature = np.random.choice(features)
                            vals = (dataset[eval(mask)] if mask else dataset).sort_values(by=best_feature)[
                                best_feature].drop_duplicates().values
                            vals = vals[1:]  # so we can't choose the min value
                            best_value = np.random.choice(vals)
                            best_groups = self.test_split(best_feature, best_value, mask)
                            if len(dataset[eval(best_groups[0])].index) < min_size or len(
                                    dataset[eval(best_groups[1])].index) < min_size:
                                best_value = None
                        except ValueError:
                            pass
                else:
                    # at other nodes can easily choose random feature + value
                    best_feature = np.random.choice(features)
                    best_value = np.random.choice(
                        (dataset[eval(mask)] if mask else dataset)[best_feature].drop_duplicates().values)

                    # split by the chosen feature - value
                    best_groups = self.test_split(best_feature, best_value, mask)

                    # if one subgroup empty, invalid split
                    if len(dataset[eval(best_groups[0])].index) < min_size or len(
                            dataset[eval(best_groups[1])].index) < min_size:
                        best_feature, best_value, best_groups = 999, 999, None
            best_score = None
        else:
            #print(f'not doing random split epsilon={epsilon} rand = {rand}' )
            # if not randomly split, choose best calculated split
            best_feature, best_value, best_score, best_groups = self.choose_split_chisquare(mask, features, min_size)

        return best_feature, best_value, best_score, best_groups

    def choose_split_fit_improvement(self, mask, features, min_size, fit_models):
        dataset = self.dataset

        # fit the model to the parent group
        r = self.get_response(dataset[eval(mask)] if mask else dataset)

        if fit_models:
            model = self.fit_model(dataset[eval(mask)] if mask else dataset)
            if(self.verbose):
                print('fitting data to parent partition')

        # initialise best split
        best_feature, best_value, best_score, best_groups = 999, 999, 0, None

        # loop through feature values
        # test_statistic = 0
        for feature in features:
            values = (dataset[eval(mask)] if mask else dataset)[feature].drop_duplicates().values
            for val in values:

                # make split
                left, right = self.test_split(feature, val, mask)
                if len(dataset[eval(left)].index) < min_size or len(dataset[eval(right)].index) < min_size:
                    continue

                # fit models to each subgroup
                r_left = self.get_response(dataset[eval(left)])
                r_right = self.get_response(dataset[eval(right)])

                # get the RMSE between model above and each potential split response
                # score = np.sqrt(np.sum((r_left.response - model.expected.exp) ** 2)
                #                                           / len(r_left.response.values))
                # calculate test statistic
                # n = size of observations
                # delta_Wj = jacobian * np.sqrt(n/i) * score
                # test_statistic = (Ic/n)*np.sqrt()**2

                #####

                if fit_models:
                    if(self.verbose):
                        print('fitting models to children')
                    model_left = self.fit_model(dataset[eval(left)])
                    model_right = self.fit_model(dataset[eval(right)])

                    # if bad model fit, print warning so it can be investigated
                    mse_l = model_left.total_rmse
                    mse_r = model_right.total_rmse
                    if (mse_l > 2e-2) & (mse_r > 2e-2):
                        print('Warning: not a good model fit. Info:')
                        print('feature:', feature, 'value:', val)
                        print('rmses:', mse_l, mse_r)
                        print('response:')
                        print(r_left)
                        print(r_right)
                        # continue

                    # calculate how much better the child models did at explaining the child group than
                    # the parent model did at explaining the child group
                    left_child_error = np.sqrt(np.sum((r_left.response - model_left.expected.exp) ** 2)
                                               / len(r_left.response.values))
                    right_child_error = np.sqrt(np.sum((r_right.response - model_right.expected.exp) ** 2) \
                                                / len(r_right.response.values))
                    left_parent_error = np.sqrt(np.sum((r_left.response - model.expected.exp) ** 2)
                                                / len(r_left.response.values))
                    right_parent_error = np.sqrt(np.sum((r_right.response - model.expected.exp) ** 2)
                                                 / len(r_right.response.values))
                    diff_between_errors = [(left_parent_error - left_child_error), (right_parent_error - right_child_error)]
                    score = max(diff_between_errors)
                else:
                    left_child_error = 0
                    right_child_error = 0
                    left_parent_error = np.sqrt(np.sum((r_left.response - r.response) ** 2)
                                                / len(r_left.response.values))
                    right_parent_error = np.sqrt(np.sum((r_right.response - r.response) ** 2)
                                                 / len(r_right.response.values))
                    diff_between_errors = [(left_parent_error - left_child_error), (right_parent_error - right_child_error)]
                    score = max(diff_between_errors)
                ######

                # we want the split that provides the best improvement for a certain subgroup
                # also don't want it to produce a split with an avrg difference of less than 0.5%
                if (score > best_score) & (score > self.sensitivity):
                    best_score = score
                    best_value = val
                    best_feature = feature
                    best_groups = left, right

        return best_feature, best_value, best_score, best_groups

    def split(self, node, features, max_depth, min_size, split_criteria, depth, epsilon, gen_tree, fit_models):
        #print(f'in tr.split fit_models={fit_models}')
        dataset = self.dataset
        # recursively split to produce tree
        if not node['groups']:
            if(self.verbose):
                print('no good split')
            return
        left, right = node['groups']
        del (node['groups'])
        # check for a no split
        if dataset[eval(left)].empty or dataset[eval(right)].empty:
            node['left'] = node['right'] = self.to_terminal(f'{left} & {right}', fit=fit_models)
            return
        # check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left, fit=fit_models), self.to_terminal(right, fit=fit_models)
            return
        # process left child

        if gen_tree:
            try:
                n = self.get_split(left, features, depth, split_criteria, epsilon, gen_tree['left'], min_size, fit_models)
            except:
                print(gen_tree)
                raise KeyError
        else:
            n = self.get_split(left, features, depth, split_criteria, epsilon, gen_tree, min_size, fit_models)
        if not n['groups']:
            node['left'] = self.to_terminal(left, fit=fit_models)
        else:
            node['left'] = n
            if gen_tree:
                self.split(node['left'], features, max_depth, min_size, split_criteria, depth + 1, epsilon,
                           gen_tree['left'], fit_models)
            else:
                self.split(node['left'], features, max_depth, min_size, split_criteria, depth + 1, epsilon, gen_tree, fit_models)
        # process right child
        if gen_tree:
            n = self.get_split(right, features, depth, split_criteria, epsilon, gen_tree['right'], min_size, fit_models)
        else:
            n = self.get_split(right, features, depth, split_criteria, epsilon, gen_tree, min_size, fit_models)
        if not n['groups']:
            node['right'] = self.to_terminal(right, fit=fit_models)
        else:
            node['right'] = n
            if gen_tree:
                self.split(node['right'], features, max_depth, min_size, split_criteria, depth + 1, epsilon,
                           gen_tree['right'], fit_models)
            else:
                self.split(node['right'], features, max_depth, min_size, split_criteria, depth + 1, epsilon, gen_tree, fit_models)

    def to_terminal(self, mask, fit):
        #print(f'in tr.to_terminal fit={fit}')
        dataset = self.dataset
        r = self.get_response(dataset[eval(mask)])

        if (fit==True):
            #print('in subbranch for fit true')
            response_model = self.fit_model(dataset[eval(mask)])
            expected = response_model.expected.exp.tolist()
            model_inf = response_model.best_models
        else:
            expected = r.response.tolist()
            model_inf = {}

        group_size = len(dataset[eval(mask)].index)
        results = {'model_info': model_inf,
                   'expected': expected,
                   'group_size': group_size,
                   'mask': mask
                   }
        return results

    def score_tree(self, tree):
        # get list of pandas dataframe masks (as string) for the resulting subgroups
        # along with model info for those groups
        self.get_end_nodes(tree)

        # calculate the "balanced accuracy" i.e. avrg recall
        self.accuracy = score_tree(self.dataset, self.end_nodes, self.days)

    def get_end_nodes(self, node):
        try:
            self.get_end_nodes(node['left'])
            self.get_end_nodes(node['right'])
        except KeyError:
            self.end_nodes.append([node['mask'], node['expected'], node['model_info']])

    def get_response(self, group):
        return get_response(group, self.days)

    def fit_model(self, df):
        response_model = mr.ModelResponse()
        if 'failures' in df:
            #print('fitting model using flexmix')
            response_model.fit(df, self.intervention_times, self.days, self.verbose)
        else:
            #print('fitting models')
            r = self.get_response(df)
            response_model.fit(r, self.intervention_times,self.days,self.verbose)
            response_model.predict(r, self.intervention_times)
        return response_model

    def test_split(self, feature, value, dataset_mask):
        if dataset_mask:
            left = f'{dataset_mask} & (dataset.{feature} < {value})'
            right = f'{dataset_mask} & (dataset.{feature} >= {value})'
        else:
            left = f'(dataset.{feature} < {value})'
            right = f'(dataset.{feature} >= {value})'
        return left, right

    def choose_split_chisquare(self, mask, features, min_size):
        dataset = self.dataset

        p_values = {}
        key = 0
        val_dict = {}
        for feature in features:
            p_values[feature] = {}
            values = (dataset[eval(mask)] if mask else dataset)[feature].drop_duplicates().values
            for val in values:
                val_dict[str(key)] = val

                # make split
                left, right = self.test_split(feature, val, mask)
                if len(dataset[eval(left)].index) < min_size or len(dataset[eval(right)].index) < min_size:
                    continue

                r_left = self.get_response(dataset[eval(left)])
                r_right = self.get_response(dataset[eval(right)])

                expected_left = r_left.response.values
                expected_right = r_right.response.values

                # fit models to each subgroup
                # model_left = self.fit_model(r_left)
                # model_right = self.fit_model(r_right)
                #
                # # if bad model fit, print warning so it can be investigated
                # mse_l = model_left.total_rmse
                # mse_r = model_right.total_rmse
                # if (mse_l > 2e-2) & (mse_r > 2e-2):
                #     print('Warning: not a good model fit. Info:')
                #     print('feature:', feature, 'value:', val)
                #     print('rmses:', mse_l, mse_r)
                #     print('response:')
                #     print(r_left)
                #     print(r_right)
                #     # continue
                # expected_left = model_left.expected.exp.values
                # expected_right = model_right.expected.exp.values

                mult = np.min([len(dataset[eval(left)].index), len(dataset[eval(right)].index)])
                l = (np.append(expected_left, 1-np.sum(expected_left)) * mult).astype(int)
                r = (np.append(expected_right, 1-np.sum(expected_right)) * mult).astype(int)
                l[l == 0] = 1
                r[r == 0] = 1
                c, p = chisquare(r, l)

                # print(feature + ' < ' + str(val) + ' p-value= ' + str(p))
                if p < self.sensitivity:
                    p_values[feature][str(key)] = p
                    key += 1

            if len(p_values[feature]) == 0:
                del (p_values[feature])

        # if at least one feature in p_values, choose one of those features
        if len(p_values) > 0:
            f = list(p_values.keys())[0]
            v = list(p_values[f].keys())[0]
        else:
            return 999, 999, 999, None

        # save this feature value pair as m
        m = f, v
        # find the feature value pair that produced the smallest p_value, replacing m with that pair
        for feature in list(p_values.keys()):
            if len(p_values[feature].keys()) > 1:
                v = min(p_values[feature], key=lambda k: p_values[feature][k])
            else:
                v = list(p_values[feature].keys())[0]
            if p_values[feature][v] < p_values[m[0]][m[1]]:
                m = feature, v
        # if any p_values are close to the minimum p_value, randomly choose one of those
        tol = 1e-05
        l = []
        for feature in list(p_values.keys()):
            for val in list(p_values[feature].keys()):
                if np.absolute(p_values[feature][val] - p_values[m[0]][m[1]]) < tol:
                    l.append([feature, val])
        if len(l) > 0:
            if len(l) > 1:
                r = np.random.randint(len(l))
            else:
                r = 0
            feat, val = l[r][0], val_dict[l[r][1]]
            v = l[r][1]

        else:
            feat, val = m[0], val_dict[m[1]]
            v = m[1]

        left, right = self.test_split(feat, val, mask)
        groups = left, right
        return feat, val, p_values[feat][v], groups
    
    def setNodeCount(self):
        #print(f"in setNodeCount with end nodes {self.end_nodes}")
        #print(f"end nodes has {len(self.end_nodes)} entries")
        #print(f"self.treee is {self.tree}")
        if(self.nodeCount==1):
            #print(f"self.nodeCount already set to 1")
            pass
        else:
            #print(f"setting nodeCount len self.end_nodes")
            self.nodeCount = len(self.end_nodes)

 
 

    # def choose_split_chisquare(self, dataset, features):
    #
    #     p_values = {}
    #     best_feature, best_value, best_score, best_groups = 999, 999, 999, None
    #     key = 0
    #     val_dict = {}
    #     for feature in features:
    #         p_values[feature] = {}
    #         for val in dataset[feature].drop_duplicates().values:
    #             val_dict[str(key)] = val
    #
    #             left, right = self.test_split(feature, val, dataset)
    #             print(feature, val)
    #             if left.empty or right.empty:
    #                 continue
    #
    #             r_left = self.get_response(left)
    #             r_right = self.get_response(right)
    #             model_left = self.fit_model(r_left)
    #             model_right = self.fit_model(r_right)
    #
    #             mse_l = model_left.total_rmse
    #             mse_r = model_right.total_rmse
    #             if (mse_l > 2e-3) & (mse_r > 2e-3):
    #                 print('Could not fit a good model to this split')
    #                 continue
    #             mult = 300
    #             c, p = chisquare((model_left.expected.exp.values * mult).astype(int),
    #                              (model_right.expected.exp.values * mult).astype(int))
    #             print(feature + ' < ' + str(val) + ' p-value= ' + str(p))
    #             p_values[feature][str(key)] = p
    #             key += 1
    #         if len(p_values[feature]) == 0:
    #             del (p_values[feature])
    #     # if at least one feature in p_values, choose one of those features
    #     if len(p_values) > 0:
    #         f = list(p_values.keys())[0]
    #         v = list(p_values[f].keys())[0]
    #     else:
    #         return {'feature': best_feature, 'value': best_value, 'groups': best_groups} # change this
    #
    #     # save this feature value pair as m
    #     m = f, v
    #     # find the feature value pair that produced the smallest p_value, replacing m with that pair
    #     for feature in list(p_values.keys()):
    #         if len(p_values[feature].keys()) > 1:
    #             v = min(p_values[feature], key=lambda k: p_values[feature][k])
    #         else:
    #             v = list(p_values[feature].keys())[0]
    #         if p_values[feature][v] < p_values[m[0]][m[1]]:
    #             m = feature, v
    #     # if any p_values are close to the minimum p_value, randomly choose one of those
    #     tol = 1e-05
    #     l = []
    #     for feature in list(p_values.keys()):
    #         for val in list(p_values[feature].keys()):
    #             if np.absolute(p_values[feature][val] - p_values[m[0]][m[1]]) < tol:
    #                 l.append([feature, val])
    #     if len(l) > 0:
    #         if len(l) > 1:
    #             r = np.random.randint(len(l))
    #         else:
    #             r = 0
    #         feat, val = l[r][0], val_dict[l[r][1]]
    #         v = l[r][1]
    #
    #     else:
    #         feat, val = m[0], val_dict[m[1]]
    #         v = m[1]
    #
    #     left, right = self.test_split(feat, val, dataset)
    #     groups = left, right
    #     return feat, val, p_values[feat][v], groups

    # -- methods here aren't currently used --
    # def plot_tree(self, dataset):
    #     # not being used since new visualisation method
    #     self.print_tree(self.tree, dataset)
    #
    # def print_tree(self, node, dataset, depth=0):
    #     try:
    #         print('%s' % (2 * depth * ' ') + '[' + node['feature'] + '<' + str(node['value']) + ']')
    #         left = dataset.loc[eval('(' + 'dataset.' + node['feature'] + '<' + str(node['value']) + ')')]
    #         right = dataset.loc[eval('(' + 'dataset.' + node['feature'] + '>=' + str(node['value']) + ')')]
    #         r_left = self.get_response(left)
    #         r_right = self.get_response(right)
    #         model_l = self.fit_model(r_left)
    #         model_r = self.fit_model(r_right)
    #         plt.plot(model_l.expected.t, model_l.expected.exp, label=node['feature'] + '<' + str(node['value']))
    #         plt.plot(model_r.expected.t, model_r.expected.exp, label=node['feature'] + '>=' + str(node['value']))
    #         plt.legend()
    #         plt.text(0.02, 0.5, ' ', fontsize=14, transform=plt.gcf().transFigure)
    #         plt.subplots_adjust(left=(0 + depth * 0.2))
    #
    #         plt.show()
    #
    #         self.print_tree(node['left'], left, depth + 1)
    #         self.print_tree(node['right'], right, depth + 1)
    #     except:
    #         print(' ')
    #         # print('%s[%s]' % (depth * ' ', 'end node'))


def get_response(group, days):
    # get the number of episodes still running at each number of days into an episode
    num_logins = np.zeros(days.size)
    num_eps = np.zeros(days.size) + group.shape[0]
    for i, day in enumerate(days):
        # num_eps[i] = (group.episodeDuration >= i).sum()
        num_logins[i] = (group.dayOfEp == day).sum()

    # proportion of open eps that log in each day
    logins = num_logins / num_eps

    data = pd.DataFrame(np.array([logins, days]).T, columns=['response', 't'])
    data.response.fillna(0, inplace=True)

    return data


def score_tree(dataset, end_nodes, days):
    if (dataset.dayOfEp == -1).sum():
        classes = len(days) + 1
        cl = days.tolist()
        cl.append(-1)
        code_map = pd.Series(np.arange(len(days) + 1), index=cl)
    else:
        classes = len(days)
        cl = days.tolist()
        code_map = pd.Series(np.arange(len(days)), index=cl)

    # loop through leaf nodes / resulting subgroups
    num_each_class = np.zeros(classes)
    num_recalled = np.zeros(classes)

    # test = np.zeros(classes)
    for i in range(len(end_nodes)):
        # get raw/unadjusted probabilities of login on each day
        raw_probs_i = end_nodes[i][1]
        # if dataset has customers who never respond, add this
        if (dataset.dayOfEp == -1).sum():
            raw_probs_i.append(1 - np.sum(raw_probs_i))

        # for each of these, need to multiply by the proportion of each class

        # get the mask for this subgroup
        mask = eval(end_nodes[i][0])

        # s = len(dataset.loc[mask].index) # size of subgroup
        # sc = np.zeros(len(cl))
        # for i, c in enumerate(cl):
        #     sc[i] = len(dataset.loc[mask&(dataset.dayOfEp == c)].index)  # size of class in subgroup
        # test = test + (np.array(raw_probs_i) * (sc / s))

        # make a list of lists num_classes * size of subgroup
        raw_probs = []
        s = len(dataset.loc[mask].index)
        for k in range(s):
            raw_probs.append(raw_probs_i)

        targets = dataset[mask].dayOfEp.map(code_map).values
        
        # robustness checks
        if (targets.dtype != np.dtype('int')):
            print(f"casting targets from  {targets.dtype} to int")
            targets = targets.astype(int)
        

        # get the 'number recalled' for this subgroup and add it to the total number recalled
        if targets.size != 0:
            one_hot = np.eye(classes)[targets]
            num_each_class = num_each_class + np.sum(one_hot, axis=0)
            num_recalled = num_recalled + np.sum((np.array(raw_probs) * one_hot), axis=0)

    if len(np.argwhere(num_each_class == 0)) > 0:
        del_idx = np.argwhere(num_each_class == 0)
        num_each_class = np.delete(num_each_class, del_idx)
        num_recalled = np.delete(num_recalled, del_idx)
    # calculate the "balanced accuracy" i.e. avrg recall
    return np.mean(num_recalled / num_each_class)
    # return np.sum(test/len(end_nodes))/classes


# class with methods to get key information from resulting leaf nodes
class SubGroup:
    def __init__(self, model_info, intervention_times):
        self.model_info = model_info
        self.intervention_times = intervention_times

    def get_proba(self, t, leave_out=None):
        # get unadjusted probabilities of customer in subgroup responding on day t[_]
        # you can specify to leave out certain interventions with leave_out
        proba = pd.DataFrame(np.array([t, np.zeros(len(t))]).T, columns=['t', 'proba'])

        try:
            for i in range(len(self.intervention_times)):
                model_type = self.model_info['curve' + str(i)]['best_model']
                params = self.model_info['curve' + str(i)]['parameters']
                model = mr.choose_model(model_type, params)

                s = len(proba.loc[(proba.t >= self.intervention_times[i])].index)

                if not leave_out or i not in leave_out:
                    proba.loc[(proba.t >= self.intervention_times[i]), 'proba'] = (
                            proba.loc[(proba.t >= self.intervention_times[i]), 'proba'].values
                            + model.predict(t[:s].astype('float64')))
        except TypeError:
            proba['proba'] = self.model_info

        return proba

    def get_adjusted_proba(self, t, leave_out=None):
        adj_proba = self.get_proba(t, leave_out)
        responded_so_far = 0
        for t, p in adj_proba.itertuples(index=False):
            adj_proba.loc[(adj_proba.t == t), 'proba'] = p / (1 - responded_so_far)
            responded_so_far = responded_so_far + p

        return adj_proba

    def proba_response_within(self, t, days, leave_out=None):
        proba = self.get_proba(t, leave_out)
        return proba.loc[proba.t < days, 'proba'].sum()


def plot_leaf_nodes(dataset, days, end_nodes, y_lim, figsize, test_set=None, merged_nodes=False, cols=3):
    N = len(end_nodes)
    rows = int(np.ceil(N / cols))
    if rows < 2:
        rows = 2
    if cols < 2:
        cols = 2
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    for i in range(N):
        mask = eval(end_nodes[i][0])
        if test_set is not None:
            mask_test = eval(end_nodes[i][0].replace('dataset', 'test_set'))
            s_test = len(test_set.loc[mask_test].index)

        s = len(dataset.loc[mask].index)

        row = int(np.floor(i / cols))
        col = i % cols
        try:
            axs[row][col].bar(days, end_nodes[i][1])
        except ValueError:
            axs[row][col].bar(np.append(days,'None'), end_nodes[i][1])

        title = f'{i+1} (size: {s})'
        if merged_nodes:
            title = f'{end_nodes[i][3]} (size: {s})'
        if test_set is not None:
            # do get response instead
            test_r = get_response(test_set.loc[mask_test], days)
            axs[row][col].bar(days, test_r.response, alpha=0.6)
            title = f'{i+1} (train: {s}, test: {s_test})'
            if merged_nodes:
                title = f'{(np.array(end_nodes[i][3])+1).tolist()} (train: {s}, test: {s_test})'
        axs[row][col].set_ylim(0, y_lim)
        axs[row][col].set_title(title)


def make_prob_df(end_nodes, intervention_times, days, time_bins=[2, 8, 22]):
    cols = ['subgroup']

    for tb in time_bins:
        cols.append(f'prob_response_{tb}_days')

    cols.append('prob_response_total')

    df = pd.DataFrame([], columns=cols)

    for i, node in enumerate(end_nodes):
        if node[2]:
            sg = SubGroup(node[2], intervention_times)
        else:
            sg = SubGroup(node[1], intervention_times)

        dta = [i+1]
        for tb in time_bins:
            d = sg.proba_response_within(days, tb)
            dta.append(d)

        total = sg.proba_response_within(days, max(days) + 2)
        dta.append(total)

        df = df.append(pd.DataFrame([dta], columns=cols), ignore_index=True)

    return df


def get_json_tree(tree, directory, days):
    d = {}
    get_dict(tree, d, days, directory)
    with open(f'{directory}/data/tt.json', 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=4)


def get_dict(node, d, days, directory, c=0, parent="null"):
    try:
        d['parent'] = parent
        d['name'] = node['feature'] + '<' + str(node['value'])
        d['edge_name'] = "null"

        d['children'] = [{}, {}]

        c = get_dict(node['left'], d['children'][1], days, directory, c, node['feature'] + '<' + str(node['value']))
        c = get_dict(node['right'], d['children'][0], days, directory, c, node['feature'] + '<' + str(node['value']))
        return c
    except:
        c += 1
        d['children'] = []
        try:
            plt.bar(days, node['expected'])
            plt.ylim(0, 1)
            plt.savefig(f'{directory}/data/node_{c}.png')
            plt.close()
            d['img'] = f'data/node_{c}.png'
            d['node_number'] = str(c)
        except:
            pass
        return c


def cluster_nodes(leaf_nodes, df, eps):
    samples = sample_nodes(df, leaf_nodes)
    distance_matrix = pairwise_distance(samples)
    clustering = DBSCAN(eps=eps, min_samples=2, metric='precomputed').fit(distance_matrix)

    return clustering.labels_


def pairwise_distance(samples):
    D = np.zeros((len(samples), len(samples)))
    for i in range(len(samples)):
        for j in range(len(samples)):
            D[i, j] = 1 - distance(samples[i], samples[j])
    return D


def sample(df, node):
    dataset = df
    mask = eval(node[0])
    size = len(dataset[mask].index)
    N = np.round(size * np.array(node[1]))
    samples = []
    for i, n in enumerate(N):
        for _ in range(int(n)):
            samples.append(i)

    # if (dataset.dayOfEp == -1).sum():
    #     p_no_response = 1 - np.sum(np.array(node[1]))
    #     n = np.round(size * p_no_response)
    #     for _ in range(int(n)):
    #         samples.append(max(samples) + 1)

    return np.array(samples)


def sample_nodes(df, nodes):
    samples = []
    for i, node in enumerate(nodes):
        samples.append(sample(df, node))
    return samples


def distance(x, y):
    #_, p = chisquare(r, l)
    _, ks_p_value = st.ks_2samp(x, y)
    # st.anderson_ksamp((x, y))
    return ks_p_value


def find_similar_leaf_nodes(leaf_nodes, dataset, days, intervention_times, eps=0.7):
    labels = cluster_nodes(leaf_nodes, dataset, eps)
    labels = pd.DataFrame(np.array([np.arange(len(labels)), labels]).T, columns=['node', 'cluster'])

    similar_nodes = []
    clusters = labels.cluster.values
    for c in np.unique(clusters):
        if c == -1:
            continue
        similar_nodes.append(labels[labels.cluster == c].node.tolist())

    merged_nodes = copy.deepcopy(leaf_nodes)
    for i in range(len(merged_nodes)):
        merged_nodes[i].append([i])

    delete = []
    for sn in similar_nodes:
        for i, n in enumerate(sn):
            if i == 0:
                continue
            merged_nodes[sn[0]][0] = '(' + merged_nodes[sn[0]][0] + ') | (' + merged_nodes[n][0] + ')'
            merged_nodes[sn[0]][3].append(n)
            delete.append(n)
        # fit the curve + get expected values for this new mask
        r = get_response(dataset[eval(merged_nodes[sn[0]][0])], days)
        response_model = mr.ModelResponse()
        response_model.fit(r, intervention_times)
        response_model.predict(r, intervention_times)

        rm = response_model.expected.exp.tolist()
        if (dataset.dayOfEp==-1).sum():
            rm.append(1-np.sum(rm))
        merged_nodes[sn[0]][1] = rm
        merged_nodes[sn[0]][2] = response_model.best_models

    mn = np.array(merged_nodes)
    merged_nodes = np.delete(mn, delete, axis=0).tolist()

    return merged_nodes




