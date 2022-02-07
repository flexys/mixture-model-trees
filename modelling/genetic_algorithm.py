import pandas as pd
import numpy as np
import modelling.tree as tr
import copy
import time
from multiprocessing import Pool, cpu_count


def gp(dataset, population_size=1, intervention_times=[], features=[],  max_depth=1, min_size=1, generations=50,tournament_size=5,
       prune_tol=0.003, mutation_rate=0.15, crossover_rate=0.6, min_improvement=0.003, gens_no_improvement=30, max_possible_fitness = 101.00, epsilon=1.,
       file_name='gp_results', test_set=None, days=None, fit_models=False, verbose=True, internal_parallelism=True):
    """

    :param dataset: pandas dataframe with features, dayOfEp. One row per episode
    :param population_size: integer size of population of trees
    :param intervention_times: list of days that mark the start of each curve section
    :param features: list of strings of the features you wish to make trees from
    :param generations: integer maximum number of generations
    :param max_depth: integer maximum depth the trees can be
    :param min_size: integer minimum size subgroup can be
    :param tournament_size: integer number of trees to compete against each other in tournaments
    :param prune_tol: float, choose tree with lower number of nodes if score of 2 trees is this amount or more similar
    :param mutation_rate: float proportion of trees each generation to mutate
    :param crossover_rate: float proportion of trees each generation to crossover
    :param min_improvement: float minimum improvement we must see each generation else stop (for x consecutive gens)
    :param epsilon: float, for initial population splits are made randomly epsilon*100% of the time
    :param file_name: string name of file to save results as (three files produced: filename_tree.txt, filename_nodes.txt, filename_acc.txt)
    :param gens_no_improvement: stop after this no. of consecutive generations with less than min_improvement in best tree
    :param fit_models: boolean, whether to fit models to the data while training or not. This speeds up training
                       considerably if False, and if a good model can be fit to every subgroup then this shouldn't affect
                       the result much as they are always fit on the final tree.
    :param days: if necessary, specify an array of days
    :param test_set: test set pandas dataframe
    :param max_possible_fitness: float,  maximum attainable fitness- can be used for earlty stopping
    :param verbose: Boolean  controls output messages for debugging vs faster batch running
    :param internal_parallelism: Boolean - has to be false if the function might be called in parallel 
    :return: tr.Tree class with attributes such as the best tree in dict form, end node information
    """

    # if (verbose):
        #print(locals())


    if(verbose):
        print(f'Generating initial population... fit_models={fit_models}')
        print(f'features are: {features}')
  
    # get days
    if days is None:
        days = dataset.loc[dataset.dayOfEp != -1].sort_values(by='dayOfEp').dayOfEp.drop_duplicates().values.astype(int)

    if (dataset.dayOfEp == -1).sum():
        prune = prune_tol * (1 / (len(days) + 1))
        min_improvement = min_improvement * (1 / (len(days) + 1))
    else:
        prune = prune_tol * (1 / len(days))
        min_improvement = min_improvement * (1 / len(days))

    tic = time.perf_counter()

    if fit_models and internal_parallelism:
        # build initial population of trees, adding each build as a task to be done in parallel
        pool = Pool(processes=(cpu_count()))

        results = [()] * population_size
        for i in range(population_size):
            results[i] = pool.apply_async(build_trees, [dataset, features, max_depth,
                                                          min_size, intervention_times, days, epsilon, fit_models])
        # store those trees in a list
        trees = [res.get(timeout=1000) for res in results]
    else:
        # parallel tasks not needed when no model fitting
        trees = [()] * population_size
        for i in range(population_size):
            trees[i] = build_trees(dataset, features, max_depth,
                                                          min_size, intervention_times, days, epsilon, fit_models)

    toc = time.perf_counter()

    if(verbose):
        print(f"Took {toc - tic:0.4f} seconds. That's {((toc - tic) / 60):0.4f} minutes.")
        print('Done generating initial population.')

    no_improvement_count = 0
    test_accuracies = np.zeros(generations)
    train_accuracies = np.zeros((3, generations))
    # loop through generations
    gensUsed = 1
    for gen in range(generations):
        # select the best tree in our list of tree
        best = select_best(trees, prune)

        # if we know the max possible fitness we can stop as soon as we reach it    
        if (  max_possible_fitness - best[1] <= prune_tol):
            if(verbose):
                print(f'Stopping, attained max possible fitness in {gen} generations')
            break 
            
        # if gen % 5 == 0:
        #     print('fitness scores:')
        #     for tre in trees:
        #         print(tre[1])
        #     print('...')

        # record how many times in a row we have had < min_improvement
        if gen > 0:
            improvement = best[1] - old_best[1]
            if improvement < min_improvement:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            if no_improvement_count == gens_no_improvement:
                if(verbose):
                    print(f'Stopping, no significant improvement for {gens_no_improvement} consecutive generations')
                break

        if test_set is not None:
            tree_model = tr.Tree(dataset, intervention_times, days)
            tree_model.build_tree(features, max_depth, min_size, split_criteria='from tree', gen_tree=best[0], fit_models=fit_models)
            test_score = tr.score_tree(test_set, tree_model.end_nodes, days)
            test_accuracies[gen] = test_score

        train_accuracies[0][gen] = worst_score(trees)
        train_accuracies[1][gen] = average_score(trees)
        train_accuracies[2][gen] = best[1]

        if(verbose):
            print('----')
            print('generation:', gen)
            print('best accuracy:', best[1])
            print('---')
            print('Holding tournaments, storing winners')

    

 

        # hold tournaments to get new, better population of trees

        parents = [()] * population_size
        parents[0] = (copy.deepcopy(best[0]), best[1]) # keep the best tree
        for i in range(population_size):
            if i > 0:
                winner = tournament(trees, tournament_size, prune, gen)
                parents[i] = (copy.deepcopy(winner[0]), winner[1])

        # create offspring by swapping over subtrees
        if(verbose):
            print('Producing offspring...')
        offspring = [0] * population_size
        
        # add this line for the case of a single member local search
        if(population_size==1):
            offspring[0] = copy.deepcopy(best[0]) # keep the best tree
            
        for parent1 in range(population_size - 1):
            parent2 = parent1 + 1
            np.random.seed()
            rand = np.random.random()
            if rand < crossover_rate:
                invalid_tree = True
                while invalid_tree:
                    p1, p2 = combine(parents[parent1][0], parents[parent2][0])
                    try:
                        p1['left'], p2['left']
                        invalid_tree = False
                    except KeyError:
                        pass

                offspring[parent1], offspring[parent2] = p1, p2
            else:
                offspring[parent1], offspring[parent2] = parents[parent1][0], parents[parent2][0]
        if(verbose):
            print('Done producing offspring.')

        # mutate trees by replacing random node with randomly created subtree
        if(verbose):
            print('Mutating')
        for i, child in enumerate(offspring):
            np.random.seed()
            rand = np.random.random()
            if rand < mutation_rate:
                offspring[i] = mutate(child, dataset, features, intervention_times, days, max_depth, min_size, epsilon, False)

        if(verbose):
            print('Evaluating children...')
        tic = time.perf_counter()

        ##JIM 23/06 changed the condition below  because I wanted to be able to execute multiple runs in parallel rather than having parallism within EA
        if internal_parallelism:
            # parallelise the process of scoring the trees
            pool = Pool(processes=(cpu_count()))
            results = [()] * population_size
            for i, child in enumerate(offspring):
                results[i] = pool.apply_async(eval_tree, [child, dataset, features, max_depth,
                                                        min_size, intervention_times, days, fit_models])

            trees = [res.get(timeout=10000) for res in results]
        else:
             # don't parallelise if no model fitting
             trees = [()] * population_size
             for i, child in enumerate(offspring):
                 trees[i] = eval_tree(child, dataset, features, max_depth,
                                                             min_size, intervention_times, days, fit_models)

        toc = time.perf_counter()

        if(verbose):
            print('Done evaluating children.')
            print(f"Took {toc - tic:0.4f} seconds to train. That's {((toc - tic) / 60):0.4f} minutes.")

        # save the new best tree
        old_best = best
        gensUsed +=1
    # end of generations loop
    
    if(verbose):
        print(f'Returning best tree with accuracy: {best[1]}')

    # select best tree
    best = select_best(trees, prune)

    # need to re-build the tree to get the leaf node info
    #Always get final fitness usinfg fitted model
    tree_model = tr.Tree(dataset, intervention_times, days)
    tree_model.build_tree(features, max_depth, min_size, split_criteria='from tree', gen_tree=best[0], fit_models=True)

    if(verbose):
        print(f'after fitting models best now has accuracy {tree_model.accuracy}')
    
    # save leaf node info
    f = open(f"{file_name}_nodes.txt", "w")
    f.write(str(tree_model.end_nodes))
    f.close()

    # # make the tree save-able
    # simplified_tree = copy.deepcopy(tree_model.tree)
    # del_end_nodes(simplified_tree)

    # save the tree
    f = open(f"{file_name}_tree.txt", "w")
    f.write(str(tree_model.tree))
    f.close()

    # save accuracies
    f = open(f"{file_name}_acc.txt", "w")
    f.write(str((train_accuracies.tolist(), test_accuracies.tolist())))
    f.close()

    # write into the tree the number generations used to create it
    tree_model.evals = gensUsed 

    #Jim: added to call to set the nodecount of the tree
    tree_model.setNodeCount()
    return tree_model


def eval_tree(child, dataset, features, max_depth, min_size, intervention_times, days, fit_models):
    """ take in tree that may be a product of crossing over and/or mutating
        make this tree adhere to max_depth, min_size
        return that tree and its score
     """
    np.random.seed()
    tree_model = tr.Tree(dataset, intervention_times, days)
    tree_model.build_tree(features, max_depth, min_size, split_criteria='from tree', gen_tree=child, fit_models=fit_models)

    return tree_model.tree, tree_model.accuracy


def build_trees(dataset, features, max_depth, min_size, intervention_times, days, epsilon, fit_models):
    """ generate a tree where splits are made randomly epsilon*100% of the time
        return that tree and its score
     """
    np.random.seed()
    tree_model = tr.Tree(dataset, intervention_times, days)
    tree_model.build_tree(features, max_depth, min_size, split_criteria='random', epsilon=epsilon, fit_models=fit_models)
    return tree_model.tree, tree_model.accuracy


def select_best(trees, prune_tol):
    """ return (tree, score) tuple for tree with best score
     """
    best_fitness = 0
    for tree, fitness in trees:
        if fitness > best_fitness:
            best = tree, fitness
            best_fitness = fitness

    # penalise unnecessarily complex trees
    best_node_count = count_nodes(best[0], c=0)
    best_tree = copy.deepcopy(best)

    for tre in trees:
        node_count = count_nodes(tre[0], c=0)
        if (best[1] - tre[1]) < prune_tol and node_count < best_node_count:
            best_tree = copy.deepcopy(tre)
    return best_tree


def worst_score(trees):
    """
     """
    worst_fitness = 100
    for tree, fitness in trees:
        if fitness < worst_fitness:
            worst_fitness = fitness
    return worst_fitness


def average_score(trees):
    """
     """
    total_fitness = 0
    for tree, fitness in trees:
        total_fitness += fitness

    return total_fitness / len(trees)


def tournament(trees, tournament_size, prune_tol, gen):
    """ hold tournament of tournament_size
        return winner
    """
    # randomly select tournament_size trees from population
    np.random.seed()
    idx = np.random.randint(0, len(trees), tournament_size)
    competitors = [()] * tournament_size
    for i, x in enumerate(idx):
        competitors[i] = trees[x]
    # out of these randomly selected trees, choose best one
    best = select_best(competitors, prune_tol)
    return best


def combine(parent1, parent2):
    # choose random subtree from parent1
    parent1_subtree_key, _ = select_random_subtree(parent1, 'parent1')
    # choose random subtree from parent2
    parent2_subtree_key, _ = select_random_subtree(parent2, 'parent2')

    # swap those subtrees
    parent1_subtree = eval(f'copy.deepcopy({parent1_subtree_key})')
    parent2_subtree = eval(f'copy.deepcopy({parent2_subtree_key})')
    exec(f'{parent1_subtree_key} = parent2_subtree')
    try:
        exec(f'{parent2_subtree_key} = parent1_subtree')
    except KeyError:
        print('Error, output debug info')
        return parent1, parent2, parent1_subtree, parent2_subtree

    return parent1, parent2


def mutate(tree, dataset, features, intervention_times, days, max_depth, min_size, epsilon, fit_models):
    # create random tree
    tree_model = tr.Tree(dataset, intervention_times, days)
    tree_model.build_tree(features, max_depth - 1, min_size, split_criteria='random', fit_models=fit_models)

    # select random node in existing tree
    random_node_key, node_number = select_random_subtree(tree, 'tree')

    # 50% swap subtree for random one, 50% of the time only swap node's feature-value for random
    np.random.seed()
    rand = np.random.rand()
    if rand < 0.35:
        try:
            # only replace the feature and value of the node
            eval(f'{random_node_key}[\'feature\']')
            exec(f'{random_node_key}[\'feature\'] = copy.deepcopy(tree_model.tree[\'feature\'])')
            exec(f'{random_node_key}[\'value\'] = copy.deepcopy(tree_model.tree[\'value\'])')
        except KeyError:
            # replace that node with the randomly created subtree
            exec(f'{random_node_key} = copy.deepcopy(tree_model.tree)')
    elif node_number > 2 and rand > 0.9:
        terminal_node = '\'terminal\''
        exec(f'{random_node_key} = {terminal_node}')
    else:
        # replace that node with the randomly created subtree
        exec(f'{random_node_key} = copy.deepcopy(tree_model.tree)')

    return tree


def select_random_subtree(tree, tree_string):
    # count the number of nodes the tree has
    count = count_nodes(tree, c=0)

    # randomly choose one of those nodes
    np.random.seed()
    rand = np.random.randint(count)

    # get a list of 'left's and 'right's to access that node
    node_key = []
    node_key = select_node(tree, node_key, rand + 1)

    # turn that list into the actual key to get the subtree from that node
    key = tree_string
    if node_key:
        for node in node_key:
            key = f'{key}[\'{node}\']'
    return key, rand


def count_nodes(node, c):
    # recursive function to count number of nodes in tree
    c += 1
    try:
        c = count_nodes(node['left'], c)
        c = count_nodes(node['right'], c)
        return c
    except KeyError:
        return c


def select_node(node, node_key, stop, c=0):
    # recursive function to get a list of 'left's and 'right's to access node
    c += 1
    if c == stop:
        return node_key
    try:
        node_key.append('left')
        nc = select_node(node['left'], node_key, stop, c)
        if type(nc) is tuple:
            node_key, c = nc
        else:
            return node_key
        del node_key[-1]

        node_key.append('right')
        nc = select_node(node['right'], node_key, stop, c)
        if type(nc) is tuple:
            node_key, c = nc
        else:
            return node_key
        del node_key[-1]

        return node_key, c
    except KeyError:
        del node_key[-1]
        return node_key, c


def prune_tree(tree, tree_string, dataset, features, intervention_times, days, prune_tol, fit_models=False):
    improvement = True
    while improvement:

        improvement = False
        base_tree = copy.deepcopy(tree)
        # count the number of nodes the tree has
        count = count_nodes(tree, c=0)
        print('node count:', count)

        tree_list = [()] * (count-1)
        for i in range(1, count):

            # get a list of 'left's and 'right's to access that node
            node_key = []
            node_key = select_node(tree, node_key, i + 1)

            # turn that list into the actual key to get the subtree from that node
            key = tree_string
            if node_key:
                for node in node_key:
                    key = f'{key}[\'{node}\']'

            terminal = '\'terminal\''
            exec(f'{key} = {terminal}')

            # evaluate the tree
            tree_model = tr.Tree(dataset, intervention_times, days)
            tree_model.build_tree(features, 20, 10, split_criteria='from tree', gen_tree=tree,
                                  fit_models=fit_models)
            # add it to list of trees
            tree_list[i-1] = (tree_model.tree, tree_model.accuracy)
            tree = copy.deepcopy(base_tree)

        best = select_best(tree_list, 0)

        #penalise unnecessarily complex trees
        best_node_count = count_nodes(best[0], c=0)
        best_tree = copy.deepcopy(best)

        for competitor in tree_list:
            node_count = count_nodes(competitor[0], c=0)
            if (best[1] - competitor[1]) < prune_tol and node_count < best_node_count:
                improvement = True
                best_tree = competitor

        tree = best_tree[0]

    tree_model = tr.Tree(dataset, intervention_times, days)
    tree_model.build_tree(features, 20, 10, split_criteria='from tree', gen_tree=best_tree[0],
                          fit_models=fit_models)

    return tree_model.tree, tree_model.accuracy


if __name__ == "__main__":

    # set variables

    dataset_file_path = 'data/synth_data.csv'

    features = ['feature_0',
                'feature_1',
                'feature_2',
                'feature_rand'
                ]

    days = np.arange(0, 30)

    intervention_times = [0, 10, 20]

    # load data

    df = pd.read_csv(dataset_file_path)

    # run evolutionary algorithm

    tic = time.perf_counter()
    tree_gp = gp(df,
                 population_size=150,
                 intervention_times=intervention_times,
                 features=features,
                 generations=700,
                 max_depth=7,
                 min_size=200,
                 tournament_size=5,
                 prune_tol=0.004,
                 mutation_rate=0.15,
                 crossover_rate=0.6,
                 min_improvement=0.004,
                 gens_no_improvement=300,
                 epsilon=1,
                 file_name='data/tree_results',
                 test_set=None,
                 days=days,
                 fit_models=False
                 )
    toc = time.perf_counter()

    print(f"Took {toc - tic:0.4f} seconds. That's {((toc - tic) / 60):0.4f} minutes.")

