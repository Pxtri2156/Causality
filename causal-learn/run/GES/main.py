from causallearn.search.ScoreBased.GES import ges
import numpy as np
import pandas as pd
# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import igraph as ig
import argparse
import os
import io

def read_data(data_path):
    # X = np.loadtxt(data_path, skiprows=1)
    X=pd.read_csv(data_path) ## Need to preprocessing
    # df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    cat_columns = X.select_dtypes(['object']).columns
    X[cat_columns] = X[cat_columns].apply(lambda x: pd.factorize(x)[0])
    bool_columns = X.select_dtypes(['boolean']).columns
    X[bool_columns] = X[bool_columns].replace({True: 1, False: 0})
    # print("After X: ", X)                                         
    X = X.to_numpy()
    return X

def predict_ges(args):
    X = read_data(args.data_path)
    # default parameters
    Record = ges(X)
    # or customized parameters
    Record = ges(X, score_func=args.score_func, maxP=args.maxP, parameters=args.parameters)
    return Record

def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()

def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    dag=True
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if not is_dag(B_est):
            # raise ValueError('B_est should be a DAG')
            dag=False
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size, 'dag': dag}

def decode_result(Record):
    graph = Record['G'].graph
    cp_graph = graph
    for i in range(graph.shape[0]):
        for j in range(i, graph.shape[1]):
            if graph[i][j]*graph[j][i]==1:
                 cp_graph[i][j], cp_graph[j][i] = 1, 1
            elif graph[i][j]*graph[j][i]== -1:
                # print("Do")
                if graph[i][j] == -1:
                    cp_graph[i][j]=1
                    cp_graph[j][i]=0
                else:
                    cp_graph[j][i]=1
                    cp_graph[i][j]=0
    return cp_graph

def main(args):
    # Load ground truth
    gt = pd.read_csv(args.gt_path, header=None).to_numpy()
    # Predict
    Record = predict_ges(args)
    graph = decode_result(Record)
    # Evaluate 
    acc = count_accuracy(gt, graph != 0 ) # Skip DAG
    print(acc)
    
    # Save results 
    results = f'Ground truth: \n {gt} \n Predict: \n {graph} \n Acc: {acc}'
    if not os.path.isdir(args.results):
        os.makedirs(args.results)
    results_file = os.path.join(args.results, 'results.txt')
    with open(results_file, "w") as f:
        f.write(results)
    # Visualize results
    visualize_path = os.path.join(args.results, 'graph.png')
    pyd = GraphUtils.to_pydot(Record['G'])
    pyd.write_png(visualize_path)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default='/dataset/Bayesian_Data/ASIA/ASIA_DATA.csv',
        type=str,
        help="The data path conntain file csv for training",
    )
    parser.add_argument(
        "--gt_path", 
        default="/dataset/Bayesian_Data/ASIA/DAGtrue_ASIA_bi.csv", 
        type=str, 
        help="File csv contain ground truth",
    )
    parser.add_argument(
        "--results",
        default='./result',
        type=str,
        help="The folder path contain results",
    )
    parser.add_argument(
        "--score_func",
        default='local_score_BDeu',
        type=str,
    )
    parser.add_argument(
        "--maxP",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--parameters",
        default=None,
        type=str,
    )
    return parser.parse_args()    

if __name__ == '__main__':
    agrs = get_parser()
    main(agrs)