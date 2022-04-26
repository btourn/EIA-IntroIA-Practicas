"""Ejercicio 5 - Average query precision."""

import numpy as np

def aqp(q_id, predict, truth):
    """
    Funcion aqp().
    
    dsasadasd.
    """    
    ar_id = np.array(q_id)
    ar_pred = np.array(predict)
    ar_truth = np.array(truth)
    max_q = np.max(ar_id)
    id_vals = np.unique(ar_id)
    aqp = 0
    for i in id_vals:
        q_i = np.where(ar_id==i)
        nq = np.size(q_i)
        nt = np.sum(ar_truth[q_i,])
        aqp += (nt/nq)/max_q
    
    return aqp


# q_id =             [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
# predicted_rank =   [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3]
# truth_relevance =  [True, False, True, False, True, True, True, False, False, False, False, False, True, False, False, True] 
# aqp(q_id, predicted_rank, truth_relevance)
