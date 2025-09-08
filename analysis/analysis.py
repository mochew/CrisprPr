import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
from utils.output import plot_matrix_rows, plot_sim_hotmap

def load_matrics(origin_matrix_path, update_matrix_base_path, seed_num):
    """
    Load the original prior matrix and a list of updated matrices produced by different random seeds.
    """
    orig_mat = np.load(origin_matrix_path)
    
    # 加载各个模型种子对应的更新矩阵
    update_matrices = []
    for seed in range(seed_num):
        update_matrix = np.load(f"{update_matrix_base_path}/update_matrix{seed}.npy")
        update_matrices.append(update_matrix)
        
    return orig_mat, update_matrices

def mk_dir(path):
    """
    Create a directory if it does not already exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def generate_agg_update_matrix(orig_mat, update_matrices):
    """
    Aggregate multiple update matrices by majority sign voting (per cell),
    then average only values that agree with the majority direction.

    Workflow
    --------
    1) Compute delta = update_mat - orig_mat for each seed.
    2) For each cell (i, j), count how many deltas are positive vs negative.
    3) If positives >= 3, average only positive deltas; if negatives >= 3, average only negative deltas.
    4) Add the aggregated delta back to `orig_mat` to get the final aggregated matrix.
    """
    updates_list = [update_mat - orig_mat for update_mat in update_matrices]
    upd_stack = np.stack(updates_list, axis=2)
    signs = np.sign(upd_stack)  # -1,0,1
    pos_count = np.sum(signs == 1, axis=2)
    neg_count = np.sum(signs == -1, axis=2)

    # 5. 根据多数派方向分别取平均（只对同方向的值取平均）
    agg_update = np.zeros_like(orig_mat)
    for i in range(orig_mat.shape[0]):
        for j in range(orig_mat.shape[1]):
            if pos_count[i, j] >= 3:
                vals = upd_stack[i, j, signs[i, j, :] == 1]
                agg_update[i, j] = np.mean(vals)
            elif neg_count[i, j] >= 3:
                vals = upd_stack[i, j, signs[i, j, :] == -1]
                agg_update[i, j] = np.mean(vals)

    agg_update_mat = orig_mat + agg_update

    return agg_update_mat



def visual_position_distributions(orig_mat, agg_update_mat, output_path):
    """
    Plot per-position distributions of the original matrix and the aggregated updated matrix.
    """
    orig_output_path = output_path + 'orig_mis_dis.png'
    update_output_path = output_path + 'update_mis_dis.png'
    plot_matrix_rows(orig_mat, 'origin', orig_output_path)
    plot_matrix_rows(agg_update_mat, 'update', update_output_path)

def vector_similarity(x, y, alpha=0.5):
    """
    Compute a blended similarity between two vectors by combining
    a normalized Euclidean-distance component and a Pearson-correlation component.

    Similarity = alpha * (1 - normalized_distance) + (1 - alpha) * pearson_r
    """
    r, _ = pearsonr(x, y)
    trend_dist = r
    d = euclidean(x, y)
    norm_d = d / (np.linalg.norm(x) + np.linalg.norm(y) + 1e-9)
    norm_d = 1 - norm_d
    return alpha * norm_d + (1 - alpha) * trend_dist

def compute_mismatch_similarity_matrix(df, alpha=0.5):
    """
    Compute a column-wise similarity matrix for mismatch categories.
    """    
    cols = df.columns
    n = len(cols)
    sim_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # print(i,j)
            sim_mat[i, j] = vector_similarity(df.iloc[:, i].values,
                                              df.iloc[:, j].values,
                                              alpha)
    return pd.DataFrame(sim_mat, index=cols, columns=cols)



def visual_mismatch_similarity(orig_mat, agg_update_mat, output_path):
    """
    Compute and visualize column-wise mismatch similarity for:
    (a) seed region rows (3..12) and
    (b) non-seed region rows (13..end).
    """   
    seed_orig_mat = orig_mat.iloc[3:13, :]
    seed_final_mat = agg_update_mat.iloc[3:13, :]

    seed_orig_sim = compute_mismatch_similarity_matrix(seed_orig_mat)
    plot_sim_hotmap(seed_orig_sim, "seed origin", output_path+"origin_seed_bp_sim.png", 0, 1)
    seed_update_sim = compute_mismatch_similarity_matrix(seed_final_mat)
    plot_sim_hotmap(seed_update_sim, "seed update", output_path+"update_seed_bp_sim.png", 0,1)

    Nseed_orig_mat = orig_mat.iloc[13:, :]
    Nseed_update_mat = agg_update_mat.iloc[13:-1, :]

    Nseed_orig_sim = compute_mismatch_similarity_matrix(Nseed_orig_mat)
    plot_sim_hotmap(Nseed_orig_sim, "Nseed origin", output_path+"origin_Nseed_bp_sim.png", 0.2, 1)
    Nseed_update_sim = compute_mismatch_similarity_matrix(Nseed_update_mat)
    plot_sim_hotmap(Nseed_update_sim, "Nseed update", output_path+"update_Nseed_bp_sim.png", 0.2, 1)



def analysis_update(origin_matrix_path, update_matrix_base_path, output_path, seed_num):
    """
    End-to-end analysis pipeline:
    1) Load original and seed-wise updated matrices.
    2) Select a subset of columns and rows (mismatches).
    3) Aggregate updates via majority-sign voting and averaging.
    4) Save violin plots (per-position distributions) and similarity heatmaps.

    """
    row_labels = ['pam3', 'pam2', 'pam1', 'Pos 1', 'Pos 2', 'Pos 3', 'Pos 4', 'Pos 5', 'Pos 6', 'Pos 7', 'Pos 8', 'Pos 9',
              'Pos 10', 'Pos 11', 'Pos 12', 'Pos 13', 'Pos 14', 'Pos 15', 'Pos 16', 'Pos 17', 'Pos 18', 'Pos 19',
              'Pos 20', 'Pos 21']
    # 可视化归一化后的 Δ 矩阵热图
    col_labels = ['AA', 'AG', 'AC', 'AT', 'A_', 'A-', 'GG', 'GA', 'GC', 'GT', 'G_', 'G-',
                'CC', 'CA', 'CG', 'CT', 'C_', 'C-', 'TT', 'TA',
                'TG', 'TC', 'T_', 'T-', '__', '_A', '_G', '_C', '_T', '_-', '--',
                '-A', '-G', '-C', '-T', '-_']

    # selec_cols = [0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 21]
    selec_cols = [1, 2, 3, 7, 8, 9, 13, 14, 15, 19, 20, 21]
    selec_labels = [col_labels[i] for i in selec_cols]

    orig_mat, update_matrices = load_matrics(origin_matrix_path, update_matrix_base_path, seed_num)

    select_orig_mat  = pd.DataFrame(orig_mat, index=row_labels, columns=col_labels).iloc[:-1, selec_cols]
    select_orig_mat = select_orig_mat.round(5)

    select_update_matrices =[pd.DataFrame(update_mat, index=row_labels, columns=col_labels).iloc[:-1, selec_cols] for update_mat in update_matrices]

    agg_update_mat = generate_agg_update_matrix(select_orig_mat, select_update_matrices)
    agg_update_mat = pd.DataFrame(agg_update_mat, index=row_labels, columns=selec_labels)
    agg_update_mat = agg_update_mat.round(5)
    mk_dir(output_path)
    visual_position_distributions(select_orig_mat, agg_update_mat, output_path)
    visual_mismatch_similarity(select_orig_mat, agg_update_mat, output_path)






# origin_matrix_path = "/wuyingfu/CrisprPr/data/matrices/MTP/origin_matrix.npy"
# update_matrix_base_path = "/wuyingfu/CrisprPr/data/matrices/MTP"
# output_path = "/wuyingfu/CrisprPr/analysis/figs/MTP/"
# seed_num = 5
# analysis_update(origin_matrix_path, update_matrix_base_path, output_path, seed_num)