import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def output_Multiple(result):
    """
    Pretty-print per-sgRNA metrics and global averaged metrics.
    """
    print("=" * 70)
    print("Model Evaluation Results (Averaged Global Metrics)".center(70))
    print("=" * 70)

    # 1. List all SG-specific metrics first
    print(f"\n【SG-Specific Metrics】 (Total: {len(result['sg_type'])} SG sequences)")
    for i in range(len(result['sg_type'])):
        print(f"\nSG #{i+1}:")
        print(f"{'  Sequence':<25}: {result['sg_type'][i]}")
        print(f"{'  ROC-AUC':<25}: {result['sg_roc_auc'][i]:.4f}")
        print(f"{'  ROC-PRC':<25}: {result['sg_roc_prc'][i]:.4f}")

    # 2. Show global metrics with explicit averaging note
    print("\n" + "-" * 70)
    print("【Global Metrics】 (Average of all SG metrics above)")
    print(f"{'Global ROC-AUC (avg)':<25}: {result['roc_auc']:.4f}")
    print(f"{'Global ROC-PRC (avg)':<25}: {result['roc_prc']:.4f}")
    print("\n" + "=" * 70)

def output_single(sg, off, predict_list):
    """
    Pretty-print a single sgRNA/off-target prediction result.
    """   
    print(sg, off, predict_list[0])
    print("-" * 70)
    print(f"{'SG Sequence':<26} {'Off-Target Sequence':<26} {'Model Prediction':<20}")
    print("-" * 70)
    # Print values (adjust formatting based on variable type: .3f for floats, etc.)
    print(
        f"{sg:<26} "
        f"{off:<26} "  # Use .3f for 3 decimal places (adjust if off is a string)
        f"{predict_list[0]}"
    )
    print("-" * 70)

def plot_matrix_rows(matrix_df, title, output_path, cmap_name="coolwarm"):
    """
    Visualize each row's distribution as a violin plot (per position).
    """
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['axes.unicode_minus'] = False

    # 转换为长格式数据
    long_df = matrix_df.reset_index().melt(
        id_vars='index',
        var_name='cols',
        value_name='Value'
    )
    long_df.rename(columns={'index': 'Position'}, inplace=True)
    n_rows = len(matrix_df.index)
    col_medians = matrix_df.median(axis=1)
    means = matrix_df.mean(axis=1)
    colors = sns.color_palette(cmap_name, n_colors=n_rows)

    plt.figure(figsize=(15, 8))
    ax = sns.violinplot(
        x='Position',
        y='Value',
        data=long_df,
        palette=colors,      # 传入颜色列表
        cut = 0,
        scale="width", # 小提琴宽度表示样本量
        # inner=None
    )
    xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
    group_means = long_df.groupby('Position')['Value'].mean()
    for i, pos in enumerate(xtick_labels):
        mean_val = group_means.loc[pos]
        ax.hlines(
            y=mean_val,
            xmin=i - 0.08, xmax=i + 0.08,  # 控制横线的宽度
            color='red',
            linewidth=2.5,
            zorder=10
        )
    plt.title(title, fontsize=20, fontweight='bold')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.xticks(rotation=45, ha='right', fontsize=18, fontweight='bold')  # 旋转x轴标签
    plt.yticks(fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_sim_hotmap(df, anno, filename, vmin, vmax):
    """
    Draw and save a heatmap of a similarity matrix with mismatch base pairs.
    """    
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, vmin=vmin,  vmax=vmax, annot=True, cmap='coolwarm', annot_kws={'fontsize': 8, "fontweight":'bold'} )
    plt.title(f'Column Similarity ({anno})', fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')


    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()