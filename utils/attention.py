import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import constants as cst


def analyze_attention_features(
    att_feature,
    feature_names=None,
    sequence_names=None,
    top_k=10,
    top_k_positions=None,
    save_dir=None,
    model_type=None,
    dataset_type=None,
    is_wandb=False,
    tag="attention",
):
    """Analyze and visualize most-attended features and sequence positions.

    Produces:
      1) Bar plot of top-k features by weighted attention mass (overall).
      2) Heatmap of per-layer/head normalized attention for those top-k features.
      3) Bar plot of top-k sequence positions by weighted attention mass.
      4) Heatmap of per-layer/head normalized attention for those top-k positions.

    Returns a summary dict with:
      - overall_weighted
      - overall_counts
      - top_k_indices
      - per_head_top_feature
      - per_head_confidence
      - position_weighted
      - top_k_positions
    """
    att_feature = np.stack(att_feature)
    att_feature = att_feature.transpose(1, 3, 0, 2, 4)
    indices = att_feature[:, :, :, 1].astype(int)
    values = att_feature[:, :, :, 0].astype(float)
    num_layers, num_heads, num_samples, num_query_features = indices.shape
    max_index = int(indices.max(initial=0))
    num_features = max(num_query_features, max_index + 1)

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(num_features)]
    else:
        if len(feature_names) < num_features:
            feature_names = list(feature_names) + [
                f"f{i}" for i in range(len(feature_names), num_features)
            ]

    if sequence_names is None:
        sequence_names = [f"p{i}" for i in range(num_query_features)]
    else:
        if len(sequence_names) < num_query_features:
            sequence_names = list(sequence_names) + [
                f"p{i}" for i in range(len(sequence_names), num_query_features)
            ]

    weighted_per_head = np.zeros((num_layers, num_heads, num_features))
    counts_per_head = np.zeros((num_layers, num_heads, num_features))
    for layer in range(num_layers):
        for head in range(num_heads):
            head_indices = indices[layer, head].reshape(-1)
            head_values = values[layer, head].reshape(-1)
            weighted_per_head[layer, head] = np.bincount(
                head_indices, weights=head_values, minlength=num_features
            )
            counts_per_head[layer, head] = np.bincount(
                head_indices, minlength=num_features
            )

    overall_weighted = weighted_per_head.sum(axis=(0, 1))
    overall_counts = counts_per_head.sum(axis=(0, 1))
    top_k = min(top_k, num_features)
    top_k_indices = np.argsort(overall_weighted)[-top_k:][::-1]

    per_head_top_feature = np.argmax(weighted_per_head, axis=2)
    per_head_confidence = weighted_per_head.max(axis=2) / (
        weighted_per_head.sum(axis=2) + 1e-8
    )

    position_weighted_per_head = values.sum(axis=2)
    position_weighted = position_weighted_per_head.sum(axis=(0, 1))
    if top_k_positions is None:
        top_k_positions = top_k
    top_k_positions = min(top_k_positions, num_query_features)
    top_k_pos_indices = np.argsort(position_weighted)[-top_k_positions:][::-1]

    if save_dir is None:
        save_dir = cst.DIR_SAVED_MODEL
    os.makedirs(save_dir, exist_ok=True)

    dataset_suffix = f"_{dataset_type}" if dataset_type else ""
    model_suffix = f"_{model_type}" if model_type else ""

    # Bar plot: top-k features by weighted attention mass
    plt.figure(figsize=(14, 6), dpi=120)
    top_k_values = overall_weighted[top_k_indices]
    top_k_labels = [feature_names[i] for i in top_k_indices]
    sns.barplot(x=top_k_labels, y=top_k_values, palette="viridis")
    plt.title("Top attended features (weighted attention mass)")
    plt.xlabel("Feature")
    plt.ylabel("Weighted attention")
    plt.xticks(rotation=45, ha="right")
    bar_path = os.path.join(
        save_dir, f"{tag}_top_features{dataset_suffix}{model_suffix}.png"
    )
    if is_wandb:
        wandb.log({f"{tag}_top_features{dataset_suffix}{model_suffix}": wandb.Image(plt)})
    plt.tight_layout()
    plt.savefig(bar_path)
    plt.close()

    # Heatmap: per-layer/head normalized attention for top-k features
    heatmap_data = np.zeros((num_layers * num_heads, top_k))
    row_labels = []
    row_idx = 0
    for layer in range(num_layers):
        for head in range(num_heads):
            head_vals = weighted_per_head[layer, head, top_k_indices]
            norm = head_vals.sum() + 1e-8
            heatmap_data[row_idx] = head_vals / norm
            row_labels.append(f"L{layer}-H{head}")
            row_idx += 1

    plt.figure(figsize=(14, 0.45 * len(row_labels) + 3), dpi=120)
    sns.heatmap(
        heatmap_data,
        xticklabels=top_k_labels,
        yticklabels=row_labels,
        cmap="mako",
        cbar_kws={"label": "Normalized attention"},
    )
    plt.title("Per-head attention distribution over top features")
    plt.xlabel("Feature")
    plt.ylabel("Layer-Head")
    plt.tight_layout()
    heatmap_path = os.path.join(
        save_dir, f"{tag}_head_heatmap{dataset_suffix}{model_suffix}.png"
    )
    if is_wandb:
        wandb.log({f"{tag}_head_heatmap{dataset_suffix}{model_suffix}": wandb.Image(plt)})
    plt.savefig(heatmap_path)
    plt.close()

    # Bar plot: top-k sequence positions by weighted attention mass
    plt.figure(figsize=(14, 6), dpi=120)
    top_pos_values = position_weighted[top_k_pos_indices]
    top_pos_labels = [sequence_names[i] for i in top_k_pos_indices]
    sns.barplot(x=top_pos_labels, y=top_pos_values, palette="crest")
    plt.title("Top sequence positions (weighted attention mass)")
    plt.xlabel("Position")
    plt.ylabel("Weighted attention")
    plt.xticks(rotation=45, ha="right")
    pos_bar_path = os.path.join(
        save_dir, f"{tag}_top_positions{dataset_suffix}{model_suffix}.png"
    )
    if is_wandb:
        wandb.log({f"{tag}_top_positions{dataset_suffix}{model_suffix}": wandb.Image(plt)})
    plt.tight_layout()
    plt.savefig(pos_bar_path)
    plt.close()

    # Heatmap: per-layer/head normalized attention for top-k positions
    pos_heatmap_data = np.zeros((num_layers * num_heads, top_k_positions))
    row_labels = []
    row_idx = 0
    for layer in range(num_layers):
        for head in range(num_heads):
            head_vals = position_weighted_per_head[layer, head, top_k_pos_indices]
            norm = head_vals.sum() + 1e-8
            pos_heatmap_data[row_idx] = head_vals / norm
            row_labels.append(f"L{layer}-H{head}")
            row_idx += 1

    plt.figure(figsize=(14, 0.45 * len(row_labels) + 3), dpi=120)
    sns.heatmap(
        pos_heatmap_data,
        xticklabels=top_pos_labels,
        yticklabels=row_labels,
        cmap="rocket",
        cbar_kws={"label": "Normalized attention"},
    )
    plt.title("Per-head attention distribution over top positions")
    plt.xlabel("Position")
    plt.ylabel("Layer-Head")
    plt.tight_layout()
    pos_heatmap_path = os.path.join(
        save_dir, f"{tag}_position_heatmap{dataset_suffix}{model_suffix}.png"
    )
    if is_wandb:
        wandb.log({f"{tag}_position_heatmap{dataset_suffix}{model_suffix}": wandb.Image(plt)})
    plt.savefig(pos_heatmap_path)
    plt.close()

    return {
        "overall_weighted": overall_weighted,
        "overall_counts": overall_counts,
        "top_k_indices": top_k_indices,
        "per_head_top_feature": per_head_top_feature,
        "per_head_confidence": per_head_confidence,
        "position_weighted": position_weighted,
        "top_k_positions": top_k_pos_indices,
        "bar_plot_path": bar_path,
        "heatmap_path": heatmap_path,
        "position_bar_plot_path": pos_bar_path,
        "position_heatmap_path": pos_heatmap_path,
    }
