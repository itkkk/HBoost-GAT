import json
from pathlib import Path
from typing import List

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import torch
from torch_geometric.nn import GAT, to_hetero
import typer

from mir_gatboost.config import PROJ_ROOT
from mir_gatboost.utils.dataset_manager import get_datasets
from mir_gatboost.utils.GATBoosting import IRBoostSH, graphs_constructor
import mir_gatboost.utils.scores_manager as sm

app = typer.Typer()


def fold_dataset_constructor(
    PD_fold: pd.DataFrame,
    EXP_fold: pd.DataFrame,
    EMB_MIRNA: pd.DataFrame,
    n_components: List[int],
    neighbors_index_path: Path,
    rebuild_index: bool,
) -> tuple[dict, dict, dict]:
    """
    Builds multi-view graphs and tabular feature splits for the current fold.
    PCA is fitted with max(n_components) components strictly on the training split
    to prevent data leakage. Each view is a slice of the resulting components.
    """
    train_mask = ~PD_fold.test.values
    test_mask = PD_fold.test.values

    EXP_fold_imputed = EXP_fold.fillna(0)

    max_components = max(n_components)

    max_possible = min(
        EXP_fold_imputed[train_mask].shape[0],
        EXP_fold_imputed[train_mask].shape[1],
    )
    if max_components > max_possible:
        logger.warning(
            f"Requested {max_components} PCA components exceeds min(n_samples, n_features)={max_possible}. "
            f"Capping to {max_possible}."
        )
        max_components = max_possible

    logger.debug(f"Fitting PCA with {max_components} components on the training split.")
    pca = PCA(n_components=max_components, random_state=42).fit(EXP_fold_imputed[train_mask])
    EXP_pca = pd.DataFrame(pca.transform(EXP_fold_imputed), index=EXP_fold_imputed.index)
    logger.debug(f"PCA output shape: {EXP_pca.shape}")

    X = {'train': {}, 'test': {}}

    disease_codes = PD_fold.disease.factorize()[0]
    y = {
        'train': pd.Series(disease_codes[train_mask], index=PD_fold[train_mask].index),
        'test':  pd.Series(disease_codes[test_mask],  index=PD_fold[test_mask].index),
    }
    graphs = {}

    for view_size in n_components:
        # Use a string key for every view. IRBoostSH selects a view each iteration via
        # np.random.choice(list(views.keys()), p=q_views) (multi-arm bandit), and
        # predict_proba checks `views_selected[t] in X.keys()`; keeping all keys as
        # strings keeps both consistent.
        view_key = str(view_size)

        neighbors_index_view_path = neighbors_index_path / view_key
        neighbors_index_view_path.mkdir(parents=True, exist_ok=True)

        EXP_lessDim = EXP_pca.iloc[:, :view_size]
        logger.debug(f"View {view_key}: feature matrix shape {EXP_lessDim.shape}")

        graphs[view_key] = graphs_constructor(
            PD_fold,
            EXP_fold,
            EMB_MIRNA,
            emb_view=EXP_lessDim,
            index_PATH=str(neighbors_index_view_path),
            rebuild_index_file=rebuild_index,
        )

        X['train'][view_key] = EXP_lessDim[train_mask]
        X['test'][view_key] = EXP_lessDim[test_mask]

    return X, y, graphs


def get_sample_weights(PD_fold: pd.DataFrame) -> pd.Series:
    """
    Computes log-based inverse-frequency sample weights to counterbalance class imbalance.
    """
    info = PD_fold.disease.value_counts()
    inv_p = 1 / (PD_fold.disease.map(lambda m: info[m]) / info.sum())
    weights = np.log(inv_p) / np.log(inv_p).sum()
    return weights


def build_model(
    PD_fold: pd.DataFrame,
    fold_graphs: dict,
    hidden_channels: int,
    num_layers: int,
    n_iter: int,
    reference_view_size: str,
) -> IRBoostSH:
    """
    Instantiates a heterogeneous GATv2 wrapped in IRBoostSH for boosted training.
    The graph metadata is inferred from the graph at reference_view_size.
    """
    logger.info(
        f"Selected Model: Heterogeneous GATv2 wrapped in IRBoostSH "
        f"(Hidden Channels: {hidden_channels}, Layers: {num_layers}, Boosting Iterations: {n_iter})"
    )

    model = GAT(
        in_channels=(-1, -1),
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        out_channels=len(PD_fold.disease.unique()),
        v2=True,
        jk='cat',
        edge_dim=1,
        add_self_loops=False,
    )
    model = to_hetero(model, fold_graphs[reference_view_size].metadata(), aggr='mean')

    _forward = model.forward
    model.forward = lambda *args, **kwargs: _forward(*args, **kwargs)['sample']

    return IRBoostSH(model, n_iter=n_iter, num_layers=num_layers)


def evaluate_model(
    PD_fold: pd.DataFrame,
    X_fold: dict,
    model: IRBoostSH,
    scores_output_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates the trained GATBoosting model on the test split.
    IRBoostSH.predict_proba returns calibrated probabilities, so stage_number=0.
    """
    disease_names = PD_fold.disease.unique().tolist()

    y_true = PD_fold[PD_fold.test].disease
    y_pred = pd.Categorical.from_codes(model.predict(X_fold['test']), disease_names)
    out = model.predict_proba(X_fold['test']).values

    # IRBoostSH outputs calibrated probabilities → stage_number=0 skips softmax
    scores = sm.classification_report_for_disease(y_true, y_pred, out, stage_number=0, output_dict=True)
    scores.sort_values(by='Support', inplace=True, ascending=False)

    averages = sm.classification_report_average(y_true, y_pred, out, stage_number=0, output_dict=True)
    averages.sort_values(by='Support', inplace=True, ascending=False)

    scores.to_csv(scores_output_path / 'scores_GAT_Boosting.csv')
    averages.to_csv(scores_output_path / 'scoresAVG_GAT_Boosting.csv')

    return scores, averages


@app.command()
def main(
    n_views: int = typer.Option(3, "--n-views", "-nv", help="Number of multi-view feature inputs for the boosting model"),
    n_components_str: str = typer.Option("512:256:128", "--n-components", help="PCA component counts per view, separated by colon, e.g. '512:256:128'"),
    hidden_channels: int = typer.Option(64, "--hidden-channels", "-hc", help="Number of hidden channels in the GAT model"),
    num_layers: int = typer.Option(1, "--num-layers", "-l", help="Number of GAT layers (all layers share the same hidden_channels)"),
    n_iter: int = typer.Option(10, "--n-iter", "-i", help="Number of IRBoostSH boosting iterations"),
    num_epoch: int = typer.Option(50, "--epochs", "-e", help="Number of training epochs per boosting iteration"),
    rebuild_index: bool = typer.Option(False, "--rebuild-index/--no-rebuild-index", help="Force rebuild of Annoy neighbor index files"),
    dataset_base_path: Path = typer.Option(PROJ_ROOT / "data" / "processed", "--dataset-path", help="Base directory path of the source dataset"),
    artifacts_path: Path = typer.Option(PROJ_ROOT / "artifacts", "--artifacts-path", help="Directory containing miRNA_embeddings.pt and Annoy indexes"),
    output_base_path: Path = typer.Option(PROJ_ROOT / "results", "--output-path", help="Base output directory for experiment metrics"),
    platform_technology: str = typer.Option("MicroArray", "--platform", help="Platform technology filter: 'MicroArray' or 'NGS'"),
):
    """
    Executes a 3-Fold Cross Validation pipeline for a GATBoosting model (Heterogeneous GATv2 + IRBoostSH).
    Views are PCA component counts; PCA is fitted with max(n_components) components
    and each view is a slice of the result.
    """
    n_components = [int(x) for x in n_components_str.split(":")]

    if len(n_components) != n_views:
        raise typer.BadParameter(
            f"--n-components must have exactly n_views={n_views} values, got {len(n_components)}: {n_components}"
        )

    logger.info("Initializing experiment configuration...")
    logger.info(f"  Views: {n_views}, Components per view: {n_components}")
    logger.info(f"  GAT Hidden Channels: {hidden_channels}, Layers: {num_layers}")
    logger.info(f"  Boosting Iterations: {n_iter}, Epochs per iteration: {num_epoch}")
    logger.info(f"  Rebuild Index: {rebuild_index}")
    logger.info(f"  Platform: {platform_technology}")

    components_str = "_".join(f"F{v}" for v in sorted(n_components, reverse=True))

    weights_str = "SW"
    model_name_folder = f"GATBoosting_{components_str}_I{n_iter}_L{num_layers}_H{hidden_channels}_{weights_str}"

    model_path = output_base_path / model_name_folder
    model_path.mkdir(parents=True, exist_ok=True)

    params = {
        "model_name_folder": model_name_folder,
        "n_views": n_views,
        "n_components": n_components,
        "hidden_channels": hidden_channels,
        "num_layers": num_layers,
        "n_iter": n_iter,
        "num_epoch": num_epoch,
        "platform_technology": platform_technology,
    }
    with open(model_path / "params.json", "w") as f:
        json.dump(params, f, indent=4)
    logger.info(f"Experiment parameters saved to: {model_path / 'params.json'}")

    global_models_path = PROJ_ROOT / "models" / model_name_folder
    global_models_path.mkdir(parents=True, exist_ok=True)

    neighbors_index_base_path = artifacts_path / "indexes" / model_name_folder
    neighbors_index_base_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Results will be saved to: {model_path}")
    logger.info(f"Trained models will be saved to: {global_models_path}")
    logger.info(f"Neighbor indexes will be stored in: {neighbors_index_base_path}")

    PD, EXP = get_datasets(dataset_base_path, platform_technology=platform_technology, discard_none_mirna=True)

    logger.debug("Loading miRNA embeddings...")
    mirna_embeddings = torch.load(artifacts_path / "miRNA_embeddings.pt", weights_only=False)
    EMB_MIRNA = pd.DataFrame(
        [mirna_embeddings[name] for name in EXP.columns],
        index=EXP.columns,
    )
    logger.debug(f"miRNA embeddings loaded: {EMB_MIRNA.shape}")

    # Reference view for graph metadata: the largest PCA view (metadata is
    # identical across views anyway). Must be a string to match the string
    # graph keys built in fold_dataset_constructor.
    reference_view_size = str(max(n_components))

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    all_scores = []
    all_averages = []

    logger.info("Starting 3-Fold Cross Validation loop...")

    for fold_counter, (train_idx, test_idx) in enumerate(skf.split(EXP, PD.disease), start=1):
        logger.info(f"--- Processing Fold {fold_counter} / 3 ---")

        fold_output_path = model_path / f"FOLD_{fold_counter}"
        fold_output_path.mkdir(parents=True, exist_ok=True)

        fold_model_save_path = global_models_path / f"FOLD_{fold_counter}"
        fold_model_save_path.mkdir(parents=True, exist_ok=True)

        neighbors_index_fold_path = neighbors_index_base_path / f"FOLD_{fold_counter}"
        neighbors_index_fold_path.mkdir(parents=True, exist_ok=True)

        PD_fold = PD.copy()
        PD_fold['test'] = False
        PD_fold.loc[PD.index[test_idx], 'test'] = True

        EXP_fold = EXP.copy()

        logger.info(f"[Fold {fold_counter}] Dataset and graph construction...")
        X_fold, y_fold, fold_graphs = fold_dataset_constructor(
            PD_fold, EXP_fold, EMB_MIRNA,
            n_components=n_components,
            neighbors_index_path=neighbors_index_fold_path,
            rebuild_index=rebuild_index,
        )

        logger.info(f"[Fold {fold_counter}] Building model...")
        boosting_model = build_model(
            PD_fold, fold_graphs,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            n_iter=n_iter,
            reference_view_size=reference_view_size,
        )

        logger.info(f"[Fold {fold_counter}] Training...")
        boosting_model.fit(
            X_fold['train'], y_fold['train'], fold_graphs,
            model_PATH=fold_model_save_path,
            num_epoch=num_epoch,
            model_NAME="GAT",
            fold_number=fold_counter,
            sample_weights=get_sample_weights(PD_fold[~PD_fold.test]),
        )

        logger.info(f"[Fold {fold_counter}] Evaluation...")
        scores, averages = evaluate_model(
            PD_fold, X_fold,
            model=boosting_model,
            scores_output_path=fold_output_path,
        )

        all_scores.append(scores)
        all_averages.append(averages)

        logger.success(f"[Fold {fold_counter}] Pipeline completed successfully.")

    logger.info("Aggregating cross-validation results...")

    combined_scores = pd.concat(all_scores, axis=0, keys=range(1, len(all_scores) + 1))
    combined_averages = pd.concat(all_averages, axis=0, keys=range(1, len(all_averages) + 1))

    mean_scores = sm.custom_agg(combined_scores)
    mean_averages = sm.custom_agg(combined_averages)

    mean_scores.to_csv(model_path / 'GLOBALscores.csv')
    mean_averages.to_csv(model_path / 'GLOBALscoresAVG.csv')

    logger.success(f"Execution pipeline for {model_name_folder} successfully finalized.")


if __name__ == "__main__":
    app()