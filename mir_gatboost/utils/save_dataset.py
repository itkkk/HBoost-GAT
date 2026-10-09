from pathlib import Path
import pickle
from typing import List

from loguru import logger
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import typer

from mir_gatboost.config import PROJ_ROOT
from mir_gatboost.utils.dataset_manager import get_datasets

app = typer.Typer()


def fold_dataset_constructor(
    PD_fold: pd.DataFrame,
    EXP_fold: pd.DataFrame,
    n_components: List[int],
    pca_save_path: Path,
) -> None:
    """
    Fits PCA on the training split and saves the transformed train/test
    DataFrames as pickle files in pca_save_path.
    Also saves the corresponding raw EXP and PD splits.
    """
    train_mask = ~PD_fold.test.values
    test_mask  =  PD_fold.test.values

    EXP_fold_imputed = EXP_fold.fillna(0)
    max_components   = max(n_components)

    max_possible = min(EXP_fold_imputed[train_mask].shape[0], EXP_fold_imputed[train_mask].shape[1])
    if max_components > max_possible:
        logger.warning(
            f"Requested {max_components} PCA components exceeds min(n_samples, n_features)={max_possible}. "
            f"Capping to {max_possible}."
        )
        max_components = max_possible

    logger.debug(f"Fitting PCA with {max_components} components on the training split.")
    pca     = PCA(n_components=max_components, random_state=42).fit(EXP_fold_imputed[train_mask])
    EXP_pca = pd.DataFrame(pca.transform(EXP_fold_imputed), index=EXP_fold_imputed.index)
    logger.debug(f"PCA output shape: {EXP_pca.shape}")

    files = {
        f"pca_train_F{max_components}.pkl": EXP_pca[train_mask],
        f"pca_test_F{max_components}.pkl":  EXP_pca[test_mask],
        "exp_train.pkl":                    EXP_fold[train_mask],
        "exp_test.pkl":                     EXP_fold[test_mask],
        "pd_train.pkl":                     PD_fold[train_mask].drop(columns="test"),
        "pd_test.pkl":                      PD_fold[test_mask].drop(columns="test"),
    }

    for filename, data in files.items():
        with open(pca_save_path / filename, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved → {pca_save_path / filename}  shape: {data.shape}")


@app.command()
def main(
    n_views: int = typer.Option(5, "--n-views", "-nv"),
    n_components_str: str = typer.Option("1024:512:256:128:256", "--n-components"),
    dataset_base_path: Path = typer.Option(PROJ_ROOT / "data" / "processed", "--dataset-path"),
    output_base_path: Path = typer.Option(PROJ_ROOT / "models", "--output-path"),
    platform_technology: str = typer.Option("MicroArray", "--platform"),
):
    """
    TEST: esegue solo il salvataggio delle PCA, EXP e PD per ogni fold (no training, no evaluation).
    """
    n_components = [int(x) for x in n_components_str.split(":")]

    if len(n_components) != n_views:
        raise typer.BadParameter(
            f"--n-components deve avere esattamente {n_views} valori, ricevuti {len(n_components)}"
        )

    components_str    = "_".join(f"F{v}" for v in sorted(n_components, reverse=True))
    model_name_folder = f"GATBoosting_{components_str}_PCA_TEST"

    output_path = output_base_path / model_name_folder
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Files will be saved under: {output_path}")

    PD, EXP = get_datasets(dataset_base_path, platform_technology=platform_technology, discard_none_mirna=True)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for fold_counter, (train_idx, test_idx) in enumerate(skf.split(EXP, PD.disease), start=1):
        logger.info(f"--- Fold {fold_counter} / 3 ---")

        fold_save_path = output_path / f"FOLD_{fold_counter}"
        fold_save_path.mkdir(parents=True, exist_ok=True)

        PD_fold = PD.copy()
        PD_fold['test'] = False
        PD_fold.loc[PD.index[test_idx], 'test'] = True

        fold_dataset_constructor(
            PD_fold, EXP.copy(),
            n_components=n_components,
            pca_save_path=fold_save_path,
        )

        logger.success(f"[Fold {fold_counter}] Completato.")

    logger.success("Test completato.")


if __name__ == "__main__":
    app()
