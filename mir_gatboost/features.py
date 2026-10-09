import collections
from collections import OrderedDict
import collections.abc

# Monkey patch for backward compatibility of the attrdict library
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence

from pathlib import Path  # noqa: E402, I001

from loguru import logger # noqa: E402
import pandas as pd # noqa: E402
from third_party.RNABERT.bert import BertForMaskedLM, BertModel, get_config # noqa: E402
import torch # noqa: E402
import typer # noqa: E402

from mir_gatboost.config import (  # noqa: E402
    ARTIFACTS_DIR,
    EXTERNAL_DATA_DIR,
)


app = typer.Typer()

BASE_MAPPING = {'A': 2, 'U': 3, 'G': 4, 'C': 5}


@app.command()
def main(
    model_config_path: Path = Path('third_party/RNABERT/RNA_bert_config.json'),
    model_weights_path: Path = Path('third_party/RNABERT/bert_mul_2.pth'),
    mirna_path: Path = EXTERNAL_DATA_DIR / 'miRNA.tsv',
    output_path: Path = ARTIFACTS_DIR / 'miRNA_embeddings.pt',
):
    logger.debug(f"Model config path: {model_config_path}")
    logger.debug(f"Model weights path: {model_weights_path}")
    logger.debug(f"miRNA sequences path: {mirna_path}")
    logger.debug(f"Output path: {output_path}")
    logger.info("Extracting miRNA features...")

    model, rnabert_config = _load_rnabert_model(model_config_path, model_weights_path)
    compute_mirna_embeddings(model, rnabert_config, mirna_path, output_path)

    logger.success("Feature extraction complete.")


def compute_mirna_embeddings(
    model: BertForMaskedLM,
    rnabert_config,
    mirna_path: Path,
    output_path: Path,
):
    logger.info("Computing miRNA embeddings...")

    device = next(model.parameters()).device

    # Load miRNA sequences, dropping rows with missing name or sequence
    mirna_df = pd.read_csv(mirna_path, sep='\t').dropna(
        subset=['miRNAName_v22', 'miRNASequence_v22']
    )
    mirna_dict = dict(zip(mirna_df['miRNAName_v22'], mirna_df['miRNASequence_v22']))
    logger.debug(f"miRNA sequences loaded: {len(mirna_dict)}")

    embeddings = {}
    skipped = 0

    with torch.no_grad():
        for name, sequence in mirna_dict.items():
            tokens = _tokenize_sequence(sequence, rnabert_config.max_position_embeddings)

            if tokens is None:
                logger.warning(f"Skipping '{name}': sequence contains unknown bases")
                skipped += 1
                continue

            encoded_layers = model(torch.tensor([tokens]).to(device))[-1]
            embeddings[name] = torch.mean(encoded_layers[0], dim=0).cpu().numpy()

    logger.debug(f"Embeddings computed: {len(embeddings)}, skipped: {skipped}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)
    logger.debug(f"Embeddings saved to: {output_path}")

    logger.success("miRNA embeddings computation complete.")


def _load_rnabert_model(config_path: Path, weights_path: Path):
    """Load and initialize the RNABERT model from config and weights files."""
    logger.info("Loading RNABERT model...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.debug(f"Using device: {device}")

    rnabert_config = get_config(str(config_path))
    rnabert_config.hidden_size = rnabert_config.num_attention_heads * rnabert_config.multiple

    model = BertModel(rnabert_config)
    model = BertForMaskedLM(rnabert_config, model).eval()

    state_dict = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(
        OrderedDict((key[7:], value) for key, value in state_dict.items())
    )
    model.to(device)

    logger.debug(f"Model loaded — hidden size: {rnabert_config.hidden_size}, device: {device}")
    logger.success("RNABERT model loaded.")

    return model, rnabert_config


def _tokenize_sequence(sequence: str, max_length: int) -> list[int] | None:
    """
    Tokenize a miRNA nucleotide sequence into integer tokens, padding to max_length.
    Returns None if the sequence contains bases outside the expected alphabet {A, U, G, C}.
    """
    try:
        tokens = [BASE_MAPPING[base] for base in sequence]
    except KeyError:
        return None
    padding = [0] * (max_length - len(tokens))
    return tokens + padding


if __name__ == "__main__":
    app()