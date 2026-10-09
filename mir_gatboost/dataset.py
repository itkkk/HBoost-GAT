import json
from pathlib import Path
import pickle
import re

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mir_gatboost.config import (
    EXTERNAL_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

app = typer.Typer()

DISEASE_MERGE_MAP = {
    'Hepatocellular carcinoma with chronic hepatitis C': 'Hepatocellular carcinoma',
    'Esophageal squamous cell carcinoma': 'Esophageal cancer',
    'Bone and soft tissue sarcoma': 'Sarcoma',
    'Colon cancer': 'Colon/Colorectal cancer',
    'Colorectal cancer': 'Colon/Colorectal cancer',
}


@app.command()
def main(
    samples_path: Path = RAW_DATA_DIR / "pd",
    exp_path: Path = RAW_DATA_DIR / "exp",
):
    logger.debug(f"Samples path: {samples_path}")
    logger.debug(f"Expression path: {exp_path}")
    logger.info("Processing dataset...")
    
    study_filtering(samples_path, exp_path)
    sample_filtering()
    data_alignment()
    data_merging()

    logger.success("Processing dataset complete.")


def study_filtering(samples_path: Path, exp_path: Path):
    logger.info("Filtering studies...")  

    # Get all pd.CSV file paths
    csv_files = list(samples_path.glob('*.csv'))
    # Get dataset names (remove "_pd" suffix and file extension)
    dataframes_names = [file.stem.split("_")[0] for file in csv_files]
    # Create a Dataframe for each file
    dataframes = [pd.read_csv(file, ) for file in csv_files]
    # Create a dictionary mapping dataset names to their corresponding DataFrames
    dataframe_dict = dict(zip(dataframes_names, dataframes))

    # Get all exp.CSV file paths
    csv_files = list(exp_path.glob('*.csv'))
    # Get miRNA dataset names (remove "_exp" suffix and file extension)
    EXP_names = [file.stem.split("_")[0] for file in csv_files]
    # Create a Dataframe for each file
    EXP_dataframes = [pd.read_csv(file, ) for file in csv_files]
    # Create a dictionary mapping exp names to their corresponding DataFrames
    EXP_dict = dict(zip(EXP_names, EXP_dataframes))
 
    geo_platforms = pd.read_csv(EXTERNAL_DATA_DIR / 'geo_platforms.csv')
    # Create a dict with (key, value) = (platform_ID, technology) 
    geo_dict = dict( zip(geo_platforms.loc[:, 'Accession'].values, geo_platforms.loc[:, 'Technology'].values))
    
    datasets = pd.read_csv(EXTERNAL_DATA_DIR / 'Datasets.tsv', sep='\t', index_col='ExomiRHub ID')
    # Platform Id conversion to Technology
    datasets['Platform Id'] = datasets['Platform Id'].map(geo_dict)

    # Take all the dataset with RT-PCR | other technology
    drop_dataframe_list = datasets[(datasets['Platform Id'] == 'RT-PCR') | (datasets['Platform Id'] == 'other')].index.to_list()

    # Extend the list with dataframe containing data duplicated in other datasets
    # see notebook 1. preprocessing_Dataset_Filtering.ipynb for more details
    drop_dataframe_list.extend([    
        'EMIR00000091',
        'EMIR00000092',
        'EMIR00000097',
        'EMIR00000101'
    ])

    # Extend the list with dataframe not related to diseases
    drop_dataframe_list.extend([
        'EMIR00000010', # Adipose mesenchmmal stem cell
        'EMIR00000011', # Human mesenchymal stem cell
        'EMIR00000023', # Hair follicle regeneration
        'EMIR00000026', # Wound healing; Myoblast formation
        'EMIR00000028', # Pregnancy-related disease
        'EMIR00000081', # Hematopoietic stem and progenitor cell expansion
        'EMIR00000095', # Sedentary and trained older adults
        'EMIR00000123', # Lethal and sublethal radiation doses
        'EMIR00000158', # Human suspended red blood cells with leukocyte reduced
        'EMIR00000149' # Unclear disease in the dataset
    ])

    logger.debug(f"Datasets to drop: {len(drop_dataframe_list)}")
    # Drop the sample datasets and miRNA dataset IN 'drop_dataframe_list'
    for key in drop_dataframe_list:
        logger.debug(f"Dataset {key} removed")
        dataframe_dict.pop(key, None)
        EXP_dict.pop(key, None)    
    
    # Map platform_id to technology in each dataframe
    for df_id, df in dataframe_dict.items():
        df['platform_id'] = df['platform_id'].map(geo_dict)

    # Renames the duplicated entries in EMIR00000177 and EMIR00000180
    # since those two datasets have been gathered from the same sample
    # in different time points, and thus have some duplicated entries 
    # with the same code (in 'Unnamed: 0' column)
    # see notebook 1. preprocessing_Dataset_Filtering.ipynb for more details
    codes_177 = set(dataframe_dict['EMIR00000177']['Unnamed: 0'])
    codes_180 = set(dataframe_dict['EMIR00000180']['Unnamed: 0'])
    duplicate_codes = codes_177 & codes_180

    for code in duplicate_codes:
        mask_177 = dataframe_dict['EMIR00000177']['Unnamed: 0'] == code
        mask_180 = dataframe_dict['EMIR00000180']['Unnamed: 0'] == code
        dataframe_dict['EMIR00000177'].loc[mask_177, 'Unnamed: 0'] = code + '_from_177'
        dataframe_dict['EMIR00000180'].loc[mask_180, 'Unnamed: 0'] = code + '_from_180'

    rename_177 = {code: code + '_from_177' for code in duplicate_codes}
    rename_180 = {code: code + '_from_180' for code in duplicate_codes}
    EXP_dict['EMIR00000177'].rename(columns=rename_177, inplace=True)
    EXP_dict['EMIR00000180'].rename(columns=rename_180, inplace=True)
    
    # Save the filtered dataframes in a pickle file for later use
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(INTERIM_DATA_DIR / 'dataframe_dict_study.pkl', 'wb') as f:
        pickle.dump(dataframe_dict, f)

    with open(INTERIM_DATA_DIR / 'EXP_dict_study.pkl', 'wb') as f:
        pickle.dump(EXP_dict, f)

    _sample_count(dataframe_dict)

    logger.success("Filtering studies complete.")


def sample_filtering():
    logger.info("Filtering samples...")

    # Read the filtered dataframes from the previous step
    with open(INTERIM_DATA_DIR / 'dataframe_dict_study.pkl', 'rb') as f:
        dataframe_dict = pickle.load(f)

    with open(INTERIM_DATA_DIR / 'EXP_dict_study.pkl', 'rb') as f:
        EXP_dict = pickle.load(f)
    
    # List of entries to filter, with the format (dataset_id, reason, column_name, bad_terms)
    # see notebook 2. preprocessing_Sample_Filtering.ipynb for more details about the entries 
    # and the reasons for filtering them
    entries = [
        ("EMIR00000090", "Disease status not clear", "disease state:ch1", "NA"),
        ("EMIR00000008", "Disease status not clear", "diagnosis:ch1", "NA"),
        ("EMIR00000067", "Disease status not clear", "disease status:ch1", "NA"),
        ("EMIR00000166", "Disease status not clear", "description", "No Glioma"),
        ("EMIR00000168", "Disease status not clear", "status:ch1", "NA"),
        ("EMIR00000115", "Non Homo Sapiens", "organism_ch1", "Rattus norvegicus"),
        ("EMIR00000034", "Non Homo Sapiens", "organism_ch1", "Mus musculus"),
        ("EMIR00000102", "Non Homo Sapiens", "organism_ch1", "Bos taurus"),
        ("EMIR00000033", "Non Molecola di solo RNA", "molecule_ch1", "polyA RNA"),
        ("EMIR00000172", "channel count > 1", "channel_count", "2"),
        ("EMIR00000016", "channel count > 1", "channel_count", "2"),
        ("EMIR00000017", "channel count > 1", "channel_count", "2"),
    ]

    def make_is_corrupted(terms):
        def is_corrupted(val):
            if pd.isnull(val):
                return True
            return any(term in str(val) for term in terms)
        return is_corrupted

    # Find samples to remove
    samples_to_remove_dict = {}

    for dataset_id, _, col_name, bad_terms in entries:

        if dataset_id not in dataframe_dict.keys():
            logger.warning(f"Dataset {dataset_id} already removed")
            continue

        df = dataframe_dict[dataset_id]

        if col_name not in df.columns:
            logger.warning(f"Column {col_name} not found in {dataset_id}")
            continue

        bad_terms_list = [val.strip() for val in bad_terms.split(";") if val != 'NA']

        corrupted_rows = df[df[col_name].apply(make_is_corrupted(bad_terms_list))]

        if not corrupted_rows.empty:
            corrupted_ids = corrupted_rows['Unnamed: 0'].to_list()
            samples_to_remove_dict.setdefault(dataset_id, []).extend(corrupted_ids)

    logger.debug(f"Dataset with samples to remove: {len(samples_to_remove_dict)}")

    # Remove the samples
    for dataset_id, samples_to_delete in samples_to_remove_dict.items():

        if dataframe_dict[dataset_id].shape[0] == len(samples_to_delete):
            dataframe_dict.pop(dataset_id, None)
            EXP_dict.pop(dataset_id, None)
            logger.debug(f"Dataset {dataset_id} removed entirely")

        else:
            df_pd = dataframe_dict[dataset_id]
            dataframe_dict[dataset_id] = df_pd[~df_pd['Unnamed: 0'].isin(samples_to_delete)]

            df_exp = EXP_dict[dataset_id]
            EXP_dict[dataset_id] = df_exp.drop(columns=samples_to_delete)

            logger.debug(f"Samples removed from {dataset_id}: {len(samples_to_delete)}")

    # Salva i pickle aggiornati
    with open(INTERIM_DATA_DIR / 'dataframe_dict_sample.pkl', 'wb') as f:
        pickle.dump(dataframe_dict, f)

    with open(INTERIM_DATA_DIR / 'EXP_dict_sample.pkl', 'wb') as f:
        pickle.dump(EXP_dict, f)

    _sample_count(dataframe_dict)

    logger.success("Filtering samples complete.")


def data_alignment():
    logger.info("Aligning data...")
    # Read the filtered dataframes from the previous step
    with open(INTERIM_DATA_DIR / 'dataframe_dict_sample.pkl', 'rb') as f:
        dataframe_dict = pickle.load(f)

    with open(INTERIM_DATA_DIR / 'EXP_dict_sample.pkl', 'rb') as f:
        EXP_dict = pickle.load(f)
    
    dataframe_dict = _remove_dictionary_columns(dataframe_dict)
    dataframe_dict = _rename_standard_columns(dataframe_dict)
    dataframe_dict = _normalize_sex_age(dataframe_dict)
    dataframe_dict = _map_sample_type(dataframe_dict)
    dataframe_dict, EXP_dict = _map_disease(dataframe_dict, EXP_dict)
    dataframe_dict, EXP_dict = _remove_non_cancer(dataframe_dict, EXP_dict)

    # Salva i pickle
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(INTERIM_DATA_DIR / 'dataframe_dict_alignment.pkl', 'wb') as f:
        pickle.dump(dataframe_dict, f)
    with open(INTERIM_DATA_DIR / 'EXP_dict_alignment.pkl', 'wb') as f:
        pickle.dump(EXP_dict, f)
    
    _sample_count(dataframe_dict)
    
    logger.success("Data alignment complete.")


def data_merging():
    logger.info("Merging data...")
    # Read the aligned dataframes from the previous step
    with open(INTERIM_DATA_DIR / 'dataframe_dict_alignment.pkl', 'rb') as f:
        dataframe_dict = pickle.load(f)

    with open(INTERIM_DATA_DIR / 'EXP_dict_alignment.pkl', 'rb') as f:
        EXP_dict = pickle.load(f)


    # Concat PD
    PD = pd.concat(dataframe_dict.values(), ignore_index=True)
    PD = PD[['Unnamed: 0', 'technology', 'sample_type', 'type', 'country', 'city',
              'sex', 'age', 'disease']].rename(columns={'Unnamed: 0': 'sample_id',
                                                         'sex': 'gender'})
    PD.set_index('sample_id', inplace=True)

    # Rename miRNA column 
    for df_id in EXP_dict:
        EXP_dict[df_id].rename(columns={'Unnamed: 0': 'miRNA'}, inplace=True)

    # Convert miRNA names to miRBase v22
    # see notebook 4. preprocessing_Dataset_Merging.ipynb for more details
    mirna_tsv = pd.read_csv(EXTERNAL_DATA_DIR / 'miRNA.tsv', sep='\t')
    mirna_name_map = dict(zip(mirna_tsv['OriginalName'], mirna_tsv['miRNAName_v22']))
 
    def convert_miRNA_name(old_name):
        old_name = old_name.strip().split('_v1')[0].split('_v2')[0]
        return mirna_name_map.get(old_name, None)
 
    for df_id in list(EXP_dict.keys()):
        EXP_dict[df_id]['miRNA'] = EXP_dict[df_id]['miRNA'].apply(convert_miRNA_name)
        EXP_dict[df_id].dropna(subset='miRNA', inplace=True)
        EXP_dict[df_id].drop_duplicates(subset='miRNA', inplace=True)
        EXP_dict[df_id] = EXP_dict[df_id].set_index('miRNA').T.rename_axis(index='sample_id')

    # Fix negative values (re-normalize log2 → log2(x+1))
    for df_id, df in EXP_dict.items():
        if df.min().min() < 0:
            original_values = np.exp2(df)
            EXP_dict[df_id] = np.log2(original_values + 1)
    
    # Fill NaN with 0 (within-study missing = absent probe) 
    for df_id in EXP_dict:
        if EXP_dict[df_id].isna().sum().sum() > 0:
            EXP_dict[df_id].fillna(value=0, inplace=True)
 
    # Concat EXP
    EXP = pd.concat(EXP_dict.values()).sort_index()
    EXP = EXP.loc[PD.index]

    # Remove diseases with < 50 samples per technology
    # see notebook 4. preprocessing_Dataset_Merging.ipynb for more details
    def drop_instance_from_dfs(drop_id_list):
        for df_id, df in dataframe_dict.items():
            mask = [instance_id not in drop_id_list for instance_id in df['Unnamed: 0']]
            dataframe_dict[df_id] = df[mask]
 
    def delete_non_represented_disease_for_tech(platform_technology):
        nonlocal PD, EXP
 
        if platform_technology == 'MicroArray Sequencing':
            mask = PD.technology != 'high-throughput sequencing'
        elif platform_technology == 'NextGeneration Sequencing':
            mask = PD.technology == 'high-throughput sequencing'
        else:
            return
 
        counts = PD.loc[mask, 'disease'].value_counts()
        counts = counts[counts < 50]
        disease_indices = PD[mask & PD['disease'].isin(counts.index)].index
 
        PD.drop(disease_indices, inplace=True)
        EXP.drop(disease_indices, inplace=True)
        drop_instance_from_dfs(disease_indices)
 
    delete_non_represented_disease_for_tech('MicroArray Sequencing')
    delete_non_represented_disease_for_tech('NextGeneration Sequencing')

    # Drop country and city columns
    PD.drop(columns=['country', 'city'], inplace=True)
 
    # Save final CSVs
    output_dir = PROCESSED_DATA_DIR
    exp_dir = output_dir / 'EXP'
    exp_dir.mkdir(parents=True, exist_ok=True)
 
    # Clear existing EXP files
    for f in exp_dir.glob('*.csv'):
        f.unlink()
 
    # Save EXP in chunks
    n_split = 13
    chunk_size = len(PD) // n_split
    for i in range(1, n_split + 1):
        start = (i - 1) * chunk_size
        end = i * chunk_size if i < n_split else len(PD)
        EXP.iloc[start:end].to_csv(exp_dir / f'EXP-{i}.csv')
 
    # Save PD
    PD.to_csv(output_dir / 'PD.csv')
 
    # Save MicroArray only
    PD_micro = PD[PD.technology != 'high-throughput sequencing']
    EXP_micro = EXP[PD.technology != 'high-throughput sequencing']
    PD_micro.to_csv(output_dir / 'PD_microArraySeq.csv')
    EXP_micro.to_csv(output_dir / 'EXP_microArraySeq.csv')

    # Save NGS only
    PD_ngs = PD[PD.technology == 'high-throughput sequencing']
    EXP_ngs = EXP[PD.technology == 'high-throughput sequencing']
    PD_ngs.to_csv(output_dir / 'PD_nextGenSeq.csv')
    EXP_ngs.to_csv(output_dir / 'EXP_nextGenSeq.csv')
 
    # Print stats
    logger.info(f"Studies remaining: {len(dataframe_dict)}")
    logger.info(f"Samples remaining: {len(PD)}")
    logger.info(f"Only Microarray: {len(PD[PD.technology != 'high-throughput sequencing'])}")
    logger.info(f"Only NGS: {len(PD[PD.technology == 'high-throughput sequencing'])}")

    logger.success("Data merging complete.")


def _sample_count(dataframe_dict: dict):
    logger.info(f"Studies remaining: {len(dataframe_dict)}")
    logger.info(f"Samples remaining: {sum([df.shape[0] for df in dataframe_dict.values()])}")


def _is_float(string):
    try:
        float(string)
        return True
    except (ValueError, TypeError):
        return False


def _sex_map_func(row_sex):
    pattern_male = re.compile(r'^m', re.IGNORECASE)
    pattern_female = re.compile(r'^f', re.IGNORECASE)
    if pd.isna(row_sex):
        return row_sex
    if pattern_male.match(row_sex.lower()):
        return 'M'
    elif pattern_female.match(row_sex.lower()):
        return 'F'
    return row_sex


def _age_map_func(row_age):
    if _is_float(row_age):
        return float(row_age)
    pattern_agey = re.compile(r'^\d+y$')
    if pattern_agey.match(str(row_age)):
        return float(str(row_age).split('y')[0])
    pattern_age_y = re.compile(r'^\d+\s+y$')
    if pattern_age_y.match(str(row_age)):
        return float(str(row_age).split(' y')[0])
    pattern_age_years = re.compile(r'^\d+\s+years$')
    if pattern_age_years.match(str(row_age)):
        return float(str(row_age).split(' y')[0])
    pattern_minor_agey = re.compile(r'^[<>]\d+y$')
    if pattern_minor_agey.match(str(row_age)):
        return float('nan')
    if str(row_age) in ['NA', 'NaN', 'n/a', 'None', 'NULL', 'nan', 'uncertain']:
        return float('nan')
    return float(row_age)



def _apply_rules(value: str, rules: list) -> str:
    """Apply a list of rules to a single cell value and return the label."""
    value_str = str(value)

    for rule in rules:

        if "contains" in rule and "label" in rule:
            keyword = rule["contains"]
            case_insensitive = rule.get("case_insensitive", False)
            haystack = value_str.lower() if case_insensitive else value_str
            needle = keyword.lower() if case_insensitive else keyword
            if needle in haystack:
                return rule["label"]

        elif "not_contains" in rule and rule.get("default_from_value"):
            keyword = rule["not_contains"]
            if keyword not in value_str.lower():
                words = value_str.split()
                if len(words) > 1:
                    return ' '.join([words[0]] + [w[0].lower() + w[1:] if w else '' for w in words[1:]])
                return value_str

        elif "contains_any" in rule and "label" in rule:
            keywords = rule["contains_any"]
            if any(kw.lower() in value_str.lower() for kw in keywords):
                return rule["label"]

        elif "default" in rule:
            return rule["default"]

        elif rule.get("default_from_value"):
            words = value_str.split()
            if len(words) > 1:
                return ' '.join([words[0]] + [w[0].lower() + w[1:] if w else '' for w in words[1:]])
            return value_str

    return None
    

def _map_labels(df: pd.DataFrame, config: dict) -> pd.Series:
    """Apply a label config to a DataFrame and return a Series of labels."""
    if "label" in config:
        return pd.Series([config["label"]] * len(df), index=df.index)
    col = config["check_column_name"]
    rules = config["rules"]
    return df[col].apply(lambda val: _apply_rules(str(val), rules))


def _remove_dictionary_columns(dataframe_dict: dict) -> dict:
    """Remove columns whose values are in 'key: value' format."""
    logger.info("Removing dictionary columns...")
    pattern = re.compile(r'^[^:]+: .+')
    for key, df in dataframe_dict.items():
        dict_cols = [
            col for col in df.columns
            if df[col].fillna('').astype(str).apply(lambda x: bool(pattern.match(x))).any()
        ]
        dataframe_dict[key].drop(columns=dict_cols, inplace=True)
    return dataframe_dict


def _rename_standard_columns(dataframe_dict: dict) -> dict:
    """Rename platform_id, contact_country, contact_city to standard names."""
    logger.info("Renaming standard columns...")
    for key in dataframe_dict:
        dataframe_dict[key].rename(columns={
            'platform_id': 'technology',
            'contact_country': 'country',
            'contact_city': 'city'
        }, inplace=True)
    return dataframe_dict


def _normalize_sex_age(dataframe_dict: dict) -> dict:
    """Normalize sex and age columns."""
    logger.info("Normalizing sex and age columns...")
    sex_pattern = re.compile(r'\b(sex|gender)\b')
    age_pattern = re.compile(r'.*\bage\b.*')

    for key, df in dataframe_dict.items():
        sex_cols = [col for col in df.columns if sex_pattern.match(col.lower())]
        if len(sex_cols) == 1:
            dataframe_dict[key][sex_cols[0]] = df[sex_cols[0]].apply(_sex_map_func)
            dataframe_dict[key].rename(columns={sex_cols[0]: 'sex'}, inplace=True)
        elif len(sex_cols) > 1:
            logger.warning(f"{key} ha più di una colonna sex/gender: {sex_cols}")

        age_cols = [col for col in df.columns if age_pattern.match(col.lower())]
        if len(age_cols) == 1:
            dataframe_dict[key][age_cols[0]] = df[age_cols[0]].apply(_age_map_func)
            dataframe_dict[key].rename(columns={age_cols[0]: 'age'}, inplace=True)
        elif len(age_cols) > 1:
            logger.warning(f"{key} ha più di una colonna age: {age_cols}")

    return dataframe_dict


def _map_sample_type(dataframe_dict: dict) -> dict:
    """Add sample_type and type columns using disease_labels.json."""
    logger.info("Mapping sample types...")

    # Load sample type configuration from JSON file
    # The JSON file has been manually created analyzing each single dataset and 
    # mapping the sample type labels to a common set of sample types, 
    # and to a common set of sample categories (type), based on the sample type 
    # (e.g. "plasma derived exosomes" is mapped to "exosomes EV" sample type and "plasma" type)
    # see notebook 3. preprocessing_Feature_Filtering.ipynb for more details 
    with open(EXTERNAL_DATA_DIR / 'sample_type_labels.json', 'r') as f:
        sample_type_config = json.load(f)

    stv_dict = {
        'Ascites derived exosomes': ('exosomes EV', 'biofluids'),
        'Cell derived exosomes': ('exosomes EV', 'cell'),
        'Cell derived extracellular vesicles': ('not-exosomes EV', 'cell'),
        'Cerebrospinal fluid derived exosomes': ('exosomes EV', 'biofluids'),
        'Fecal exosomes': ('exosomes EV', 'biofluids'),
        'Follicular fluid derived exosomes': ('exosomes EV', 'biofluids'),
        'Liquid milk derived exosomes': ('exosomes EV', 'biofluids'),
        'Plasma circulating miRNA': ('circulating', 'plasma'),
        'Plasma derived exosomes': ('exosomes EV', 'plasma'),
        'Plasma derived extracellular vesicles': ('not-exosomes EV', 'plasma'),
        'Saliva derived exosomes': ('exosomes EV', 'biofluids'),
        'Serum circulating miRNA': ('circulating', 'serum'),
        'Serum derived exosomes': ('exosomes EV', 'serum'),
        'Serum derived extracellular vesicles': ('not-exosomes EV', 'serum'),
        'Tissue derived exosomes': ('exosomes EV', 'tissue'),
        'Umbilical cord blood derived exosomes': ('exosomes EV', 'biofluids'),
        'Urinary derived exosomes': ('exosomes EV', 'biofluids'),
        'Urinary derived extracellular vesicles': ('not-exosomes EV', 'biofluids'),
        'Whole blood circulating miRNA': ('circulating', 'whole blood'),
        'Whole cell': ('whole cell', pd.NA),
    }

    for df_id, config in sample_type_config.items():
        if df_id not in dataframe_dict:
            continue
        dataframe_dict[df_id]['sample_type'] = _map_labels(dataframe_dict[df_id], config)

    def sample_type_values(sample_type):
        return stv_dict.get(sample_type, (pd.NA, pd.NA))

    for df_id in sample_type_config:
        if df_id not in dataframe_dict:
            continue
        dataframe_dict[df_id][['sample_type', 'type']] = (
            dataframe_dict[df_id]['sample_type']
            .apply(sample_type_values)
            .apply(pd.Series)
        )

    return dataframe_dict


def _map_disease(dataframe_dict: dict, EXP_dict: dict) -> tuple[dict, dict]:
    """Add disease column using disease_labels.json."""
    logger.info("Mapping diseases...")

    # Load disease configuration from JSON file
    # The JSON file has been manually created analyzing each single dataset, 
    # and mapping the disease labels to a common set of disease names (or "Healthy control" for healthy samples)
    # see notebook 3. preprocessing_Feature_Filtering.ipynb for more details
    with open(EXTERNAL_DATA_DIR / 'disease_labels.json', 'r') as f:
        disease_config = json.load(f)

    for df_id, config in disease_config.items():
        if df_id not in dataframe_dict:
            continue
        dataframe_dict[df_id]['disease'] = _map_labels(dataframe_dict[df_id], config)

    # Merge near-synonym labels (applied after mapping because some labels, e.g. "Sarcoma",
    # come from default_from_value and are not literal in the JSON)
    for df in dataframe_dict.values():
        if 'disease' in df.columns:
            df['disease'] = df['disease'].replace(DISEASE_MERGE_MAP)

    return dataframe_dict, EXP_dict


def _remove_non_cancer(dataframe_dict: dict, EXP_dict: dict) -> tuple[dict, dict]:
    """Remove datasets or samples with only non-oncological diseases."""
    logger.info("Removing non-cancer samples...")

    # see notebook 3. preprocessing_Feature_Filtering.ipynb for more details
    NON_cancer_diseases = {
        'Acquired immune deficiency syndrome', 'Adult temporal lobe epilepsy',
        'Benign ovarian disease', 'Bronchial disease', 'Bronchopulmonary dysplasis',
        'Cholelithiasi', 'Chronic hepatitis', 'Chronic hepatitis B', 'Chronic hepatitis C',
        'Chronic pancreatitis', 'Cognitive decline', 'Early podocyte injury',
        'Fulminant myocarditis', 'Gestational diabetes mellitus', 'Hand,foot and mouth disease',
        'Heat stroke', 'Immune related disease', 'Inflammatory periodontal ligament disease',
        'Intracranial atherosclerosis associated with antiangiogenesis', 'Kawasaki disease',
        'Liver cirrhosis', 'Multiple sclerosis disease', 'Negative prostate biopsy',
        'Non-alcoholic steatohepatitis', 'OV_others', 'Obesity-related disease', 'Oocyte quality',
        'Oral lichen planus', 'Oxalate nephropathy', 'Pediatric cystic fibrosis',
        'Perioperative myocardial injury', 'Polycystic ovary syndrome', 'Pre-term birth',
        'Preeclampsia', 'Preterm birth pregnancies', 'Primary human T lymphoblasts',
        'Skin disease', 'Spinal Cord dysfunction', 'Subarachnoid hemorrhage', 'Tuberculosis',
        'Type 1 autoimmune pancreatitis', 'Vascular disease',
        'Ventricular Septal Defect in the fetus'
    }

    for df_id, df in dataframe_dict.copy().items():

        diseases_list = set(df['disease'].unique())
        if 'Healthy control' in diseases_list:
            diseases_list.remove('Healthy control')

        ds_num = len(diseases_list)

        if ds_num > 0:
            ds_NON_cancer_num = sum([
                True if pd.isna(d) else d in NON_cancer_diseases
                for d in diseases_list
            ])

            if ds_num == ds_NON_cancer_num:
                dataframe_dict.pop(df_id, None)
                EXP_dict.pop(df_id, None)
                logger.debug(f"Dataset {df_id} removed (only non-oncological diseases)")

            elif ds_NON_cancer_num > 0:
                mask = [True if pd.isna(d) else d in NON_cancer_diseases for d in df['disease']]
                sample_id_to_drop = df[mask]['Unnamed: 0']
                dataframe_dict[df_id] = df[~np.array(mask)]
                EXP_dict[df_id].drop(columns=sample_id_to_drop, inplace=True)
                logger.debug(f"Dataset {df_id} modified: removed non-oncological samples")

    return dataframe_dict, EXP_dict


if __name__ == "__main__":
    app()
