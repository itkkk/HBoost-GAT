import glob
import os
from typing import Literal, Optional

from annoy import AnnoyIndex
import numpy as np
import pandas as pd


def get_datasets(dataset_PATH= "DEFAULT_PATH/", platform_technology: Optional[Literal['MicroArray', 'NGS']] = None, discard_none_mirna= True):
    """
    Return a specific dataset partion.

    Parameters
    ----------
    dataset_PATH : str, default='DEFAULT_PATH/'
        Path where to find the dataset .csv files.
    platform_technology : {'MicroArray', 'NGS'}, optional
        If set, select samples obtained only for the specified technology.
    discard_none_mirna : bool, default=True
        Dataset EXP will not contain miRNAs with all missing values. It works only if `platform_technology` is 'MicroArray'.

    Returns
    -------
    PD : pandas.DataFrame
        Dataframe containing sample metadata.
    EXP : pandas.DataFrame
        Dataframe containing miRNA expression levels in samples.
    """

    # PARAMETERS CHECK AREA---

    allowed_pt = [None, 'MicroArray', 'NGS']
    if platform_technology not in allowed_pt:
        raise ValueError(f"Invalid platform_technology value: {platform_technology}. Must be one of {allowed_pt}.")

    # ---END parameters check area

    PD = pd.read_csv(os.path.join(dataset_PATH, 'PD.csv'), index_col="sample_id")

    # Concat the slitted files
    EXP_csv_files = glob.glob( os.path.join(dataset_PATH, 'EXP', '*.csv') )
    EXP = pd.concat( [pd.read_csv(file, index_col='sample_id') for file in EXP_csv_files] )

    if platform_technology == "MicroArray":

        mask = PD.loc[PD.technology != "high-throughput sequencing", :].index
        
        # Select MicroArray instances only
        PD_ma = PD.loc[mask, :]
        EXP_ma = EXP.loc[mask, :]

        if discard_none_mirna:
            # Select miRNAs with at least one expression level
            EXP_ma = EXP_ma.loc[:, EXP_ma.isna().all(axis=0) == False]

        return PD_ma.drop(columns='technology'), EXP_ma

    elif platform_technology == "NGS":

        mask = PD.loc[PD.technology == "high-throughput sequencing", :].index

        # Select NGS instances only
        PD_ngs = PD.loc[mask, :]
        EXP_ngs = EXP.loc[mask, :]

        return PD_ngs.drop(columns='technology'), EXP_ngs.fillna(value=0)    
    
    else:
        return PD, EXP





def get_neighbors(df, DIM_neighborhood= 1, neighbors_index_PATH="DEFAULT_INDEX_FILE_PATH/", name_index_file= "DEFAULT_NAME.ann", rebuild_index_file= False, split: Optional[Literal['train', 'test']] = None, test: Optional[list] = None):

    """
    For each instance in the dataset, find the specific number of nearest neighbors.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to be indexed. All features must be numeric.
    DIM_neighborhood : int, default=1
        Number of neighbors to find for each instance (neighborhood size).
    neighbors_index_PATH : str, default='./DEFAULT_INDEX_FILE_PATH/'
        Path where to create or find the index file.
    name_index_file : str, default='DEFAULT_NAME.ann'
        Name of the index file to build or use.
    rebuild_index_file : bool, default=False
        Whether to rebuild (overwrite) the index file.
    split : {'train', 'test'}, optional
        If set, find neighborhoods only for the specified subset.
    test : array-like of bool, shape (n_samples,), optional
        Boolean mask indicating training/test instances. Required if `split` is not None.

    Returns
    -------
    central_nodes : list of int
        Indices of the central (query) nodes.
    neighbor_nodes : list of int
        Indices of the neighboring nodes for each central node.
    distances : list of float
        Distances between each central node and its neighbors.
    """
    
    # PARAMETERS CHECK AREA---

    allowed_splits = [None, 'train', 'test']
    if split not in allowed_splits:
        raise ValueError(f"Invalid split value: {split}. Must be one of {allowed_splits}.")
    
    if not test is None: 
        if len(test) != df.shape[0]:
            raise ValueError(f"Invalid test dimension: {len(test)}. Must be equal to df.shape[0] {df.shape[0]}.")
        if (not isinstance(test[0], (bool, np.bool_))):
            raise ValueError(f"Invalid test elements type: {type(test[0])}. Must be bool type.")
    
    # ---END parameters check area
    

    if split is None:
        INDEX_PATH = neighbors_index_PATH + name_index_file
    else:
        INDEX_PATH = neighbors_index_PATH + split +'_'+ name_index_file

    NUM_F = df.shape[1]     # Instance features number
    sample_list= []         # Instances on which neighborhoods are calculated
    neighbors_list= []      # Instances belonging to the neighborhoods
    distances_list= []

    
    # If index file doesn't already exist or you want to force the reconstruction, generate it
    if (not os.path.isfile(INDEX_PATH)) | rebuild_index_file:
    
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
            # print(f"Deleted: {INDEX_PATH}")   # DEBUG
        else:
            pass
            # print(f"Not found: {INDEX_PATH}") # DEBUG
        
        t = AnnoyIndex(NUM_F, 'angular')

        for i, (index, row) in enumerate(df.iterrows()):
            match split:
                case 'train':
                        # Add trainig instances only
                        if not test[i]:
                            t.add_item(i, row.values)

                case 'test' | _:
                        t.add_item(i, row.values)

        t.build(10) # 10 trees
        t.save(INDEX_PATH)



    u = AnnoyIndex(NUM_F, 'angular')
    u.load(INDEX_PATH) # super fast, will just mmap the file

    match split:
        case 'train':

            OFFSET = 1  # Enables dropping a node (itself) from the neighborhood

            for i in range(df.shape[0]):
                
                # Find the neighborhood for trainig instances only
                if not test[i]:
                    nl, dl = u.get_nns_by_item(i, DIM_neighborhood + OFFSET, include_distances=True)
                    
                    if i in nl:    
                        
                        # Drop node i from the neighborhood
                        dl.pop(nl.index(i))
                        nl.remove(i)
                    else:

                        # Drop farthest neighbor node
                        dl.pop(-1)
                        nl.pop(-1)                  
                    
                    sample_list += [i] * DIM_neighborhood
                    neighbors_list += nl
                    distances_list += dl

        case 'test':

            test = np.array(test)
            OFFSET = test.sum() # test instances number

            for i in range(df.shape[0]):
                
                # Find the neighborhood for test instances only
                if test[i]:
                    nl, dl  = u.get_nns_by_item(i, DIM_neighborhood + OFFSET, include_distances=True)
                    nl, dl = np.array(nl), np.array(dl)

                    # Discard the test instances from the neighborhood and select the firt N nearest training instances
                    train_mask = test[nl] == False # Fancy Indexing

                    nl = nl[train_mask][:DIM_neighborhood]
                    dl = dl[train_mask][:DIM_neighborhood]
                    
                    sample_list += [i] * DIM_neighborhood
                    neighbors_list += nl.tolist()
                    distances_list += dl.tolist()

        case _:

            for i in range(df.shape[0]):
                nl, dl  = u.get_nns_by_item(i, DIM_neighborhood, include_distances=True)
                
                sample_list += [i] * DIM_neighborhood
                neighbors_list += nl
                distances_list += dl

    

    return sample_list, neighbors_list, distances_list


def get_sample_weights(PD_fold: pd.DataFrame) -> pd.Series:
    """
    Computes log-based inverse-frequency sample weights to counterbalance class imbalance.
    """
    info = PD_fold.disease.value_counts()
    inv_p = 1 / (PD_fold.disease.map(lambda m: info[m]) / info.sum())
    weights = np.log(inv_p) / np.log(inv_p).sum()
    return weights