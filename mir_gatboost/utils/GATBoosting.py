import copy
import os
import random

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import cross_val_predict
from sklearn.utils import shuffle
import torch
from torch import compile
import torch.nn.functional as F
from torch.nn.functional import cross_entropy
from torch_geometric import seed_everything
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import ToUndirected
from tqdm import tqdm

from mir_gatboost.utils.dataset_manager import get_neighbors
from mir_gatboost.utils.loss_functions import FocalLoss


class FocalLossWeighed(FocalLoss):

    def forward(self, logits, labels, weights):
        r"""Forward pass to compute the Focal Loss. 

        Args:
            logits: Predictions (logits) from the model.
                       Shape:
                         - multi-class: (batch_size, num_classes)
            labels: Ground truth labels.
                        Shape:
                         - multi-class: (batch_size)
        """
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = torch.pow((1 - pt), self.gamma.to(logits.device)) * ce_loss

        # return whole focal_loss
        return (focal_loss * weights).sum()


def graphs_constructor(PD, EXP, EMB_MIRNA, emb_view, index_PATH="PATH_DI_DEFAULT/",  rebuild_index_file=False):
    graph = HeteroData()

    # SAMPLE node    
    
    # Set features for 'sample' nodes 
    graph['sample'].x = torch.tensor(emb_view.values, dtype=torch.float)
    
    # Set class labels
    graph['sample'].y = torch.tensor(PD.disease.factorize()[0])
    
    # Set train and test masks
    graph['sample'].train_mask = torch.tensor(~PD['test'].values)
    graph['sample'].test_mask = torch.tensor(PD['test'].values)

    # MIRNA node
    
    # Set Feature    
    graph['mirna'].x = torch.tensor(EMB_MIRNA.values)

    # SAMPLE <-> MIRNA edges
    rows, cols = np.where(~np.isnan(EXP))
    graph['sample', 'mirna'].edge_index = torch.tensor(np.array([rows, cols]))
    graph['sample', 'mirna'].edge_attr = torch.tensor(EXP.values[rows, cols], dtype=torch.float)
    graph = ToUndirected()(graph)

    # SAMPLE <-> SAMPLE edges
    sl_train, nl_train, dl_train = get_neighbors(
        emb_view,
        DIM_neighborhood=100,
        neighbors_index_PATH=index_PATH,
        name_index_file="sample_neighbors.ann",
        rebuild_index_file=rebuild_index_file,
        split='train',
        test=PD.test.to_list()
    )
    sl_test, nl_test, dl_test = get_neighbors(
        emb_view,
        DIM_neighborhood=100,
        neighbors_index_PATH=index_PATH,
        name_index_file="sample_neighbors.ann",
        rebuild_index_file=rebuild_index_file,
        split='test',
        test=PD.test.to_list()
    )
    sample_list = sl_train + sl_test
    neighbors_list = nl_train + nl_test
    distances_list = dl_train + dl_test
    graph['sample', 'sample'].edge_index = torch.tensor([neighbors_list, sample_list])
    graph['sample', 'sample'].edge_attr = torch.tensor(distances_list, dtype=torch.float)

    # MIRNA <-> MIRNA edges
    miRNA_list, mirna_neighbors_list, mirna_distances_list = get_neighbors(
        EMB_MIRNA,
        DIM_neighborhood=10,
        neighbors_index_PATH=index_PATH,
        name_index_file="miRNA_neighbors.ann",
        rebuild_index_file=rebuild_index_file,
        split=None
    )
    graph['mirna', 'mirna'].edge_index = torch.tensor([mirna_neighbors_list, miRNA_list])
    graph['mirna', 'mirna'].edge_attr = torch.tensor(mirna_distances_list, dtype=torch.float)

    graph.validate(raise_on_error=True)
    return graph


class BoostSH(BaseEstimator, ClassifierMixin):

    def __init__(self, base_estimator, n_iter=10, learning_rate=1., num_layers=1):
        """
        Boost SH: Boosting classification algorithm for multiview with shared weights.
        Greedy approach in which each view is tested to evaluate the one with larger edge.

        Arguments:
            base_estimator: Base GNN classifier to use on each view.
            n_iter: Number of boosting iterations.
            learning_rate: Learning rate for boosting (default: 1).
            num_layers: Number of GAT layers; controls NeighborLoader sampling depth (default: 1).
        """
        super(BoostSH, self).__init__()
        self.base_estimator = base_estimator
        self.views = {}

        self.models = []
        self.classes = []
        self.alphas = []
        self.views_selected = []
        self.weights = []
        self.eps = 10 ** (-6)

        self.n_iter = n_iter
        self.epoch = 100
        self.learning_rate = learning_rate
        
        self.num_layers = num_layers
        self.graphs = {}
        self.model_PATH = ""
        self.model_NAME = ""

    def fit(self, X, y, graphs, model_PATH, model_NAME, num_epoch=100, forecast_cv=None, sample_weights=None):
        """
        Fit the model by adding models in an adaboost fashion.

        Arguments:
            X: Dict of pd.DataFrame — views to use for the task.
            y: pd.Series — labels; index must be contained in view union.
            forecast_cv: Number of folds used to estimate the edge
                (default: None — performance computed on training set).
        """
        self.epoch = num_epoch
        self.check_input(X, y)

        self.views = copy.deepcopy(X)
        self.classes = np.unique(y)

        index = self.__index_union__(self.views)
        y = y.reindex(index)

        # Initialize distribution over weights
        self.initialize_weights(sample_weights, index)
        self.weights = pd.Series(self.weights)
        self.graphs = graphs
        self.model_PATH = model_PATH
        self.model_NAME = model_NAME

        for t in range(self.n_iter):

            if self.weights.sum() == 0:
                break

            self.weights /= self.weights.sum()

            selected = {'max_edge': -np.inf}
            for v in self.views:
                mask = self.views[v].index.tolist()
                weak_forecast = self.__compute_weak_forecast__(self.views[v], v, t, y[mask], self.weights[mask], forecast_cv, True)
                tmp = 2 * ((weak_forecast['forecast'] == y[mask].values).float() - .5).cpu().numpy()
                edge = (self.weights[mask].values * tmp).sum()
                if edge > selected['max_edge']:
                    selected = {'max_edge': edge, 'view': v, 'forecast': weak_forecast['forecast'], 'mask': mask,
                                'model': weak_forecast['model'], 'tmp': tmp}

            if (1 - selected['max_edge']) < self.eps:
                alpha = self.learning_rate * .5 * 10.
            else:
                alpha = self.learning_rate * .5 * np.log((1 + selected['max_edge']) / (1 - selected['max_edge']))

            # Update weights
            self.weights[selected['mask']] *= np.exp(- alpha * selected['tmp'])

            self.models.append(selected['model'])
            self.alphas.append(alpha)
            self.views_selected.append(selected['view'])

            logger.info(f'Iteration {t}')
            logger.info(f'Winning view {selected["view"]}')
            logger.info(f'Edge {selected["max_edge"]}')

        return

    def initialize_weights(self, sample_weights=None, index=None):
        if sample_weights is None:
            # Initialize uniform distribution over weights
            self.weights = pd.Series(1, index=index)
        else:
            # Assign pre-defined distribution over weights
            self.weights = pd.Series(sample_weights, index=index)

    def __compute_weak_forecast__(self, data, model_num, iter_num, labels, weights, forecast_cv=None, return_model=False, fold_number=None):
        weak_forecast = dict()
        model = copy.deepcopy(self.base_estimator)
        graph_copy = copy.deepcopy(self.graphs[model_num])

        graph_copy['sample'].w = torch.zeros(graph_copy['sample'].num_nodes)
        graph_copy['sample'].w[graph_copy['sample'].train_mask] = torch.tensor(weights.values, dtype=torch.float)

        seed = 42
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        compile(model)
        model.to(device)

        focal_loss = FocalLossWeighed(gamma=1, reduction='mean')

        # Hamilton et al., 2017 (GraphSAGE)
        mirna_to_sample_neighbors = [-1] + [25] * (self.num_layers - 1)
        sample_to_mirna_neighbors = [-1] + [25] * (self.num_layers - 1)

        train_loader = NeighborLoader(
            graph_copy,
            input_nodes=('sample', graph_copy['sample'].train_mask),
            num_neighbors={
                ("sample", "to", "mirna"): sample_to_mirna_neighbors,
                ("sample", "to", "sample"): [-1] * self.num_layers,
                ("mirna", "to", "mirna"): [-1] * self.num_layers,
                ('mirna', 'rev_to', 'sample'): mirna_to_sample_neighbors,
            },
            subgraph_type='directional',
            batch_size=512,
            shuffle=True
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        infoTraining = {
            'model': None,
            'optimizer': None,
            'epoch': None,
            'loss': None,
            'gamma': None
        }

        model.train()
        for epoch in range(1, self.epoch + 1):
            total_loss = 0
            out = []
            id_nodes = []

            for batch in tqdm(train_loader):
                batch['sample', 'to', 'mirna'].edge_index = batch['mirna', 'rev_to', 'sample'].edge_index.flip(0)
                batch['sample', 'to', 'mirna'].edge_attr = batch['mirna', 'rev_to', 'sample'].edge_attr
                batch = batch.to(device)

                optimizer.zero_grad()
                out_batch = model(batch.x_dict, batch.edge_index_dict, edge_attr=batch.edge_attr_dict)
                out.append(out_batch['sample'][:batch['sample'].batch_size])
                id_nodes.append(batch["sample"].n_id[:batch['sample'].batch_size])
                loss = focal_loss(out_batch['sample'], batch['sample'].y, batch['sample'].w)
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * batch['sample'].batch_size

            out = torch.cat(out, dim=0)
            id_nodes = torch.cat(id_nodes, dim=0)
            total_loss /= int(graph_copy['sample'].train_mask.sum())

            logger.info(f'[Iteration {iter_num}] - [Fold {fold_number}] - Epoch: {epoch:03d}, Loss: {total_loss:.4f}')

            if epoch % 20 == 0:
                infoTraining['model'] = model.state_dict()
                infoTraining['optimizer'] = optimizer.state_dict()
                infoTraining['epoch'] = epoch
                infoTraining['loss'] = total_loss
                infoTraining['gamma'] = list(focal_loss.parameters())[0].item()
                logger.info(f'[Iteration {iter_num}] - [Fold {fold_number}] - Saving checkpoint for epoch {epoch}')
                torch.save(infoTraining, os.path.join(self.model_PATH, self.model_NAME + f'{model_num}_{iter_num}.pt'))

        sorted_id_nodes, sorter_indices = torch.sort(id_nodes)
        out = out[sorter_indices]
        weak_forecast['forecast'] = out.argmax(dim=1).detach().cpu()
        if return_model:
            weak_forecast['model'] = model

        return weak_forecast

    def predict_proba(self, X):
        index = self.__index_union__(X)
        predictions = np.zeros((len(index), len(self.classes)))

        # Hamilton et al., 2017 (GraphSAGE)
        mirna_to_sample_neighbors = [-1] + [25] * (self.num_layers - 1)
        sample_to_mirna_neighbors = [-1] + [25] * (self.num_layers - 1)

        for t in range(len(self.models)):
            if self.views_selected[t] in X.keys():
                X_test = X[self.views_selected[t]]
                int_index = [list(index).index(i) for i in X_test.index]

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                checkpoint = torch.load(
                    os.path.join(self.model_PATH, self.model_NAME + f"{self.views_selected[t]}_{t}.pt"),
                    map_location=device
                )

                model = copy.deepcopy(self.base_estimator)
                model.load_state_dict(checkpoint['model'])
                model.to(device)

                graph_copy = copy.deepcopy(self.graphs[self.views_selected[t]])
                test_loader = NeighborLoader(
                    graph_copy,
                    input_nodes=('sample', graph_copy['sample'].test_mask),
                    num_neighbors={
                        ("sample", "to", "mirna"): sample_to_mirna_neighbors,
                        ("sample", "to", "sample"): [-1] * self.num_layers,
                        ("mirna", "to", "mirna"): [-1] * self.num_layers,
                        ('mirna', 'rev_to', 'sample'): mirna_to_sample_neighbors,
                    },
                    subgraph_type='directional',
                    batch_size=400,
                    shuffle=False
                )

                out = []
                model.eval()
                with torch.no_grad():
                    for batch in tqdm(test_loader):
                        batch['sample', 'to', 'mirna'].edge_index = batch['mirna', 'rev_to', 'sample'].edge_index.flip(0)
                        batch['sample', 'to', 'mirna'].edge_attr = batch['mirna', 'rev_to', 'sample'].edge_attr
                        batch = batch.to(device)
                        out_batch = model(batch.x_dict, batch.edge_index_dict, edge_attr=batch.edge_attr_dict)
                        out.append(out_batch['sample'][:batch['sample'].batch_size])

                predictions[int_index] += self.alphas[t] * F.softmax(torch.cat(out, dim=0).cpu(), dim=1).numpy()

        predictions = pd.DataFrame(predictions, index=index).loc[X_test.index]
        return predictions / predictions.sum(axis=1).to_numpy()[:, None]

    def predict(self, X):
        self.check_X(X)
        assert len(self.models) > 0, 'Model not trained'
        return self.predict_proba(X).idxmax(axis=1)

    def check_input(self, X, y):
        self.check_X(X)
        self.check_y(y)

    def check_X(self, X):
        assert isinstance(X, dict), "Not right format for X"
        for key in X.keys():
            assert isinstance(X[key], pd.DataFrame)
            assert not X[key].empty, "Empty dataframe"

    def check_y(self, y):
        assert isinstance(y, pd.Series), "Not right format for y"
        assert not y.empty, "Empty dataframe"

    def __index_union__(self, views):
        self.check_X(views)
        index = set([])
        for view in views:
            index = index.union(set(views[view].index))
        return list(index)


class IRBoostSH(BoostSH):

    def __init__(self, base_estimator, n_iter=10, learning_rate=1., num_layers=1, sigma=0.15, gamma=0.3):
        """
        IRBoost SH: Boosting classification for multiview with shared weights.
        Multi-arm bandit approach in which a view is selected at each iteration.

        Arguments:
            base_estimator: Base GNN classifier to use on each view.
            n_iter: Number of boosting iterations.
            learning_rate: Learning rate for boosting (default: 1).
            num_layers: Number of GAT layers; controls NeighborLoader sampling depth (default: 1).
            sigma: Exploration parameter for the bandit (default: 0.15).
            gamma: Exploitation/exploration tradeoff for the bandit (default: 0.3).
        """
        super(IRBoostSH, self).__init__(base_estimator, n_iter, learning_rate, num_layers)
        self.sigma = sigma
        self.gamma = gamma

    def fit(self, X, y, graphs, model_PATH, model_NAME, num_epoch=100, forecast_cv=None, sample_weights=None, fold_number=None):
        """
        Fit the model by adding models in a boosting fashion.

        Arguments:
            X: Dict of pd.DataFrame — views to use for the task.
            y: pd.Series — labels; index must be contained in view union.
            forecast_cv: Number of folds used to estimate the edge
                (default: None — performance computed on training set).
        """
        self.epoch = num_epoch
        self.check_input(X, y)

        self.views = copy.deepcopy(X)
        self.classes = np.unique(y)
        K = len(self.views)
        possible_views = list(self.views.keys())

        index = self.__index_union__(self.views)
        y = y.reindex(index)

        self.initialize_weights(sample_weights, index)
        self.weights = pd.Series(self.weights)

        self.graphs = graphs
        self.model_PATH = model_PATH
        self.model_NAME = model_NAME

        p_views = pd.Series(np.exp(self.sigma * self.gamma / 3 * np.sqrt(self.n_iter / K)), index=possible_views)

        for t in range(self.n_iter):

            if self.weights.sum() == 0:
                break

            self.weights /= self.weights.sum()

            # Bandit selection of best view
            q_views = (1 - self.gamma) * p_views / p_views.sum() + self.gamma / K
            selected_view = np.random.choice(possible_views, p=q_views)

            mask = self.views[selected_view].index.tolist()
            weak_forecast = self.__compute_weak_forecast__(
                self.views[selected_view], selected_view, t,
                y[mask], self.weights[mask], forecast_cv, True, fold_number=fold_number
            )

            tmp = 2 * ((weak_forecast['forecast'] == y[mask].values).float() - .5).cpu().numpy()
            edge = (self.weights[mask].values * tmp).sum()

            if (1 - edge) < self.eps:
                alpha = self.learning_rate * .5 * 10.
            elif edge <= 0:
                alpha = 0
            else:
                alpha = self.learning_rate * .5 * np.log((1 + edge) / (1 - edge))

            self.weights[mask] *= np.exp(- alpha * tmp)

            # Update arm probability
            r_views = pd.Series(0.0, index=possible_views)
            square = np.sqrt(1 - edge ** 2) if edge < 1 else 0
            r_views[selected_view] = (1 - square) / q_views[selected_view]
            p_views *= np.exp(self.gamma / (3 * K) * (r_views + self.sigma / (q_views * np.sqrt(self.n_iter * K))))

            self.models.append(weak_forecast['model'])
            self.alphas.append(alpha)
            self.views_selected.append(selected_view)

            logger.info(f'Iteration {t}')
            logger.info(f'Winning view {selected_view}')
            logger.info(f'Edge {edge}')

        return

    def view_weights(self):
        """Return relative importance of the different views in the final decision."""
        assert len(self.models) > 0, 'Model not trained'
        view_weights = pd.DataFrame({"view": self.views_selected, "alpha": np.abs(self.alphas)})
        return (view_weights.groupby('view').sum() / np.sum(np.abs(self.alphas))).sort_values('alpha')

    def save_views(self, path):
        df = pd.DataFrame(self.views_selected)
        file = path if str(path).endswith('xlsx') else os.path.join(path, 'winning views.xlsx')
        df.to_excel(file)
