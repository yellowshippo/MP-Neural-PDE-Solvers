import pathlib

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
from torch import nn
from typing import Tuple
from torch_geometric.data import Data
from torch_cluster import radius_graph


NS_STEP = 40
NS_DT = .1
NS_DX = (0.3337 + 0.8592) / 45
T_RESOLUTION = NS_STEP

DEBUG = False


class NpyDataset(Dataset):
    """Load samples of an PDE Dataset, get items according to PDE"""

    def __init__(
            self,
            path: pathlib.Path,
            pde: str,
            mode: str,
            load_all: bool = False,
            time_window: int = 10) -> None:
        """Initialize the dataset object
        Args:
            path: path to dataset
            pde: string of PDE ('CE' or 'WE')
            mode: [train, valid, test]
            base_resolution: base resolution of the dataset [nt, nx]
            super_resolution: super resolution of the dataset [nt, nx]
            load_all: load all the data into memory
            time_window: time window
        Returns:
            None
        """
        super().__init__()
        self.mode = mode
        self.pde = pde
        self.dtype = torch.float64
        self.tw = time_window
        if self.pde == 'ns':
            self.steps = NS_STEP
            self.nt = NS_STEP + self.tw  # Take into account padding
            self.dt = NS_DT
            self.tmin = 0.
            self.tmax = self.nt * self.dt  # Take into account padding
            self.required_file_name = 'nodal_U_step0.npy'
        self.data_directories = self._collect_directories(path)
        return

    def _collect_directories(self, path):
        if (path / 'list.csv').is_file():
            print(f"Reading list in: {path}")
            with open(path / 'list.csv') as f:
                lines = f.readlines()
            return [pathlib.Path(line.strip()) for line in lines]
        else:
            print(f"Searching data in: {path}")
            return [
                f.parent for f in path.glob(f"**/{self.required_file_name}")]

    def __len__(self):
        return len(self.data_directories)

    def __getitem__(self, idx: int) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Get data item
        Args:
            idx (int): data index
        Returns:
            torch.Tensor: numerical baseline trajectory
            torch.Tensor: downprojected high-resolution trajectory
                          (used for training)
            torch.Tensor: spatial coordinates
            list: equation specific parameters
        """
        if f'{self.pde}' == 'ns':
            data_directory = self.data_directories[idx]
            x = np.load(data_directory / 'node.npy')
            u_base = torch.from_numpy(np.stack(
                [
                    np.concatenate(
                        [
                            np.load(data_directory / f"nodal_U_step{i}.npy")[
                                ..., 0],
                            np.load(data_directory / f"nodal_p_step{i}.npy"),
                        ], axis=-1)
                    for i
                    in [0] * (self.tw - 1) + list(range(self.steps + 1))],
                # Pad first step for temporal bundling
                axis=0))
            u_super = u_base.clone()

            dirichlet = np.concatenate(
                [
                    np.load(data_directory / 'nodal_boundary_U.npy')[..., 0],
                    np.load(data_directory / 'nodal_boundary_p.npy'),
                ], axis=-1)
            dirichlet_flag = ~np.isnan(dirichlet)
            dirichlet[np.isnan(dirichlet)] = 0.
            normal = np.load(data_directory / 'normal.npy')[..., 0]
            scipy_adj = sp.load_npz(data_directory / 'nodal_adj.npz')
            adj = torch.sparse_coo_tensor(
                np.stack([scipy_adj.row, scipy_adj.col], axis=0),
                scipy_adj.data, scipy_adj.shape)

            variables = {
                'dirichlet': torch.from_numpy(dirichlet),  # [n_node, 4]
                'dirichlet_flag': torch.from_numpy(
                    dirichlet_flag.astype(np.float32)),  # [n_node, 4]
                'normal': torch.from_numpy(normal),  # [n_node, 3]
                'level1': torch.from_numpy(np.load(
                    data_directory / 'nodal_level1.npy')),  # [n_node, 1]
                'level2': torch.from_numpy(np.load(
                    data_directory / 'nodal_level2.npy')),  # [n_node, 1]
                'level0p5': torch.from_numpy(np.load(
                    data_directory / 'nodal_level0p5.npy')),  # [n_node, 1]
                'adj': adj,  # [n_node, n_node], sparse
            }

            return u_base, u_super, x, variables

        else:
            raise Exception("Wrong experiment: {self.pde}")


class GraphCreator(nn.Module):

    def __init__(
            self,
            pde: str,
            neighbors: int = 2,
            time_window: int = 5) -> None:
        """
        Initialize GraphCreator class
        Args:
            pde (PDE): PDE at hand [CE, WE, ...]
            neighbors (int): how many neighbors the graph has in each direction
            time_window (int): how many time steps are used for PDE prediction
        Returns:
            None
        """
        super().__init__()
        self.pde = pde
        self.n = neighbors
        self.tw = time_window
        if self.pde == 'ns':
            self.nt = NS_STEP + self.tw  # Take into account padding
            self.dt = NS_DT
            self.dx = NS_DX
            self.tmin = 0.
            self.tmax = NS_STEP * NS_DT
            self.t_res = T_RESOLUTION + self.tw
        else:
            raise ValueError(f"Invalid pde type: {self.pde}")

        assert isinstance(self.n, int)
        assert isinstance(self.tw, int)

        return

    def create_data(
        self, datapoints: torch.Tensor, steps: list) -> Tuple[
            torch.Tensor, torch.Tensor]:
        """
        Getting data for PDE training at different time points
        Args:
            datapoints (torch.Tensor):
                trajectory [batch, time, node, feature]-shaped tensor.
            steps (list):
                list of different starting points for each batch entry
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: input data and label
        """
        data = torch.Tensor()
        labels = torch.Tensor()
        for (dp, step) in zip(datapoints, steps):
            d = dp[step - self.tw:step]
            label = dp[step:self.tw + step]
            data = torch.cat((data, d[None, :]), 0)
            labels = torch.cat((labels, label[None, :]), 0)

        return data, labels

    def create_graph(
            self,
            data: torch.Tensor,  # [batch, time, node, feature]
            labels: torch.Tensor,  # [batch, time, node, feature]
            x: torch.Tensor,  # [batch, node, feature]
            variables: dict,
            steps: list) -> Data:
        """
        Getting graph structure out of data sample
        previous timesteps are combined in one node
        Args:
            data (torch.Tensor): input data tensor
            labels (torch.Tensor): label tensor
            x (torch.Tensor): spatial coordinates tensor
            variables (dict): dictionary of equation specific parameters
            steps (list):
                list of different starting points for each batch entry
        Returns:
            Data: Pytorch Geometric data graph
        """
        t = torch.linspace(0, self.tmax, self.nt)
        nx = x.shape[-2]

        u, x_pos, t_pos, y, batch = torch.Tensor(), torch.Tensor(), \
            torch.Tensor(), torch.Tensor(), torch.Tensor()
        for b, (data_batch, labels_batch, step) in enumerate(
                zip(data, labels, steps)):
            u = torch.cat(
                (u, torch.transpose(torch.cat(
                    [d[None, :] for d in data_batch]), 0, 1)))
            y = torch.cat(
                (y, torch.transpose(torch.cat(
                    [label[None, :] for label in labels_batch]), 0, 1)))
            x_pos = torch.cat((x_pos, x[0]))
            t_pos = torch.cat((t_pos, torch.ones(nx, 1) * t[step]))
            batch = torch.cat((batch, torch.ones(nx) * b))

        # Calculate the edge_index
        if f'{self.pde}' == 'ns':
            radius = (self.n + .01) * self.dx
            edge_index = radius_graph(
                x_pos, r=radius, batch=batch.long(), loop=False)
            # NOTE: max_num_neighbors is 32 by default

        else:
            raise ValueError(f"Invalid PDE type: {self.pde}")

        graph = Data(x=u, edge_index=edge_index)
        graph.y = y
        graph.pos = torch.cat((t_pos, x_pos), -1)
        graph.batch = batch.long()

        # Equation specific parameters
        if self.pde == 'ns':
            graph.parameters = torch.cat([
                v for k, v in variables.items()
                if k not in ['adj', 'data_directory']], -1)

        else:
            raise ValueError(f"Invalid PDE type: {self.pde}")

        return graph

    def create_next_graph(
            self,
            graph: Data,
            pred: torch.Tensor,
            labels: torch.Tensor,
            steps: list) -> Data:
        """
        Getting new graph for the next timestep
        Method is used for unrolling and when applying the pushforward trick
        during training

        Args:
            graph (Data): Pytorch geometric data object
            pred (torch.Tensor):
                prediction of previous timestep ->  input to next timestep
            labels (torch.Tensor): labels of previous timestep
            steps (list):
                list of different starting points for each batch entry
        Returns:
            Data: Pytorch Geometric data graph
        """
        # Output is the new input
        graph.x = torch.cat(
            (graph.x.reshape(len(pred), -1), pred), 1)[:, pred.shape[-1]:]
        nx = pred.shape[0]
        t = torch.linspace(self.tmin, self.tmax, self.nt)
        # Update labels and input timesteps
        y, t_pos = torch.Tensor(), torch.Tensor()
        for (labels_batch, step) in zip(labels, steps):
            y = torch.cat((
                y, torch.transpose(
                    torch.cat([label[None, :] for label in labels_batch]),
                    0, 1)))
            t_pos = torch.cat((t_pos, torch.ones(nx) * t[step]))
        graph.y = y
        graph.pos[:, 0] = t_pos

        return graph
