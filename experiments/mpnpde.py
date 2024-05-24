# import pathlib

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, InstanceNorm

# import experiments.train_helper as helper


LEN_U_MIXTURE = 5  # (u, p, alpha)
N_PARAM_MIXTURE = 15  # (4, 4, 3, 3, 1), `variables` in common.datasets
DIM = 3

DEBUG = False


class Swish(nn.Module):
    """
    Swish activation function
    """
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)


class GNN_Layer(MessagePassing):
    """
    Message passing layer
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        time_window: int,
        n_variables: int,
        pde: str,
    ):
        """
        Initialize message passing layers
        Args:
            in_features (int): number of node input features
            out_features (int): number of node output features
            hidden_features (int): number of hidden features
            time_window (int): number of input/output timesteps
                               (temporal bundling)
            n_variables (int): number of equation specific parameters used in
                               the solver
        """
        super(GNN_Layer, self).__init__(node_dim=-2, aggr='mean')
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        self.pde = pde
        if self.pde == 'mixture':
            len_u = LEN_U_MIXTURE
        else:
            raise ValueError(f"Invalid PDE type: {self.pde}")

        self.message_net_1 = nn.Sequential(
            nn.Linear(
                2 * in_features + time_window * len_u + DIM + n_variables,
                hidden_features),
            Swish()
        )
        # raise ValueError(self.message_net_1[0])

        self.message_net_2 = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            Swish()
        )
        self.update_net_1 = nn.Sequential(
            nn.Linear(
                in_features + hidden_features + n_variables, hidden_features),
            Swish()
        )
        self.update_net_2 = nn.Sequential(
            nn.Linear(hidden_features, out_features),
            Swish()
        )
        self.norm = InstanceNorm(hidden_features)

    def forward(self, x, u, pos, variables, edge_index, batch):
        """
        Propagate messages along edges
        """
        x = self.propagate(edge_index, x=x, u=u, pos=pos, variables=variables)
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        """
        Message update following formula 8 of the paper
        """
        message = self.message_net_1(
            torch.cat(
                (x_i, x_j, u_i - u_j, pos_i - pos_j, variables_i), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x, variables):
        """
        Node update following formula 9 of the paper
        """
        update = self.update_net_1(
            torch.cat((x, message, variables), dim=-1))
        update = self.update_net_2(update)
        if self.in_features == self.out_features:
            return x + update
        else:
            return update


class MP_PDE_Solver(torch.nn.Module):
    """
    MP-PDE solver class
    """
    def __init__(
        self,
        pde: str,
        time_window: int = 25,
        hidden_features: int = 128,
        hidden_layer: int = 6,
        eq_variables: dict = {}
    ):
        """
        Initialize MP-PDE solver class.
        It contains 6 MP-PDE layers with skip connections
        The input graph to the forward pass has the shape
        [batch*n_nodes, time_window].
        The output graph has the shape [batch*n_nodes, time_window].
        Args:
            pde (PDE): PDE at hand
            time_window (int): number of input/output timesteps
                               (temporal bundling)
            hidden features (int): number of hidden features
            hidden_layer (int): number of hidden layers
            eq_variables (dict): dictionary of equation specific parameters
        """
        super(MP_PDE_Solver, self).__init__()
        # 1D decoder CNN is so far designed time_window = [20,25,50]
        # assert(time_window == 25 or time_window == 20 or time_window == 50)
        self.pde = pde
        self.out_features = time_window
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.time_window = time_window
        self.eq_variables = eq_variables

        if self.pde == 'mixture':
            self.n_u = LEN_U_MIXTURE
            self.n_parameter_feature = N_PARAM_MIXTURE
        else:
            raise ValueError(f"Invalid PDE type: {self.pde}")

        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=self.time_window,
            n_variables=self.n_parameter_feature + 1,
            # ^^ variables = eq_variables + time
            pde=self.pde,
        ) for _ in range(self.hidden_layer - 1)))

        # The last message passing last layer has a fixed output size to make
        # the use of the decoder 1D-CNN easier
        self.gnn_layers.append(
            GNN_Layer(
                in_features=self.hidden_features,
                hidden_features=self.hidden_features,
                out_features=self.hidden_features,
                time_window=self.time_window,
                n_variables=self.n_parameter_feature + 1,
                pde=self.pde,
            )
        )

        self.embedding_mlp = nn.Sequential(
            nn.Linear(
                self.time_window * self.n_u + DIM + 1  # + 1 being time
                + self.n_parameter_feature,
                self.hidden_features),
            Swish(),
            nn.Linear(self.hidden_features, self.hidden_features),
            Swish()
        )

        # Decoder CNN, maps to different outputs (temporal bundling)
        if (self.time_window == 8):
            if self.hidden_features == 128:
                self.output_mlp = nn.Sequential(
                    nn.Conv1d(1, 8, 16, stride=6),
                    Swish(),
                    nn.Conv1d(8, self.n_u, 12, stride=1)
                )
            elif self.hidden_features == 64:
                self.output_mlp = nn.Sequential(
                    nn.Conv1d(1, 8, 16, stride=3),
                    Swish(),
                    nn.Conv1d(8, self.n_u, 10, stride=1)
                )
            elif self.hidden_features == 32:
                self.output_mlp = nn.Sequential(
                    nn.Conv1d(1, 8, 1, stride=1),
                    Swish(),
                    nn.Conv1d(8, self.n_u, 25, stride=1)
                )
            else:
                raise ValueError(
                    f"Invalid hidden_features: {self.hidden_features}")
        elif (self.time_window == 4):
            if self.hidden_features == 128:
                self.output_mlp = nn.Sequential(
                    nn.Conv1d(1, 8, 16, stride=6),
                    Swish(),
                    nn.Conv1d(8, self.n_u, 10, stride=3)
                )
            elif self.hidden_features == 64:
                self.output_mlp = nn.Sequential(
                    nn.Conv1d(1, 8, 61, stride=1),
                    Swish(),
                    nn.Conv1d(8, self.n_u, 1, stride=1)
                )
            elif self.hidden_features == 32:
                self.output_mlp = nn.Sequential(
                    nn.Conv1d(1, 8, 10, stride=1),
                    Swish(),
                    nn.Conv1d(8, self.n_u, 20, stride=1)
                )
            else:
                raise ValueError(
                    f"Invalid hidden_features: {self.hidden_features}")
        elif (self.time_window == 2):
            if self.hidden_features == 128:
                self.output_mlp = nn.Sequential(
                    nn.Conv1d(1, 8, 15, stride=54),
                    Swish(),
                    nn.Conv1d(8, self.n_u, 2, stride=1)
                )
            elif self.hidden_features == 64:
                self.output_mlp = nn.Sequential(
                    nn.Conv1d(1, 8, 1, stride=5),
                    Swish(),
                    nn.Conv1d(8, self.n_u, 12, stride=1)
                )
            elif self.hidden_features == 32:
                self.output_mlp = nn.Sequential(
                    nn.Conv1d(1, 8, 1, stride=2),
                    Swish(),
                    nn.Conv1d(8, self.n_u, 15, stride=1)
                )
            else:
                raise ValueError(
                    f"Invalid hidden_features: {self.hidden_features}")
        else:
            raise ValueError(f"Invalid time_window: {self.time_window}")
        return

    def __repr__(self):
        return 'GNN'

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of MP-PDE solver class.
        The input graph has the shape [batch*n_nodes, time_window].
        The output tensor has the shape [batch*n_nodes, time_window].
        Args:
            data (Data): Pytorch Geometric data graph
        Returns:
            torch.Tensor: data output
        """
        u = torch.reshape(data.x, (data.x.shape[0], -1))
        # if DEBUG:
        #     helper.save_mixture(u.detach().numpy(), pathlib.Path('tmp'))
        # Encode and normalize coordinate information
        pos = data.pos
        pos_x = pos[:, 1:]
        pos_t = pos[:, 0][:, None] / self.eq_variables['tmax']
        edge_index = data.edge_index
        batch = data.batch

        variables = torch.cat(
            (
                pos_t,    # time is treated as equation variable
                data.parameters[0]
            ), -1)

        # Encoder and processor (message passing)
        node_input = torch.cat((u, pos_x, variables), -1)
        h = self.embedding_mlp(node_input)
        for i in range(self.hidden_layer):
            h = self.gnn_layers[i](h, u, pos_x, variables, edge_index, batch)

        # Decoder (formula 10 in the paper)
        dt = (torch.ones(1, self.time_window) * self.eq_variables['dt']).to(
            h.device)
        dt = torch.cumsum(dt, dim=1)
        dt = torch.cat([
            torch.cat([dt[:, [i_t]]] * self.n_u, -1)
            for i_t in range(dt.shape[-1])], -1)
        # [batch*n_nodes, hidden_dim]
        # -> 1DCNN([batch*n_nodes, 1, hidden_dim])
        # -> [batch*n_nodes, time_window * n_u]
        diff = torch.reshape(self.output_mlp(h[:, None]), (len(pos_x), -1))

        repeated_u = torch.cat(
            [u[:, -self.n_u:]] * (u.shape[-1] // self.n_u), -1)
        if DEBUG:
            print(f"repeated_u: {repeated_u.shape}")
            print(f"dt: {dt.shape}")
            print(f"diff: {diff.shape}")

        out = repeated_u + dt * diff
        if DEBUG:
            print('At forward')
            print(u[100:110, 4:8])
            # helper.save_mixture(
            #     out.cpu().detach().numpy(), pathlib.Path('tmp'))
            # helper.save_mixture(
            #     diff.cpu().detach().numpy(), pathlib.Path('tmp/diff'))

        return out
