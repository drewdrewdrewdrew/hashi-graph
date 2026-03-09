"""Backbone GNN for Hashi Puzzle Solver."""

import torch
from torch.nn import Dropout, LayerNorm, ModuleList, ReLU
import torch.nn.functional as func
from torch_geometric.nn import TransformerConv, GATConv, GINEConv


class GraphBackbone(torch.nn.Module):
    """
    Backbone GNN that performs message passing across the graph.
    Supports Transformer, GAT, and GINE architectures.
    """

    def __init__(
        self,
        node_input_dim: int,
        hidden_channels: int,
        num_layers: int,
        heads: int = 8,
        dropout: float = 0.25,
        edge_dim: int | None = None,
        gnn_type: str = "transformer",
    ):
        super().__init__()
        self.gnn_type = gnn_type
        self.dropout = dropout
        self.convs = ModuleList()
        self.norms = ModuleList()

        curr_dim = node_input_dim

        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            out_heads = 1 if is_last and num_layers > 1 else heads
            concat = not is_last or num_layers == 1

            if gnn_type == "transformer":
                conv = TransformerConv(
                    curr_dim,
                    hidden_channels,
                    heads=out_heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    concat=concat,
                )
            elif gnn_type == "gat":
                conv = GATConv(
                    curr_dim,
                    hidden_channels,
                    heads=out_heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    concat=concat,
                )
            elif gnn_type == "gine":
                # GINE needs an MLP for the message function
                nn = torch.nn.Sequential(
                    torch.nn.Linear(curr_dim, hidden_channels),
                    ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels),
                )
                conv = GINEConv(nn, edge_dim=edge_dim)
            else:
                msg = f"Unknown GNN type: {gnn_type}"
                raise ValueError(msg)

            self.convs.append(conv)
            
            norm_dim = hidden_channels * (out_heads if concat else 1)
            self.norms.append(LayerNorm(norm_dim))
            curr_dim = norm_dim

        self.final_dim = curr_dim

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply GNN layers."""
        for conv, norm in zip(self.convs, self.norms, strict=True):
            h_in = h
            
            if self.gnn_type == "gine":
                # GINE expects edge_attr if edge_dim was provided
                h = conv(h, edge_index, edge_attr=edge_attr)
            else:
                h = conv(h, edge_index, edge_attr=edge_attr)
            
            h = norm(h)
            h = func.relu(h)
            h = func.dropout(h, p=self.dropout, training=self.training)

            # Skip connection if dimensions match
            if h_in.shape == h.shape:
                h = h + h_in

        return h
