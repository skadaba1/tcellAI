import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, coalesce, global_add_pool
from attentions import SparseAttention
from graph_obs.graph_utils import create_edges_and_weights, custom_add_self_loops
from embeddings import ShallowGeneEmbedding, ShallowAttentionEmbedding
from losses import loss_fn, loss_fn_ce
import torch.nn.functional as F

class DynamicGATLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, threshold=0.0, heads=1, concat=True, negative_slope=0.2, dropout=0.0, num_sparse_connections=128):
        super(DynamicGATLayer, self).__init__(aggr='mean')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.multihead_attn = SparseAttention(in_channels, heads, num_sparse_connections=num_sparse_connections)

        # Set on the fly
        self.attention_scores = None

        # Learnable weights
        self.update_mlp_msg_trf = nn.Sequential(
            nn.Linear(in_channels*2, in_channels*2 // 2, bias=False),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(in_channels*2 // 2, out_channels, bias=False)
        )

        self.update_mlp_msg_att = nn.Sequential(
            nn.Linear(in_channels*2, in_channels*2 // 2, bias=False),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(in_channels*2 // 2, out_channels, bias=False)
        )

        self.update_mlp_neighbor = nn.Sequential(
            nn.Linear(in_channels*2, in_channels*2 // 2, bias=False),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(in_channels*2 // 2, in_channels, bias=False)
        )

        self.update_mlp_att = nn.Sequential(
            nn.Linear(in_channels*2, in_channels*2 // 2, bias=False),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(in_channels*2 // 2, in_channels, bias=False)
        )

        # # Learnable attention threshold, arbitrary
        self.threshold = torch.tensor(threshold)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize Regressor weights
        for layer in self.update_mlp_msg_trf:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)

        # Initialize Classifier weights
        for layer in self.update_mlp_msg_att:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)

        # Initialize Classifier weights
        for layer in self.update_mlp_neighbor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)

        # Initialize Classifier weights
        for layer in self.update_mlp_att:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)

    def forward(self, x, decoupled_emb, edge_index, mask, attention_init):
        out, updated_edge_index, edge_penalty, attention_returned, decoupled_emb, edge_weights = self.dynamic_propagate(x, decoupled_emb, edge_index, mask, attention_init)
        return out, updated_edge_index, decoupled_emb, edge_penalty, attention_returned, edge_weights

    def dynamic_propagate(self, x, decoupled_emb, edge_index, mask, attention_init):
        '''Custom propogate implementation to handle dynamic edge_index '''
        edge_index, _ = remove_self_loops(edge_index)
        num_nodes = x.size(0)
        edge_penalty = 0.0

        # Compute new edges and penalties for each node
        attention_scores = self.multihead_attn(decoupled_emb)
        attention_scores = attention_scores.sum(dim=0) #+ attention_init

        temperature = 0.5
        hard = False
        new_edge_index, edge_weights, probs = create_edges_and_weights(attention_scores, self.threshold, temperature, hard)

        updated_edge_index = torch.cat([edge_index, new_edge_index], dim=1)
        edge_weights = torch.cat((probs[edge_index[0], edge_index[1]], edge_weights))

        updated_edge_index, edge_weights = custom_add_self_loops(num_nodes, updated_edge_index, edge_weights, probs)

        # Coalesce to remove duplicate edges and sum their weights
        updated_edge_index, edge_weights = coalesce(
            updated_edge_index, edge_attr=edge_weights, num_nodes=num_nodes
        )

        if mask is not None:
          # **Filter edges based on the mask**
          valid_nodes = mask.nonzero(as_tuple=True)[0]
          valid_node_mask = (mask[updated_edge_index[0]] & mask[updated_edge_index[1]])
          updated_edge_index_masked = updated_edge_index[:, valid_node_mask]
          edge_weights_masked = edge_weights[valid_node_mask]
        else:
          updated_edge_index_masked = updated_edge_index
          edge_weights_masked = edge_weights

        # Calculate messages using the updated edge index
        messages = self.propagate(edge_index=updated_edge_index_masked, x=x, edge_weights=edge_weights_masked, mode='val')
        att_messages = self.propagate(edge_index=updated_edge_index, x=decoupled_emb, edge_weights=edge_weights, mode='att')

        # penalizes entropy, rewards peaked attention distribution
        edge_penalty += -torch.sum(probs * torch.log(probs + 1e-10))

        return messages, updated_edge_index, edge_penalty, attention_scores, att_messages, edge_weights

    def message(self, x_i, x_j, edge_weights, edge_index_i, edge_index_j, mode, **kwargs):
        '''
        Args:
        - x_i: neighbor node feature (target)
        - x_j: source node feature
        - edge_weights: edge weights
        - mode: 'val' or 'att', (whether to use MLP for passing value embeddings or attention embeddings)
        - kwargs: additional arguments for propagation

        Returns:
        - message: message that is passed from source node to target node
        '''
        source_target = torch.cat([x_i, x_j], dim=1)
        # value embeddings
        if(mode == 'val'):
          message = self.update_mlp_neighbor(source_target)
          return edge_weights.view(-1,1)*message

        # attention embeddings also shared since attention can be influenced at the sub-graph level
        elif(mode == 'att'):
          message = self.update_mlp_att(source_target)
          return edge_weights.view(-1,1)*message
        else:
          # fallback, should not get called
          return x_j

    def update(self, aggr_out, x, mode):
        '''
        Args:
        - aggr_out: aggregated messages from neighbors
        - x: node features
        - mode: 'val' or 'att', (whether to use MLP for passing value embeddings or attention embeddings)

        Returns:
        - aggr_out: updated node features
        '''
        aggr_out_x = torch.cat([aggr_out, x], dim=1)
        if(mode == 'val'):
          aggr_out = self.update_mlp_msg_trf(aggr_out_x)
        elif(mode == 'att'):
          aggr_out = self.update_mlp_msg_att(aggr_out_x)
        else:
          pass # fallback should not get called
        return aggr_out

class DynamicGAT(torch.nn.Module):
    '''
    Stacked DynamicGAT layers with trainable heads for downstream regression and classification tasks.
    '''

    def __init__(self, num_genes, in_channels, embedding_dim, hidden_channels, out_channels, num_layers, threshold, n_classes=4, heads=4, dropout_prob=0.0, negative_slope=0.2, num_sparse_connections=128, sc_embed=None):
        super(DynamicGAT, self).__init__()
        self.embedding = ShallowGeneEmbedding(input_dim=in_channels, embedding_dim=embedding_dim, init_embeddings=None)
        self.decoupled_embedding = ShallowAttentionEmbedding(num_embeddings=num_genes, embedding_dim=embedding_dim, init_embeddings=sc_embed)  # separate embedding

        self.layers = nn.ModuleList()
        self.layer_norms_val = nn.ModuleList()
        self.layer_norms_att = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.negative_slope = negative_slope

        for i in range(num_layers):
            if i == 0:
                self.layers.append(DynamicGATLayer(embedding_dim, hidden_channels, threshold=threshold, heads=heads, negative_slope=negative_slope, num_sparse_connections=num_sparse_connections))
                self.skip_connections.append(
                      nn.Linear(embedding_dim, hidden_channels, bias=False),
                    )
            elif i == num_layers - 1:
                self.layers.append(DynamicGATLayer(hidden_channels, out_channels, threshold=threshold, heads=heads, negative_slope=negative_slope, num_sparse_connections=num_sparse_connections))
                self.skip_connections.append(
                    nn.Linear(embedding_dim, out_channels, bias=False),
                )
            else:
                self.layers.append(DynamicGATLayer(hidden_channels, hidden_channels, threshold=threshold, heads=heads, negative_slope=negative_slope, num_sparse_connections=num_sparse_connections))
                self.skip_connections.append(
                        nn.Linear(embedding_dim, hidden_channels, bias=False)
                )

            # DynamicGAT mulitplies output dim by heads

            # Add Layer Normalization and Dropout for each layer
            self.layer_norms_val.append(nn.LayerNorm(hidden_channels if i < num_layers - 1 else out_channels))
            self.layer_norms_att.append(nn.LayerNorm(hidden_channels if i < num_layers - 1 else out_channels))
            self.dropouts.append(nn.Dropout(p=dropout_prob))

        # Regressor with Uniform Initialization
        self.regressor = nn.Sequential(
            nn.Linear(out_channels, out_channels//2, bias=True),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout_prob),
            nn.Linear(out_channels//2, 1, bias=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(out_channels, out_channels//2, bias=True),
            nn.LeakyReLU(negative_slope=negative_slope),  # LeakyReLU to prevent dead neurons
            nn.Dropout(dropout_prob),
            nn.Linear(out_channels//2, n_classes, bias=True),
        )

        # stack DiffPool layers together
        self.reset_parameters()
        self.num_genes = num_genes


    def reset_parameters(self):
        # Initialize Regressor weights
        for layer in self.regressor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)

        # Initialize Classifier weights
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)

        for skip in self.skip_connections:
            if isinstance(skip, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)

    def forward(self, x, edge_index, batch, mask=None, labels=None, response=None, compute_loss=False, return_edges=False):

        decoupled_emb = self.decoupled_embedding(x)
        x = self.embedding(x.view(self.num_genes, -1))

        x_init = x
        loss = 0.0
        edge_penalities = 0.0
        ce_loss = 0.0

        attention_init = torch.zeros(self.num_genes, self.num_genes, device=x.device, requires_grad=True)
        for i, layer in enumerate(self.layers):
            x, edge_index, decoupled_emb, edge_penality, attention_init, edge_weights = layer(x, decoupled_emb, edge_index, mask=mask, attention_init=attention_init)
            # Apply Layer Normalization and Dropout

            x = self.layer_norms_val[i](x)
            x = self.skip_connections[i](x_init) + x
            x = self.dropouts[i](x)

            decoupled_emb = self.layer_norms_att[i](decoupled_emb)
            decoupled_emb = self.dropouts[i](decoupled_emb)

            # Accumulate edge penalties
            edge_penalities += edge_penality

        self.attention_final = attention_init # for use after forward pass
        self.x_final = x # for use after forward pass

        if compute_loss:
            if(mask is not None):
              num_masked_nodes = mask.numel() - mask.sum().item()
              for node_idx in (~mask).nonzero(as_tuple=True)[0]:
                  # Pool the neighborhood of the masked node

                  neighbor_idxs = edge_index[1][edge_index[0] == node_idx]  # Get neighbors of the masked node
                  # print(neighbor_idxs.shape)

                  if neighbor_idxs.numel() == 0:  # No neighbors

                      # Assign a high differentiable penalty loss for isolated nodes
                      node_loss = loss_fn(x*0, labels[node_idx].unsqueeze(0))
                  else:

                      # Compute the neighborhood embedding if neighbors exist
                      neighbor_embeddings = x[neighbor_idxs]

                      # apply attention from attention_init and sum
                      weights = F.softmax(attention_init[node_idx, neighbor_idxs].unsqueeze(0), dim=1)
                      pooled_embedding = (weights @ neighbor_embeddings)
                      if torch.isnan(pooled_embedding).any():
                          # Handling NaN values
                          continue

                      # Make prediction for the current masked node
                      pred = self.regressor(pooled_embedding)  # Adding batch dimension for single node

                      # Compute the loss for the current masked node
                      node_loss = loss_fn(pred, labels[node_idx].unsqueeze(0))

                  # Accumulate the loss for each masked node
                  loss += node_loss if not torch.isnan(node_loss) else 0.0
              loss /= (num_masked_nodes if num_masked_nodes > 0 else 1)

            x = global_add_pool(x, batch)
            if(response is not None):
                response = response.view(1,-1)
                response_idx = torch.argmax(response, dim=-1)
                logits = self.classifier(x).view(1,-1)
                ce_loss = loss_fn_ce(logits, response_idx)

            if(return_edges):
                return x, loss, edge_penalities, ce_loss, edge_index

            return x, loss, edge_penalities, ce_loss

        x = global_add_pool(x, batch)
        if(return_edges):
            return x, edge_index
        return x