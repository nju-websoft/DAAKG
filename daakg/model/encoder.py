import torch
import torch.nn as nn


# --- Main Models: Encoder ---
class Encoder(nn.Module):
    def __init__(self, name, hiddens, heads, activation, feat_drop, attn_drop, negative_slope, bias):
        super(Encoder, self).__init__()
        self.name = name
        self.hiddens = hiddens
        self.heads = heads
        self.num_layers = len(hiddens) - 1
        self.gnn_layers = nn.ModuleList()
        self.activation = activation
        self.feat_drop = feat_drop
        for l in range(0, self.num_layers):
            if self.name == "gcn-align":
                self.gnn_layers.append(
                    GCNAlign_GCNConv(in_channels=self.hiddens[l], out_channels=self.hiddens[l+1], improved=False, cached=True, bias=bias)
                )
            elif self.name == "naea":
                self.gnn_layers.append(
                    NAEA_GATConv(in_channels=self.hiddens[l]*self.heads[l-1], out_channels=self.hiddens[l+1], heads=self.heads[l], concat=True, negative_slope=negative_slope, dropout=attn_drop, bias=bias)
                )
            elif self.name == "kecg":
                self.gnn_layers.append(
                    KECG_GATConv(in_channels=self.hiddens[l]*self.heads[l-1], out_channels=self.hiddens[l+1], heads=self.heads[l], concat=False, negative_slope=negative_slope, dropout=attn_drop, bias=bias)
                )
            # elif self.name == "SLEF-DESIGN":
            #     self.gnn_layers.append(
            #         SLEF-DESIGN_Conv()
            #     )
            else:
                raise NotImplementedError("bad encoder name: " + self.name)
        if self.name == "naea":
            self.weight = Parameter(torch.Tensor(self.hiddens[0], self.hiddens[-1]))
            nn.init.xavier_normal_(self.weight)
        # if self.name == "SLEF-DESIGN":
        #     '''SLEF-DESIGN: extra parameters'''

    def forward(self, edges, x, r=None):
        edges = edges.t()
        if self.name == "alinet":
            stack = [F.normalize(x, p=2, dim=1)]
            for l in range(self.num_layers):
                x = F.dropout(x, p=self.feat_drop, training=self.training)
                x_ = self.gnn_layers[l](x, edges)
                stack.append(F.normalize(x_, p=2, dim=1))
                x = x_
                if l != self.num_layers - 1:
                    x = self.activation(x)
            return torch.cat(stack, dim=1)
        elif self.name == "naea":
            for l in range(self.num_layers):
                x = F.dropout(x, p=self.feat_drop, training=self.training)
                x_ = self.gnn_layers[l](x, edges, r)
                x = x_
                if l != self.num_layers - 1:
                    x = self.activation(x)
            x = torch.sigmoid(x)
            return (x, self.weight)
        # elif self.name == "SLEF-DESIGN":
        #     '''SLEF-DESIGN: special encoder forward'''
        else:
            for l in range(self.num_layers):
                x = F.dropout(x, p=self.feat_drop, training=self.training)
                x_ = self.gnn_layers[l](x, edges)
                x = x_
                if l != self.num_layers - 1:
                    x = self.activation(x)
            return x            

    def __repr__(self):
        return '{}(name={}): {}'.format(self.__class__.__name__, self.name, "\n".join([layer.__repr__() for layer in self.gnn_layers]))
# --- Main Models: Encoder end ---