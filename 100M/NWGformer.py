import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_scatter import scatter, scatter_add

class oursgat(torch.nn.Module):
    def __init__(self, input_dim, output_dim, P=1, M=2,sys=False,use_weight=False):
        super(oursgat,self).__init__()
        self.lin = torch.nn.Linear(input_dim, output_dim)
        self.lin2 = torch.nn.Linear(input_dim, output_dim)
        self.use_weight = use_weight
        if self.use_weight :
            self.w = torch.nn.Linear(input_dim, 1)
        self.P = P
        self.M = M
        self.usesys = sys#if SYS---->GCN;else:---->SAGE
        self.eps = 1e-10

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.lin2.reset_parameters()
        if self.use_weight:
            self.w.reset_parameters()

    def forward(self, x, edge_index):
        V = self.lin(x)
        Q = F.relu(self.lin2(x))
        out = self.localgat(Q, edge_index, V)
        return out

    def localgat(self, V, edge_index, x):
        if self.use_weight:
            h = self.w(V) ** self.P
        else:
            h = F.layer_norm(V, [V.size(-1)])
            h = (torch.sum(h, dim=1).unsqueeze(1)) ** self.P
        src, dst = edge_index[0], edge_index[1]
        h_src = h[src]  # (E, d)
        h_dst = h[dst]  # (E, d)
        m = (torch.max(h_src) - torch.min(h_src)) * self.M
        attention = (torch.cos((torch.pi / (2 * m + self.eps)) * (h_src - h_dst)).sum(dim=1))
        if self.usesys:
            norm_attention = self.sparse_row_normalize_sys(edge_index, attention, V.shape[0])
        else:
            norm_attention = self.sparse_row_normalize_l1(edge_index, attention, V.shape[0])
        messages = norm_attention.unsqueeze(1) * x[dst]  # (E, d)
        out = scatter(messages, src, dim=0, dim_size=V.size(0), reduce='sum')

        return out

    def sparse_row_normalize_l1(self, indices, values, n):
        row_sum = scatter_add(values, indices[0], dim_size=n)
        normalized_values = values / (row_sum[indices[0]]) + self.eps
        return normalized_values

    def sparse_row_normalize_sys(self, indices, values, n):
        row_sum = scatter_add(values, indices[0], dim_size=n) ** (-0.5) + self.eps
        col_sum = scatter_add(values, indices[1], dim_size=n) ** (-0.5) + self.eps
        normalized_values = (row_sum[indices[0]]) * values * (col_sum[indices[1]])
        return normalized_values


class NWGformerF(torch.nn.Module):
    def __init__(self, input_dim, output_dim, droupout=0.2, P=2, M=1, bn=False, ln=False, res=False,use_weight=False, gnn='gat'):
        super(NWGformerF, self).__init__()
        self.linq1 = torch.nn.Linear(input_dim, output_dim)
        self.linv1 = torch.nn.Linear(input_dim, output_dim)
        if gnn == 'gat' and P != 0:
            self.conv = oursgat(input_dim, output_dim)
        else:
            self.conv = GCNConv(input_dim, output_dim, cached=False, normalize=True)
        self.use_weight = use_weight
        if self.use_weight:
            self.w = torch.nn.Linear(input_dim, 1)
        self.P = P
        self.M = M
        self.bn = bn
        self.ln = ln
        if self.bn:
            self.bn = torch.nn.BatchNorm1d(output_dim)
        if self.ln:
            self.ln = torch.nn.LayerNorm(output_dim)
        self.res = res
        self.p = droupout
        self.eps = 1e-10

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.linq1.reset_parameters()
        self.linv1.reset_parameters()
        if self.use_weight:
            self.w.reset_parameters()
        if self.bn:
            self.bn.reset_parameters()
        if self.ln:
            self.ln.reset_parameters()

    def forward(self, x, edge_index):
        Q = F.relu(self.linq1(x))
        V = self.linv1(x)
        if self.res:
            out = (self.former(Q, V) + self.conv(x, edge_index)) + V
        else:
            out = (self.former(Q, V) + self.conv(x, edge_index))
        if self.bn:
            out = self.bn(out)
        elif self.ln:
            out = self.ln(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.p, training=self.training)
        return out

    def former(self, Q, V):
        if self.use_weight:
            H = (self.w(Q)) ** self.P
            Q = F.layer_norm(Q, [Q.size(-1)])
            Q = F.relu(Q)
        else:
            Q = F.layer_norm(Q, [Q.size(-1)])
            Q = F.relu(Q)
            H = (torch.sum(Q, dim=1).unsqueeze(1)) ** self.P
        m = (torch.max(H) - torch.min(H)) * self.M
        H = (torch.pi / (2 * m)) * H
        cosQ = (Q * torch.cos(H))
        sinQ = (Q * torch.sin(H))
        out = torch.mm(cosQ, torch.mm(cosQ.T, V)) + torch.mm(sinQ, torch.mm(sinQ.T, V))
        norm = torch.mm(cosQ, (torch.sum(cosQ, dim=0)).unsqueeze(1)) + torch.mm(sinQ,(torch.sum(sinQ, dim=0)).unsqueeze(1)) + self.eps
        return (out / norm)

    def Softmax(self, Q, V):
        A = F.softmax(torch.mm(Q,Q.T), dim=1)
        out = torch.mm(A, V)
        return out

class MPNNs(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, local_layers=3,
                 dropout=0.5, heads=1, pre_ln=False, pre_linear=False, res=False, ln=False, bn=False, jk=False,
                 gnn='gat'):
        super(MPNNs, self).__init__()
        self.dropout = dropout
        self.pre_ln = pre_ln
        self.pre_linear = pre_linear
        self.res = res
        self.ln = ln
        self.bn = bn
        self.jk = jk
        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()
        self.lin_in = torch.nn.Linear(in_channels, hidden_channels)
        if self.pre_linear:
            self.former = NWGformerF(hidden_channels, hidden_channels, droupout=self.dropout, P=2, M=1, bn=self.bn,
                                     ln=self.ln, res=self.res, gnn=gnn)
        else:
            self.former = NWGformerF(in_channels, hidden_channels, droupout=self.dropout, P=2, M=1, bn=self.bn,
                                     ln=self.ln, res=self.res, gnn=gnn)

        if not self.pre_linear:
            if gnn == 'gat':
                self.local_convs.append(oursgat(hidden_channels, hidden_channels))
            elif gnn == 'sage':
                self.local_convs.append(SAGEConv(hidden_channels, hidden_channels))
            else:
                self.local_convs.append(GCNConv(hidden_channels, hidden_channels,
                                                cached=False, normalize=True))
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(in_channels))
            local_layers = local_layers - 1

        for _ in range(local_layers):
            if gnn == 'gat':
                self.local_convs.append(oursgat(hidden_channels, hidden_channels))
            elif gnn == 'sage':
                self.local_convs.append(SAGEConv(hidden_channels, hidden_channels))
            else:
                self.local_convs.append(GCNConv(hidden_channels, hidden_channels,
                                                cached=False, normalize=True))
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(hidden_channels))

        self.pred_local = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        self.lin_in.reset_parameters()
        self.pred_local.reset_parameters()
        self.former.reset_parameters()

    def forward(self, x, edge_index):
        # x = F.normalize(x,p=1)
        if self.pre_linear:
            x = self.lin_in(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.former(x, edge_index)
        x_final = 0
        for i, local_conv in enumerate(self.local_convs):
            if self.res:
                x = local_conv(x, edge_index) + self.lins[i](x)
            else:
                x = local_conv(x, edge_index)
            if self.ln:
                x = self.lns[i](x)
            elif self.bn:
                x = self.bns[i](x)
            else:
                pass
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.jk:
                x_final = x_final + x
            else:
                x_final = x
        x = self.pred_local(x_final)
        return x


class NWGformerT(torch.nn.Module):
    def __init__(self, input_dim, output_dim, droupout=0.2, P=1, M=2, alpha=0.8, bn=False, ln=False, res=False,use_weight=False,gnn='gat'):
        super(NWGformerT, self).__init__()
        self.linq1 = torch.nn.Linear(input_dim, output_dim)
        self.linv1 = torch.nn.Linear(input_dim, output_dim)
        if gnn == 'gat' and P != 0:
            self.conv = oursgat(input_dim, output_dim)
        else:
            self.conv = GCNConv(input_dim, output_dim, cached=False, normalize=True)
        self.use_weight = use_weight
        if self.use_weight:
            self.w = torch.nn.Linear(input_dim, 1)
        self.P = P
        self.M = M
        self.alpha = alpha
        self.bn = bn
        self.ln = ln
        if self.bn:
            self.bn = torch.nn.BatchNorm1d(output_dim)
        if self.ln:
            self.ln = torch.nn.LayerNorm(output_dim)
        self.res = res
        self.p = droupout
        self.eps = 1e-10

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.linq1.reset_parameters()
        self.linv1.reset_parameters()
        if self.use_weight:
            self.w.reset_parameters()
        if self.bn:
            self.bn.reset_parameters()
        if self.ln:
            self.ln.reset_parameters()
    def forward(self, x, edge_index, S):
        Q = F.sigmoid(self.linq1(x))
        V = self.linv1(x)
        if self.res:
            out = (self.former(Q, V, S) + self.conv(x, edge_index)) + V
        else:
            out = (self.former(Q, V, S) + self.conv(x, edge_index))
        if self.bn:
            out = self.bn(out)
        elif self.ln:
            out = self.ln(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.p, training=self.training)
        return out

    def former(self, Q, V, S):
        if self.use_weight:
            xH = (self.w(Q)) ** self.P
            Q = F.layer_norm(Q, [Q.size(-1)])
            Q = F.relu(Q)
        else:
            Q = F.layer_norm(Q, [Q.size(-1)])
            Q = F.relu(Q)
            xH = (torch.sum(Q, dim=1).unsqueeze(1)) ** self.P
        xm = (torch.max(xH) - torch.min(xH)) * self.M
        xH = (torch.pi / (2 * xm + self.eps)) * xH
        xcosQ = (Q * torch.cos(xH))
        xsinQ = (Q * torch.sin(xH))
        xout = torch.mm(xcosQ, torch.mm(xcosQ.T, V)) + torch.mm(xsinQ, torch.mm(xsinQ.T, V))
        xnorm = torch.mm(xcosQ, (torch.sum(xcosQ, dim=0)).unsqueeze(1)) + torch.mm(xsinQ,(torch.sum(xsinQ, dim=0)).unsqueeze(1))
        Hs = S
        ms = (torch.max(Hs) - torch.min(Hs)) * self.M
        Hs = (torch.pi / (2 * ms + self.eps)) * Hs
        scosQ = (Q * torch.cos(Hs))
        ssinQ = (Q * torch.sin(Hs))
        sout = torch.mm(scosQ, torch.mm(scosQ.T, V)) + torch.mm(ssinQ, torch.mm(ssinQ.T, V))
        snorm = torch.mm(scosQ, (torch.sum(scosQ, dim=0)).unsqueeze(1)) + torch.mm(ssinQ,(torch.sum(ssinQ, dim=0)).unsqueeze(1))
        return (xout * self.alpha + sout * (1 - self.alpha)) / (
                    xnorm * self.alpha + snorm * (1 - self.alpha) + self.eps)


class MPNNTs(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, local_layers=3,
                 dropout=0.5, heads=1, pre_ln=False, pre_linear=False, res=False, ln=False, bn=False, jk=False,
                 gnn='gat'):
        super(MPNNTs, self).__init__()
        self.dropout = dropout
        self.pre_ln = pre_ln
        self.pre_linear = pre_linear
        self.res = res
        self.ln = ln
        self.bn = bn
        self.jk = jk
        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()
        self.lin_in = torch.nn.Linear(in_channels, hidden_channels)
        if self.pre_linear:
            self.former = NWGformerT(hidden_channels, hidden_channels, droupout=self.dropout, P=2, M=1, alpha=0.3,
                                     bn=self.bn, ln=self.ln, res=self.res, gnn=gnn)
        else:
            self.former = NWGformerT(in_channels, hidden_channels, droupout=self.dropout, P=2, M=1, alpha=0.8,
                                     bn=self.bn, ln=self.ln, res=self.res, gnn=gnn)
        if not self.pre_linear:
            if gnn == 'gat':
                self.local_convs.append(oursgat(hidden_channels, hidden_channels))
            elif gnn == 'sage':
                self.local_convs.append(SAGEConv(hidden_channels, hidden_channels))
            else:
                self.local_convs.append(GCNConv(hidden_channels, hidden_channels,
                                                cached=False, normalize=True))
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(in_channels))
            local_layers = local_layers - 1

        for _ in range(local_layers):
            if gnn == 'gat':
                self.local_convs.append(oursgat(hidden_channels, hidden_channels))
            elif gnn == 'sage':
                self.local_convs.append(SAGEConv(hidden_channels, hidden_channels))
            else:
                self.local_convs.append(GCNConv(hidden_channels, hidden_channels,
                                                cached=False, normalize=True))
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(hidden_channels))
        self.pred_local = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        self.lin_in.reset_parameters()
        self.pred_local.reset_parameters()
        self.former.reset_parameters()

    def forward(self, x, edge_index):
        x, s = x[:, :-1], x[:, -1:]
        x = F.normalize(x, p=1)
        if self.pre_linear:
            x = self.lin_in(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.former(x, edge_index, s)
        x_final = 0
        for i, local_conv in enumerate(self.local_convs):
            if self.res:
                x = local_conv(x, edge_index) + self.lins[i](x)
            else:
                x = local_conv(x, edge_index)
            if self.ln:
                x = self.lns[i](x)
            elif self.bn:
                x = self.bns[i](x)
            else:
                pass
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.jk:
                x_final = x_final + x
            else:
                x_final = x
        x = self.pred_local(x_final)
        return x
