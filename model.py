import torch.nn as nn
import torch
import torch.nn.functional as F


class STGAMAM(nn.Module):
    def __init__(self, feature_size, adj, num_layers=2, dropout=0.1):
        super(STGAMAM, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=7, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.GAT = GAT(in_features=feature_size, hidden_features=16, out_features=feature_size, dropout=0.1, alpha=0.2,
                       nheads=6)
        self.adj = adj
        self.nNode = len(adj)

        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, st_fusion_adj, device):
        GAT_x_t = []
        for i in range(self.nNode):
            selected_matrices = []
            for j in range(i, len(src), self.nNode):
                selected_matrices.append(src[j:j + 1, :, :])
            result = torch.cat(selected_matrices, dim=0)
            mask = self._generate_square_subsequent_mask(len(result)).to(device)
            GAT_x_t.append(self.transformer_encoder(result, mask))
        GAT_x = torch.cat(GAT_x_t, dim=0)
        output = []
        for i in range(48):
            t_GAT_x = GAT_x[i::48].squeeze(1)
            if output == []:
                output = self.GAT(t_GAT_x, self.adj)
            else:
                output = torch.cat((output, self.GAT(t_GAT_x, self.adj)), dim=0)

        t_GAT_x = []
        for i in range(45, 48):
            if t_GAT_x == []:
                t_GAT_x = GAT_x[i::48].squeeze(1)
            else:
                t_GAT_x = torch.cat((t_GAT_x, GAT_x[i::48].squeeze(1)), dim=0)

        output2 = self.GAT(t_GAT_x, st_fusion_adj)

        decoder_output = []
        for i in range(self.nNode):
            t_output = torch.cat((output[i::8], output2[i::8]), dim=0)

            if decoder_output == []:
                decoder_output = self.decoder(t_output)[-1]
            else:
                decoder_output = torch.cat((decoder_output, self.decoder(t_output)[-1]), dim=0)

        decoder_output = decoder_output.unsqueeze(1)
        return decoder_output


class GAT(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(in_features, hidden_features, dropout=dropout, alpha=alpha, concat=True)
                           for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(hidden_features * nheads, out_features, dropout=dropout, alpha=alpha,
                                           concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, hidden_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = hidden_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, hidden_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * hidden_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h,
                      self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
