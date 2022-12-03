import torch as T
import torch.nn as Tnn
import torch.nn.functional as TnnF
from mlp import MLP 

class CNNgraph(Tnn.Module):
    def __init__(self, Nlayers, Nmlp_layers, D_input, D_hidden, D_output, R_dropupout, TF, Gpooling_method, Npooling_method, device):

        super(CNNgraph, self).__init__()

        self.Nlayers = Nlayers 
        self.R_dropupout = R_dropupout
        self.TF = TF
        self.Gpooling_method = Gpooling_method
        self.Npooling_method = Npooling_method
        self.device = device


        self.eps = Tnn.Parameter(T.zeros(self.Nlayers-1))
        
        self.pred_lin = Tnn.ModuleList()

        self.mlps = Tnn.ModuleList()
        self.I_final_pred = Tnn.ModuleList()

        for l in range(Nlayers):
            if l == 0:
                self.pred_lin = self.pred_lin + [Tnn.Linear(D_input, D_output)]
            else:
                self.pred_lin = self.pred_lin + [Tnn.Linear(D_hidden, D_output)]

        for l in range(self.Nlayers-1):
            if l == 0:
                self.mlps = self.mlps + [MLP(Nmlp_layers, D_input, D_hidden, D_hidden)]
            else:
                self.mlps = self.mlps + [MLP(Nmlp_layers, D_hidden, D_hidden, D_hidden)]

            self.I_final_pred = self.I_final_pred + [Tnn.BatchNorm1d(D_hidden)]


    def N_maxpool(self, graph_b):

        N_max = max([graph.N_max for graph in graph_b])

        C_graph = []
        list = [0]

        for i, graph in enumerate(graph_b):

            list = list + [list[i] + len(graph.g)]
            pad_N = []

            for j in range(len(graph.neighbors)):

                temp_pad = [n + list[i] for n in graph.neighbors[j]]
                temp_pad.extend([-1]*(N_max - len(temp_pad)))

                if not self.TF:
                    temp_pad = temp_pad + [j + list[i]]

                pad_N = pad_N + [temp_pad]

            C_graph.extend(pad_N)

        return T.LongTensor(C_graph)


    def N_sum_mean_pooling(self, graph_b):

        sparse_matrix = []
        list = [0]

        for i, graph in enumerate(graph_b):

            list = list + [list[i] + len(graph.g)]
            sparse_matrix = sparse_matrix + [graph.edge_mat + list[i]]

        adj_S_mat = T.cat(sparse_matrix, 1)
        ones = T.ones(adj_S_mat.shape[1])

        if not self.TF:
            node_index = list[-1]
            sle = T.LongTensor([range(node_index), range(node_index)])
            ones1 = T.ones(node_index)
            adj_S_mat = T.cat([adj_S_mat, sle], 1)
            ones = T.cat([ones, ones1], 0)

        adjacentBlock = T.sparse.FloatTensor(adj_S_mat, ones, T.Size([list[-1],list[-1]]))

        return adjacentBlock.to(self.device)


    def graph_pooling(self, graph_b):
          
        list = [0]

        for i, graph in enumerate(graph_b):
            list = list + [list[i] + len(graph.g)]

        index = []
        node = []
        for i, graph in enumerate(graph_b):
            
            if self.Gpooling_method == "average":
                node.extend([1./len(graph.g)]*len(graph.g))
            else:
                node.extend([1]*len(graph.g))

            index.extend([[i, j] for j in range(list[i], list[i+1], 1)])
        index = T.LongTensor(index).transpose(0,1)
        node = T.FloatTensor(node)
        Gpooling = T.sparse.FloatTensor(index, node, T.Size([len(graph_b), list[-1]]))
        
        return Gpooling.to(self.device)


    def MaxPooling(self, h, padded_neighborurhood):

        temp = T.min(h,dim = 0)[0]
        temp_h = T.cat([h, temp.reshape((1, -1)).to(self.device)])
        poolElem = T.max(temp_h[padded_neighborurhood],dim = 1)[0]

        return poolElem


    def next_eps(self, h, l, padded_neighborurhood = None, adjacentBlock = None):
        # l l   
        if self.Npooling_method == "max":
            N_pool = self.MaxPooling(h, padded_neighborurhood)
        else:
            N_pool = T.spmm(adjacentBlock, h)
            if self.Npooling_method == "average":
                avg = T.spmm(adjacentBlock, T.ones((adjacentBlock.shape[0], 1)).to(self.device))
                N_pool = N_pool/avg

        N_pool = N_pool + (1 + self.eps[l])*h
        poolElem = self.mlps[l](N_pool)
        h = self.I_final_pred[l](poolElem)
        h = TnnF.relu(h)

        return h


    def next_layer(self, h, l, padded_neighborurhood = None, adjacentBlock = None):

        if self.Npooling_method == "max":
            N_pool = self.MaxPooling(h, padded_neighborurhood)

        else:
            N_pool = T.spmm(adjacentBlock, h)
            if self.Npooling_method == "average":
                avg = T.spmm(adjacentBlock, T.ones((adjacentBlock.shape[0], 1)).to(self.device))
                N_pool = N_pool/avg

        poolElem = self.mlps[l](N_pool)
        h = self.I_final_pred[l](poolElem)
        h = TnnF.relu(h)

        return h



    def forward(self, graph_b):

        Concatenete_X = T.cat([graph.features for graph in graph_b], 0).to(self.device)
        Gpooling = self.graph_pooling(graph_b)

        if self.Npooling_method == "max":
            padded_neighborurhood = self.N_maxpool(graph_b)
        else:
            adjacentBlock = self.N_sum_mean_pooling(graph_b)

        H = [Concatenete_X]
        h = Concatenete_X

        for l in range(self.Nlayers-1):
            if self.Npooling_method == "max" and self.TF:
                h = self.next_eps(h, l, padded_neighborurhood = padded_neighborurhood)
            elif not self.Npooling_method == "max" and self.TF:
                h = self.next_eps(h, l, adjacentBlock = adjacentBlock)
            elif not self.TF and self.Npooling_method == "max":
                h = self.next_layer(h, l, padded_neighborurhood = padded_neighborurhood)
            elif not self.TF and  not self.Npooling_method == "max":
                h = self.next_layer(h, l, adjacentBlock = adjacentBlock)
            H = H + [h]

        c = 0
    
        for l, h in enumerate(H):
            Gpooling_h = T.spmm(Gpooling, h)
            c = c + TnnF.dropout(self.pred_lin[l](Gpooling_h), self.R_dropupout, training = self.training)

        return c
 
