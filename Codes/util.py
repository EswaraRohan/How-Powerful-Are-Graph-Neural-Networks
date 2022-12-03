import numpy as np
import networkx as nw
import random 
import torch as T

from sklearn.model_selection import StratifiedKFold

 
class S2VGraph(object):
    
    def __init__(self,g, label, flags = None,features = None):
        
        ### g --> A network graph
        ### label --> An integer graph label
        ### flags --> A list of integer node tags
        ### features --> A torch float tensor, one-hot representation of the tag is used as input to neural nets

        self.label = label
        self.g = g
        self.flags = flags
        self.features = 0
        self.neighbors = list()
        self.edge_mat = 0
        
        self.N_max = 0
        

def load_data(dataset, degree_tag):
        
        ### data --> name of dataset

        print('Loading Data')
        g_list = list()
        label_dict = dict()
        feat_dict = dict()
        
        with open('dataset/%s/%s.txt' %(dataset,dataset),'r') as f:
            
            n_g = int(f.readline().strip())
            
            for i in range(n_g):
                
                row = f.readline().strip().split()
                num, k = [int(w) for w in row]
                
                if not k in label_dict:
                    
                    len_map = len(label_dict)
                    label_dict[k] = len_map
                    
                g = nw.Graph()
                flags = list()
                features = list()
                n_edges = 0
                
                for node in range(num):
                    
                    g.add_node(node)
                    row = f.readline().strip().split()
                    temp = int(row[1])+ 2
                    
                    if(len(row) != temp):
                        
                        row = [int(w) for w in row[:temp]]
                        attr = np.array([float(w) for w in row[temp:]])
                        
                    else:
                        
                        row = [int(w) for w in row]
                        attr = None
                        
                    if not row[0] in feat_dict:
                        
                        len_map = len(feat_dict)
                        feat_dict[row[0]] = len_map
                        
                    flags = flags + [feat_dict[row[0]]]
                    
                    
                    if len(row) < temp:
                        
                        features = features + [attr]
                        
                    
                    n_edges = n_edges + row[1]
                    
                    for edge in range(2, len(row)):
                        
                        g.add_edge(node,row[edge])
                        
                
                if features == []:  ## needed to verify 
                    
                    features = None
                    flag = 0
                        
                        
                else:
                    
                    features = np.stack(features)
                    flag = 1
                
                # print(len(g),num)
                assert len(g) == num
                
                # g_list = g_list + [S2VGraph(g,k,flags)]
                g_list.append(S2VGraph(g, k, flags))
            # print(g_list)
                 
                        
        for g in g_list:
           
            g.neighbors = [[] for i in range(len(g.g))]
            
            for i,j in g.g.edges():
                
                g.neighbors[i] = g.neighbors[i] + [j]
                g.neighbors[j] = g.neighbors[j] + [i]

            list_degree = list()
            
            for i in range(len(g.g)):
                
                g.neighbors[i] = g.neighbors[i]
                list_degree = list_degree + [len(g.neighbors[i])]
            
            g.N_max = max(list_degree)

            g.label = label_dict[g.label]
            
            edges = [list(pair) for pair in g.g.edges()]
            edges.extend([[i,j] for j,i in edges])

            list_deg = list(dict(g.g.degree(range(len(g.g)))).values())
            
            g.edge_mat = T.LongTensor(edges).transpose(0,1)
            
            
        
        if degree_tag:
            
            for g in g_list:
                
                g.flags = list(dict(g.g.degree).values)
                
        

        tagset = set([])
        
        for g in g_list:
            # print("1x1")
            tagset = tagset.union(set(g.flags))
            
        tagset = list(tagset)
        tag_index = {tagset[i]:i for i in range(len(tagset))}
        


        for g in g_list:
            # print("x")
            g.features = T.zeros(len(g.flags),len(tagset))
            g.features[range(len(g.flags)),[tag_index[tag] for tag in g.flags]] = 1
            # print("y")

        
        print(" # classes : %d" % len(label_dict))
        print(" # maximum node tag: %d " % len(tagset))
        
        print('# data: %d' %len(g_list))
        
        
        vec = [g_list, len(label_dict)]
        
        return vec
                
                       
    
            
    
    
def separate_data(list_graph,seed,idx):
        
        assert 0 <= idx and idx < 10, "Index for folding must be from 0 to 9."
        
        fold = StratifiedKFold(n_splits=10,shuffle=True,random_state=seed) # same as k-fold but we use straitified sampling
        
        list_index = list()
        
        labels = [Graph.label for Graph in list_graph]
        
        
        for i in fold.split(np.zeros(len(labels)),labels):
            
            list_index = list_index + [i]
            
        index_training,index_testing = list_index[idx]
        
        list_graph_test = [list_graph[i] for i in index_testing]
        list_graph_train = [list_graph[i] for i in index_training]
        
        list_graph_train_test = [list_graph_train,list_graph_test]
        
        return list_graph_train_test
        
