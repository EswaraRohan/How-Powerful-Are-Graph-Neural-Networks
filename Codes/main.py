import argparse
from util import load_data, separate_data
import torch as T
import torch.nn as Tnn
import torch.nn.functional as TnnF
import torch.optim as torchoptim
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from CNNgraph import CNNgraph

criteria = Tnn.CrossEntropyLoss()

al = []
traacc = []
teaacc = []
def training(Graph_Train,Opt,epoch,arg,model,device):
    model.train()
    NoOfIterations = arg.iters_per_epoch
    LossAccumulation = 0
    p = tqdm(range(NoOfIterations),unit='batch')
    
    for i in p:
        rand = np.random.permutation(len(Graph_Train))[:arg.batch_size]
        Graph_Batch = [Graph_Train[idx] for idx in rand]
        Label = T.LongTensor([graph.label for graph in Graph_Batch]).to(device)
        Output = model(Graph_Batch)
        Loss = criteria(Output,Label)

        if Opt is not None:
            Opt.zero_grad()
            Loss.backward()
            Opt.step()
            
        Loss = Loss.detach().cpu().numpy()    
        LossAccumulation = LossAccumulation + Loss
        p.set_description('epoch: %d' %(epoch))
        
    AverageLoss = LossAccumulation/NoOfIterations
    print("Average Loss - %f" %(AverageLoss))
    al.append(AverageLoss)
    return AverageLoss


def data(model,Graphs,Size_Batch = 64):
    idx = np.arange(len(Graphs))    
    op = []
    model.eval()
    for k in range(0,len(Graphs),Size_Batch):
        rand = idx[k:k+Size_Batch]
        if len(rand) == 0:
            continue
        op = op + [(model([Graphs[var] for var in rand]).detach())]
    return T.cat(op,0)


def testing(Graph_Train,Graph_Test,model,device):
    model.eval()
    
    op = data(model,Graph_Train)
    Prediction = op.max(1,keepdim=True)[1]
    Label = T.LongTensor([graph.label for graph in Graph_Train]).to(device)
    crt = Prediction.eq(Label.view_as(Prediction)).sum().cpu().item()
    
    tra = float(len(Graph_Train))
    TrainingAccuracy = crt/tra

    op = data(model, Graph_Test)
    Prediction = op.max(1, keepdim=True)[1]
    Label = T.LongTensor([graph.label for graph in Graph_Test]).to(device)
    crt = Prediction.eq(Label.view_as(Prediction)).sum().cpu().item()
    
    tes = float(len(Graph_Test))
    TestingAccuracy = crt/tes
    
    print("Testing Accuracy: %f Training Accuracy: %f" % (TestingAccuracy,TrainingAccuracy))
    traacc.append(TrainingAccuracy)
    teaacc.append(TestingAccuracy)
    return TrainingAccuracy,TrainingAccuracy

def main():
    
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--iters_per_epoch', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fold_idx', type=int, default=0)
    parser.add_argument('--Nlayers', type=int, default=5)
    parser.add_argument('--Nmlp_layers', type=int, default=2)
    parser.add_argument('--D_hidden', type=int, default=64)
    parser.add_argument('--R_dropupout', type=float, default=0.5)
    parser.add_argument('--Gpooling_method', type=str, default="sum", choices=["sum", "average"])
    parser.add_argument('--Npooling_method', type=str, default="sum", choices=["sum", "average", "max"])
    # parser.add_argument('--Npooling_method', type=str, default="sum", choices=["sum", "average", "max"])
    parser.add_argument('--TF', action="store_false")
    # parser.add_argument('--TF', action="store_true")
    parser.add_argument('--degree_as_tag', action="store_true")
    parser.add_argument('--filename', type = str, default = "")
    arg = parser.parse_args()

    T.manual_seed(0)
    np.random.seed(0)   
    if T.cuda.is_available():
        device = T.device("cuda:" + str(arg.device))
        T.cuda.manual_seed_all(0)
    else:
        device = T.device("cpu")
    

    Graphs,num_classes = load_data(arg.dataset,arg.degree_as_tag)
    Graph_Train,Graph_Test = separate_data(Graphs,arg.seed,arg.fold_idx)
    model = CNNgraph(arg.Nlayers, arg.Nmlp_layers, Graph_Train[0].features.shape[1], arg.D_hidden, num_classes, arg.R_dropupout, arg.TF, arg.Gpooling_method, arg.Npooling_method, device).to(device)
    Opt = torchoptim.Adam(model.parameters(), lr=arg.lr)
    schedule = torchoptim.lr_scheduler.StepLR(Opt,step_size=50,gamma=0.5)


    for epoch in range (arg.epochs):
        schedule.step()
        AverageLoss = training(Graph_Train,Opt,epoch+1,arg,model,device)
        TrainingAccuracy,TestingAccuracy = testing(Graph_Train,Graph_Test,model,device)

        if not arg.filename == "":
            with open(arg.filename,'w') as file:
                file.write("%f %f %f" % (AverageLoss,TrainingAccuracy,TestingAccuracy))
                file.write("\n")
        #print(model.eps)

    plt.subplot(1,2,1)
    xaxis = np.arange(arg.epochs)+1
    plt.plot(xaxis,traacc,label='Training Accuracy')
    plt.plot(xaxis,teaacc,label='Testing Accuracy')
    plt.ylim(0, 1.1)
    #plt.plot(xaxis,al,label='Average loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracies')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(xaxis,al)
    plt.title('Average loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig('X.png')




if __name__ == '__main__':
    main()
    
