import torch as T
import torch.nn as Tnn
import torch.nn.functional as TnnF

## Multilayer perceptron for linear output, used by GraphCNN

class MLP(Tnn.Module): 

    def __init__(self,Nlayers,D_input,D_hidden,D_output):
        
        ## Nlayers --> Number of layers in neural Networks
        ## D_input --> Dimensionality of input features
        ## D_hidden --> Dimensionality of hidden features
        ## D_output --> Number of classes for prediction
         
        super(MLP,self).__init__() ## refers to a subclass of Neural network
        
        self.Nlayers = Nlayers
        self.islinear = True # Keeping it is linear model as default
        
        # Linear model
        if Nlayers == 1:
            self.linear = Tnn.Linear(D_input,D_output)
            
        
        elif Nlayers < 1:
            raise ValueError("Nlayers should be positive")
        
        # Multilayer model
        else:
            
            self.islinear = False   
            self.linears = Tnn.ModuleList()  # Making a list of modules for both
            self.I_final_pred = Tnn.ModuleList() # linears and final Predicted values
            
            self.linears = self.linears + [Tnn.Linear(D_input,D_hidden)] # Appending layer 1 to linears list
            
            hidden_Layers = Nlayers - 2  # number of hidden kayers
            
            # Iterating through all hidden layers and 
            # appending each layer values
            
            for i in range(hidden_Layers):
                
                self.linears = self.linears + [Tnn.Linear(D_hidden,D_hidden)]
                
            self.linears = self.linears + [Tnn.Linear(D_hidden,D_output)] # For the output layer
            
            
            for i in range(Nlayers-1):
                self.I_final_pred = self.I_final_pred + [Tnn.BatchNorm1d((D_hidden))] 
                
    def forward(self,x):
        
        # If multilayer perceptron
        if not(self.islinear):
            
            temp = x
            for i in range(self.Nlayers - 1):
                temp = TnnF.relu(self.I_final_pred[i](self.linears[i](temp)))
            return self.linears[self.Nlayers-1](temp)
        # if linear model
        else:
            
             return self.linear(x)

