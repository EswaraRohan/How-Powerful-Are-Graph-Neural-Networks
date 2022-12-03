# SMAI-Project-Team-22

## Installation Requirements

pip install scikit-learn matplotlib Pillow 

pip install torch

pip install tqdm

pip install numpy

pip install networkx


## Running code

Open the Codes file

Unzip the dataset file  --> unzip dataset.zip

Run the main.py file  --> python3 main.py

(this runs the using data set MUTAG which is taken as default)

Rename the output plot file obtained(X.png) to store it.

### To run other datasets (for sum pooling)

python3 main.py --dataset "PROTEINS"

python3 main.py --dataset "PTC"

python3 main.py --dataset "NCI1"

###  To compare with max/ mean pooling methods

for a data set selected (in default)

use comands 

python3 main.py --Npooling_method "average"    
                 or
<br> python3 main.py --Npooling_method "max" 

Or 

change Npooling_method default to average or max which
  
gives plots respectively

### For multiple changes 

Ex:-  python3 main.py --dataset "PROTEINS"  --Npooling_method "max" 

can continue changing required features

