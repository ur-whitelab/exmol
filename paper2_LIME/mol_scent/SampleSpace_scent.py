#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Descriptor attribution using exmol


# In[1]:


#Imports
#!pip install matplotlib numpy pandas seaborn jax jaxlib dm-haiku tensorflow exmol
import exmol
import tensorflow as tf
import seaborn as sns
import jax.numpy as jnp
import jax
import jax.experimental.optimizers as opt
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import haiku as hk
import numpy as np
import rdkit, rdkit.Chem, rdkit.Chem.rdDepictor, rdkit.Chem.Draw
import mordred, mordred.descriptors
import sklearn.metrics
from IPython.display import display, SVG
from rdkit.Chem.Draw import MolToImage as mol2img, DrawMorganBit  # type: ignore
from rdkit.Chem import rdchem, MACCSkeys, AllChem  # type: ignore
import sys

import warnings
warnings.filterwarnings('ignore')
sns.set_context('notebook')
sns.set_style('dark',  {'xtick.bottom':True, 'ytick.left':True, 'xtick.color': '#666666', 'ytick.color': '#666666',
                        'axes.edgecolor': '#666666', 'axes.linewidth':     0.8 , 'figure.dpi': 300})
color_cycle = ['#1BBC9B', '#F06060', '#5C4B51', '#F3B562', '#6e5687']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=color_cycle) 
np.random.seed(0)
tf.random.set_seed(0)


# ### GNN Model Related Code
# 
# GNN model using molecular scent dataset from Leffingwell Odor Datset (https://zenodo.org/record/4085098#.YTfYwy1h29Y)
# 
# Code below modified from example code given in the "Predicting DFT Energies with GNNs" and "Interpretability and Deep Learning" sections of "Deep Learning for Molecules and Materials" textbook (https://whitead.github.io/dmol-book/applied/QM9.html)

# In[2]:


#Parameters for GNN model
node_feat_length = 256
message_feat_length = 256
graph_feat_length = 512
weights_stddevGNN = 0.01


#Code to load data & generate graphs + labels for all molecules in dataset
#Load data --> file uploaded to jhub (locally stored)
scentdata = pd.read_csv('leffingwell_data_shuffled.csv')

#Code to generate list of all scent labels (scentClasses)
numMolecules = len(scentdata.odor_labels_filtered)
numClasses = 112 #No odorless class
scentClasses = []
moleculeScentList = []
for i in range(numMolecules):
    scentString = scentdata.odor_labels_filtered[i]
    temp = scentString.replace('[', '')
    temp = temp.replace(']','')
    temp = temp.replace('\'','')
    temp = temp.replace(' ','')
    scentList = temp.split(',')
    if('odorless' in scentList):
        scentList.remove('odorless')
    moleculeScentList.append(scentList)
    for j in range(len(scentList)):
        if (not(scentList[j] in scentClasses)):
            scentClasses.append(scentList[j])


#Check to make sure read in data properly & created scentClasses & moleculeScentList correctly
print(f'Is the number of scent classes 112?: {len(scentClasses)==112}')
print(f'Is the number of molecules 3523?: {len(moleculeScentList)==3523}')

def gen_smiles2graph(sml):
    '''Argument for the RD2NX function should be a valid SMILES sequence
    returns: the graph
    '''
    m = rdkit.Chem.MolFromSmiles(sml)
    m = rdkit.Chem.AddHs(m)
    order_string = {rdkit.Chem.rdchem.BondType.SINGLE: 1,
                    rdkit.Chem.rdchem.BondType.DOUBLE: 2,
                    rdkit.Chem.rdchem.BondType.TRIPLE: 3,
                    rdkit.Chem.rdchem.BondType.AROMATIC: 4}
    N = len(list(m.GetAtoms()))
    nodes = np.zeros((N,node_feat_length))
    for i in m.GetAtoms():
        nodes[i.GetIdx(), i.GetAtomicNum()] = 1
        #Add in whether atom is in a ring or not for one-hot encoding
        if(i.IsInRing()):
            nodes[i.GetIdx(), -1] = 1
    
    adj = np.zeros((N,N))
    for j in m.GetBonds():
        u = min(j.GetBeginAtomIdx(),j.GetEndAtomIdx())
        v = max(j.GetBeginAtomIdx(),j.GetEndAtomIdx())        
        order = j.GetBondType()
        if order in order_string:
            order = order_string[order]
        else:
            raise Warning('Ignoring bond order' + order)
        adj[u, v] = 1       
        adj[v, u] = 1 
    adj += np.eye(N)
    return nodes, adj

#Function that creates label vector given list of strings describing scent of molecule as input
#Each index in label vector corresponds to specific scent -> if output has a 0 at index i, then molecule does not have scent i
#If label vector has 1 at index i, then molecule does have scent i

def createLabelVector(scentsList):
    #Find class index in label vector that each scent corresponds to & update label for that molecule to 1
    labelVector = np.zeros(numClasses)
    for j in range(len(scentsList)):
        #Find class index
        classIndex = scentClasses.index(scentsList[j])
        #print(classIndex)
        #print(scentsList[j])
        #print(scentClasses[classIndex])
        #Update label vector
        labelVector[classIndex] = 1
    return labelVector

def generateGraphs():
    for i in range(numMolecules):
        graph = gen_smiles2graph(scentdata.smiles[i])   
        labels = createLabelVector(moleculeScentList[i])
        yield graph, labels

#Check that generateGraphs() works for 1st molecule
#print(gen_smiles2graph(scentdata.SMILES[0]))
#print(scentdata.SENTENCE[0].split(','))
#print(np.nonzero(createLabelVector(scentdata.SENTENCE[0].split(','))))
#print(scentClasses[89])
data = tf.data.Dataset.from_generator(generateGraphs, output_types=((tf.float32, tf.float32), tf.float32), 
                                      output_shapes=((tf.TensorShape([None, node_feat_length]), tf.TensorShape([None, None])), tf.TensorShape([None])))


# In[3]:


class GNNLayer(hk.Module): #TODO: If increase number of layers, stack features & new_features and shrink via dense layer

    def __init__(self, output_size, name=None):
        super().__init__(name=name)
        self.output_size = output_size

    def __call__(self, inputs):
        # split input into nodes, edges & features
        nodes, edges, features = inputs
        #Nodes is of shape (N, Nf) --> N = # atoms, Nf = node_feature_length
        #Edges is of shape (N,N) (adjacency matrix)
        #Features is of shape (Gf) --> Gf = graph_feature_length

        graph_feature_len = features.shape[-1] #graph_feature_len (Gf)
        node_feature_len = nodes.shape[-1] #node_feature_len (Nf)
        message_feature_len = message_feat_length #message_feature_length (Mf)
        
        #Initialize weights
        w_init = hk.initializers.RandomNormal(stddev = weights_stddevGNN)
        
        #we is of shape (Nf,Mf)
        we = hk.get_parameter("we", shape=[node_feature_len, message_feature_len], init=w_init)
        
        #b is of shape (Mf)
        b = hk.get_parameter("b", shape=[message_feature_len], init=w_init)
        
        #wv is of shape (Mf,Nf)
        wv = hk.get_parameter("wv", shape=[message_feature_len, node_feature_len], init=w_init)
        
        #wu is of shape (Nf,Gf)
        wu = hk.get_parameter("wu", shape=[node_feature_len, graph_feature_len], init=w_init)
        
        # make nodes be N x N x Nf so we can just multiply directly (N = number of atoms)
        # ek is now shaped N x N x Mf
        ek = jax.nn.leaky_relu(b + 
            jnp.repeat(nodes[jnp.newaxis,...], nodes.shape[0], axis=0) @ we * edges[..., None])

        #Uncomment lines below to update edges
        #Update edges, use jnp.any to have new_edges be of shape N x N
        #new_edges = jnp.any(ek, axis=-1) 
        
        #Normalize over edge features w/layer normalization
        #new_edges = hk.LayerNorm(axis=[0,1], create_scale=False, create_offset=False, eps=1e-05)(new_edges)
    
        # take sum over neighbors to get ebar shape = Nf x Mf
        ebar = jnp.sum(ek, axis=1)
        
        # dense layer for new nodes to get new_nodes shape = N x Nf
        new_nodes = jax.nn.leaky_relu(ebar @ wv) + nodes #Use leaky ReLU 
        
        #Normalize over node features w/layer normalization
        new_nodes = hk.LayerNorm(axis=[0,1], create_scale=False, create_offset=False, eps=1e-05)(new_nodes)
        
        # sum over nodes to get shape features so global_node_features shape = Nf
        global_node_features = jnp.sum(new_nodes, axis=0)
        
        # dense layer for new features so new_features shape = Gf
        new_features = jax.nn.leaky_relu(global_node_features  @ wu) + features #Use leaky ReLU for activation
        
        return new_nodes, edges, new_features

    
def model_fn(x):
    nodes, edges = x
    features = jnp.ones(graph_feat_length)
    x = nodes, edges, features
    
    #NOTE: If edited config.num_GNN_layers, need to edit code below (increase or decrease # times have x = GNNLayer(...))
    # 4 GNN layers
    x = GNNLayer(output_size=graph_feat_length)(x)
    x = GNNLayer(output_size=graph_feat_length)(x)
    x = GNNLayer(output_size=graph_feat_length)(x)
    x = GNNLayer(output_size=graph_feat_length)(x)
    
    # 2 dense layers
    logits = hk.Linear(numClasses)(x[-1])
    #logits = jax.nn.relu(logits) #ReLU activation between dense layer
    logits = hk.Linear(numClasses)(logits)

    return logits #Model now returns logits

model = hk.without_apply_rng(hk.transform(model_fn))

#Initialize model
rng = jax.random.PRNGKey(0)
sampleData = data.take(1)
for dataVal in sampleData: #Look into later how to get larger set
    (nodes_i, edges_i), yi = dataVal
nodes_i = nodes_i.numpy()
edges_i = edges_i.numpy()

yi = yi.numpy()
xi = (nodes_i,edges_i)

params = model.init(rng, xi)


# In[4]:


#Load optimal parameters for GNN model 
print('Edit fileName to change parameters being loaded')
fileName = 'optParams_dry-waterfall-17.npy' #Currently optimal parameters, edit when get better model
paramsArr = jnp.load(fileName, allow_pickle = True)
opt_params =  {'gnn_layer': {'b': paramsArr[0], 'we': paramsArr[1], 'wu': paramsArr[2], 'wv': paramsArr[3]},'gnn_layer_1': {'b': paramsArr[4], 'we': paramsArr[5], 'wu': paramsArr[6], 'wv': paramsArr[7]},'gnn_layer_2': {'b': paramsArr[8], 'we': paramsArr[9], 'wu': paramsArr[10], 'wv': paramsArr[11]}, 'gnn_layer_3': {'b': paramsArr[12], 'we': paramsArr[13], 'wu': paramsArr[14], 'wv': paramsArr[15]}, 'linear': {'b': paramsArr[16], 'w': paramsArr[17]} , 'linear_1': {'b': paramsArr[18], 'w': paramsArr[19]}}


# In[5]:


#Read in threshold values for each scent class (in test set) that maximizes F1 score
thresholds = pd.read_csv('ThresholdsForMaxF1_OdorlessClassRemoved_dry-waterfall-17.csv')


# In[6]:


def my_model(smilesString, scentString):
    molecularGraph = gen_smiles2graph(smilesString) 
    pos = scentClasses.index(scentString)
    thresholdIndex_scent = thresholds.index[thresholds.Scent==scentString].tolist()
    threshold = thresholds.Threshold[thresholdIndex_scent].tolist()[0] #Threshold is the one that maximizes the F1 score
    pred = jax.nn.sigmoid(model.apply(opt_params, molecularGraph))[pos]
    if(pred > threshold): 
        pred = 1
    else:
        pred = 0
    return pred


# ### Generate Sample Space around Top 10 Scents ('fruity', 'green', 'sweet', 'floral', 'fatty', 'herbal', 'apple', 'sulfurous', 'waxy', 'fresh')

# In[57]:


#Function takes in a SMILES string for a molecule and returns the scent class indices that the GNN model predicts the molecule belongs to
def my_model_allScents(smilesString):
    molecularGraph = gen_smiles2graph(smilesString) 
    predictions = jax.nn.sigmoid(model.apply(opt_params, molecularGraph))
    hard_predictions = (np.empty(numClasses)).tolist()
    scents = []
    for i in range(numClasses):
        scentString = scentClasses[i]
        thresholdIndex_scent = thresholds.index[thresholds.Scent==scentString].tolist()
        threshold = thresholds.Threshold[thresholdIndex_scent].tolist()[0] #Threshold is the one that maximizes the F1 score
        if(predictions[i] > threshold): 
            scents.append(scentString)
            hard_predictions[i] = 1
        else:
            hard_predictions[i] = 0
    return scents, hard_predictions


# In[58]:


#For each positive example, generate sample space (using STONED) around that example (use that example as the base molecule)
##then combine all of the sample spaces together 
space_total = []
scent = sys.argv[1]

for i in range(numMolecules): #sample space created using positive examples
    molecule = scentdata.smiles[i]
    if(moleculeScentList[i].count(scent) == 1): 
        sampleSpace = exmol.sample_space(molecule, lambda smi, sel: my_model(smi,scent), batched=False, preset='medium', num_samples=200)
        space_total.extend(sampleSpace)


# In[59]:


#Uncomment to test function that returns all scents of the model
'''
s, p = my_model_allScents('CC(=O)C1=NCCCC1')
for i in s:
    print(my_model('CC(=O)C1=NCCCC1', i))
    print(i)

print(p)
'''


# In[ ]:


space_total = pd.DataFrame(space_total)

#Get predictions for all scent classes for all molecules in the space
predictions = []
mols = space_total['smiles']
for mol in mols:
    scents, preds = my_model_allScents(mol)
    predictions.append(preds)

#Save the space along with scent class predictions to a csv file
space_total[scentClasses] = predictions
scentFileName = f'space_{scent}.csv'
space_total.to_csv(scentFileName)
                             
space_total.head()





