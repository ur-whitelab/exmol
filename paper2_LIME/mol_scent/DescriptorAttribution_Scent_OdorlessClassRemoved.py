#!/usr/bin/env python
# coding: utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from rdkit.Chem import MolFromSmarts, MolFromSmiles
from rdkit import DataStructs

# import mordred, mordred.descriptors
import scipy.stats as ss
import sklearn.metrics
from IPython.display import display, SVG
from rdkit.Chem.Draw import MolToImage as mol2img, DrawMorganBit  # type: ignore
from rdkit.Chem import rdchem, MACCSkeys, AllChem  # type: ignore

import warnings

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-scent", help="scent class to analyse")
parser.add_argument("-space_file", help="scent samples csv file")
args = parser.parse_args()

scent = args.scent
space_file = args.space_file

warnings.filterwarnings("ignore")
sns.set_context("notebook")
sns.set_style(
    "dark",
    {
        "xtick.bottom": True,
        "ytick.left": True,
        "xtick.color": "#666666",
        "ytick.color": "#666666",
        "axes.edgecolor": "#666666",
        "axes.linewidth": 0.8,
        "figure.dpi": 300,
    },
)
color_cycle = ["#1BBC9B", "#F06060", "#5C4B51", "#F3B562", "#6e5687"]
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=color_cycle)
np.random.seed(0)
tf.random.set_seed(0)


# ### GNN Model Related Code
#
# GNN model using molecular scent dataset from Leffingwell Odor Datset (https://zenodo.org/record/4085098#.YTfYwy1h29Y)
#
# Code below modified from example code given in the "Predicting DFT Energies with GNNs" and "Interpretability and Deep Learning" sections of "Deep Learning for Molecules and Materials" textbook (https://whitead.github.io/dmol-book/applied/QM9.html)

# In[2]:


# Parameters for GNN model
node_feat_length = 256
message_feat_length = 256
graph_feat_length = 512
weights_stddevGNN = 0.01


# Code to load data & generate graphs + labels for all molecules in dataset
# Load data --> file uploaded to jhub (locally stored)
scentdata = pd.read_csv("leffingwell_data_shuffled.csv")

# Code to generate list of all scent labels (scentClasses)
numMolecules = len(scentdata.odor_labels_filtered)
numClasses = 112  # No odorless class
scentClasses = []
moleculeScentList = []
for i in range(numMolecules):
    scentString = scentdata.odor_labels_filtered[i]
    temp = scentString.replace("[", "")
    temp = temp.replace("]", "")
    temp = temp.replace("'", "")
    temp = temp.replace(" ", "")
    scentList = temp.split(",")
    if "odorless" in scentList:
        scentList.remove("odorless")
    moleculeScentList.append(scentList)
    for j in range(len(scentList)):
        if not (scentList[j] in scentClasses):
            scentClasses.append(scentList[j])

# Check to make sure read in data properly & created scentClasses & moleculeScentList correctly
print(f"Is the number of scent classes 113?: {len(scentClasses)==112}")
print(f"Is the number of molecules 3523?: {len(moleculeScentList)==3523}")


def gen_smiles2graph(sml):
    """Argument for the RD2NX function should be a valid SMILES sequence
    returns: the graph
    """
    m = rdkit.Chem.MolFromSmiles(sml)
    m = rdkit.Chem.AddHs(m)
    order_string = {
        rdkit.Chem.rdchem.BondType.SINGLE: 1,
        rdkit.Chem.rdchem.BondType.DOUBLE: 2,
        rdkit.Chem.rdchem.BondType.TRIPLE: 3,
        rdkit.Chem.rdchem.BondType.AROMATIC: 4,
    }
    N = len(list(m.GetAtoms()))
    nodes = np.zeros((N, node_feat_length))
    for i in m.GetAtoms():
        nodes[i.GetIdx(), i.GetAtomicNum()] = 1
        # Add in whether atom is in a ring or not for one-hot encoding
        if i.IsInRing():
            nodes[i.GetIdx(), -1] = 1

    adj = np.zeros((N, N))
    for j in m.GetBonds():
        u = min(j.GetBeginAtomIdx(), j.GetEndAtomIdx())
        v = max(j.GetBeginAtomIdx(), j.GetEndAtomIdx())
        order = j.GetBondType()
        if order in order_string:
            order = order_string[order]
        else:
            raise Warning("Ignoring bond order" + order)
        adj[u, v] = 1
        adj[v, u] = 1
    adj += np.eye(N)
    return nodes, adj


# Function that creates label vector given list of strings describing scent of molecule as input
# Each index in label vector corresponds to specific scent -> if output has a 0 at index i, then molecule does not have scent i
# If label vector has 1 at index i, then molecule does have scent i


def createLabelVector(scentsList):
    # Find class index in label vector that each scent corresponds to & update label for that molecule to 1
    labelVector = np.zeros(numClasses)
    for j in range(len(scentsList)):
        # Find class index
        classIndex = scentClasses.index(scentsList[j])
        # print(classIndex)
        # print(scentsList[j])
        # print(scentClasses[classIndex])
        # Update label vector
        labelVector[classIndex] = 1
    return labelVector


def generateGraphs():
    for i in range(numMolecules):
        graph = gen_smiles2graph(scentdata.smiles[i])
        labels = createLabelVector(moleculeScentList[i])
        yield graph, labels


# Check that generateGraphs() works for 1st molecule
# print(gen_smiles2graph(scentdata.SMILES[0]))
# print(scentdata.SENTENCE[0].split(','))
# print(np.nonzero(createLabelVector(scentdata.SENTENCE[0].split(','))))
# print(scentClasses[89])
data = tf.data.Dataset.from_generator(
    generateGraphs,
    output_types=((tf.float32, tf.float32), tf.float32),
    output_shapes=(
        (tf.TensorShape([None, node_feat_length]), tf.TensorShape([None, None])),
        tf.TensorShape([None]),
    ),
)


# In[3]:


class GNNLayer(
    hk.Module
):  # TODO: If increase number of layers, stack features & new_features and shrink via dense layer
    def __init__(self, output_size, name=None):
        super().__init__(name=name)
        self.output_size = output_size

    def __call__(self, inputs):
        # split input into nodes, edges & features
        nodes, edges, features = inputs
        # Nodes is of shape (N, Nf) --> N = # atoms, Nf = node_feature_length
        # Edges is of shape (N,N) (adjacency matrix)
        # Features is of shape (Gf) --> Gf = graph_feature_length

        graph_feature_len = features.shape[-1]  # graph_feature_len (Gf)
        node_feature_len = nodes.shape[-1]  # node_feature_len (Nf)
        message_feature_len = message_feat_length  # message_feature_length (Mf)

        # Initialize weights
        w_init = hk.initializers.RandomNormal(stddev=weights_stddevGNN)

        # we is of shape (Nf,Mf)
        we = hk.get_parameter(
            "we", shape=[node_feature_len, message_feature_len], init=w_init
        )

        # b is of shape (Mf)
        b = hk.get_parameter("b", shape=[message_feature_len], init=w_init)

        # wv is of shape (Mf,Nf)
        wv = hk.get_parameter(
            "wv", shape=[message_feature_len, node_feature_len], init=w_init
        )

        # wu is of shape (Nf,Gf)
        wu = hk.get_parameter(
            "wu", shape=[node_feature_len, graph_feature_len], init=w_init
        )

        # make nodes be N x N x Nf so we can just multiply directly (N = number of atoms)
        # ek is now shaped N x N x Mf
        ek = jax.nn.leaky_relu(
            b
            + jnp.repeat(nodes[jnp.newaxis, ...], nodes.shape[0], axis=0)
            @ we
            * edges[..., None]
        )

        # Uncomment lines below to update edges
        # Update edges, use jnp.any to have new_edges be of shape N x N
        # new_edges = jnp.any(ek, axis=-1)

        # Normalize over edge features w/layer normalization
        # new_edges = hk.LayerNorm(axis=[0,1], create_scale=False, create_offset=False, eps=1e-05)(new_edges)

        # take sum over neighbors to get ebar shape = Nf x Mf
        ebar = jnp.sum(ek, axis=1)

        # dense layer for new nodes to get new_nodes shape = N x Nf
        new_nodes = jax.nn.leaky_relu(ebar @ wv) + nodes  # Use leaky ReLU

        # Normalize over node features w/layer normalization
        new_nodes = hk.LayerNorm(
            axis=[0, 1], create_scale=False, create_offset=False, eps=1e-05
        )(new_nodes)

        # sum over nodes to get shape features so global_node_features shape = Nf
        global_node_features = jnp.sum(new_nodes, axis=0)

        # dense layer for new features so new_features shape = Gf
        new_features = (
            jax.nn.leaky_relu(global_node_features @ wu) + features
        )  # Use leaky ReLU for activation

        return new_nodes, edges, new_features


def model_fn(x):
    nodes, edges = x
    features = jnp.ones(graph_feat_length)
    x = nodes, edges, features

    # NOTE: If edited config.num_GNN_layers, need to edit code below (increase or decrease # times have x = GNNLayer(...))
    # 4 GNN layers
    x = GNNLayer(output_size=graph_feat_length)(x)
    x = GNNLayer(output_size=graph_feat_length)(x)
    x = GNNLayer(output_size=graph_feat_length)(x)
    x = GNNLayer(output_size=graph_feat_length)(x)

    # 2 dense layers
    logits = hk.Linear(numClasses)(x[-1])
    # logits = jax.nn.relu(logits) #ReLU activation between dense layer
    logits = hk.Linear(numClasses)(logits)

    return logits  # Model now returns logits


model = hk.without_apply_rng(hk.transform(model_fn))

# Initialize model
rng = jax.random.PRNGKey(0)
sampleData = data.take(1)
for dataVal in sampleData:  # Look into later how to get larger set
    (nodes_i, edges_i), yi = dataVal
nodes_i = nodes_i.numpy()
edges_i = edges_i.numpy()

yi = yi.numpy()
xi = (nodes_i, edges_i)

params = model.init(rng, xi)


# Load optimal parameters for GNN model
print("Edit fileName to change parameters being loaded")
fileName = "optParams_dry-waterfall-17.npy"  # Currently optimal parameters, edit when get better model
paramsArr = jnp.load(fileName, allow_pickle=True)
opt_params = {
    "gnn_layer": {
        "b": paramsArr[0],
        "we": paramsArr[1],
        "wu": paramsArr[2],
        "wv": paramsArr[3],
    },
    "gnn_layer_1": {
        "b": paramsArr[4],
        "we": paramsArr[5],
        "wu": paramsArr[6],
        "wv": paramsArr[7],
    },
    "gnn_layer_2": {
        "b": paramsArr[8],
        "we": paramsArr[9],
        "wu": paramsArr[10],
        "wv": paramsArr[11],
    },
    "gnn_layer_3": {
        "b": paramsArr[12],
        "we": paramsArr[13],
        "wu": paramsArr[14],
        "wv": paramsArr[15],
    },
    "linear": {"b": paramsArr[16], "w": paramsArr[17]},
    "linear_1": {"b": paramsArr[18], "w": paramsArr[19]},
}


# Read in threshold values for each scent class (in test set) that maximizes F1 score
thresholds = pd.read_csv("ThresholdsForMaxF1_OdorlessClassRemoved_dry-waterfall-17.csv")


def my_model(smilesString, scentString):
    molecularGraph = gen_smiles2graph(smilesString)
    pos = scentClasses.index(scentString)
    thresholdIndex_scent = thresholds.index[thresholds.Scent == scentString].tolist()
    threshold = thresholds.Threshold[thresholdIndex_scent].tolist()[
        0
    ]  # Threshold is the one that maximizes the F1 score
    pred = jax.nn.sigmoid(model.apply(opt_params, molecularGraph))[pos]
    if pred > threshold:
        pred = 1
    else:
        pred = 0
    return pred


def lime_explain(
    examples,
    descriptor_type="MACCS",
    return_beta=True,
    multiple_bases=None,
):
    """From given :obj:`Examples<Example>`, find descriptor t-statistics (see
    :doc: `index`)
    :param examples: Output from :func: `sample_space`
    :param descriptor_type: Desired descriptors, choose from 'Classic', 'ECFP' 'MACCS'
    :return_beta: Whether or not the function should return regression coefficient values
    :param multiple_bases: Consider multiple bases for explanation (default: infer from examples)
    """
    if multiple_bases is None:
        multiple_bases = exmol.exmol._check_multiple_bases(examples)

    # add descriptors
    examples = exmol.add_descriptors(
        examples, descriptor_type, multiple_bases=multiple_bases
    )
    # weighted tanimoto similarities
    w = np.array([1 / (1 + (1 / (e.similarity + 0.000001) - 1) ** 5) for e in examples])
    # Only keep nonzero weights
    non_zero = w > 10 ** (-6)
    nonzero_w = w[non_zero]
    # create a diagonal matrix of w
    N = nonzero_w.shape[0]
    diag_w = np.zeros((N, N))
    np.fill_diagonal(diag_w, nonzero_w)
    # get feature matrix
    x_mat = np.array([list(e.descriptors.descriptors) for e in examples])[
        non_zero
    ].reshape(N, -1)
    # remove zero variance columns
    y = (
        np.array([e.yhat for e in examples])
        .reshape(len(examples))[non_zero]
        .astype(float)
    )
    # remove bias
    y -= np.mean(y)
    # compute least squares fit
    xtinv = np.linalg.pinv(
        (x_mat.T @ diag_w @ x_mat)
        + 0.001 * np.identity(len(examples[0].descriptors.descriptors))
    )
    beta = xtinv @ x_mat.T @ (y * nonzero_w)
    # compute standard error in beta
    yhat = x_mat @ beta
    resids = yhat - y
    SSR = np.sum(resids**2)
    se2_epsilon = SSR / (len(examples) - len(beta))
    se2_beta = se2_epsilon * xtinv
    # now compute t-statistic for existence of coefficients
    tstat = beta * np.sqrt(1 / np.diag(se2_beta))
    # Set tstats for base, to be used later
    examples[0].descriptors.tstats = tstat
    # Return beta (feature weights) which are the fits if asked for
    if return_beta:
        return beta
    else:
        return None


# compute dot product with labels
def cosine_similarity_base(df, bases, llists):
    df["label_dot"] = np.array(0.0)
    for j, row in df.iterrows():
        if j in bases:
            base = j
            df["label_dot"][j] = 1
        else:
            # cosine similarity
            if np.all(llists[j] == 0):
                df["label_dot"][j] = 0
                continue
            df["label_dot"][j] = (
                llists[base]
                @ llists[j]
                / np.linalg.norm(llists[base])
                / np.linalg.norm(llists[j])
            )
    return df


def createExampleListfromDataFrame(data):
    exampleList = []  # list[exmol.Example]()
    for i in range(len(data.index)):
        # using weighted tanimoto with dot product
        exampleList.append(
            exmol.Example(
                data.smiles.tolist()[i],
                data.selfies.tolist()[i],
                data.label_similarity.tolist()[i],
                data.yhat.tolist()[i],
                data.index.tolist()[i],
                data.position.tolist()[i],
                data.is_origin.tolist()[i],
                data.cluster.tolist()[i],
                data.label.tolist()[i],
            )
        )
    return exampleList


def multiple_aromatic_rings(mol):
    ri = mol.GetRingInfo()
    count = 0
    for bondRing in ri.BondRings():
        flag = True
        for id in bondRing:
            if not mol.GetBondWithIdx(id).GetIsAromatic():
                flag = False
                continue
        if flag:
            count += 1
    return True if count > 1 else False


def get_text_explanations(examples):
    """Take an example and convert t-statistics into text explanations"""
    from importlib_resources import files
    import exmol.lime_data

    print(examples[0].descriptors.descriptor_type.lower())
    if examples[0].descriptors.descriptor_type.lower() != "maccs":
        raise ValueError("Text explaantions only work for MACCS descriptors")

    # Take t-statistics, rank them
    tstats = list(examples[0].descriptors.tstats)
    d_importance = {
        a: [b, i]
        for i, a, b in zip(
            np.arange(len(examples[0].descriptors.descriptors)),
            examples[0].descriptors.descriptor_names,
            tstats,
        )
        if not np.isnan(b)
    }
    d_importance = dict(
        sorted(d_importance.items(), key=lambda item: abs(item[1][0]), reverse=True)
    )

    # get significance value - if >significance, then important else weakly important?
    w = np.array([1 / (1 + (1 / (e.similarity + 0.000001) - 1) ** 5) for e in examples])
    effective_n = np.sum(w) ** 2 / np.sum(w**2)
    T = ss.t.ppf(0.975, df=effective_n)

    # get a substructure match!! Is it in the molecule?
    mk = files(exmol.lime_data).joinpath("MACCSkeys.txt")
    with open(str(mk), "r") as f:
        desc_smarts = {
            x.strip().split("\t")[-1]: x.strip().split("\t")[-2]
            for x in f.readlines()[1:]
        }
    mol = MolFromSmiles(examples[0].smiles)

    # text explanation
    positive_exp = "Positive features:\n"
    negative_exp = "Negative features:\n"
    for i, (k, v) in enumerate(zip(d_importance.keys(), d_importance.values())):

        if i == 5:
            break

        Match = False
        if k.lower() == "are there multiple aromatic rings?":
            match = multiple_aromatic_rings(mol)
        else:
            patt = MolFromSmarts(desc_smarts[k])
            if len(mol.GetSubstructMatch(patt)) > 0:
                match = True

        if match:
            if abs(v[0]) > 4:
                imp = "Very Important\n"
            elif abs(v[0]) >= T:
                imp = "Important\n"
            else:
                continue
            if v[0] > 0:
                positive_exp += f"{k} " + "Yes. " + imp
            else:
                negative_exp += f"{k} " + "Yes. " + imp
        else:
            continue

    return positive_exp + negative_exp


# Load files with preciously sampled space and labeled with scents

df = pd.read_csv(space_file, usecols=np.arange(1, 11))
# vanilla_df['lsimilarity'] = np.array(0.)
labels = pd.read_csv(space_file, usecols=np.append([1], np.arange(11, 123)))
llists = labels.to_numpy()[:, 1:]
bases = list(df[df["is_origin"] == True].index)

df = cosine_similarity_base(df, bases, llists)
df["label_similarity"] = df["similarity"] * df["label_dot"]

df.to_csv(f"{scent}_samples_label_weighted_similarity.csv")

samples = createExampleListfromDataFrame(df)

lime_explain(samples, descriptor_type="ECFP")

svg = exmol.plot_descriptors(
    samples, output_file=f"plots/{scent}_ecfp.svg", return_svg=True
)

lime_explain(samples, descriptor_type="MACCS")
exmol.plot_descriptors(samples, output_file=f"plots/{scent}_maccs.svg")

prompt = (
    get_text_explanations(samples)
    + f"Explanation: Molecules have {scent} smell because"
)

with open(f"prompts/{scent}.txt", "w+") as f:
    f.write(prompt)
