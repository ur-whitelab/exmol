# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import numpy as np
import tensorflow as tf
import selfies as sf
import exmol
from dataclasses import dataclass
from rdkit.Chem.Draw import rdDepictor

rdDepictor.SetPreferCoordGen(True)
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
soldata = pd.read_csv(
    "https://github.com/whitead/dmol-book/raw/master/data/curated-solubility-dataset.csv"
)
features_start_at = list(soldata.columns).index("MolWt")
np.random.seed(0)



soldata = soldata.sample(frac=0.01, random_state=0).reset_index(drop=True)



basic = set(exmol.get_basic_alphabet())
data_vocab = set(sf.get_alphabet_from_selfies([s for s in selfies_list if s is not None]))
vocab = ['[Nop]']
vocab.extend(list(data_vocab.union(basic)))
vocab_stoi = {o:i for o,i in zip(vocab, range(len(vocab)))}
#vocab_stoi['[nop]'] = 0 

def selfies2ints(s):
    result = []
    for token in sf.split_selfies(s):
        if token == '.':
            continue #?
        if token in vocab_stoi:
            result.append(vocab_stoi[token])
        else:
            result.append(np.nan)
            #print('Warning')
    return result

def ints2selfies(v):
    return ''.join([vocab[i] for i in v])

# test them out
s = selfies_list[0]
print('selfies:', s)
v = selfies2ints(s)
print('selfies2ints:', v)
so = ints2selfies(v)
print('ints2selfes:', so)
assert so == s.replace('.','') #make sure '.' is removed from Selfies string during assertion


@dataclass
class Config:
    vocab_size: int
    example_number: int
    batch_size: int
    buffer_size: int
    embedding_dim: int
    rnn_units: int
    hidden_dim: int


config = Config(
    vocab_size=len(vocab),
    example_number=len(selfies_list),
    batch_size=16,
    buffer_size=10000,
    embedding_dim=256,
    hidden_dim=128,
    rnn_units=128,
)



# now get sequences
encoded = [selfies2ints(s) for s in selfies_list if s is not None]
padded_seqs = tf.keras.preprocessing.sequence.pad_sequences(encoded, padding="post")

# Now build dataset
data = tf.data.Dataset.from_tensor_slices(
    (padded_seqs, soldata.Solubility.iloc[[bool(s) for s in selfies_list]].values)
)
# now split into val, test, train and batch
N = len(data)
split = int(0.1 * N)
test_data = data.take(split).batch(config.batch_size)
nontest = data.skip(split)
val_data, train_data = nontest.take(split).batch(config.batch_size), nontest.skip(
    split
).shuffle(config.buffer_size).batch(config.batch_size).prefetch(
    tf.data.experimental.AUTOTUNE
)



def uncompiled_model():
    "function for building uncompiled solubility RNN"
    model = tf.keras.Sequential()

    # make embedding and indicate that 0 should be treated as padding mask
    model.add(
        tf.keras.layers.Embedding(
          input_dim=config.vocab_size, output_dim=config.embedding_dim, mask_zero=True
      )
  )

    # RNN layer
    model.add(tf.keras.layers.GRU(config.rnn_units))
    # a dense hidden layer
    model.add(tf.keras.layers.Dense(config.hidden_dim, activation="relu"))
    # regression, so no activation
    model.add(tf.keras.layers.Dense(1))

    #model.summary() 
    return model 


def compile_model():
    compile_model = uncompiled_model()
  
    compile_model.compile(tf.optimizers.Adam(1e-4), loss="mean_squared_error")
    return compile_model



#result = compile_model().fit(train_data, validation_data=val_data, epochs=100, verbose=2)