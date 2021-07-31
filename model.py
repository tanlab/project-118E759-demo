import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow.keras.models import load_model
import json
import pandas as pd
import numpy as np
import torch
import sys
import copy
# import tensorflow as tf
from jtnn import *
from rdkit import Chem

sys.path.append('./jtnn')
vocab = [x.strip("\r\n ") for x in open("unique_canonical_train_vocab.txt")]
vocab = Vocab(vocab)

hidden_size = 450
latent_size = 56
depth = 3
stereo = True

def get_gene_names(mode, cell_line):
    f = open(f"./Multi-task_models/multi-task_{mode}_models/Gene_list/{cell_line}_multi_task_gene_list_up.txt", "rt")
    up_genes = f.read()
    up_genes = up_genes[:-1]
    f.close()

    f = open(f"./Multi-task_models/multi-task_{mode}_models/Gene_list/{cell_line}_multi_task_gene_list_dn.txt", "rt")
    dn_genes = f.read()
    dn_genes = dn_genes[:-1]
    f.close()

    gene_list_ud = up_genes.split('\n') + (dn_genes.split('\n'))
    return up_genes.split('\n'), dn_genes.split('\n'), gene_list_ud

def binarize(floats, genes):
    r = []
    for i in range(len(floats)):
        if floats[i] > 0.5:
            r.append(genes[i])
    return r

def get_results(smiles, mode="jtvae", cell_line="A549"):
    if mode == "jtvae":

        model_jtvae = JTNNVAE(vocab, hidden_size, latent_size, depth, stereo=stereo)
        model_jtvae.cpu()
        model_jtvae.load_state_dict(torch.load("Models/model.iter-9-6000", map_location=torch.device('cpu'))) 

        koku = pd.DataFrame(columns=list(range(56)))
        dec_smiles = model_jtvae.reconstruct(smiles, DataFrame=koku)
        del dec_smiles
    else:
        mol = Chem.MolFromSmiles(smiles)
        koku = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024, useChirality=True).ToBitString()
        koku = np.fromstring(koku,'u1') - ord('0')
        koku = koku.reshape(1,1024)
        
    up_model = load_model(f"./Multi-task_models/multi-task_{mode}_models/Models/{cell_line}_multi_task_model_up.h5")
    dn_model = load_model(f"./Multi-task_models/multi-task_{mode}_models/Models/{cell_line}_multi_task_model_dn.h5")

    up_genes, dn_genes, _  = get_gene_names(mode, cell_line)
    

    up_floats = []
    _ = [up_floats.append(x[0][0]) for x in up_model.predict(koku)]

    dn_floats = []
    _ = [dn_floats.append(x[0][0]) for x in dn_model.predict(koku)]

    up = binarize(up_floats, up_genes)
    dn = binarize(dn_floats, dn_genes)


    r_dataframe = pd.concat((pd.DataFrame(np.sort(np.asarray(up)).reshape(-1,1),columns=["Up Regulated Genes"]), \
        pd.DataFrame(np.sort(np.asarray(dn)).reshape(-1,1),columns=["Down Regulated Genes"])),axis=1).fillna('', inplace=False)
    return r_dataframe