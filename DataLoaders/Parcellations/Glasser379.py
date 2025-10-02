# =====================================================================================
# Methods to input Glasser379 Parcellation data
#
# =====================================================================================
import warnings
import numpy as np
import pandas as pd

from DataLoaders.Parcellations.parcellation import Parcellation
from DataLoaders.Parcellations.atlas import Atlas

from DataLoaders.WorkBrainFolder import *
glasserParcellationFolder = WorkBrainDataFolder / "_Parcellations/Glasser360/"


class Glasser379(Parcellation):
    def get_coords(self):
        # ----------------- coordinates, but only for the 360 version...
        cog = np.loadtxt(glasserParcellationFolder + 'Glasser360_coords.txt')
        return cog

    def get_region_labels(self):
        # ----------------- node labels
        with open(glasserParcellationFolder + 'glasser379NodeNames.txt', 'r') as file:
            node_names = [line.strip() for line in file]
        # ----------------- node long labels
        columnames = ['id', 'reg name']
        df = pd.read_csv(glasserParcellationFolder + 'Anatomical-Labels.csv', names=columnames, header=None)
        nlist = df[columnames[1]].tolist()
        fullnames = nlist + nlist + node_names[360:378] + ['Brainstem']
        return fullnames

    def get_region_short_labels(self):
        # ----------------- node labels
        with open(glasserParcellationFolder + 'glasser379NodeNames.txt', 'r') as file:
            node_names = [line.strip() for line in file]
        return node_names

    def get_cortices(self):
        # ----------------- Cortices
        df = pd.read_csv(glasserParcellationFolder + 'HCP-MMP1_UniqueRegionList.csv')
        clist = df['cortex'][0:180].tolist()
        cortex = clist + clist + ['Subcortical'] * 18 + ['Brainstem']
        return cortex

    def get_lobes(self):
        # ----------------- Lobes
        df = pd.read_csv(glasserParcellationFolder + 'HCP-MMP1_UniqueRegionList.csv')
        clist = df['Lobe'][0:180].tolist()
        lobes = clist + clist + ['Subcortical'] * 18 + ['Brainstem']
        return lobes

    def get_RSN(self, useLR=False):
        raise NotImplemented('Unfinished implementation!')
        indicesFileParcellationRSN = f'../../Data_Produced/Parcellations/Glasser360RSN_{"14" if useLR else "7"}_indices.csv'

    def get_atlas_MNI(self):
        warnings.warn('Using Atlas from Glasser360 instead of Glasser379')
        return Atlas('Glasser360')

