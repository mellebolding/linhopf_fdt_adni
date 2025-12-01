import os
import numpy as np
import src.data_processing.filterps as filterps
from src.data_loaders.ADNI_A import loadBurden
import src.data_loaders.ADNI_B as ADNI_B


def load_group_data(DL, group_name, DL_type, SC=None):
    """Load all subjects from one group"""
    subject_ids = DL.get_groupSubjects(group_name)
    if DL_type == 'DL_A': 
        factor = 1
    else: 
        factor = 1
    group_data = []
    
    for subject_id in subject_ids:
        data = DL.get_subjectData(subject_id)
        group_data.append({
            'subject_id': subject_id,
            'group': group_name,
            'MRI': data[subject_id]['timeseries'].T*factor,
            'SC': data[subject_id]['SC'] if 'SC' in data[subject_id] else SC,
            'f_diff': filterps.calc_H_freq(
                data[subject_id]['timeseries'].T, 3000, 
                filterps.FiltPowSpetraVersion.v2021
            )[0],
            'ABeta': data[subject_id]['ABeta'],
            'Tau': data[subject_id]['Tau']
        })
    
    return group_data

def prepare_group_data(all_data, NPARCELLS):
    """Prepare averaged data for each group"""
    group_data = {}
    for group in set(d['group'] for d in all_data):
        group_subjects = [d for d in all_data if d['group'] == group]
        group_data[group] = {
            'n_subjects': len(group_subjects),
            'ts': np.array([d['MRI'][:,:NPARCELLS].T for d in group_subjects]),
            'SC': np.mean([d['SC'][:NPARCELLS, :NPARCELLS] for d in group_subjects], axis=0) if all(d['SC'] is not None for d in group_subjects) else None,
            'f_diff': np.mean([d['f_diff'][:NPARCELLS] for d in group_subjects], axis=0)
        }
    return group_data

def loadProteins(df, DL_type, protein, repo_root,filt=False):
    "protein must be 'Tau' or 'Amyloid'"
    protein = f'{protein}'
    if DL_type == 'DL_A':
        df[protein] = [loadBurden(subID, protein, repo_root + '/data/ADNI-A_DATA/', normalize=False) 
                    if subID != 'nan' else np.nan for subID in df['subject_id']]
        group_means = df.groupby('group')[protein].agg(lambda x: np.mean(np.stack(x.dropna().tolist()), axis=0))
        nan_mask = df[protein].isna()
        df.loc[nan_mask, protein] = df.loc[nan_mask, 'group'].map(group_means)
    if DL_type == 'DL_B1':
        if protein == 'Amyloid':
            protein = 'ABeta'
        DL = ADNI_B.ADNI_B_Alt(['HC', 'AD'])
        HC_IDs = DL.get_groupSubjects('HC')
        AD_IDs = DL.get_groupSubjects('AD')
        df[protein] = [DL.get_subjectData(subID)[subID][protein] if subID in HC_IDs + AD_IDs else np.nan for subID in df['subject_id']]
        group_means = df.groupby('group')[protein].agg(lambda x: np.mean(np.stack(x.dropna().tolist()), axis=0))
        nan_mask = df[protein].isna()
        df.loc[nan_mask, protein] = df.loc[nan_mask, 'group'].map(group_means)
    if DL_type == 'DL_B2':
        if protein == 'Amyloid':
            protein = 'ABeta'
        DL = ADNI_B.ADNI_B_Alt(['HC', 'MCI(AB-)', 'MCI(AB+)', 'AD'], filt=filt)
        HC_IDs = DL.get_groupSubjects('HC')
        MCI_AB_minus_IDs = DL.get_groupSubjects('MCI(AB-)')
        MCI_AB_plus_IDs = DL.get_groupSubjects('MCI(AB+)')
        AD_IDs = DL.get_groupSubjects('AD')
        df[protein] = [DL.get_subjectData(subID)[subID][protein] if subID in HC_IDs + MCI_AB_minus_IDs + MCI_AB_plus_IDs + AD_IDs else np.nan for subID in df['subject_id']]
        group_means = df.groupby('group')[protein].agg(lambda x: np.mean(np.stack(x.dropna().tolist()), axis=0))
        nan_mask = df[protein].isna()
        df.loc[nan_mask, protein] = df.loc[nan_mask, 'group'].map(group_means)
    return df