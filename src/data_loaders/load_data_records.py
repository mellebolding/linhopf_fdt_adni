import os
import numpy as np
import src.data_processing.filterps as filterps

def load_group_data(DL, group_name):
    """Load all subjects from one group"""
    subject_ids = DL.get_groupSubjects(group_name)
    group_data = []
    
    for subject_id in subject_ids:
        data = DL.get_subjectData(subject_id, printInfo=False)
        group_data.append({
            'subject_id': subject_id,
            'group': group_name,
            'MRI': data[subject_id]['timeseries'].T,
            'SC': data[subject_id]['SC'],
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
            'SC': np.mean([d['SC'][:NPARCELLS, :NPARCELLS] for d in group_subjects], axis=0),
            'f_diff': np.mean([d['f_diff'][:NPARCELLS] for d in group_subjects], axis=0)
        }
    return group_data