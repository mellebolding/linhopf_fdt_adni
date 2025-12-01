# -*- coding: utf-8 -*-
"""
Author: Kiret Dhindsa
Contact: kiretd@gmail.com or dhindsaj@charite.de
Last Content Modification: September 15, 2020
Last Update (clean-up for publication to Ebrains): October 2, 2025

Cleaned code to run nested cross-validation experiment from the published paper
(Triebkorn et al. 2022). 

If you use this code, please cite the paper and the Github Repository at
[LINK]

Triebkorn, Paul, et al. "Brain simulation augments machine‐learning–based classification 
of dementia." Alzheimer's & Dementia: Translational Research & Clinical Interventions 
8.1 (2022): e12303.

GITHUB REPO

"""


# %% Import Libraries
# basic dataset libraries
import os
import numpy as np
import pandas as pd
import scipy
import scipy.io as sio
import seaborn as sns

# machine learning libraries
from sklearn.feature_selection import SelectKBest, SelectFromModel, f_classif
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import metrics, svm, base
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, metrics, base
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Statistics
from scipy.stats import ttest_rel, shapiro, mannwhitneyu, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

# plotting libraries
import matplotlib.pyplot as plt

# util libraries
import time

np.random.seed(123)
USE_GI = False # Flag for using graph index features
USE_REG_MEANS = False # flag for using abeta and tau region-wise means vs. all regions

# %% Load Data
def get_data():
    '''
    Extract and organize data from file.
    '''
    filedata = sio.loadmat('Feature_matrices.mat')

    # Correct volumes for subject 4
    corrected_volumes = sio.loadmat('corrected_volumes.mat')
    filedata['Volumes'] = corrected_volumes['Volumes']
    
    sim_varnames = ['Bifurcations','Capacity33','Mean_real_freq_global',
                    'Mean_unreal_freq_global','High_d33','Low_d33','Freq_reg']
    
    # all but Freq_reg should just include the last column
    for varname in sim_varnames[:-1]:
        filedata[varname] = filedata[varname][:,2]

    emp_varnames = ['Tau_reg','tPET','PET','Abeta_reg','Volumes']
    
    FRA = filedata.pop('Freq_reg_allgc')
    MMSE = filedata.pop('MMSE')
    Volume_names = filedata.pop('Volume_names')
    
    Y = filedata.pop('Gnew').flatten()
    
    filedata.pop('__globals__')
    filedata.pop('__header__')
    filedata.pop('__version__')


    return (filedata, Y, sim_varnames, emp_varnames, FRA, MMSE, Volume_names)


from src.data_loaders.load_data_records import loadProteins
# ==================== DATA SETUP ====================
filename = f"FDT_results_DL_B1_N400_sigTrue_aTrue.npz"
repo_root = os.getcwd() 
save_path = os.path.join(repo_root, "data", "FDT_DATA")
fdt_data = np.load(os.path.join(save_path, filename), allow_pickle=True)
df = pd.DataFrame({k: fdt_data[k].tolist() for k in fdt_data.files})
df = loadProteins(df, 'DL_B1', 'Tau', repo_root)
df = loadProteins(df, 'DL_B1', 'ABeta', repo_root)

# Create feature matrices - each has shape (N_subjects, N_parcels)
Xdict = {
    'I_norm2': np.stack(df['I_norm2'].values),
    'X_norm2': np.stack(df['X_norm2'].values),
    'ABeta': np.stack(df['ABeta'].values),
    'Tau': np.stack(df['Tau'].values),
}

# Setup labels
group_map = {'HC': 1, 'AD': 2}
group_classes = ['HC', 'AD']
Y = df['group'].map(group_map).values

emp_varnames = ['ABeta', 'Tau']
sim_varnames = ['I_norm2', 'X_norm2']
good_sim = sim_varnames

Nclass = np.unique(Y).shape[0]
Nsubj = Y.shape[0]

# Correct vectors with dim==1
for key in Xdict.keys():
    if len(Xdict[key].shape) == 1:
        Xdict[key] = np.expand_dims(Xdict[key], axis=1)

Ystr = [group_classes[int(c)-1] for c in Y]

# Setup region/feature names - FIXED: proper parcel numbering
N_parcels = Xdict['I_norm2'].shape[1]
degrees = np.ones(N_parcels)
regions = [str(i) for i in range(N_parcels)]  # Just use parcel numbers: 0, 1, 2, ...
volnames = regions


# ==================== HELPER FUNCTIONS ====================
def extendFeatureNames(varnames, Xdict, regions):
    """
    Creates individual names for each feature column
    
    Example:
    - 'ABeta' with 400 parcels → ['ABeta_0', 'ABeta_1', ..., 'ABeta_399']
    - 'I_norm2' with 400 parcels → ['I_norm2_0', 'I_norm2_1', ..., 'I_norm2_399']
    """
    names = []
    for var in varnames:
        n = Xdict[var].shape[1]
        if n == 1:
            names.append(var)
        else:         
            names.extend([f'{var}_{regions[i]}' for i in range(n)])
    return names


def clean_features(X, featnames=None):
    """Remove features with zero variance or too many repeated values"""
    # _, mc = scipy.stats.mode(X, axis=0, keepdims=True)
    rmidx = np.where(np.std(X, axis=0) == 0)[0]
    # rmidx = np.append(rmidx, np.where(mc[0] > 10)[0], axis=0)
    # rmidx = np.unique(rmidx)
    
    X = np.delete(X, rmidx, axis=1)    
    if featnames is not None:
        trimmednames = [f for i, f in enumerate(featnames) if i not in rmidx]
        return X, trimmednames, rmidx
    else:
        return X, None, rmidx


def getFeatureMatrix(Xdict, varnames):
    """
    Constructs feature matrix from Xdict using variable names
    Returns: X (cleaned), feature_names, removed_indices
    """
    # Build feature matrix
    # X = np.concatenate([Xdict[features] for features in varnames], axis=1)
    feature_names = extendFeatureNames(varnames, Xdict, regions)
    for var in varnames:
        print(var, Xdict[var].shape)
    X = np.concatenate([Xdict[features] for features in varnames], axis=1)
    print("X combined shape:", X.shape)
    # Clean features and track which were removed
    # Xclean, cleaned_names, removed_idx = clean_features(X, feature_names)
    
    # if Xclean.shape[1] > 0:
    #     return Xclean, cleaned_names, removed_idx
    # else:
    return X, feature_names, []

def analyze_feature_importance(FI, FX, FN, feature_names_all, title=''):
    """
    Comprehensive feature importance analysis
    
    Returns:
    - Selection frequency for each feature
    - Mean importance for each feature
    - Top features by both metrics
    """
    n_features = len(feature_names_all)
    n_folds = len(FI)
    
    # Initialize counters
    selection_count = np.zeros(n_features)
    importance_sum = np.zeros(n_features)
    importance_count = np.zeros(n_features)
    
    # Aggregate across all folds
    for fold_idx in range(n_folds):
        fi = FI[fold_idx]
        fidx = FX[fold_idx]
        
        # Count selections
        selection_count[fidx] += 1
        
        # Sum importances (only for selected features)
        importance_sum[fidx] += fi[fidx]
        importance_count[fidx] += 1
    
    # Calculate metrics
    selection_freq = selection_count / n_folds * 100  # Percentage
    mean_importance = np.divide(importance_sum, importance_count, 
                                where=importance_count>0, 
                                out=np.zeros_like(importance_sum))
    
    # Create summary dataframe
    df_features = pd.DataFrame({
        'Feature': feature_names_all,
        'Selection_Frequency_%': selection_freq,
        'Mean_Importance': mean_importance,
        'Times_Selected': selection_count.astype(int)
    })
    
    # Sort by selection frequency
    df_features = df_features.sort_values('Selection_Frequency_%', ascending=False)
    
    print(f"\n{'='*60}")
    print(f"FEATURE IMPORTANCE ANALYSIS - {title}")
    print(f"{'='*60}")
    print(f"\nTop 20 features by selection frequency:")
    print(df_features.head(20).to_string(index=False))
    
    # Analyze by feature type
    print(f"\n{'='*60}")
    print("FEATURE TYPE BREAKDOWN:")
    print(f"{'='*60}")
    for var in set([name.split('_')[0] for name in feature_names_all]):
        var_features = df_features[df_features['Feature'].str.startswith(var)]
        n_selected = (var_features['Times_Selected'] > 0).sum()
        avg_freq = var_features['Selection_Frequency_%'].mean()
        print(f"{var:12s}: {n_selected:4d} parcels selected at least once (avg freq: {avg_freq:.1f}%)")
    
    return df_features


# ==================== MAIN CLASSIFICATION ====================
# ==================== IMPROVED PIPELINE FOR IMBALANCED DATA ====================
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import svm, metrics, base
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== IMPROVED MULTI-MODAL FEATURE SELECTION ====================

def balanced_multimodal_feature_selection(X, Y, feature_names_all, varnames, n_features_target=30):
    """
    Ensures balanced feature selection across multiple modalities.
    
    Instead of treating all features equally, this:
    1. Runs separate RF on each modality
    2. Selects proportional number of features from each
    3. Ensures combined features represent all data types
    
    Parameters:
    -----------
    X : array (n_samples, n_features)
    Y : array (n_samples,)
    feature_names_all : list of str
    varnames : list of str (e.g., ['ABeta', 'Tau', 'I_norm2', 'X_norm2'])
    n_features_target : int - total features to select
    
    Returns:
    --------
    selected_idx : array of indices
    selected_names : list of feature names
    importance_dict : dict with importance per modality
    """
    
    # Group features by modality
    modality_features = {}
    for var in varnames:
        modality_features[var] = [i for i, name in enumerate(feature_names_all) 
                                  if name.startswith(var + '_')]
    
    # Calculate how many features to select from each modality
    # Option 1: Proportional to number of features
    # total_features = sum(len(v) for v in modality_features.values())
    # n_per_modality = {var: max(5, int(n_features_target * len(idx) / total_features))
    #                   for var, idx in modality_features.items()}
    total_features = 40
    n_per_modality = {}
    n_per_modality['ABeta'] = 15 # Set a fixed, high number
    n_per_modality['Tau'] = 15 
    n_per_modality['I_norm2'] = 5 # Set a fixed, lower number
    n_per_modality['X_norm2'] = 5
    
    # Option 2: Equal representation (uncomment to use)
    # n_per_modality = {var: n_features_target // len(varnames) for var in varnames}
    
    print(f"\nBalanced feature selection plan:")
    for var, n in n_per_modality.items():
        print(f"  {var}: {n} features from {len(modality_features[var])} available")
    
    # Train separate RF for each modality and select top features
    selected_idx = []
    importance_dict = {}
    
    clf = RFC(n_estimators=200, max_depth=5, min_samples_leaf=4, 
              class_weight='balanced', random_state=999)
    
    for var in varnames:
        idx = modality_features[var]
        if len(idx) == 0:
            continue
            
        X_modality = X[:, idx]
        clf.fit(X_modality, Y)
        
        # Get feature importances
        fi = clf.feature_importances_
        importance_dict[var] = fi
        
        # Select top K features from this modality
        k = n_per_modality[var]
        top_k_local = np.argsort(fi)[::-1][:k]
        
        # Convert to global indices
        top_k_global = [idx[i] for i in top_k_local]
        selected_idx.extend(top_k_global)
        
        print(f"  {var}: Selected {len(top_k_global)} features, "
              f"importance range [{fi[top_k_local[-1]]:.4f}, {fi[top_k_local[0]]:.4f}]")
    
    selected_idx = np.array(selected_idx)
    selected_names = [feature_names_all[i] for i in selected_idx]
    
    return selected_idx, selected_names, importance_dict


def improved_nestedCV_multimodal(Xdict, Y, varnames, strategy='balanced', 
                                 selection_method='balanced'):
    """
    Improved nested CV with better multi-modal feature handling.
    
    Parameters:
    -----------
    selection_method : str
        'balanced' - Select proportionally from each modality
        'unified' - Use standard RF on all features (original method)
        'hybrid' - Combine both approaches
    """
    
    X, feature_names_all, _ = getFeatureMatrix(Xdict, varnames)
    
    print(f"\nFeature Selection Method: {selection_method}")
    print(f"Total features: {X.shape[1]}")
    
    # Identify modalities
    n_modalities = len(varnames)
    print(f"Modalities: {varnames}")
    
    unique, counts = np.unique(Y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")
    
    # === OUTPUTS ===
    SC, FI, FX, FN = [], [], [], []
    CM, YH, YT, TI = [], [], [], []
    metrics_per_fold = []
    modality_selection_counts = {var: [] for var in varnames}
    
    # === SETUP ===
    scoring = 'balanced_accuracy'
    ssc = RobustScaler()
    
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
    cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=888)
    
    clf_inner = RFC(random_state=999, class_weight='balanced')
    clf_outer = svm.SVC(random_state=999, class_weight='balanced', probability=True)
    
    use_smote = (strategy == 'smote')
    if use_smote:
        smote = SMOTE(random_state=999, k_neighbors=3)
    
    fold_num = 0
    for train_outer, test_outer in cv_outer.split(X, Y):
        fold_num += 1
        
        Xtrain, Xtest = X[train_outer], X[test_outer]
        print("X_train: ", Xtrain)
        Ytrain, Ytest = Y[train_outer], Y[test_outer]
        unique, counts = np.unique(Ytrain, return_counts=True)
        n_minority = np.min(counts)
        
        print(f"\n{'='*60}")
        print(f"Fold {fold_num}/10 - Train: {np.bincount(Ytrain)}, Test: {np.bincount(Ytest)}")
        
        # ===== INNER CV: Feature selection =====
        pipeline_inner = Pipeline([
            ('Scaler', ssc),
            ('Classifier', clf_inner)
        ])
        
        inner_params = [{
            'Classifier__n_estimators': [200],
            'Classifier__max_depth': [3, 5, None],
            'Classifier__min_samples_leaf': [2, 4],
            'Classifier__max_features': ['sqrt'],
        }]
        
        grid_inner = GridSearchCV(
            pipeline_inner, inner_params, cv=cv_inner,
            scoring=scoring, n_jobs=-1, verbose=0
        )
        
        grid_inner.fit(Xtrain, Ytrain)
        
        # Extract best scaler
        for obj in grid_inner.best_estimator_.steps:
            if obj[0] == 'Scaler':
                ssc_inner_best = base.clone(obj[1])
        
        # Scale the training data for feature selection
        Xtrain_scaled = ssc_inner_best.fit_transform(Xtrain)
        
        # === FEATURE SELECTION STRATEGY ===
        #n_minority = np.min(np.bincount(Ytrain))
        
        if selection_method == 'balanced':
            # BALANCED: Select proportionally from each modality
            if n_modalities > 1:
                n_total = min(50, n_minority * 2)  # Increase for combined
                fidx, selected_names, importance_dict = balanced_multimodal_feature_selection(
                    Xtrain_scaled, Ytrain, feature_names_all, varnames, n_total
                )
                # Create combined importance vector for consistency
                fi = np.zeros(len(feature_names_all))
                for var, importance in importance_dict.items():
                    var_idx = [i for i, name in enumerate(feature_names_all) 
                              if name.startswith(var + '_')]
                    fi[var_idx] = importance
            else:
                # Single modality - use standard method
                n_total = min(30, n_minority * 2)
                clf_temp = RFC(n_estimators=200, max_depth=5, class_weight='balanced', 
                              random_state=999)
                clf_temp.fit(Xtrain_scaled, Ytrain)
                fi = clf_temp.feature_importances_
                fidx = np.argsort(fi)[::-1][:n_total]
                selected_names = [feature_names_all[i] for i in fidx]
                importance_dict = {varnames[0]: fi}
        
        elif selection_method == 'unified':
            # UNIFIED: Standard RF on all features (original method)
            n_total = min(30 if n_modalities == 1 else 50, n_minority * 2)
            clf_temp = RFC(n_estimators=600, max_depth=10, class_weight='balanced', 
                          random_state=999)
            clf_temp.fit(Xtrain_scaled, Ytrain)
            fi = clf_temp.feature_importances_
            fidx = np.argsort(fi)[::-1][:n_total]
            selected_names = [feature_names_all[i] for i in fidx]
            importance_dict = {'all': fi}
        
        elif selection_method == 'hybrid':
            # HYBRID: Select from both balanced and unified approaches
            n_balanced = min(30, n_minority)
            n_unified = min(20, n_minority)
            
            # Get balanced selection
            fidx_bal, _, importance_dict = balanced_multimodal_feature_selection(
                Xtrain_scaled, Ytrain, feature_names_all, varnames, n_balanced
            )
            
            # Get unified selection
            clf_temp = RFC(n_estimators=200, max_depth=5, class_weight='balanced', 
                          random_state=999)
            clf_temp.fit(Xtrain_scaled, Ytrain)
            fi = clf_temp.feature_importances_
            fidx_uni = np.argsort(fi)[::-1][:n_unified]
            
            # Combine (remove duplicates)
            fidx = np.unique(np.concatenate([fidx_bal, fidx_uni]))
            selected_names = [feature_names_all[i] for i in fidx]
            importance_dict['unified'] = fi
        
        else:
            raise ValueError(f"Unknown selection_method: {selection_method}")
        
        # Track modality representation
        for var in varnames:
            n_selected = sum(1 for name in selected_names if name.startswith(var + '_'))
            modality_selection_counts[var].append(n_selected)
        
        FI.append(fi)
        FX.append(fidx)
        FN.append(selected_names)
        
        print(f"Selected {len(fidx)} features:")
        for var in varnames:
            n = sum(1 for name in selected_names if name.startswith(var + '_'))
            pct = 100 * n / len(fidx) if len(fidx) > 0 else 0
            print(f"  {var}: {n} ({pct:.1f}%)")
        
        # ===== OUTER CV: Classification =====
        if use_smote:
            pipeline_outer = ImbPipeline([
                ('Scaler', ssc_inner_best),
                ('SMOTE', smote),
                ('Classifier', clf_outer)
            ])
        else:
            pipeline_outer = Pipeline([
                ('Scaler', ssc_inner_best),
                ('Classifier', clf_outer)
            ])
        
        outer_params = [{
            'Classifier__kernel': ['rbf'],
            'Classifier__C': [0.1, 1, 10],
            'Classifier__gamma': ['scale', 0.01, 0.1]
        }]
        
        outer_split = -np.ones((len(X),))
        outer_split[test_outer] = 0
        
        grid_outer = GridSearchCV(
            pipeline_outer, outer_params,
            cv=PredefinedSplit(outer_split),
            scoring=scoring, n_jobs=-1, verbose=0
        )
        
        grid_outer.fit(X[:, fidx], Y)
        SC.append(grid_outer.best_score_)
        
        # Extract best model
        for obj in grid_outer.best_estimator_.steps:
            if obj[0] == 'Scaler':
                ssc_outer_best = base.clone(obj[1])
            elif obj[0] == 'Classifier':
                clf_outer_best = base.clone(obj[1])
        
        # Final prediction
        if use_smote:
            pipeline_opt = ImbPipeline([
                ('Scaler', ssc_outer_best),
                ('SMOTE', smote),
                ('Classifier', clf_outer_best)
            ])
        else:
            pipeline_opt = Pipeline([
                ('Scaler', ssc_outer_best),
                ('Classifier', clf_outer_best)
            ])
        
        pipeline_opt.fit(Xtrain[:, fidx], Ytrain)
        yhat = pipeline_opt.predict(Xtest[:, fidx])
        yprob = pipeline_opt.predict_proba(Xtest[:, fidx])[:, 1]
        
        # Metrics
        cm = metrics.confusion_matrix(Ytest, yhat, labels=[1, 2])
        tn, fp, fn, tp = cm.ravel()
        
        fold_metrics = {
            'balanced_accuracy': metrics.balanced_accuracy_score(Ytest, yhat),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'f1_ad': metrics.f1_score(Ytest, yhat, pos_label=2, zero_division=0),
            'f1_hc': metrics.f1_score(Ytest, yhat, pos_label=1, zero_division=0),
            'auc': metrics.roc_auc_score((Ytest == 2).astype(int), yprob)
        }
        
        metrics_per_fold.append(fold_metrics)
        CM.append(cm)
        YH.append(yhat)
        YT.append(Ytest)
        TI.append(test_outer)
        
        print(f"BA: {fold_metrics['balanced_accuracy']:.3f}, "
              f"Sens: {fold_metrics['sensitivity']:.3f}, "
              f"Spec: {fold_metrics['specificity']:.3f}")
    
    # Print summary of modality usage
    print(f"\n{'='*60}")
    print("MODALITY USAGE SUMMARY (across all folds)")
    print(f"{'='*60}")
    for var in varnames:
        counts = modality_selection_counts[var]
        print(f"{var}:")
        print(f"  Mean: {np.mean(counts):.1f} features selected per fold")
        print(f"  Range: [{np.min(counts)}, {np.max(counts)}]")
    
    return SC, FI, FX, FN, CM, YH, YT, TI, feature_names_all, metrics_per_fold, modality_selection_counts


def plot_modality_usage(modality_counts, varnames, title=''):
    """
    Visualize how features from each modality are used across folds.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Bar plot of average usage
    means = [np.mean(modality_counts[var]) for var in varnames]
    stds = [np.std(modality_counts[var]) for var in varnames]
    
    axes[0].bar(varnames, means, yerr=stds, capsize=5, alpha=0.7)
    axes[0].set_ylabel('Number of Features Selected')
    axes[0].set_title('Average Feature Selection per Modality')
    axes[0].grid(axis='y', alpha=0.3)
    
    # 2. Box plot across folds
    data_to_plot = [modality_counts[var] for var in varnames]
    axes[1].boxplot(data_to_plot, labels=varnames)
    axes[1].set_ylabel('Number of Features Selected')
    axes[1].set_title('Feature Selection Distribution Across Folds')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Modality Usage - {title}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'Modality_Usage_{title}.png', dpi=300, bbox_inches='tight')
    plt.show()


def compare_selection_methods(Xdict, Y, varnames):
    """
    Compare different feature selection strategies.
    """
    methods = ['unified', 'balanced', 'hybrid']
    results = {}
    
    print(f"\n{'='*70}")
    print(f"COMPARING FEATURE SELECTION METHODS FOR: {varnames}")
    print(f"{'='*70}")
    
    for method in methods:
        print(f"\n{'='*70}")
        print(f"METHOD: {method.upper()}")
        print(f"{'='*70}")
        
        SC, FI, FX, FN, CM, YH, YT, TI, feat_names, metrics, mod_counts = \
            improved_nestedCV_multimodal(Xdict, Y, varnames, selection_method=method)
        
        results[method] = {
            'SC': SC, 'FI': FI, 'FX': FX, 'FN': FN,
            'CM': CM, 'metrics': metrics, 'feat_names': feat_names,
            'modality_counts': mod_counts
        }
        
        # Print performance
        ba = np.mean([m['balanced_accuracy'] for m in metrics])
        sens = np.mean([m['sensitivity'] for m in metrics])
        spec = np.mean([m['specificity'] for m in metrics])
        
        print(f"\n{method.upper()} Results:")
        print(f"  Balanced Accuracy: {ba:.3f} ± {np.std([m['balanced_accuracy'] for m in metrics]):.3f}")
        print(f"  Sensitivity: {sens:.3f} ± {np.std([m['sensitivity'] for m in metrics]):.3f}")
        print(f"  Specificity: {spec:.3f} ± {np.std([m['specificity'] for m in metrics]):.3f}")
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_to_plot = ['balanced_accuracy', 'sensitivity', 'specificity']
    titles = ['Balanced Accuracy', 'Sensitivity (AD)', 'Specificity (HC)']
    
    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        data = []
        for method in methods:
            values = [m[metric] for m in results[method]['metrics']]
            data.append(values)
        
        bp = axes[idx].boxplot(data, labels=methods, patch_artist=True)
        axes[idx].set_ylabel('Score')
        axes[idx].set_title(title)
        axes[idx].set_ylim([0, 1])
        axes[idx].axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Color boxes
        colors = ['lightblue', 'lightgreen', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
    
    plt.suptitle('Feature Selection Method Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('Method_Comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


# ==================== HELPER: ANALYZE FEATURE SYNERGY ====================

def analyze_feature_synergy(FN_emp, FN_sim, FN_com):
    """
    Analyze how empirical and simulated features work together.
    
    This shows:
    1. Which features are selected when alone vs combined
    2. Whether combined model uses complementary information
    """
    
    print(f"\n{'='*60}")
    print("FEATURE SYNERGY ANALYSIS")
    print(f"{'='*60}")
    
    # Flatten all feature selections across folds
    emp_features = set()
    for fold_features in FN_emp:
        emp_features.update(fold_features)
    
    sim_features = set()
    for fold_features in FN_sim:
        sim_features.update(fold_features)
    
    com_features = set()
    for fold_features in FN_com:
        com_features.update(fold_features)
    
    print(f"\nUnique features selected (across all folds):")
    print(f"  Empirical alone: {len(emp_features)}")
    print(f"  Simulated alone: {len(sim_features)}")
    print(f"  Combined: {len(com_features)}")
    
    # Check overlap
    com_emp = [f for f in com_features if any(f.startswith(p) for p in ['ABeta', 'Tau'])]
    com_sim = [f for f in com_features if any(f.startswith(p) for p in ['I_norm2', 'X_norm2'])]
    
    print(f"\nIn combined model:")
    print(f"  Empirical features: {len(com_emp)} ({100*len(com_emp)/len(com_features):.1f}%)")
    print(f"  Simulated features: {len(com_sim)} ({100*len(com_sim)/len(com_features):.1f}%)")
    
    # Check if combined uses new features
    new_features = com_features - emp_features - sim_features
    print(f"\nNovel features (not in either individual model): {len(new_features)}")
    
    return {
        'emp_only': emp_features,
        'sim_only': sim_features,
        'combined': com_features,
        'novel': new_features
    }



# ===== OPTION 3: Statistical Comparison =====
def compare_performance_stats(metrics1, metrics2, metrics3, labels=['Emp', 'Sim', 'Com']):
    """Statistical comparison of classification performance"""
    from scipy.stats import friedmanchisquare, wilcoxon
    
    print(f"\n{'='*60}")
    print("STATISTICAL PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    # Extract balanced accuracy from each
    ba1 = [m['balanced_accuracy'] for m in metrics1]
    ba2 = [m['balanced_accuracy'] for m in metrics2]
    ba3 = [m['balanced_accuracy'] for m in metrics3]
    
    print(f"\nBalanced Accuracy:")
    print(f"  {labels[0]}: {np.mean(ba1):.3f} ± {np.std(ba1):.3f}")
    print(f"  {labels[1]}: {np.mean(ba2):.3f} ± {np.std(ba2):.3f}")
    print(f"  {labels[2]}: {np.mean(ba3):.3f} ± {np.std(ba3):.3f}")
    
    # Friedman test (non-parametric repeated measures)
    stat, p = friedmanchisquare(ba1, ba2, ba3)
    print(f"\nFriedman test: χ²={stat:.3f}, p={p:.4f}")
    
    if p < 0.05:
        print("Significant difference detected. Post-hoc pairwise tests:")
        
        # Pairwise Wilcoxon tests
        pairs = [(ba1, ba2, f"{labels[0]} vs {labels[1]}"),
                 (ba1, ba3, f"{labels[0]} vs {labels[2]}"),
                 (ba2, ba3, f"{labels[1]} vs {labels[2]}")]
        
        for b1, b2, name in pairs:
            stat, p = wilcoxon(b1, b2)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {name}: W={stat:.1f}, p={p:.4f} {sig}")
    
    # Effect size (for Combined vs others)
    from scipy.stats import mannwhitneyu
    
    print(f"\nEffect sizes (Mann-Whitney U):")
    u1, _ = mannwhitneyu(ba3, ba1, alternative='greater')
    u2, _ = mannwhitneyu(ba3, ba2, alternative='greater')
    
    # Convert to rank-biserial correlation (effect size)
    n1, n2 = len(ba3), len(ba1)
    r1 = 1 - (2*u1) / (n1 * n2)
    r2 = 1 - (2*u2) / (n1 * n2)
    
    print(f"  {labels[2]} vs {labels[0]}: r={r1:.3f} ({'small' if abs(r1)<0.3 else 'medium' if abs(r1)<0.5 else 'large'})")
    print(f"  {labels[2]} vs {labels[1]}: r={r2:.3f} ({'small' if abs(r2)<0.3 else 'medium' if abs(r2)<0.5 else 'large'})")
    
    return {'ba1': ba1, 'ba2': ba2, 'ba3': ba3}




# ===== OPTION 4: Visualize Feature Selection Patterns =====
def visualize_feature_selection_patterns(FX_list, feature_names_all, varnames, title=''):
    """
    Heatmap showing which features are selected across folds.
    """
    n_folds = len(FX_list)
    n_features = len(feature_names_all)
    
    # Create binary matrix: 1 if feature selected in fold, 0 otherwise
    selection_matrix = np.zeros((n_folds, n_features))
    for fold_idx, fidx in enumerate(FX_list):
        selection_matrix[fold_idx, fidx] = 1
    
    # Sort features by total selection frequency
    selection_freq = selection_matrix.sum(axis=0)
    sorted_idx = np.argsort(selection_freq)[::-1][:100]  # Top 100
    
    # Create feature type labels
    feature_types = []
    for idx in sorted_idx:
        name = feature_names_all[idx]
        for var in varnames:
            if name.startswith(var + '_'):
                feature_types.append(var)
                break
    
    # Plot
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # Plot heatmap
    im = ax.imshow(selection_matrix[:, sorted_idx].T, aspect='auto', cmap='YlOrRd')
    
    ax.set_xlabel('Fold Number', fontsize=12)
    ax.set_ylabel('Feature Index (sorted by frequency)', fontsize=12)
    ax.set_title(f'Feature Selection Patterns - {title}', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Selected (1) / Not Selected (0)', fontsize=10)
    
    # Add feature type color bar on right
    from matplotlib.patches import Rectangle
    colors = {'ABeta': 'coral', 'Tau': 'salmon', 
              'I_norm2': 'steelblue', 'X_norm2': 'lightblue'}
    
    for i, ftype in enumerate(feature_types):
        rect = Rectangle((n_folds + 0.5, i - 0.5), 1, 1, 
                        facecolor=colors.get(ftype, 'gray'))
        ax.add_patch(rect)
    
    ax.set_xlim(-0.5, n_folds + 1.5)
    
    plt.tight_layout()
    plt.savefig(f'Feature_Selection_Pattern_{title}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Visualize selection patterns



# ===== FINAL SUMMARY TABLE =====
def create_summary_table(metrics_emp, metrics_sim, metrics_com):
    """Create publication-ready summary table"""
    
    metrics_names = ['balanced_accuracy', 'sensitivity', 'specificity', 
                     'f1_ad', 'f1_hc', 'auc']
    
    summary_data = []
    for label, metrics in [('Empirical', metrics_emp), 
                           ('Simulated', metrics_sim), 
                           ('Combined', metrics_com)]:
        row = {'Model': label}
        for metric in metrics_names:
            values = [m[metric] for m in metrics]
            row[metric] = f"{np.mean(values):.3f} ± {np.std(values):.3f}"
        summary_data.append(row)
    
    df_summary = pd.DataFrame(summary_data)
    
    # Rename columns for readability
    df_summary.columns = ['Model', 'Balanced Acc', 'Sensitivity', 
                         'Specificity', 'F1 (AD)', 'F1 (HC)', 'AUC']
    
    print(f"\n{'='*80}")
    print("FINAL SUMMARY TABLE")
    print(f"{'='*80}")
    print(df_summary.to_string(index=False))
    
    # Save to Excel
    df_summary.to_excel('Final_Summary_Table.xlsx', index=False)
    print("\nSaved to: Final_Summary_Table.xlsx")
    
    return df_summary



def plot_comprehensive_results(metrics_per_fold, CM, title=''):
    """
    Create comprehensive visualization of results
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Confusion Matrix (aggregate)
    cm_total = np.sum(CM, axis=0)
    cm_norm = cm_total.astype('float') / cm_total.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['HC', 'AD'], yticklabels=['HC', 'AD'],
                ax=axes[0, 0])
    axes[0, 0].set_title('Normalized Confusion Matrix')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # 2. Balanced Accuracy Distribution
    ba_scores = [m['balanced_accuracy'] for m in metrics_per_fold]
    axes[0, 1].hist(ba_scores, bins=10, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(np.mean(ba_scores), color='red', linestyle='--',
                       label=f'Mean: {np.mean(ba_scores):.3f}')
    axes[0, 1].set_xlabel('Balanced Accuracy')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Balanced Accuracy Distribution')
    axes[0, 1].legend()
    
    # 3. Sensitivity vs Specificity
    sens = [m['sensitivity'] for m in metrics_per_fold]
    spec = [m['specificity'] for m in metrics_per_fold]
    axes[0, 2].scatter(spec, sens, alpha=0.6)
    axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0, 2].set_xlabel('Specificity (HC Recall)')
    axes[0, 2].set_ylabel('Sensitivity (AD Recall)')
    axes[0, 2].set_title('Sensitivity vs Specificity')
    axes[0, 2].set_xlim([0, 1])
    axes[0, 2].set_ylim([0, 1])
    
    # 4. Per-class F1 scores
    f1_ad = [m['f1_ad'] for m in metrics_per_fold]
    f1_hc = [m['f1_hc'] for m in metrics_per_fold]
    x = ['AD (minority)', 'HC (majority)']
    means = [np.mean(f1_ad), np.mean(f1_hc)]
    stds = [np.std(f1_ad), np.std(f1_hc)]
    axes[1, 0].bar(x, means, yerr=stds, capsize=5)
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Per-Class F1 Scores')
    axes[1, 0].set_ylim([0, 1])
    
    # 5. Metrics comparison
    metric_names = ['Balanced\nAccuracy', 'Sensitivity\n(AD)', 'Specificity\n(HC)', 'PPV', 'NPV']
    metric_keys = ['balanced_accuracy', 'sensitivity', 'specificity', 'ppv', 'npv']
    means = [np.mean([m[k] for m in metrics_per_fold]) for k in metric_keys]
    stds = [np.std([m[k] for m in metrics_per_fold]) for k in metric_keys]
    
    axes[1, 1].bar(range(len(means)), means, yerr=stds, capsize=5)
    axes[1, 1].set_xticks(range(len(means)))
    axes[1, 1].set_xticklabels(metric_names, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('All Metrics Comparison')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Chance')
    
    # 6. ROC curve (if available)
    if metrics_per_fold[0].get('auc') is not None:
        auc_scores = [m['auc'] for m in metrics_per_fold if m['auc'] is not None]
        axes[1, 2].hist(auc_scores, bins=10, edgecolor='black', alpha=0.7)
        axes[1, 2].axvline(np.mean(auc_scores), color='red', linestyle='--',
                          label=f'Mean AUC: {np.mean(auc_scores):.3f}')
        axes[1, 2].set_xlabel('AUC')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('AUC Distribution')
        axes[1, 2].legend()
    else:
        axes[1, 2].text(0.5, 0.5, 'AUC not available', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].axis('off')
    
    plt.suptitle(f'Classification Results - {title}', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig(f'Comprehensive_Results_{title}.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_detailed_report(FI, FX, FN, feature_names_all, metrics_per_fold, title=''):
    """
    Create Excel report with all results
    """
    # Feature importance
    df_features = analyze_feature_importance(FI, FX, FN, feature_names_all, title)
    
    # Metrics summary
    metrics_summary = {}
    for key in metrics_per_fold[0].keys():
        if key != 'auc' or metrics_per_fold[0]['auc'] is not None:
            values = [m[key] for m in metrics_per_fold if m[key] is not None]
            metrics_summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
    
    df_metrics = pd.DataFrame(metrics_summary).T
    
    # Save to Excel with multiple sheets
    with pd.ExcelWriter(f'Detailed_Report_{title}.xlsx') as writer:
        df_features.to_excel(writer, sheet_name='Feature_Importance', index=False)
        df_metrics.to_excel(writer, sheet_name='Performance_Metrics')
        
        # Per-fold detailed results
        df_fold_metrics = pd.DataFrame(metrics_per_fold)
        df_fold_metrics.to_excel(writer, sheet_name='Per_Fold_Metrics', index=False)
    
    print(f"\nDetailed report saved to: Detailed_Report_{title}.xlsx")
    return df_features, df_metrics

# ==================== RUN CLASSIFICATION ====================
def inspect_feature_set(varnames, label):
    X, featnames, removed_idx = getFeatureMatrix(Xdict, varnames)
    print(f"\n--- {label} ---")
    print("varnames:", varnames)
    print("X shape:", X.shape)
    print("n removed features:", len(removed_idx))
    print("first 20 feature names:", featnames[:20])
    # count features by prefix (ABeta/Tau/I_norm2/X_norm2)
    from collections import Counter
    prefixes = [name.split('_')[0] for name in featnames]
    print("feature counts by prefix:", Counter(prefixes))
    # show some stats per prefix
    for pref in sorted(set(prefixes)):
        ix = [i for i,n in enumerate(featnames) if n.startswith(pref)]
        vals = X[:, ix]
        print(f" {pref}: {len(ix)} cols, mean(abs) per col (first 5):", np.mean(np.abs(vals), axis=0)[:5])

print("\n" + "="*70)
print("RUNNING WITH BALANCED FEATURE SELECTION")
print("="*70)

# Empirical features only
sf1, FI1, FX1, FN1, CM1, YH1, YT1, TI1, feat_names1, metrics1, mod1 = \
    improved_nestedCV_multimodal(Xdict, Y, emp_varnames, selection_method='unified')

# Simulated features only  
sf2, FI2, FX2, FN2, CM2, YH2, YT2, TI2, feat_names2, metrics2, mod2 = \
    improved_nestedCV_multimodal(Xdict, Y, sim_varnames, selection_method='unified')

# Combined with BALANCED selection (ensures both types represented)
sf3, FI3, FX3, FN3, CM3, YH3, YT3, TI3, feat_names3, metrics3, mod3 = \
    improved_nestedCV_multimodal(Xdict, Y, emp_varnames + sim_varnames, 
                                 selection_method='balanced')  # <-- KEY CHANGE
visualize_feature_selection_patterns(FX3, feat_names3, emp_varnames + sim_varnames, 
                                    title='Combined')
summary = create_summary_table(metrics1, metrics2, metrics3)
# Visualize results
# plot_comprehensive_results(metrics1, CM1, title='Empirical')
# plot_comprehensive_results(metrics2, CM2, title='Simulated')
# plot_comprehensive_results(metrics3, CM3, title='Combined_Balanced')

# Show modality usage in combined model
plot_modality_usage(mod3, emp_varnames + sim_varnames, title='Combined_Balanced')

# # Create reports
# create_detailed_report(FI1, FX1, FN1, feat_names1, metrics1, title='Empirical')
# create_detailed_report(FI2, FX2, FN2, feat_names2, metrics2, title='Simulated')
# create_detailed_report(FI3, FX3, FN3, feat_names3, metrics3, title='Combined_Balanced')

# Analyze synergy
synergy = analyze_feature_synergy(FN1, FN2, FN3)
# Run statistical comparison



# ===== OPTION 2: Comprehensive Comparison (Run All Methods) =====
print("\n" + "="*70)
print("COMPREHENSIVE METHOD COMPARISON")
print("="*70)

results = compare_selection_methods(Xdict, Y, emp_varnames + sim_varnames)

# Access results for each method
unified_results = results['unified']
balanced_results = results['balanced']
hybrid_results = results['hybrid']

# Create detailed reports for each
for method in ['unified', 'balanced', 'hybrid']:
    r = results[method]
    create_detailed_report(
        r['FI'], r['FX'], r['FN'], r['feat_names'], r['metrics'], 
        title=f'Combined_{method.capitalize()}'
    )
    plot_modality_usage(
        r['modality_counts'], emp_varnames + sim_varnames, 
        title=f'Combined_{method.capitalize()}'
    )
stats = compare_performance_stats(metrics1, metrics2, metrics3, 
                                  labels=['Empirical', 'Simulated', 'Combined'])

inspect_feature_set(emp_varnames, 'EMPIRICAL')
inspect_feature_set(sim_varnames, 'SIMULATED')
inspect_feature_set(sim_varnames+emp_varnames, 'COMBINED')
sf1, FI1, FX1, FN1, CM1, YH1, YT1, TI1, feat_names1, metrics1 = improved_nestedCV(Xdict, Y, emp_varnames)
sf2, FI2, FX2, FN2, CM2, YH2, YT2, TI2, feat_names2, metrics2 = improved_nestedCV(Xdict, Y, sim_varnames)
sf3, FI3, FX3, FN3, CM3, YH3, YT3, TI3, feat_names3, metrics3 = improved_nestedCV(Xdict, Y, emp_varnames + sim_varnames)
print(metrics1[0].keys())
plot_comprehensive_results(metrics1, CM1, title='Empirical')
plot_comprehensive_results(metrics2, CM2, title='Simulated')
plot_comprehensive_results(metrics3, CM3, title='Combined')

create_detailed_report(FI1, FX1, FN1, feat_names1, metrics1, title='Empirical')
create_detailed_report(FI2, FX2, FN2, feat_names2, metrics2, title='Simulated')
create_detailed_report(FI3, FX3, FN3, feat_names3, metrics3, title='Combined')

# Analyze and plot results
# print("\n" + "="*60)
# print("GENERATING VISUALIZATIONS AND REPORTS")
# print("="*60)

# df_emp = plot_FeatureFrequency_Top(FI1, FX1, FN1, feat_names1, title='Empirical')
# df_sim = plot_FeatureFrequency_Top(FI2, FX2, FN2, feat_names2, title='Simulated')
# df_com = plot_FeatureFrequency_Top(FI3, FX3, FN3, feat_names3, title='Combined')

# # Print performance summary
# print("\n" + "="*60)
# print("PERFORMANCE SUMMARY")
# print("="*60)
# print(f"Empirical:  F1 = {np.mean(sf1):.3f} ± {np.std(sf1):.3f}")
# print(f"Simulated:  F1 = {np.mean(sf2):.3f} ± {np.std(sf2):.3f}")
# print(f"Combined:   F1 = {np.mean(sf3):.3f} ± {np.std(sf3):.3f}")


# %% Classification Functions
# def clean_features(X, featnames=None):
#     '''
#     Removes deficient columns from feature matrices
#     '''
#     rmidx = np.where(np.std(X, axis=0) == 0)[0]
    
#     X = np.delete(X, rmidx, axis=1)    
#     if featnames is not None:
#         trimmednames = [f for i,f in enumerate(featnames) if i not in rmidx]
#         return X, trimmednames, rmidx
#     else:
#         return X, None, rmidx

    
# def getFeatureMatrix(Xdict, varnames):
#     '''
#     Constructs a feature matrix from Xdict using the variable names in varnames
#     '''
#     X = np.concatenate([Xdict[features] for features in varnames], axis=1) 
#     Xclean = clean_features(X)
    
    
#     if Xclean.shape[1] > 0:
#         X = Xclean
#     if X.shape[1] == 0:
#         scores = [np.nan, np.nan]
#         est = None
#         return scores, est
#     return Xclean

def nestedCV(Xdict, Y, varnames):
    '''
    Performs nested cross-validation for simultaneous model selection and 
    feature selection using SVM with Random Forest for feature selection.
    '''
    # Get the faeture matrix according to inputted variable names
    if USE_REG_MEANS:
        X = getFeatureMatrix(Xdict, varnames, opt='abeta_tau_reg_means')
    else:
        X = getFeatureMatrix(Xdict, varnames)

    idx = np.arange(0, X.shape[0], 1)
    
    # Define outputs
    SC = [] # Outer classifier scores
    FI = [] # Feature Importances
    FX = [] # Index of top features
    CM = [] # Test set confusion matrices
    YH = [] # Predicted Class Labels
    YT = [] # True Class Labels
    TI = [] # Test index
    
    # Set up components of the ML pipeline
    scoring = 'f1_weighted' #'accuracy'
    
    ssc = RobustScaler()
    clf_inner = RFC(random_state=999)
    clf_outer = svm.SVC(random_state=999)
    
    cv_inner = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=777)
    cv_outer = StratifiedShuffleSplit(n_splits=100, test_size=0.25, random_state=888)
    
    for train_outer, test_outer in cv_outer.split(X, Y):
        Xtrain, Xtest = X[train_outer], X[test_outer]
        Ytrain, Ytest = Y[train_outer], Y[test_outer]
        
        # Set up Inner CV loop using a grid search CV
        pipeline_inner = Pipeline(memory=None,
                                  steps=[('Scaler', ssc),
                                         ('Classifier', clf_inner)])
        
        inner_params = [{'Classifier__class_weight': ['balanced'],
                         'Classifier__criterion': ['entropy'], # gini or entropy
                         'Classifier__n_estimators': [10, 50],
                         'Classifier__max_depth': [None],
                         'Classifier__min_samples_split': [3, 4],
                         'Classifier__min_samples_leaf': [2, 3],
                         'Classifier__max_features': ['sqrt'],#,'log2',None,
                         }]        
    
        # Run Inner CV loop using a grid search CV
        grid_inner = GridSearchCV(pipeline_inner, inner_params, cv=cv_inner, 
                                  scoring=scoring, verbose=False, n_jobs=-1)
        grid_inner.fit(Xtrain, Ytrain)
    
        # Get best model and its feature importances (GINI index) - try entropy instead
        for obj in grid_inner.best_estimator_.steps:
#            print(obj[1])
            if obj[0] is 'Scaler':
                ssc_inner_best = base.clone(obj[1], safe=True)
            elif obj[0] is 'Feature_sel':
                sel_inner_best = base.clone(obj[1], safe=True)
            elif obj[0] is 'Classifier':
                clf_inner_best = base.clone(obj[1], safe=True)
                fi = obj[1].feature_importances_

        # Get top features
        nfeat = np.sum(fi>0)
        fidx = np.argsort(fi)[:-nfeat-1:-1]
        FI.append(fi)
        FX.append(fidx)
        
        # Set up outer CV lop
        pipeline_outer = Pipeline(memory=None,
                                  steps=[('Scaler', ssc_inner_best),
                                         ('Classifier', clf_outer)])
        
        outer_params = [{'Classifier__kernel': ['rbf'],
                         'Classifier__degree': [2],
                         'Classifier__gamma': [1e-2, 1e-1, 1],
                         'Classifier__C': [0.01, 0.1, 1, 10, 100]},
                        {'Classifier__kernel':['poly'],
                         'Classifier__degree': [2,3],
                         'Classifier__gamma': ['scale'],
                         'Classifier__C': [0.01, 0.1, 1, 10, 100]}]
        
        # set test set manually
        outer_split = -np.ones((len(idx),))
        outer_split[test_outer] = 0
        
        # Run outer CV loop
        grid_outer = GridSearchCV(pipeline_outer, outer_params, cv=PredefinedSplit(outer_split), 
                                  scoring=scoring, verbose=False, n_jobs=-1)#, refit=False)
        
        grid_outer.fit(X[:,fidx], Y)
        SC.append(grid_outer.best_score_)
        
        # return predicted
        for obj in grid_outer.best_estimator_.steps:
            if obj[0] is 'Scaler':
                ssc_outer_best = base.clone(obj[1], safe=True)
            elif obj[0] is 'Feature_sel':
                sel_outer_best = base.clone(obj[1], safe=True)
            elif obj[0] is 'Classifier':
                clf_outer_best = base.clone(obj[1], safe=True)
                
        
        pipeline_opt = Pipeline(memory=None,
                                steps=[('Scaler', ssc_outer_best),
                                       ('Classifier', clf_outer_best)])
            
        pipeline_opt.fit(Xtrain[:,fidx], Ytrain)
        yhat = pipeline_opt.predict(Xtest[:,fidx])

        YH.append(yhat)        
        CM.append(metrics.confusion_matrix(Ytest-1, yhat-1))#, ['HC','MCI','AD']))  
        YT.append(Ytest)
        TI.append(test_outer)
        
        print(grid_outer.best_score_)
    return SC, FI, FX, CM, YH, YT, TI 

def NestedCV_SVM__(X, Y, varnames):
    '''
    Performs nested cross-validation for simultaneous model selection and 
    feature selection using SVM with univariate feature selection.
        - sel_method: feature selection method ('f_classif','lda')
    '''
        # Get the faeture matrix according to inputted variable names
    if USE_REG_MEANS:
        X = getFeatureMatrix(Xdict, varnames, opt='abeta_tau_reg_means')
    else:
        X = getFeatureMatrix(Xdict, varnames)
        
    idx = np.arange(0, X.shape[0], 1)
    
    # Define outputs
    SC = [] # Outer classifier scores
    FI = [] # Feature Importances
    FX = [] # Index of top features
    YH = [] # Predicted Class Labels
    
    # Set up components of the ML pipeline
    scoring = 'f1_weighted' #'accuracy'
    
    ssc = RobustScaler()
    sel = SelectKBest(f_classif)
    clf_inner = svm.SVC(random_state=999)
    clf_outer = svm.SVC(random_state=999)
    
    cv_inner = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=777)
    cv_outer = StratifiedShuffleSplit(n_splits=100, test_size=0.25, random_state=111)
    
    for train_outer, test_outer in cv_outer.split(X, Y):
        Xtrain, Xtest = X[train_outer], X[test_outer]
        Ytrain, Ytest = Y[train_outer], Y[test_outer]
        
        # Set up Inner CV loop using a grid search CV
        pipeline_inner = Pipeline(memory=None,
                                  steps=[('Scaler', ssc),
                                         ('Feature_sel', sel),
                                         ('Classifier', clf_inner)])
        
        inner_params = [{'Feature_sel__k': [5, 10, 15, 20, 30, 40]}]#,
#                        {'Feature_sel__k': [5, 10, 15, 20, 30, 40]}]   
    
        # Run Inner CV loop using a grid search CV
        grid_inner = GridSearchCV(pipeline_inner, inner_params, cv=cv_inner, 
                                  scoring=scoring, verbose=True, n_jobs=8)
        grid_inner.fit(Xtrain, Ytrain)
    
        # Get best model and its feature importances (GINI index) - try entropy instead
        for obj in grid_inner.best_estimator_.steps:
            if obj[0] is 'Scaler':
                ssc_inner_best = base.clone(obj[1], safe=True)
            elif obj[0] is 'Feature_sel':
                sel_inner_best = base.clone(obj[1], safe=True)
                fi = obj[1].scores_
            elif obj[0] is 'Classifier':
                clf_inner_best = base.clone(obj[1], safe=True)

        # Get top features
        nfeat = sel_inner_best.k
        fidx = np.argsort(fi)[:-nfeat-1:-1]
        FI.append(fi)
        FX.append(fidx)
        
        # Set up outer CV lop
        pipeline_outer = Pipeline(memory=None,
                                  steps=[('Scaler', ssc_inner_best),
                                         ('Classifier', clf_outer)])
        
        outer_params = [{'Classifier__kernel': ['rbf'],
                         'Classifier__degree': [2],
                         'Classifier__gamma': [1e-3, 1e-2, 1e-1, 1],
                         'Classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
                        {'Classifier__kernel':['poly'],
                         'Classifier__degree': [2,3,4],
                         'Classifier__gamma': ['scale'],
                         'Classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
        
        # set test set manually
        outer_split = -np.ones((len(idx),))
        outer_split[test_outer] = 0
        
        # Run outer CV loop
        grid_outer = GridSearchCV(pipeline_outer, outer_params, cv=PredefinedSplit(outer_split), 
                                  scoring=scoring, verbose=True, n_jobs=8)#, refit=False)
        
        grid_outer.fit(X[:,fidx], Y)
        SC.append(grid_outer.best_score_)
        
        # return predicted
        for obj in grid_outer.best_estimator_.steps:
            if obj[0] is 'Scaler':
                ssc_outer_best = base.clone(obj[1], safe=True)
            elif obj[0] is 'Feature_sel':
                sel_outer_best = base.clone(obj[1], safe=True)
            elif obj[0] is 'Classifier':
                clf_outer_best = base.clone(obj[1], safe=True)
                
        
        pipeline_opt = Pipeline(memory=None,
                                steps=[('Scaler', ssc_outer_best),
                                       ('Classifier', clf_outer_best)])
            
        pipeline_opt.fit(Xtrain[:,fidx], Ytrain)
        
        yhat = pipeline_opt.predict(Xtest[:,fidx])        
        YH.append(metrics.confusion_matrix(Ytest-1, yhat-1))#, ['HC','MCI','AD']))
                
        print(grid_outer.best_score_)
    return SC, FI, FX, YH


def NestedCV_RFC___(X, Y, varnames):
    '''
    Performs nested cross-validation for simultaneous model selection and 
    feature selection using RFC with univariate feature selection.
    '''
        # Get the faeture matrix according to inputted variable names
    if USE_REG_MEANS:
        X = getFeatureMatrix(Xdict, varnames, opt='abeta_tau_reg_means')
    else:
        X = getFeatureMatrix(Xdict, varnames)
        
    idx = np.arange(0, X.shape[0], 1)
    
    # Define outputs
    SC = [] # Outer classifier scores
    FI = [] # Feature Importances
    FX = [] # Index of top features
    YH = [] # Predicted Class Labels
    
    # Set up components of the ML pipeline
    scoring = 'f1_weighted' #'accuracy'
    
    ssc = RobustScaler()
    clf_inner = RFC(random_state=999)
    clf_outer = RFC(random_state=999)
    
    cv_inner = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=777)
    cv_outer = StratifiedShuffleSplit(n_splits=100, test_size=0.25, random_state=111)
    
    for train_outer, test_outer in cv_outer.split(X, Y):
        Xtrain, Xtest = X[train_outer], X[test_outer]
        Ytrain, Ytest = Y[train_outer], Y[test_outer]
        
        # Set up Inner CV loop using a grid search CV
        pipeline_inner = Pipeline(memory=None,
                                  steps=[('Scaler', ssc),
                                         ('Classifier', clf_inner)])
        
        inner_params = [{'Classifier__class_weight': ['balanced'],
                         'Classifier__criterion': ['gini', 'entropy'],
                         'Classifier__n_estimators': [10, 50, 100, 200],
                         'Classifier__max_depth': [None],
                         'Classifier__min_samples_split': [2, 3, 4, 5],
                         'Classifier__min_samples_leaf': [1, 2, 3],
                         'Classifier__max_features': ['sqrt','log2',None],
                         'Classifier__oob_score': [True],
                         }]
    
        # Run Inner CV loop using a grid search CV
        grid_inner = GridSearchCV(pipeline_inner, inner_params, cv=cv_inner, 
                                  scoring=scoring, verbose=True, n_jobs=8)
        grid_inner.fit(Xtrain, Ytrain)
    
        # Get best model and its feature importances (GINI index) - try entropy instead
        for obj in grid_inner.best_estimator_.steps:
#            print(obj[1])
            if obj[0] is 'Scaler':
                ssc_inner_best = base.clone(obj[1], safe=True)
            elif obj[0] is 'Feature_sel':
                sel_inner_best = base.clone(obj[1], safe=True)
            elif obj[0] is 'Classifier':
                clf_inner_best = base.clone(obj[1], safe=True)
                fi = obj[1].feature_importances_

        # Get top features
        nfeat = np.sum(fi>0)
        fidx = np.argsort(fi)[:-nfeat-1:-1]
        FI.append(fi)
        FX.append(fidx)
        
        # Set up outer CV lop
        pipeline_outer = Pipeline(memory=None,
                                  steps=[('Scaler', ssc_inner_best),
#                                         ('Feature_sel', sel_inner_best)
                                         ('Classifier', clf_inner_best)])
        
        outer_params = [{'Classifier__class_weight': ['balanced'],
                        'Classifier__criterion': ['gini', 'entropy'],
                        'Classifier__n_estimators': [10, 50, 100, 200],
                        'Classifier__max_depth': [None],
                        'Classifier__min_samples_split': [2, 3, 4, 5],
                        'Classifier__min_samples_leaf': [1, 2, 3],
                        'Classifier__max_features': [None],
                        'Classifier__oob_score': [True, False],
                         }]
        
        # set test set manually
        outer_split = -np.ones((len(idx),))
        outer_split[test_outer] = 0
        
        # Run outer CV loop
        grid_outer = GridSearchCV(pipeline_outer, outer_params, cv=PredefinedSplit(outer_split), 
                                  scoring=scoring, verbose=True, n_jobs=8)#, refit=False)
        
        grid_outer.fit(X[:,fidx], Y)
        SC.append(grid_outer.best_score_)
        
        # return predicted
        for obj in grid_outer.best_estimator_.steps:
            if obj[0] is 'Scaler':
                ssc_outer_best = base.clone(obj[1], safe=True)
            elif obj[0] is 'Feature_sel':
                sel_outer_best = base.clone(obj[1], safe=True)
            elif obj[0] is 'Classifier':
                clf_outer_best = base.clone(obj[1], safe=True)
                
        
        pipeline_opt = Pipeline(memory=None,
                                steps=[('Scaler', ssc_outer_best),
#                                       ('Feature_sel', sel_outer_best),
                                       ('Classifier', clf_outer_best)])
            
        pipeline_opt.fit(Xtrain[:,fidx], Ytrain)
        yhat = pipeline_opt.predict(Xtest[:,fidx])
        YH.append(metrics.confusion_matrix(Ytest-1, yhat-1))#, ['HC','MCI','AD']))
                
        print(grid_outer.best_score_)
        print(clf_inner_best)
        print(clf_outer_best)
    return SC, FI, FX, YH

# %% Run Classification Experiment
CLASSIFIER = 'SVM' # 'SVM', 'RFC', 'NEST'
DIM_REDUCTION = 'f_classif' # 'f_classif','pca', 'lda', 'l1', 'rfc'
if CLASSIFIER == 'SVM':
    sf1, FI1, FX1, YH1 = NestedCV_SVM__(Xdict, Y, emp_varnames)
    sf2, FI2, FX2, YH2 = NestedCV_SVM__(Xdict, Y, good_sim + [sim_varnames[-1]])
    sf3, FI3, FX3, YH3 = NestedCV_SVM__(Xdict, Y, emp_varnames + good_sim + [sim_varnames[-1]])
elif CLASSIFIER == 'RFC':
    sf1, FI1, FX1, YH1 = NestedCV_RFC___(Xdict, Y, emp_varnames)
    sf2, FI2, FX2, YH2 = NestedCV_RFC___(Xdict, Y, good_sim + [sim_varnames[-1]])
    sf3, FI3, FX3, YH3 = NestedCV_RFC___(Xdict, Y, emp_varnames + good_sim + [sim_varnames[-1]])
elif CLASSIFIER == 'NEST':
    sf1, FI1, FX1, CM1, YH1, YT1, TI1 = nestedCV(Xdict, Y, emp_varnames)
    sf2, FI2, FX2, CM2, YH2, YT2, TI2 = nestedCV(Xdict, Y, good_sim + [sim_varnames[-1]])
    sf3, FI3, FX3, CM3, YH3, YT3, TI3 = nestedCV(Xdict, Y, emp_varnames + good_sim + [sim_varnames[-1]])

# Save and load results from experiment
def saveResults():
    '''
    Saves classification results to .npz
    '''
    timestr = time.strftime('%Y-%m-%d-%H-%M')
    savefolder = 'Results/'
    gi = 'withGI' if USE_GI else 'noGI'
    savename = 'Results_' + CLASSIFIER + '_' + DIM_REDUCTION + '_' + gi + '_' + timestr

    np.savez(savefolder + savename + '.npz',
             F1_emp=sf1, FI_emp=FI1, FX_emp=FX1, YH_emp=YH1, YT_emp=YT1, TI_emp=TI1,
             F1_sim=sf2, FI_sim=FI2, FX_sim=FX2, YH_sim=YH2, YT_sim=YT2, TI_sim=TI2,
             F1_com=sf3, FI_com=FI3, FX_com=FX3, YH_com=YH3, YT_com=YT3, TI_com=TI3)
	
    r = np.load(savefolder + savename + '.npz', allow_pickle=True)
    sio.savemat(savefolder + savename + '.mat', mdict={key:r[key] for key in r.keys()})
    return None

def loadResults(fname):
    '''
    Loads a previously saved classification result from a .npz file.
    '''
    r = np.load('Results/'+fname)
    return r
    
# saveResults()


# %% Plot Results
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False):#, title=None):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    '''
    cmap=plt.cm.Blues
    
    if y_pred is None:
        if y_true.ndim == 2:
            cm = y_true
    elif y_pred.ndim == 1 and y_true.ndim == 1:
        # Compute confusion matrix
        cm = metrics.confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred).astype(int)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return None

def plotScores(scores, stype='f1'):
    '''
    Plot classification score
    '''
    # barplot with errorbars
    means = [np.mean(scores[i]) for i in range(len(scores))]
    sterr = [np.std(scores[i])/np.sqrt(len(scores[i])) for i in range(len(scores))]
    
    plt.figure()
    plt.bar(['Empirical','Simulated','Combined'], means, yerr=sterr)
    plt.ylim([0,1])
#    for tick in plt.gca().xaxis.get_major_ticks():
#        tick.label.set_fontsize(14)
#    
    # add text for each score
    for i, v in enumerate(scores):
        plt.text(i - 0.1, 0.4, '{:.{}f}'.format(np.mean(v),2),
                 size=12)
    
    # add title
    if stype == 'f1':
        plt.title('Weighted F1 score', fontsize=14)
    elif stype == 'acc':
        plt.title('Percent Accuracy', fontsize=14)
    
    savename = 'NestedCV_{}_Performance_{}_{}_2.png'.format(CLASSIFIER, 'noGI', stype)
    plt.savefig(savename, bbox_inches='tight', pad_inches=0.1, dpi=400)
    return None

plot_confusion_matrix(np.sum(CM1,axis=0), None, normalize=True, classes=np.array(group_classes))
plot_confusion_matrix(np.sum(CM2,axis=0), None, normalize=True, classes=np.array(group_classes))
plot_confusion_matrix(np.sum(CM3,axis=0), None, normalize=True, classes=np.array(group_classes))
print('after plotting confusion matrices')
acc1 = [np.trace(cm)/np.sum(cm) for cm in CM1]
acc2 = [np.trace(cm)/np.sum(cm) for cm in CM2]
acc3 = [np.trace(cm)/np.sum(cm) for cm in CM3]

plotScores([sf1,sf2,sf3], stype='f1')
# plotScores([acc1,acc2,acc3], stype='acc')
print('after plotting scores')


# %% Feature Importance


def plot_FeatureFrequency_Top(Xdict, FI, FX, varnames, title=None):
    # Get feature names    
    X = np.concatenate([Xdict[features] for features in varnames], axis=1) 
    Xclean, featurenames = clean_features(X, extendFeatureNames(varnames, regions, volnames))    

    # Selected Feature Historgram
    fidx = [idx for cvrun in FX for idx in cvrun]
    tmp = [x[0][x[1]] for x in zip(FI,FX)]
    fimp = [imp for x in tmp for imp in x]
    fcount = np.bincount(fidx, minlength=len(FI[0]))[0:]
    
    n50 = np.sum(fcount>50)
    sortInd = fcount.argsort()[-n50:][::-1]
    fcount_sort = fcount[sortInd]
    
    fname_sort = [featurenames[i] if featurenames[i].startswith('Volume') else featurenames[i][:-4] for i in sortInd]
#    region_sort = [regions[i][:-4] for i in sortInd]

    fig, ax = plt.subplots()
    fig.set_size_inches(20,14)

    plt.bar(np.arange(0, len(fcount_sort)), fcount_sort)
    plt.xticks(range(len(fcount_sort)), fname_sort, rotation=90, fontsize=16)
#            plt.xlim([-1, n+1])
    plt.tick_params(axis='x',direction='in',pad=-350)
    plt.gcf().subplots_adjust(bottom=0.2)
    
    plt.title('Feature Selection Frequency ({})'.format(title), fontsize=14)
    plt.tight_layout()
    plt.savefig('Feature_Selection_Frequency_{}.png'.format(title), 
                bbox_inches='tight', pad_inches=0.1, dpi=400)

 
plot_FeatureFrequency_Top(Xdict, FI1, FX1, emp_varnames, title='Empirical') 
plot_FeatureFrequency_Top(Xdict, FI2, FX2, good_sim + [sim_varnames[-1]], title='Simulated')
plot_FeatureFrequency_Top(Xdict, FI3, FX3, emp_varnames + good_sim + [sim_varnames[-1]], title='Combined')
print('after plotting feature importance')
# --- if using results loaded from previously saved file ---
# r = loadResults(fname)
# plot_FeatureFrequency_Top(Xdict, r['FI_emp'], r['FX_emp'], emp_varnames, title='Empirical') 
# plot_FeatureFrequency_Top(Xdict, r['FI_sim'], r['FX_sim'], good_sim + [sim_varnames[-1]], title='Simulated')
# plot_FeatureFrequency_Top(Xdict, r['FI_com'], r['FX_com'], emp_varnames + good_sim + [sim_varnames[-1]], title='Combined')


# %% Save feature imortance stats to excel
def save_FeatureStats(Xdict, FI, FX, FSET):
    from openpyxl import load_workbook
    
    # Get feature names
    if FSET == 'emp':
        varnames = emp_varnames
        sheet_name = 'Empirical Feature Set'
    elif FSET == 'sim':
        varnames = good_sim + [sim_varnames[-1]]
        sheet_name = 'Simulated Feature Set'
    elif FSET == 'com':
        varnames = emp_varnames + good_sim + [sim_varnames[-1]]
        sheet_name = 'Combined Feature Set'
    
    X = np.concatenate([Xdict[features] for features in varnames], axis=1)
    Xclean, featurenames = clean_features(X, extendFeatureNames(varnames, regions, volnames))
    
    # Feature importance metrics
    fidx = [idx for cvrun in FX for idx in cvrun]
    fcount = np.bincount(fidx, minlength=len(FI[0]))[0:] # selection frequency
    fimp = np.mean(FI, axis=0) # entropy criterion
    
    # Save to excel    
    df = pd.DataFrame(np.stack([fcount, fimp], axis=1), index=featurenames,
                      columns=['Selection Frequency','Entropy Criterion'])
    
    book = load_workbook('Results/Feature_Importance_Full.xlsx')
    writer = pd.ExcelWriter('Results/Feature_Importance_Full.xlsx')
    writer.book = book
    
    df.to_excel(writer, sheet_name=sheet_name)
    writer.save()
    return None

# --- if using results loaded from previously saved file ---
# r = loadResults(fname)
# save_FeatureStats(Xdict, r['FI_emp'], r['FX_emp'], 'emp')
# save_FeatureStats(Xdict, r['FI_sim'], r['FX_sim'], 'sim')
# save_FeatureStats(Xdict, r['FI_com'], r['FX_com'], 'com')


# %% Some helpful statistical tests if needed
def mctest(y1, y2, ytrue):
    '''
    Performs McNemar's test to compare two classifiers.
    '''
    table = [[((y1==ytrue)&(y2==ytrue)).sum(), ((y1==ytrue)&(y2!=ytrue)).sum()],
              [((y1!=ytrue)&(y2==ytrue)).sum(), ((y1!=ytrue)&(y2!=ytrue)).sum()]]
    result = mcnemar(table, exact=True)
    print('McNemar Test Result: Chi-squared(1) = {}, p = {}'.format(result.statistic, result.pvalue))
    return None

def shapiro_wilk_test(f1_emp, f1_sim, f1_com):
    '''
    Peforms a shapiro wilk test of normality on performance results.
    '''
    s, p = shapiro(f1_emp)
    print('Shapiro Wilk test of F1 (Empirical): s = {:.2f}, p = {:.2f}'.format(s,p))
    
    s, p = shapiro(f1_sim)
    print('Shapiro Wilk test of F1 (Simulated): s = {:.2f}, p = {:.2f}'.format(s,p))
    
    s, p = shapiro(f1_com)
    print('Shapiro Wilk test of F1 (Combined): s = {:.2f}, p = {:.2f}'.format(s,p))
    return None
    
def ttest_compare(f1_emp, f1_sim, f1_com):
    '''
    Performs a t-test comparison of feature set performance.
    '''
    tstat, p = ttest_rel(f1_emp, f1_sim)
    print('Related-samples t-test of F1 scores [Empirical vs. Simulated]: t(98) = {}, p = {}'.format(tstat,p))
    
    tstat, p = ttest_rel(f1_emp, f1_com)
    print('Related-samples t-test of F1 scores [Empirical vs. Combined]: t(98) = {}, p = {}'.format(tstat,p))
    
    tstat, p = ttest_rel(f1_sim, f1_com)
    print('Related-samples t-test of F1 scores [Simulated vs. Combined]: t(98) = {}, p = {}'.format(tstat,p))
    return None

def utest(f1_emp, f1_sim, f1_com):
    '''
    Performs a Mann-Whitney U-test comparison of feature set performance.
    '''
    u, p = mannwhitneyu(f1_emp, f1_sim)
    print('Mann-Whitney U-test of F1 scores [Empirical vs. Simulated]: U(98) = {}, p = {}'.format(u,p))
    
    u, p = mannwhitneyu(f1_emp, f1_com)
    print('Mann-Whitney U-test of F1 scores [Empirical vs. Combined]: U(98) = {}, p = {}'.format(u,p))
    
    u, p = mannwhitneyu(f1_sim, f1_com)
    print('Mann-Whitney U-test of F1 scores [Simulated vs. Combined]: U(98) = {}, p = {}'.format(u,p))
    return None

def wsrtest(f1_emp, f1_sim, f1_com):
    '''
    Performs a Wilcoxon signed-rank test comparison of feature set performance.
    '''
    u, p = wilcoxon(f1_emp, f1_sim)
    print('Wilcoxon signed-rank test of F1 scores [Empirical vs. Simulated]: U(98) = {}, p = {}'.format(u,p))
    
    u, p = wilcoxon(f1_emp, f1_com)
    print('Wilcoxon signed-rank test of F1 scores [Empirical vs. Combined]: U(98) = {}, p = {}'.format(u,p))
    
    u, p = wilcoxon(f1_sim, f1_com)
    print('Wilcoxon signed-rank testt of F1 scores [Simulated vs. Combined]: U(98) = {}, p = {}'.format(u,p))
    return None
