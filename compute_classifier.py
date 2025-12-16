"""
This file computes nested cross-validation classification using Random Forest
feature selection and SVM classification. Original code is by Kiret Dhindsa and used
in Triebkorn et al. 2022. It has been adapted to work with the FDT data computed
from the LinHopf model and the model-free approach. - Melle Bolding 
"""

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

# basic dataset libraries
import os
import numpy as np
import pandas as pd
from src.data_loaders.load_data_records import loadProteins

# machine learning libraries
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import metrics, svm, base

np.random.seed(123)

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



def getFeatureMatrix(Xdict, varnames):
    """
    Constructs feature matrix from Xdict using variable names
    Returns: X (cleaned), feature_names, removed_indices
    """
    # Build feature matrix
    X = np.concatenate([Xdict[features] for features in varnames], axis=1)
    
    # Get feature names BEFORE cleaning
    feature_names = extendFeatureNames(varnames, Xdict, regions)

    return X, feature_names, []


# ==================== MAIN CLASSIFICATION ====================
def nestedCV(Xdict, Y, varnames):
    """
    Nested cross-validation with Random Forest feature selection + SVM classification.
    """
    
    # =========================================================================
    # FEATURE NAMES VS MODALITY KEYS
    # =========================================================================
    
    if varnames[0] in Xdict.keys():
        print(f"Input detected as Modality Keys. Loading full modalities: {varnames}")
        X, feature_names_all, removed_idx = getFeatureMatrix(Xdict, varnames)
    else:
        print(f"Input detected as Specific Feature List ({len(varnames)} features).")

        
        # Infer which modalities are needed based on the feature prefixes
        required_modalities = set()
        for fname in varnames:
            found_match = False
            for key in Xdict.keys():
                # Check if feature name starts with a known modality key
                if fname.startswith(key + '_'): 
                    required_modalities.add(key)
                    found_match = True
                    break
            if not found_match:
                print(f"Warning: Could not identify modality for feature '{fname}'")

        # Load the full universe of features for these modalities
        print(f"Constructing base matrix from: {list(required_modalities)}...")
        X_full, names_full, _ = getFeatureMatrix(Xdict, list(required_modalities))
       
        
        # Check if the requested name exists in the generated list
        if varnames[0] in names_full:
            print("Match found, problem might be specific missing features")
        else:
            print("The naming convention differs")
        print("-" * 40 + "\n")


        # Create a map for fast index lookup
        name_to_idx = {name: i for i, name in enumerate(names_full)}
        
        # Find indices of the requested features
        indices_to_keep = []
        valid_feature_names = []
        
        for fname in varnames:
            if fname in name_to_idx:
                indices_to_keep.append(name_to_idx[fname])
                valid_feature_names.append(fname)
            else:
                print(f"Warning: Feature '{fname}' not found in generated matrix.")
        
        # Slice the matrix
        X = X_full[:, indices_to_keep]
        feature_names_all = valid_feature_names
        
        print(f"Matrix sliced. Final shape: {X.shape}")

    # =========================================================================
    
    print(f"Total features after processing: {X.shape[1]}")
    if len(feature_names_all) > 0:
        print(f"Feature names example: {feature_names_all[:5]}")
    
    idx = np.arange(0, X.shape[0], 1)
    
    # Output storage
    SC = []  # Scores
    FI = []  # Feature importances
    FX = []  # Feature indices
    FN = []  # Feature names
    CM = []  # Confusion matrices
    YH = []  # Predicted labels
    YT = []  # True labels
    TI = []  # Test indices
    
    # Setup
    scoring = 'f1_weighted'
    ssc = RobustScaler()
    clf_inner = RFC(random_state=999)
    clf_outer = svm.SVC(random_state=999)
    
    cv_inner = StratifiedShuffleSplit(n_splits=20, test_size=0.25, random_state=777)
    cv_outer = StratifiedShuffleSplit(n_splits=100, test_size=0.25, random_state=888)
    
    fold_num = 0
    for train_outer, test_outer in cv_outer.split(X, Y):
        fold_num += 1
        Xtrain, Xtest = X[train_outer], X[test_outer]
        Ytrain, Ytest = Y[train_outer], Y[test_outer]
        
        # ===== INNER CV: Feature selection =====
        pipeline_inner = Pipeline(memory=None,
                                  steps=[('Scaler', ssc),
                                         ('Classifier', clf_inner)])
        
        inner_params = [{
            'Classifier__class_weight': ['balanced'],
            'Classifier__criterion': ['entropy'],
            'Classifier__n_estimators': [10, 50],
            'Classifier__max_depth': [None],
            'Classifier__min_samples_split': [3, 4],
            'Classifier__min_samples_leaf': [2, 3],
            'Classifier__max_features': ['sqrt'],
        }]        
    
        grid_inner = GridSearchCV(pipeline_inner, inner_params, cv=cv_inner, 
                                  scoring=scoring, verbose=False, n_jobs=-1)
        grid_inner.fit(Xtrain, Ytrain)
    
        # Extract best model components
        for obj in grid_inner.best_estimator_.steps:
            if obj[0] == 'Scaler':
                ssc_inner_best = base.clone(obj[1], safe=True)
            elif obj[0] == 'Classifier':
                clf_inner_best = base.clone(obj[1], safe=True)
                fi = obj[1].feature_importances_

        # Feature Selection Logic
        # Safety check: if X has fewer features than K=50, select all available
        n_feats_available = Xtrain.shape[1]
        K = min(50, n_feats_available)
        
        if n_feats_available > 0:
            fidx = np.argsort(fi)[::-1][:K]
        else:
            fidx = np.array([], dtype=int)
        
        # Store results
        FI.append(fi)
        FX.append(fidx)
        
        # Map indices back to names
        if len(fidx) > 0:
            selected_names = [feature_names_all[i] for i in fidx]
        else:
            selected_names = []
        FN.append(selected_names)
        
        if fold_num == 1:
            print(f"\nFold 1 - Selected {len(fidx)} features from input pool.")
            if len(selected_names) > 0:
                print(f"Top examples: {selected_names[:5]}")
        
        # ===== OUTER CV: SVM classification =====
        pipeline_outer = Pipeline(memory=None,
                                  steps=[('Scaler', ssc_inner_best),
                                         ('Classifier', clf_outer)])
        
        outer_params = [
            {
                'Classifier__kernel': ['rbf'],
                'Classifier__degree': [2],
                'Classifier__gamma': [1e-2, 1e-1, 1],
                'Classifier__C': [0.01, 0.1, 1, 10, 100]
            },
            {
                'Classifier__kernel': ['poly'],
                'Classifier__degree': [2, 3],
                'Classifier__gamma': ['scale'],
                'Classifier__C': [0.01, 0.1, 1, 10, 100]
            }
        ]
        
        outer_split = -np.ones((len(idx),))
        outer_split[test_outer] = 0
        
        grid_outer = GridSearchCV(pipeline_outer, outer_params, 
                                  cv=PredefinedSplit(outer_split), 
                                  scoring=scoring, verbose=False, n_jobs=-1)
        
        # Guard clause for empty feature selection
        if len(fidx) > 0:
            grid_outer.fit(X[:, fidx], Y)
            score = grid_outer.best_score_
            
            # Extract best outer model
            for obj in grid_outer.best_estimator_.steps:
                if obj[0] == 'Scaler':
                    ssc_outer_best = base.clone(obj[1], safe=True)
                elif obj[0] == 'Classifier':
                    clf_outer_best = base.clone(obj[1], safe=True)
            
            # Final prediction
            pipeline_opt = Pipeline(memory=None,
                                    steps=[('Scaler', ssc_outer_best),
                                           ('Classifier', clf_outer_best)])
            pipeline_opt.fit(Xtrain[:, fidx], Ytrain)
            yhat = pipeline_opt.predict(Xtest[:, fidx])
        else:
            # Fallback if no features were selected
            score = 0
            yhat = np.full_like(Ytest, fill_value=Ytrain[0])
            print(f"Fold {fold_num}: No features selected.")

        SC.append(score)
        YH.append(yhat)        
        CM.append(metrics.confusion_matrix(Ytest-1, yhat-1))
        YT.append(Ytest)
        TI.append(test_outer)
        
        if fold_num % 10 == 0:
            print(f"Fold {fold_num}/100 - Score: {score:.3f}")
    
    return SC, FI, FX, FN, CM, YH, YT, TI, feature_names_all


# ==================== DATA SETUP ====================
model_type = 'modelbased'
if model_type == 'modelbased':
    filename = f"FDT_results_DL_B2_N400_sigTrue_aTrue_filt.npz"
    output_file = 'nested_cv_results_modelbased.npz'
if model_type == 'modelfree':
    filename = f"FDT_results_DL_B2_N400_modelfree.npz"
    output_file = 'nested_cv_results_modelfree.npz'

filt=True
repo_root = os.getcwd() 
save_path = os.path.join(repo_root, "data", "FDT_DATA")
fdt_data = np.load(os.path.join(save_path, filename), allow_pickle=True)
df = pd.DataFrame({k: fdt_data[k].tolist() for k in fdt_data.files})
df = loadProteins(df, 'DL_B2', 'Tau', repo_root,filt=filt)
df = loadProteins(df, 'DL_B2', 'ABeta', repo_root,filt=filt)
group_map = {'HC': 1, 'MCI(AB+)': 2, 'AD': 3}
group_classes = ['HC', 'MCI(AB+)', 'AD']
df = df[df['group'].isin(group_classes)]

Xdict = {
    'I_norm2': np.stack(df['I_norm2'].values),
    'ABeta': np.stack(df['ABeta'].values),
    'Tau': np.stack(df['Tau'].values),
}
Y = df['group'].map(group_map).values

emp_varnames = ['ABeta', 'Tau']
sim_varnames = ['I_norm2']
good_sim = sim_varnames

Nclass = np.unique(Y).shape[0]
Nsubj = Y.shape[0]

# Correct vectors with dim==1
for key in Xdict.keys():
    if len(Xdict[key].shape) == 1:
        Xdict[key] = np.expand_dims(Xdict[key], axis=1)

Ystr = [group_classes[int(c)-1] for c in Y]

# Setup region/feature names
N_parcels = Xdict['I_norm2'].shape[1]
degrees = np.ones(N_parcels)
regions = [str(i) for i in range(N_parcels)]
volnames = regions


# ==================== RUN CLASSIFICATION ====================
# empirical run
sf1, FI1, FX1, FN1, CM1, YH1, YT1, TI1, feat_names1 = nestedCV(Xdict, Y, emp_varnames)
# simulated run
sf2, FI2, FX2, FN2, CM2, YH2, YT2, TI2, feat_names2 = nestedCV(Xdict, Y, sim_varnames)

# select top 40 features from each
flat_features1 = [name for fold in FN1 for name in fold]
flat_features2 = [name for fold in FN2 for name in fold]
flat_features1 = flat_features1[:40]
flat_features2 = flat_features2[:40]
unique_combined_features = list(set(flat_features1 + flat_features2))

# combined run
sf3, FI3, FX3, FN3, CM3, YH3, YT3, TI3, feat_names3 = nestedCV(Xdict, Y, unique_combined_features)

# save results
np.savez(
    output_file,
    sf1=sf1,
    FI1=FI1,
    FX1=FX1,
    FN1=FN1,
    CM1=CM1,
    YH1=YH1,
    YT1=YT1,
    TI1=TI1,
    feat_names1=feat_names1,
    sf2=sf2,
    FI2=FI2,
    FX2=FX2,
    FN2=FN2,
    CM2=CM2,
    YH2=YH2,
    YT2=YT2,
    TI2=TI2,
    feat_names2=feat_names2,
    sf3=sf3,
    FI3=FI3,
    FX3=FX3,
    FN3=FN3,
    CM3=CM3,
    YH3=YH3,
    YT3=YT3,
    TI3=TI3,
    feat_names3=feat_names3,
)