import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
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

    # Sort by mean importance
    df_features = df_features.sort_values('Mean_Importance', ascending=False)

    print(f"\n{'='*60}")
    print(f"FEATURE IMPORTANCE ANALYSIS - {title}")
    print(f"{'='*60}")
    print(f"\nTop 20 features by mean importance:")
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


def plot_FeatureFrequency_Top(FI, FX, FN, feature_names_all, title=None):
    """
    Plot top features by selection frequency with proper names
    """
    # Get feature statistics
    df_features = analyze_feature_importance(FI, FX, FN, feature_names_all, title)
    
    # Get top 50 features
    top_features = df_features.head(50)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(range(len(top_features)), top_features['Mean_Importance'])
    ax.set_xticks(range(len(top_features)))
    ax.set_xticklabels(
    top_features['Feature'], 
    rotation=60, 
    fontsize=9,
    ha='right'  # <-- Explicitly set alignment to center the text visually
    )
    ax.tick_params(
    axis='x',         # Apply to X-axis
    which='both',     # Apply to major and minor ticks
    top=True,         # Show ticks on the top
    bottom=False,      # Keep ticks on the bottom (set to False to hide them)
    labelbottom=True, # Keep labels on the bottom (set to False to hide them)
    labeltop=False,    # Do not show labels on the top (usually unnecessary)
    direction='in',  # Ticks point inwards
    )

    ax.tick_params(
    axis='y',         # Apply to Y-axis
    which='both',
    right=True,       # Show ticks on the right
    left=False,        # Keep ticks on the left
    labelleft=True,   # Keep labels on the left
    labelright=False,  # Do not show labels on the right
    direction='in',  # Ticks point inwards
    )
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)    
    ax.set_xlim(0 - 0.5, len(top_features) - 0.5)
    ax.set_ylabel('FI', fontsize=14)
    ax.set_xlabel('Features', fontsize=13)
    ax.set_title(f'Top 50 Features by Selection Frequency - {title}', fontsize=16)

    
    # Color bars by feature type
    colors = {'I_norm2': "#31A86D",  
              'ABeta': "#140BB7", 'Tau': "#E33B0D"}
    #'X_norm2': "#0F8F14",

    for i, (idx, row) in enumerate(top_features.iterrows()):
        # Example: 'I_norm2_1' -> ['I_norm2', '1']
        feature_root = row['Feature'].rsplit('_', 1)[0]
        bars[i].set_color(colors.get(feature_root, 'gray'))
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=feat) 
                      for feat, color in colors.items()]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'Feature_Selection_Frequency_{title}.png', 
                bbox_inches='tight', dpi=400)
    plt.show()
    
    # Save detailed results to Excel
    df_features.to_excel(f'Feature_Importance_Detailed_{title}.xlsx', index=False)
    print(f"\nDetailed results saved to: Feature_Importance_Detailed_{title}.xlsx")
    
    return df_features

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
    cbar = ax.figure.colorbar(im, ax=ax, format=lambda x, _: f'{x*100:.0f}')
    cbar.ax.set_ylabel('%', rotation=0, ha='left', fontweight='bold')
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j] * 100 if normalize else cm[i, j]
            ax.text(j, i, format(value, fmt),
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
    plt.show()
    return None

def plotScores2(scores, stype='f1'):
    from scipy.stats import wilcoxon
    '''
    Plot classification score with statistical significance testing
    '''
    
    # barplot with errorbars
    means = [np.mean(scores[i]) for i in range(len(scores))]
    sterr = [np.std(scores[i])/np.sqrt(len(scores[i])) for i in range(len(scores))]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Empirical','Simulated','Combined'], means, yerr=sterr)
    plt.ylim([0,1])
    
    # add text for each score
    for i, v in enumerate(scores):
        plt.text(i - 0.1, 0.4, '{:.{}f}'.format(np.mean(v),2),
                 size=12)
    
    # add title
    if stype == 'f1':
        plt.title('Weighted F1 score', fontsize=14)
    elif stype == 'acc':
        plt.title('Percent Accuracy', fontsize=14)
    
    # Statistical significance testing
    print(f"\n{'='*60}")
    print(f"STATISTICAL SIGNIFICANCE TESTING - {stype.upper()}")
    print(f"{'='*60}")
    
    # Paired comparisons using Wilcoxon signed-rank test
    comparisons = [
        ('Combined', 'Empirical', scores[2], scores[0]),
        ('Combined', 'Simulated', scores[2], scores[1]),
        ('Empirical', 'Simulated', scores[0], scores[1])
    ]
    
    for name1, name2, score1, score2 in comparisons:
        stat, p_value = wilcoxon(score1, score2)
        mean_diff = np.mean(score1) - np.mean(score2)
        
        # Format p-value
        if p_value < 0.001:
            p_str = "P < 0.001"
        else:
            p_str = f"P = {p_value:.3f}"
        
        print(f"{name1} vs {name2}:")
        print(f"  Mean difference: {mean_diff:+.3f}")
        print(f"  {p_str}")
        print()
    
    savename = 'NestedCV_{}_Performance_{}_{}_2.png'.format(CLASSIFIER, 'noGI', stype)
    plt.savefig(savename, bbox_inches='tight', pad_inches=0.1, dpi=400)
    plt.show()
    return None

def extendFeatureNames(varnames, regions, volnames):
    '''
    Extends feature names so that each column has its own name.
    '''
    names = []
    for var in varnames:
        n = Xdict[var].shape[1]
        if n == 1:
            names.append(var)
        elif var == 'Volumes':
            names.extend([var+'_'+volnames[i] for i in range(n)])
        else:         
            names.extend([var+'_'+regions[i] for i in range(n)])
    return names


# Load the file
loaded_data = np.load('nested_cv_results.npz', allow_pickle=True)

# Access variables by the name you gave them when saving
sf1 = loaded_data['sf1']
FI1 = loaded_data['FI1']
feat_names1 = loaded_data['feat_names1']
FX1 = loaded_data['FX1']
FN1 = loaded_data['FN1']
CM1 = loaded_data['CM1']
YH1 = loaded_data['YH1']
YT1 = loaded_data['YT1']
TI1 = loaded_data['TI1']
sf2 = loaded_data['sf2']
FI2 = loaded_data['FI2']
feat_names2 = loaded_data['feat_names2']
FX2 = loaded_data['FX2']
FN2 = loaded_data['FN2']
CM2 = loaded_data['CM2']
YH2 = loaded_data['YH2']
YT2 = loaded_data['YT2']
TI2 = loaded_data['TI2']
sf3 = loaded_data['sf3']
FI3 = loaded_data['FI3']
feat_names3 = loaded_data['feat_names3']
FX3 = loaded_data['FX3']
FN3 = loaded_data['FN3']
CM3 = loaded_data['CM3']
YH3 = loaded_data['YH3']
YT3 = loaded_data['YT3']
TI3 = loaded_data['TI3']

print(FN2.shape)
print(FI2.shape)
print(FI3[0])

# Note: feat_names1_loaded might be a 0-D array if it was a list/dict. 
# Access its content like this:
if feat_names1.ndim == 0:
    feat_names1_list = feat_names1.item()
else:
    feat_names1_list = feat_names1

print("Data loaded successfully.")
group_classes = ['HC', 'MCI(AB+)', 'AD']
CLASSIFIER = 'SVM'
plot_confusion_matrix(np.sum(CM1,axis=0), None, normalize=True, classes=np.array(group_classes))
plot_confusion_matrix(np.sum(CM2,axis=0), None, normalize=True, classes=np.array(group_classes))
plot_confusion_matrix(np.sum(CM3,axis=0), None, normalize=True, classes=np.array(group_classes))
df_emp = plot_FeatureFrequency_Top(FI1, FX1, FN1, feat_names1, title='Empirical')
df_sim = plot_FeatureFrequency_Top(FI2, FX2, FN2, feat_names2, title='Simulated')
df_com = plot_FeatureFrequency_Top(FI3, FX3, FN3, feat_names3, title='Combined')

# print(df_com['Mean_Importance'])
# print(f"Empirical:  F1 = {np.mean(sf1):.3f} ± {np.std(sf1):.3f}")
# print(f"Simulated:  F1 = {np.mean(sf2):.3f} ± {np.std(sf2):.3f}")
# print(f"Combined:   F1 = {np.mean(sf3):.3f} ± {np.std(sf3):.3f}")

plotScores([sf1,sf2,sf3], stype='f1')

import numpy as np
import matplotlib.pyplot as plt
import os
from nilearn import surface, datasets, plotting
import nibabel as nib
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.ticker import FixedLocator, MaxNLocator
from nilearn.datasets import load_fsaverage
from nilearn.surface import SurfaceImage
from nilearn.datasets import load_fsaverage_data
import pandas as pd


import numpy as np
import matplotlib.pyplot as plt
import os
from nilearn import surface, datasets, plotting
import nibabel as nib
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from nilearn.datasets import load_fsaverage
from nilearn.surface import SurfaceImage
from nilearn.datasets import load_fsaverage_data
import pandas as pd


def brain_map_feature_importance2(df_features, 
                                 modalities=['Tau', 'ABeta', 'I_norm2', 'X_norm2'],
                                 metric='Selection_Frequency_%',
                                 atlas_path=None,
                                 save_path=None,
                                 title_suffix="",
                                 vmin=None,
                                 vmax=None):
    """
    Visualizes feature importance with a separate figure for each modality.
    Each figure contains 4 views (posterior, left lateral, anterior, right lateral)
    with an individual colorbar.
    """
    
    import os
    import numpy as np
    import nibabel as nib
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, ListedColormap, LinearSegmentedColormap
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator
    from nilearn import plotting, datasets, surface
    from nilearn.surface import SurfaceImage
    from nilearn.datasets import load_fsaverage, load_fsaverage_data
    

    TOTAL_PARCELS = 400

    # ==================== MODALITY CONFIG ====================
    modality_config = {
        'Tau':     {'label': 'Tau', 'cmap': 'Reds'},
        'ABeta':   {'label': 'Amyloid-beta',  'cmap': 'Blues'},
        'I_norm2': {'label': 'Integral violation',   'cmap': 'Greens'},
        'X_norm2': {'label': 'χ',   'cmap': 'YlOrBr'}
    }

    modalities = [m for m in modalities if m in modality_config]

    print(f"\n{'='*60}")
    print(f"Plotting modalities: {modalities}")
    print(f"{'='*60}")

    # ==================== PARSE FEATURES ====================
    modality_features = {mod: [] for mod in modalities}

    for _, row in df_features.iterrows():
        feature_name = row['Feature']
        try:
            parts = feature_name.rsplit('_', 1)
            if len(parts) == 2:
                modality = parts[0]
                parcel_idx = int(parts[1])
                if modality in modality_features:
                    modality_features[modality].append({
                        'parcel': parcel_idx,
                        'value': row[metric]
                    })
        except (ValueError, IndexError):
            pass

    # ==================== CREATE PARCEL ARRAYS ====================
    modality_data = {}
    modality_ranges = {}

    for modality in modalities:
        parcel_values = np.zeros(TOTAL_PARCELS)

        for feat in modality_features[modality]:
            parcel_idx = feat['parcel']
            if 0 <= parcel_idx < TOTAL_PARCELS:
                parcel_values[parcel_idx] = feat['value']

        modality_data[modality] = parcel_values

        nz = parcel_values[parcel_values > 0]
        if len(nz) > 0:
            mod_vmin = vmin if vmin is not None else np.min(nz)
            mod_vmax = vmax if vmax is not None else np.max(nz)
            modality_ranges[modality] = (mod_vmin, mod_vmax)
            print(f"{modality}: {len(nz)} parcels, range [{mod_vmin:.3f}, {mod_vmax:.3f}]")
        else:
            modality_ranges[modality] = (0, 1)
            print(f"{modality}: No features")

    # ==================== LOAD ATLAS ====================
    if atlas_path is None:
        atlas_path = os.path.join(
            'data/ADNI-B_DATA/N238rev',
            'Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'
        )

    if not os.path.exists(atlas_path):
        raise FileNotFoundError(f"Parcellation file not found: {atlas_path}")

    parcel_img = nib.load(atlas_path)
    parcel_data = parcel_img.get_fdata()

    # ==================== LOAD FSAVERAGE ====================
    print("Loading fsaverage surfaces...")
    fsaverage = datasets.fetch_surf_fsaverage()
    fsaverage_meshes = load_fsaverage()
    
    curv_sign = load_fsaverage_data(data_type="curvature")
    for hemi, data in curv_sign.data.parts.items():
        curv_sign.data.parts[hemi] = np.sign(data)

    # ==================== PLOT EACH MODALITY IN SEPARATE FIGURE ====================
    for modality in modalities:
        values = modality_data[modality]
        
        # Create volume for this modality
        parcel_values_full = np.zeros_like(parcel_data)
        for i in range(TOTAL_PARCELS):
            if values[i] > 0:
                nifti_label = i + 1
                parcel_values_full[parcel_data == nifti_label] = values[i]
        
        mod_img = nib.Nifti1Image(parcel_values_full, affine=parcel_img.affine)
        
        # Project to surface
        surface_img = SurfaceImage.from_volume(
            mesh=fsaverage_meshes["pial"],
            volume_img=mod_img,
        )
        
        texture_left = surface.vol_to_surf(mod_img, fsaverage.pial_left)
        texture_right = surface.vol_to_surf(mod_img, fsaverage.pial_right)
        
        mod_vmin, mod_vmax = modality_ranges[modality]
        cmap_name = modality_config[modality]['cmap']
        label = modality_config[modality]['label']
        
        # Create custom colormap: light grey at 0, transitioning to color at max
        original_cmap = plt.get_cmap(cmap_name)
        n_colors = 256
        colors = np.zeros((n_colors, 4))
        
        # Light grey color for zero (RGB: 0.85, 0.85, 0.85)
        grey_color = np.array([0.85, 0.85, 0.85, 1.0])

        # Adjust threshold: grey for first 5% of range, then transition to color
        grey_threshold = 0.05  # Grey up to 5% of the range

        for i in range(n_colors):
            t = i / (n_colors - 1)
            
            if t < grey_threshold:
            # Stay grey in the low range
                colors[i] = grey_color
            else:
            # Remap to color range, starting from the grey threshold
                color_t = (t - grey_threshold) / (1 - grey_threshold)
                colors[i] = original_cmap(color_t)
        
        custom_cmap = ListedColormap(colors)
        
        # ==================== CREATE FIGURE FOR THIS MODALITY ====================
        print(f"Creating figure for {modality}...")
        fig = plt.figure(figsize=(20, 5))
        # gs = GridSpec(1, 5, figure=fig,
        #               width_ratios=[1, 1, 1, 1, 0.15],
        #               hspace=0.1, wspace=0.05)
    
    # Adjusted GridSpec width_ratios to allocate space for the left subtitle
        gs = GridSpec(1, 6, figure=fig,
            width_ratios=[0.01, 1, 1, 1, 1, 0.15], # Further reduced left margin
            hspace=0.0, wspace=0.02) # Small wspace to bring plots closer

        # Calculate metric display string once
        metric_display = metric.replace('_', ' ').replace('%', '(%)')
        
        # 0. Subtitle/Metric Display Area (New addition)
        # Use the first column (index 0) of the GridSpec for the left text
        ax_metric = fig.add_subplot(gs[0, 0])
        
        # Place the text centered vertically in this subplot
        ax_metric.text(1.0, 0.5,  # Move text all the way to the right edge (1.0)
            label, 
            fontsize=14, 
            fontweight='bold', 
            ha='right',  # Align to right to bring closer to plot
            va='center', 
            rotation=90, # Rotate text for a vertical subtitle effect
            transform=ax_metric.transAxes)
            
        ax_metric.axis('off') # Hide the axes for a clean appearance
        
        # View 1: Posterior
        ax1 = fig.add_subplot(gs[0, 1], projection='3d')
        plotting.plot_surf_stat_map(
            stat_map=surface_img,
            surf_mesh=fsaverage_meshes['pial'],
            hemi='both',
            view='posterior',
            colorbar=False,
            cmap=custom_cmap,
            bg_map=curv_sign,
            vmin=0,
            vmax=mod_vmax,
            axes=ax1,
            darkness=0.5,
            title='Posterior'
        )
        
        # View 2: Left Lateral
        ax2 = fig.add_subplot(gs[0, 2], projection='3d')
        plotting.plot_surf_stat_map(
            fsaverage.pial_left,
            texture_left,
            hemi='left',
            title='Left Lateral',
            view='lateral',
            colorbar=False,
            cmap=custom_cmap,
            bg_map=fsaverage.sulc_left,
            vmin=0,
            vmax=mod_vmax,
            axes=ax2,
            darkness=0.5
        )
        
        # View 3: Anterior
        ax3 = fig.add_subplot(gs[0, 3], projection='3d')
        plotting.plot_surf_stat_map(
            stat_map=surface_img,
            surf_mesh=fsaverage_meshes['pial'],
            hemi='both',
            view='anterior',
            colorbar=False,
            cmap=custom_cmap,
            bg_map=curv_sign,
            vmin=0,
            vmax=mod_vmax,
            axes=ax3,
            darkness=0.5,
            title='Anterior'
        )
        
        # View 4: Right Lateral
        ax4 = fig.add_subplot(gs[0, 4], projection='3d')
        plotting.plot_surf_stat_map(
            fsaverage.pial_right,
            texture_right,
            hemi='right',
            title='Right Lateral',
            view='lateral',
            colorbar=False,
            cmap=custom_cmap,
            bg_map=fsaverage.sulc_right,
            vmin=0,
            vmax=mod_vmax,
            axes=ax4,
            darkness=0.5
        )
        
        # Colorbar
        # cbar_ax = fig.add_axes([0.88, 0.2, 0.015, 0.6])
        cbar_ax = fig.add_axes([0.88, 0.31, 0.015, 0.36]) 
        
        norm = Normalize(vmin=0, vmax=mod_vmax)
        sm = ScalarMappable(cmap=custom_cmap, norm=norm)
        sm.set_array([])
        
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.ax.tick_params(labelsize=11)
        cbar.locator = FixedLocator([0, mod_vmax/2, mod_vmax])
        cbar.update_ticks()
        
        # Set tick labels to bold
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight('bold')
        
        cbar.ax.set_title(f'FI', fontsize=12, pad=10, fontweight='bold')

        # ==================== TITLE ====================
        title_text = f'Feature Importance: {label}{title_suffix}'
        fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.98)
        
        # ==================== SAVE ====================
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            
            fig_name = f"brainmap_{modality}_{metric}.png"
            full_path = os.path.join(save_path, fig_name)
            
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Saved to: {full_path}")
        
        plt.show()
        plt.close(fig)
    
    return modality_data

brain_map_feature_importance2(
    df_com,
    modalities=['Tau','ABeta','I_norm2'],
    metric='Mean_Importance',
    vmin=0,
    vmax=0.025,
)
# brain_map_feature_importance(
#     df_com,
#     modalities=['Tau', 'ABeta',],
#     metric='Mean_Importance',
#     save_path='./results/brain_maps',
# )
# brain_map_feature_importance(
#     df_emp,
#     modalities=['I_norm2'],
#     metric='Mean_Importance',
#     save_path='./results/brain_maps',
# )