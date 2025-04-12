import os 
import numpy as np 
import json 
import matplotlib.pyplot as plt
import pickle
from scipy.stats import norm
import logging
import itertools
from recalibration import get_std_recalibrator
import seaborn as sns
from tqdm import tqdm
import torch 
import math 
from matplotlib.ticker import MaxNLocator

models = ['actfound_fusion', 'actfound_transfer', 'ADKT-IFT', 'protonet', 'DKT', 'RF', 'GPST'] 
metric_name = "r2"
# metric_name = "rmse"
fsmol = dict()
for i in [16, 32, 64, 128]:
    fsmol[i] = dict()
    for x in models:
        with open(os.path.join("./test_results/result_indomain/fsmol", x, f"sup_num_{i}.json"), "r") as f:
            res = json.load(f)
        fsmol[i][x] = []
        for k in res:
            d = np.mean([float(data[metric_name]) for data in res[k]])
            fsmol[i][x].append(d)

def process_data(folder_size, method, mode = 'test'):
    """
    Loads and aggregates prediction, variance, and true value data from multiple folds.
    """
    all_predictions = []
    all_variances = []
    all_true_values = []

    for fold in range(10):
        pred_file = f'./fsmol_archive/{folder_size}/Fold_{fold}/{mode}_{method}_prediction_dic.pkl'
        label_file = f'./fsmol_archive/{folder_size}/Fold_{fold}/{mode}_label_dic.pkl'

        with open(pred_file, 'rb') as f:
            dic = pickle.load(f)
        with open(label_file, 'rb') as f:
            label_dic = pickle.load(f)

        for assay_id in dic.keys():
            prediction_data = dic[assay_id].get('prediction')
            variance_data = dic[assay_id].get('variance')
            y_true = label_dic.get(assay_id)

            predictions = np.mean(np.array(prediction_data)[:20], axis=0)
            variances = np.mean(np.array(variance_data)[:20], axis=0)
            if not np.all(variances > 0):
                continue
            all_predictions.extend(predictions[:1000])
            all_variances.extend(variances[:1000])
            all_true_values.extend(y_true[:1000])
    return np.array(all_predictions), np.sqrt(np.array(all_variances)), np.array(all_true_values)

recaliarate_model_dic_all = dict()
shots = [16, 32, 64, 128]
model_ls =  ['FP', 'FPaugment', 'FPaugmentRGB', 'FPRGB', 'GraphGAT', 'GraphGIN', 'GraphSAGE', 'GraphGCN']
for shot in shots:
    recaliarate_model_dic_all[shot] = dict()
    for model in tqdm(model_ls):
        predictions_val, std_val, true_values_val = process_data(shot, model, mode='valid')
        std_recalibrator = get_std_recalibrator(predictions_val, std_val, true_values_val)
        recaliarate_model_dic_all[shot][model] = std_recalibrator

shots = [16, 32, 64, 128]
model_pred_dist = dict()
model_uncertainty_dist = dict()
for shot in shots:
    model_pred_dist[shot] = dict()
    model_uncertainty_dist[shot] = dict()
    r2_fold_ls = []
    r2_fold_ls_dynamic = []
    mse_fold_ls = []
    # mode = 'valid'
    mode = 'test'
    model_ls =  ['FP', 'FPaugment', 'FPaugmentRGB', 'FPRGB', 'GraphGAT', 'GraphGIN', 'GraphSAGE', 'GraphGCN']
    recaliarate_model_dic = recaliarate_model_dic_all[shot]
    # model_ls = ['FP', 'FPRGB', 'GraphSAGE', 'FPaugment', 'GraphGAT']
    for fold_id in range(10):
        model_pred_dist[shot][fold_id] = dict()
        model_uncertainty_dist[shot][fold_id] = dict()
        with open(f'../Result_for_publication/fsmol_archive/{shot}/Fold_{fold_id}/{mode}_label_dic.pkl', 'rb') as f:
            label_dic = pickle.load(f)
        model_dic = dict()
        for model in model_ls:
            with open(f'../Result_for_publication/fsmol_archive/{shot}/Fold_{fold_id}/{mode}_{model}_prediction_dic.pkl', 'rb') as f:
                prediction_dic_temp = pickle.load(f)
            model_dic[model] = prediction_dic_temp
        r2_all = []
        r2_all_dynamic = []
        for assay_id in label_dic:
            label = label_dic[assay_id]
            prediction_per_model_dic = {}
            for idx, model in enumerate(model_ls):
                prediction_per_model_dic[model] = {}
                model_prediction_temp = []
                model_prediction_temp_sharpley = []
                model_pred_uncertainty = []
                # num_predictions = len(model_dic[model][assay_id]['prediction'])
                num_predictions = 20
                for i in range(num_predictions):
                    model_prediction_temp_sharpley.append(model_dic[model][assay_id]['prediction'][i])
                    model_prediction_temp.append(model_dic[model][assay_id]['prediction'][i])
                    model_pred_uncertainty.append(model_dic[model][assay_id]['variance'][i])
                model_prediction_temp = np.mean(np.array(model_prediction_temp), axis=0)
                model_prediction_temp_sharpley = np.mean(np.array(model_prediction_temp_sharpley), axis=0)
                model_pred_uncertainty = np.mean(np.array(model_pred_uncertainty), axis=0)
                prediction_per_model_dic[model]['prediction'] = model_prediction_temp
                prediction_per_model_dic[model]['prediction_dynamic'] = model_prediction_temp_sharpley
                prediction_per_model_dic[model]['variance'] = model_pred_uncertainty
            prediction_all = []
            prediction_all_dynamic = []
            uncertainty_all = []
            model_pred_dist[shot][fold_id][assay_id] = []
            for model in model_ls:
                prediction_all.append(prediction_per_model_dic[model]['prediction'])
                r2_temp = np.corrcoef(label, prediction_per_model_dic[model]['prediction'])[0,1] ** 2
                model_pred_dist[shot][fold_id][assay_id].append(r2_temp)
                prediction_all_dynamic.append(prediction_per_model_dic[model]['prediction_dynamic'])
                uncertainty_recailbrated = recaliarate_model_dic[model](prediction_per_model_dic[model]['variance'])
                # uncertainty_recailbrated = np.clip(uncertainty_recailbrated, 0.001, 3)
                # uncertainty_all.append(1 / uncertainty_recailbrated ** 4)
                uncertainty_all.append(np.exp(-5 * uncertainty_recailbrated))
            prediction_all = np.mean(np.array(prediction_all), axis=0)
            prediction_all_dynamic = np.array(prediction_all_dynamic)
            uncertainty_all = np.array(uncertainty_all)
            uncertainty_sum = np.sum(uncertainty_all, axis=0)
            uncertainty_norm = uncertainty_all/uncertainty_sum
            model_uncertainty_dist[shot][fold_id][assay_id] = uncertainty_norm
            prediction_all_dynamic = np.diag(uncertainty_norm.T.dot(prediction_all_dynamic))
            r2 = np.corrcoef(label, prediction_all)[0,1]
            r2_dynamic = np.corrcoef(label, prediction_all_dynamic)[0,1]
            if r2 < 0:
                r2 = 0
            else:
                r2 = r2**2
            if r2_dynamic < 0:
                r2_dynamic = 0
            else:
                r2_dynamic = r2_dynamic**2
            r2_all.append(r2)
            r2_all_dynamic.append(r2_dynamic)
        r2_fold_ls.append(r2_all)
        r2_fold_ls_dynamic.append(r2_all_dynamic)
    fsmol[shot]['EDKT_dynamic'] = np.mean(r2_fold_ls_dynamic, axis=0).tolist()
    fsmol[shot]['EDKT_average'] = np.mean(r2_fold_ls, axis=0).tolist()

sharpley_dic = dict()
for shot in shots:
    sharpley_dic[shot] = dict()
    def shapley_value_func(model_ls, reference_ls):
        if len(model_ls) == 0:
            return 0
        return_ls = []
        for fold_id in range(10):
            r2_ls = []
            included_model_result = dict()
            with open(f'../Result_for_publication/fsmol/{shot}/Fold_{fold_id}/valid_label_dic.pkl', 'rb') as f:
                label_dic = pickle.load(f)
            for model in model_ls:
                ## load model prediction dic
                with open(f'../Result_for_publication/fsmol/{shot}/Fold_{fold_id}/valid_{model}_prediction_dic.pkl', 'rb') as f:
                    prediction_dic = pickle.load(f)
                included_model_result[model] = prediction_dic
            for assay_id in label_dic:
                label = label_dic[assay_id]
                prediction = 0
                for model in included_model_result:
                    # model_num = len(included_model_result[model][assay_id]['prediction'])
                    model_num = 20
                    model_pred = np.vstack([included_model_result[model][assay_id]['prediction'][i] for i in range(model_num)])
                    model_pred_mean = np.mean(model_pred, axis=0)
                    prediction += model_pred_mean
                r2_ls.append(np.corrcoef(label, prediction)[0,1]**2)
            return_ls.append(np.mean(r2_ls))
        return np.exp(np.mean(np.array(return_ls)) * 20)
        # return np.mean(np.array(return_ls))

    def calculate_shapley(target_index, index_list, value_function):
        """
        Calculate Shapley value for a target index given a list of indices
        
        Args:
            target_index: The index to calculate Shapley value for
            index_list: List of all indices
            value_function: Function that takes a list of indices and returns a value
        
        Returns:
            Shapley value for the target index
        """
        import itertools
        
        n = len(index_list)
        shapley_value = 0
        
        # Remove target_index from index_list
        other_indices = [idx for idx in index_list if idx != target_index]
        
        # Consider all possible coalition sizes
        for size in range(len(other_indices) + 1):
            # Get all possible coalitions of current size
            for coalition in itertools.combinations(other_indices, size):
                coalition = list(coalition)
                # Calculate marginal contribution
                val_with = value_function(coalition + [target_index], index_list)
                val_without = value_function(coalition, index_list)
                marginal = val_with - val_without
                # Calculate weight for this coalition size
                weight = (math.factorial(size) * math.factorial(n - size - 1)) / math.factorial(n)
                shapley_value += marginal * weight
        return shapley_value
    def normalize_list(lst):
        total = np.sum(lst)
        return [x/total for x in lst]
    included_model = ['FP', 'FPaugment', 'FPaugmentRGB', 'FPRGB', 'GraphGAT', 'GraphGIN', 'GraphSAGE', 'GraphGCN']
    # included_model = ['FP', 'FPRGB', 'GraphSAGE', 'GraphGCN', 'GraphGAT', 'GraphGIN', 'FPaugment']
    sharpley_ls_origin = []
    sharpley_ls = []
    for target_model in included_model:
        sharpley_value = calculate_shapley(target_model, included_model, shapley_value_func)
        print(f"Shapley value for {target_model}:{sharpley_value}")
        sharpley_ls_origin.append(sharpley_value)
        if sharpley_value < 0:
            sharpley_value = 0
        sharpley_ls.append(sharpley_value)
    for target_model, value_norm, value in zip(included_model, normalize_list(sharpley_ls), sharpley_ls_origin):
        sharpley_dic[shot][target_model] = (value_norm, value)

# Set style and color parameters
plt.style.use('default')
sns.set_theme(style="whitegrid")
colors = sns.color_palette("husl", 8)

shots = [16, 32, 64, 128]
model_names = ['ECFP', 'ECFP/MACCS', 'ECFP/MACCS with RBF', 'ECFP with RBF', 'GAT', 'GIN', 'SAGE', 'GCN']

# Create figure - remove the legend space from GridSpec
fig = plt.figure(figsize=(20, 20), dpi=300)  # Adjusted figure size
gs = plt.GridSpec(3, 4, figure=fig)  # Changed to 3 rows instead of 4

# Create a custom legend
legend_ax = fig.add_subplot(gs[1, :])
legend_ax.axis('off')
legend_handles = [plt.Rectangle((0,0),1,1, fc=colors[i]) for i in range(len(model_names))]
legend_ax.legend(legend_handles, model_names, loc='center', ncol=4, 
                frameon=True, fontsize=12, title='Model Types',
                title_fontsize=14)

# Prepare data structure for plots
all_results = {
    'performance': np.zeros((len(model_names), len(shots))),
    'uncertainty': np.zeros((len(model_names), len(shots))),
    'shap': np.zeros((len(model_names), len(shots)))
}

# Fill data structures
for idx, shot in enumerate(shots):
    per_model_result = []
    per_model_result_uncertainty = []
    for assay_id in label_dic:
        fold_result = []
        fold_result_uncertainty = []
        for fold_id in range(10):
            fold_result.append(model_pred_dist[shot][fold_id][assay_id])
            fold_result_uncertainty.append(model_uncertainty_dist[shot][fold_id][assay_id].mean(axis=1))
        fold_result = np.mean(np.array(fold_result), axis=0)
        per_model_result.append(fold_result)
        fold_result_uncertainty = np.mean(np.array(fold_result_uncertainty), axis=0)
        per_model_result_uncertainty.append(fold_result_uncertainty)
    
    per_model_result = np.mean(np.array(per_model_result), axis=0)
    per_model_result_uncertainty = np.mean(np.array(per_model_result_uncertainty), axis=0)
    
    # Store results
    all_results['performance'][:, idx] = per_model_result
    all_results['uncertainty'][:, idx] = per_model_result_uncertainty
    all_results['shap'][:, idx] = [sharpley_dic[shot][model_name][0] for model_name in model_ls]

def add_labels_to_polar(ax, angles, values, labels, offset=0.1):
    """Helper function to add labels to polar plot with smart positioning"""
    max_value = max(values)
    for angle, value, label in zip(angles[:-1], values[:-1], labels):
        # Calculate label position
        angle_deg = np.rad2deg(angle)
        if angle_deg >= 0 and angle_deg <= 45:
            ha, va = 'left', 'bottom'
        elif angle_deg > 45 and angle_deg <= 135:
            ha, va = 'left', 'center'
        elif angle_deg > 135 and angle_deg <= 225:
            ha, va = 'right', 'center'
        elif angle_deg > 225 and angle_deg <= 315:
            ha, va = 'right', 'center'
        else:
            ha, va = 'right', 'bottom'
            
        # Add offset to avoid overlapping
        label_radius = value + max_value * offset
        
        # Add value annotation
        ax.text(angle, label_radius, f'{value:.3f}', 
                ha=ha, va=va, fontsize=8, rotation=np.rad2deg(angle)-90)

# Plotting
for idx, shot in enumerate(shots):
    # Performance Plot (Polar)
    ax1 = fig.add_subplot(gs[0, idx], projection='polar')
    
    # Prepare data for radar plot
    angles = np.linspace(0, 2*np.pi, len(model_names), endpoint=False)
    values = all_results['performance'][:, idx]
    
    # Close the plot by appending the first value
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
    # Plot the radar chart
    ax1.plot(angles, values, 'o-', linewidth=2, color='blue')
    ax1.fill(angles, values, alpha=0.25, color='blue')
    
    # Set the labels with rotation
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(model_names, size=8)
    ax1.set_ylim(all_results['performance'][:, idx].min() - 0.02, all_results['performance'][:, idx].max() + 0.02)
    
    # Add value annotations with smart positioning
    add_labels_to_polar(ax1, angles, values, model_names)
    
    # ax1.set_title(f'{shot}-shot Learning\nModel Performance (R²)', 
    #              fontsize=14, fontweight='bold', pad=20)
    
    # Add gridlines
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Uncertainty Plot (Line plot)
    ax2 = fig.add_subplot(gs[2, idx])
    x = np.arange(len(model_names))
    ax2.plot(x, all_results['uncertainty'][:, idx], 'o-', color='red')
    ax2.fill_between(x, 0, all_results['uncertainty'][:, idx], alpha=0.2, color='red')
    
    # Customize uncertainty plot
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right', size=8)
    ax2.set_title(f'{shot}-shot Learning\nUncertainty Analysis', 
                 fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Weight', fontsize=12, fontweight='bold')
    ax2.set_ylim(0.07, 0.165)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add value annotations
    for i, value in enumerate(all_results['uncertainty'][:, idx]):
        ax2.text(i, value + 0.01, f'{value:.3f}', 
                ha='center', va='bottom', fontsize=8)

    # SHAP Values Plot (Polar)
    ax3 = fig.add_subplot(gs[1, idx], projection='polar')
    
    # Prepare data for radar plot
    values = all_results['shap'][:, idx]
    values = np.concatenate((values, [values[0]]))
    
    # Plot the radar chart
    ax3.plot(angles, values, 'o-', linewidth=2, color='green')
    ax3.fill(angles, values, alpha=0.25, color='green')
    
    # Set the labels
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(model_names, size=8)
    
    # Add value annotations with smart positioning
    add_labels_to_polar(ax3, angles, values, model_names, offset=0.15)
    
    ax3.set_title(f'{shot}-shot Learning\nSHAP Values', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add gridlines
    ax3.grid(True, linestyle='--', alpha=0.7)

# Add a super title
# fig.suptitle('Comprehensive Model Analysis Across Different Shot Learning Scenarios', 
#             fontsize=16, fontweight='bold', y=0.95)

plt.savefig('performance_analysis.pdf', 
            bbox_inches='tight',
            dpi=300,
            format='pdf')

# Adjust layout
plt.tight_layout()
plt.show()

# Data preparation
shot_sizes = [16, 32, 64, 128]
# models = ['EDKT_sharpley', 'EDKT_average', 'EDKT_FP', 'EDKT_GraphGAT', 'EDKT_GraphSAGE', 'EDKT_GraphGIN', 'actfound_fusion', 'actfound_transfer', 'DKT']
models = ['EDKT_dynamic', 'EDKT_average','actfound_fusion', 'actfound_transfer', 'ADKT-IFT', 'protonet', 'DKT']
# Create color palette (from light to dark)
# colors = ['#c3e6c4', '#97d5bb', '#6cc4b9', '#45b4c2', '#3182bd', '#756bb1', '#b8860b']
# colors = ['#c3e6c4', '#97d5bb', '#6cc4b9', '#45b4c2', '#3182bd', '#756bb1', '#b8860b', '#008080', '#800020']
colors = ['#c3e6c4', '#97d5bb', '#6cc4b9', '#45b4c2', '#3182bd', '#756bb1', '#b8860b']

# Assuming fsmol is your data dictionary with structure fsmol[shot_size][model] = list_of_values
# Calculate means and standard errors
means = {shot: [np.mean(fsmol[shot][model]) for model in models] for shot in shot_sizes}
sems = {shot: [np.std(fsmol[shot][model])/np.sqrt(len(fsmol[shot][model])) for model in models] 
        for shot in shot_sizes}

# Reset to default style
plt.style.use('default')

# Manual style settings for publication quality
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8,
    'axes.linewidth': 0.5,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'legend.frameon': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.axisbelow': True
})

# Data preparation
shot_sizes = [16, 32, 64, 128]
models = ['EDKT_dynamic', 'EDKT_average', 'actfound_fusion', 
          'actfound_transfer', 'ADKT-IFT', 'protonet', 'DKT']

# Color palette (Nature style)
colors = ['#2166AC', '#4393C3', '#92C5DE', 
          '#D1E5F0', '#F7F7F7', '#FDDBC7', '#F4A582']

# Create figure with the right aspect ratio
fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300, facecolor='white')

# Bar width and positions
bar_width = 0.11
x = np.arange(len(shot_sizes))

# Plot bars for each model
for idx, model in enumerate(models):
    model_means = [means[shot][idx] for shot in shot_sizes]
    model_sems = [sems[shot][idx] for shot in shot_sizes]
    
    ax.bar(x + idx*bar_width, 
           model_means,
           bar_width,
           color=colors[idx],
           label=model,
           yerr=model_sems,
           capsize=2,
           error_kw={'linewidth': 0.5, 
                    'capthick': 0.5})

# Customize axes
ax.set_ylabel('Coefficient of determination (r²)', 
             labelpad=10)
ax.set_ylim(0.1, 0.51)
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

# Customize x-axis
ax.set_xticks(x + (len(models)-1)*bar_width/2)
ax.set_xticklabels(['16-shot', '32-shot', '64-shot', '128-shot'])
ax.set_xlabel('Number of training examples', labelpad=10)

# Title
# ax.set_title('FS-Mol Performance Comparison', 
#              pad=20, fontsize=10)

# Legend
legend = ax.legend(bbox_to_anchor=(0.5, -0.25),
                  loc='lower center',
                  borderaxespad=0,
                  ncol=4,
                  columnspacing=1)

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

# Adjust layout
plt.tight_layout()

# Save with high DPI
plt.savefig('fsmol_comparison.pdf', 
            bbox_inches='tight',
            dpi=300,
            format='pdf')
# plt.savefig('fsmol_comparison.png', 
#             bbox_inches='tight',
#             dpi=300)

plt.show()


def get_pqsar_average_result(group_id):
    pqsar_result = dict()
    pqsar_result['valid'] = dict()
    pqsar_result['test'] = dict()
    for file in os.listdir(f'./pQSAR/group_{group_id}'):
        if 'prediction' in file:
            model_name = file.split('_')[1]
            with open(f'./pQSAR/group_{group_id}/' + file, 'rb') as f:
                pqsar_result[file.split('_')[0]][model_name] = pickle.load(f)
        elif 'label' in file:
            if 'valid' in file:
                continue
            with open(f'./pQSAR/group_{group_id}/' + file, 'rb') as f:
                pqsar_result['label'] = pickle.load(f)
    r2_ls = []
    for assay_id in pqsar_result['label']:
        label = pqsar_result['label'][assay_id]
        predictions = []
        for model in pqsar_result['test']:
            predictions.append(np.mean(np.array(pqsar_result['test'][model][assay_id]['prediction'][:20]), axis=0))
        prediction_all = np.mean(np.vstack(predictions), axis=0)
        r2 = np.corrcoef(label, prediction_all)[0,1]**2
        r2_ls.append(r2)
    rmse_ls = []
    for assay_id in pqsar_result['label']:
        label = pqsar_result['label'][assay_id]
        predictions = []
        for model in pqsar_result['test']:
            predictions.append(np.mean(np.array(pqsar_result['test'][model][assay_id]['prediction'][:20]), axis=0))
        prediction_all = np.mean(np.vstack(predictions), axis=0)
        # Corrected RMSE calculation
        rmse = np.sqrt(np.mean((label - prediction_all)**2))
        rmse_ls.append(rmse)
    return r2_ls, rmse_ls

valid_result = dict()
valid_result['valid'] = dict()
valid_result['test'] = dict()
group_id = 1
for file in os.listdir(f'./pQSAR/group_{group_id}'):
    if 'prediction' in file:
        model_name = file.split('_')[1]
        with open(f'./pQSAR/group_{group_id}/' + file, 'rb') as f:
            valid_result[file.split('_')[0]][model_name] = pickle.load(f)
    elif 'label' in file:
        if 'test' in file:
            continue
        with open(f'./pQSAR/group_{group_id}/' + file, 'rb') as f:
            valid_result['label'] = pickle.load(f)

std_recalibrator_pqsar_dict = dict()
for model in valid_result['valid']:
    predictions_val = []
    std_val = []
    true_values_val = []
    for assay_id in valid_result['label']:
        label = valid_result['label'][assay_id].tolist()
        predictions_val.extend(np.mean(np.array(valid_result['valid'][model][assay_id]['prediction'][:20]), axis=0).tolist())
        std_val.extend(np.mean(np.array(valid_result['valid'][model][assay_id]['variance'][:20]), axis=0).tolist())
        true_values_val.extend(label)
    std_recalibrator_pqsar_dict[model]= get_std_recalibrator(np.array(predictions_val), np.array(std_val), np.array(true_values_val))

def get_pqsar_dynamic_result(group_id):
    pqsar_result = dict()
    pqsar_result['valid'] = dict()
    pqsar_result['test'] = dict()
    for file in os.listdir(f'./pQSAR/group_{group_id}'):
        if 'prediction' in file:
            model_name = file.split('_')[1]
            with open(f'./pQSAR/group_{group_id}/' + file, 'rb') as f:
                pqsar_result[file.split('_')[0]][model_name] = pickle.load(f)
        elif 'label' in file:
            if 'valid' in file:
                continue
            with open(f'./pQSAR/group_{group_id}/' + file, 'rb') as f:
                pqsar_result['label'] = pickle.load(f)
    r2_ls = []
    for assay_id in pqsar_result['label']:
        label = pqsar_result['label'][assay_id]
        predictions = []
        variances = []
        model_num = 0
        for model in pqsar_result['test']:
            predictions.append(np.mean(np.array(pqsar_result['test'][model][assay_id]['prediction'][:20]), axis=0))
            variances_temp = np.mean(np.array(pqsar_result['test'][model][assay_id]['variance'][:20]), axis=0)
            # variances_calibrated = std_recalibrator_pqsar_dict[model](variances_temp)
            variances.append(np.exp(-5 * variances_temp))

            model_num += 1
        # prediction_all = np.mean(np.vstack(predictions), axis=0)
        predictions = np.array(predictions)
        variances = np.array(variances)
        variances_sum = np.sum(variances, axis=0)
        variances_norm = variances/variances_sum
        prediction_all = np.diag(variances_norm.T.dot(predictions))
        r2 = np.corrcoef(label, prediction_all)[0,1]**2
        r2_ls.append(r2)
    rmse_ls = []
    for assay_id in pqsar_result['label']:
        label = pqsar_result['label'][assay_id]
        predictions = []
        variances = []
        for model in pqsar_result['test']:
            predictions.append(np.mean(np.array(pqsar_result['test'][model][assay_id]['prediction'][:20]), axis=0))
            variances_temp = np.mean(np.array(pqsar_result['test'][model][assay_id]['variance'][:20]), axis=0)
            # variances_calibrated = std_recalibrator_pqsar_dict[model](variances_temp)
            variances.append(np.exp(-5 * variances_temp))
        # prediction_all = np.mean(np.vstack(predictions), axis=0)
        predictions = np.array(predictions)
        variances = np.array(variances)
        variances_sum = np.sum(variances, axis=0)
        variances_norm = variances/variances_sum
        prediction_all = np.diag(variances_norm.T.dot(predictions))
        # Corrected RMSE calculation
        rmse = np.sqrt(np.mean((label - prediction_all)**2))
        rmse_ls.append(rmse)
    return r2_ls, rmse_ls

my_model_result = get_pqsar_average_result(1)
my_model_result_dynamic = get_pqsar_dynamic_result(1)

models = ['actfound', 'actfound_fusion', 'actfound_transfer', 'protonet', 'DKT', 'RF', 'GPST']
pqsar = dict()
for x in models:
    pqsar[x] = dict()
    metric_name = "r2"
    with open(os.path.join("./test_results/result_indomain/pqsar", x, f"sup_num_0.75.json"), "r") as f:
        res = json.load(f)
    pqsar[x]['r2'] = []
    for k in res:
        d = np.mean([float(data[metric_name]) for data in res[k]])
        pqsar[x]['r2'].append(d)
    metric_name = "rmse"
    with open(os.path.join("./test_results/result_indomain/pqsar", x, f"sup_num_0.75.json"), "r") as f:
        res = json.load(f)
    pqsar[x]['rmse'] = []
    for k in res:
        d = np.mean([float(data[metric_name]) for data in res[k]])
        pqsar[x]['rmse'].append(d)

all_result = dict()
for model in pqsar:
    all_result[model] = dict()
    # print(model, np.mean(pqsar[model]))
    all_result[model]['r2'] = np.mean(pqsar[model]['r2'])
    all_result[model]['r2_std'] = np.std(pqsar[model]['r2'] / np.sqrt(len(pqsar[model]['r2'])))
    all_result[model]['rmse'] = np.mean(pqsar[model]['rmse'])
    all_result[model]['rmse_std'] = np.std(pqsar[model]['rmse'] / np.sqrt(len(pqsar[model]['rmse'])))
all_result['EDKT_average'] = dict()
all_result['EDKT_average']['r2'] = np.mean(my_model_result[0])
all_result['EDKT_average']['r2_std'] = np.std(my_model_result[0]) / np.sqrt(len(my_model_result[0]))
all_result['EDKT_average']['rmse'] = np.mean(my_model_result[1])
all_result['EDKT_average']['rmse_std'] = np.std(my_model_result[1]) / np.sqrt(len(my_model_result[1]))
all_result['EDKT_dynamic'] = dict()
all_result['EDKT_dynamic']['r2'] = np.mean(my_model_result_dynamic[0])
all_result['EDKT_dynamic']['r2_std'] = np.std(my_model_result_dynamic[0]) / np.sqrt(len(my_model_result_dynamic[0]))
all_result['EDKT_dynamic']['rmse'] = np.mean(my_model_result_dynamic[1])
all_result['EDKT_dynamic']['rmse_std'] = np.std(my_model_result_dynamic[1]) / np.sqrt(len(my_model_result_dynamic[1]))

# Reset to default style
plt.style.use('default')

# Manual style settings for publication quality
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8,
    'axes.linewidth': 0.5,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'legend.frameon': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.axisbelow': True
})

# Data preparation
models = ['EDKT_dynamic','EDKT_average', 'actfound', 'actfound_fusion', 'actfound_transfer', 
          'protonet', 'DKT', 'RF', 'GPST']
values_r2 = []
errors_r2 = []
values_rmse = []
errors_rmse = []

for model in models:
    values_r2.append(all_result[model]['r2'])
    errors_r2.append(all_result[model]['r2_std'])
    values_rmse.append(all_result[model]['rmse'])
    errors_rmse.append(all_result[model]['rmse_std'])

# Create gradient colors
colors = plt.cm.GnBu(np.linspace(0.3, 0.9, len(values_r2)))

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 3), dpi=300, facecolor='white')

# Bar width and positions
bar_width = 1
x = np.arange(len(models))

# Plot R² (left subplot)
bars1 = ax1.bar(x, values_r2, bar_width, color=colors)
ax1.errorbar(x, values_r2, yerr=errors_r2, fmt='none', color='black', 
             capsize=3, capthick=1, linewidth=1)

# Plot RMSE (right subplot)
bars2 = ax2.bar(x, values_rmse, bar_width, color=colors)
ax2.errorbar(x, values_rmse, yerr=errors_rmse, fmt='none', color='black', 
             capsize=3, capthick=1, linewidth=1)

# Customize axes for both subplots
for ax in [ax1, ax2]:
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

# Set specific y-limits and labels
ax1.set_ylim(0.15, 0.55)
ax1.set_ylabel('R²', rotation=90, labelpad=10, ha='right', fontsize=5)
ax2.set_ylim(0.65, 1.05)
# Adjust RMSE y-limits based on your data
ax2.set_ylabel('RMSE', rotation=90, labelpad=10, ha='right', fontsize=5)

# Remove subplot titles
ax1.set_title('')
ax2.set_title('')

# Add main title if needed
# fig.suptitle('pQSAR-ChEMBL', y=1, fontsize=7)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save with high DPI
plt.savefig('pQSAR_comparison.pdf', 
            bbox_inches='tight',
            dpi=300,
            format='pdf')
# plt.savefig('model_comparison.png', 
#             bbox_inches='tight',
#             dpi=300)

plt.show()