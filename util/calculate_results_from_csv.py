import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, recall_score


epoch = 0
path_to_csv = './saved_models/methodnwhead_datasetchexpert_archdensenet121_pretrainedTrue_lr1e-05_bs64_projdim0_nshot2_nwayNone_wd0.0001_seed1964_classCardiomegaly'
overall_path = path_to_csv + '/output_csv/model_output_epoch_' + str(epoch) + '.csv'
# Load the CSV file into a pandas DataFrame
df = pd.read_csv(overall_path)

# Metrics for the whole dataset
y_true = df['Ground Truth']
y_pred = df['Prediction']
prob_class_1 = df['Probability Class 1']

balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, prob_class_1)
recall = recall_score(y_true, y_pred)

# Store overall metrics
overall_metrics = {
    'Balanced Accuracy': balanced_accuracy,
    'F1 Score': f1,
    'AUC': auc,
    'Recall': recall
}

# Calculate metrics for each gender subgroup
gender_metrics = {}

for gender in [0, 1]:  # 0 for Male, 1 for Female
    subgroup = df[df['Gender'] == gender]
    y_true_subgroup = subgroup['Ground Truth']
    y_pred_subgroup = subgroup['Prediction']
    prob_class_1_subgroup = subgroup['Probability Class 1']
    
    balanced_accuracy_subgroup = balanced_accuracy_score(y_true_subgroup, y_pred_subgroup)
    f1_subgroup = f1_score(y_true_subgroup, y_pred_subgroup)
    auc_subgroup = roc_auc_score(y_true_subgroup, prob_class_1_subgroup)
    recall_subgroup = recall_score(y_true_subgroup, y_pred_subgroup)
    
    gender_str = 'Female' if gender == 1 else 'Male'
    gender_metrics[gender_str] = {
        'Balanced Accuracy': balanced_accuracy_subgroup,
        'F1 Score': f1_subgroup,
        'AUC': auc_subgroup,
        'Recall': recall_subgroup
    }

# Equal Opportunity metric (difference in recall)
male_recall = gender_metrics['Male']['Recall']
female_recall = gender_metrics['Female']['Recall']
equal_opportunity_difference = female_recall - male_recall

# Demographic Parity Difference
male_positive_rate = df[df['Gender'] == 0]['Prediction'].mean()
female_positive_rate = df[df['Gender'] == 1]['Prediction'].mean()
demographic_parity_difference = female_positive_rate - male_positive_rate

# Disparate Impact
disparate_impact = female_positive_rate / male_positive_rate if male_positive_rate > 0 else 0

# Results
print(f"Overall Metrics: {overall_metrics}")
print(f"Gender Metrics: {gender_metrics}")
print(f"Equal Opportunity Difference: {equal_opportunity_difference}")
print(f"Demographic Parity Difference: {demographic_parity_difference}")
print(f"Disparate Impact: {disparate_impact}")
