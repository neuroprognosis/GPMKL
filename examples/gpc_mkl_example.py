from src.models.gpc_mkl import GaussianProcessClassifier
import numpy as np
import pandas as pd
from utils.kernels import Linear, RBF
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix
from utils.nifti_vector import read_nifti
from sklearn.model_selection import train_test_split

smooth = 6
save_path = ''
absolute_path = ''
data_path = ''
mask = ''


# read data by feature (GM, WM, CSF brain tissue types)
GM, sub1 = read_nifti(absolute_path, data_path, 'wp1_*', mask, smooth)
WM, sub2 = read_nifti(absolute_path, data_path, 'wp2_*', mask, smooth)
CSF, sub2 = read_nifti(absolute_path, data_path, 'wp3_*', mask, smooth)


GM_train, GM_test = train_test_split(GM)
CSF_train = pd.DataFrame(CSF, index=GM_train.index)
CSF_test = pd.DataFrame(CSF, index=GM_test.index)

# data into List
mkl_data = [GM_train, CSF_train]
# target variable
predict_variable = 'abeta_c'
auc_score = []
specificity = []
sensitivity = []


# Initialization a kernel
kernel = Linear(signal_variance=0.1, kernel_scaling=True)
# Initialization of Gaussian process MKL classifier
gpc = GaussianProcessClassifier(kernel, mkl_data, sub1[predict_variable], optimization=True)
# Training the GPC MKL model
fit = gpc.fit(GM_train.index)
# Prediction - Test sample
mean_predict, predict_target = gpc.predict(GM_test.index)
true_target = pd.DataFrame(sub1[predict_variable], index=GM_test.index)
outer_true_predict = np.array(true_target).ravel()
# Probablity - Test sample
outer_prob = gpc.predict_probablity(GM_test.index)
# Evaluation metrics
fp, tp, thresholds = roc_curve(outer_true_predict, outer_prob)
optimal = np.argmax(tp - fp)
AUC = roc_auc_score(outer_true_predict, outer_prob)
pre, rec, thresholds = precision_recall_curve(outer_true_predict, outer_prob)
confusion = confusion = confusion_matrix(outer_true_predict, np.where(mean_predict > thresholds[optimal], 1, -1))
TP, TN, FP, FN = confusion[1, 1], confusion[0, 0], confusion[0, 1], confusion[1, 0]
auc_score.append(AUC)
specificity.append(TN / (TN + FP))
sensitivity.append(TP / float(FN + TP))

# storing the result for plotting
result = pd.DataFrame()
result['auc_score'] = auc_score
result['specificity'] = specificity
result['sensitivity'] = sensitivity
result.to_csv(save_path+'prediction_abeta.csv')
