from src.models.gpr_mkl import GaussianProcessRegression
import numpy as np
import pandas as pd
from utils.kernels import Linear, RBF
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix
from utils.nifti_vector import read_nifti
from sklearn.model_selection import train_test_split

smooth = 4
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
predict_variable = 'memory'
correlation_score = []
r2_score = []


# Initialization a kernel
kernel = Linear(signal_variance=0.1, kernel_scaling=True)
# Initialization of Gaussian process MKL regression
gpr = GaussianProcessRegression(kernel, mkl_data, sub1[predict_variable], noise=0.2, optimization=True)
# Training the GPR MKL model
fit = gpr.fit(GM_train.index)
# Prediction - Test sample
predict_target = gpr.predict(GM_test.index)
true_target = pd.DataFrame(sub1[predict_variable], index=GM_test.index)

# Evaluation metrics
score  = gpr.score(true_target)
correlation_score.append(score[0])
r2_score.append(score[1])
# storing the result for plotting
result = pd.DataFrame()
result['Correlation'] = correlation_score
result['R2_score'] = r2_score
result.to_csv(save_path+'prediction_memory.csv')
