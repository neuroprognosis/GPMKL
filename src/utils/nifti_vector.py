"""
 Author: Nemali Aditya <aditya.nemali@dzne.de>
==================
Conversion of nii image data to numpy data
==================
This module contains functions that recursively reads all subjects .nii images from a folder, filters image files,
applies specific mask, applies degree of smoothness to the data and results subject id, data matrix of all subjects
"""

from nilearn.input_data import MultiNiftiMasker
import os
import re
import numpy as np
import pandas as pd
from datetime import datetime



def read_nifti(absolute_path, data_path, regex_filter, mask=None, smooth=None):
     start_time = datetime.now()
     absolute_path = str(absolute_path)
     regex_filter = str(regex_filter)
     if (mask == None):
         niftimasker = MultiNiftiMasker(smoothing_fwhm=smooth, n_jobs=-2)
     else:
         mask = absolute_path + str(mask)
         niftimasker = MultiNiftiMasker(mask_img=mask, smoothing_fwhm=smooth, n_jobs=-2)
     # mask images and get data
     data_folder = absolute_path + str(data_path)
     list_path = []
     subjs = []

     for files in os.listdir(data_folder):
         if re.match(regex_filter, files):
             image_path = data_folder + files
             list_path.append(image_path)
             subject_name = os.path.basename(files).split('_')[3]
             subject_name = os.path.basename(subject_name).split('.')[0]
             subjs.append(subject_name)
     print("Reading "+ regex_filter +" volume data please wait.........")
     list_path = np.sort(list_path)
     subjs = np.sort(subjs)
     x = niftimasker.fit_transform(list_path)
     x = np.vstack(x)
     subjects_id  = pd.DataFrame()
     subjects_id['subjs'] = subjs
     data_matrix = pd.DataFrame(x, dtype='float64')
     end_time = datetime.now()
     time = end_time-start_time
     print("Total time in minutes :", float(time.total_seconds()/60))
     return data_matrix, subjects_id
