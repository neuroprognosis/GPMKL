# Individualized Gaussian Process-based Prediction of Memory Performance and Biomarker Status in Ageing and Alzheimer's disease.

This code is the official implementation of the paper:
> **Individualized Gaussian Process-based Prediction of Memory Performance and Biomarker Status in Ageing and Alzheimer's disease**<br>
> A. Nemali, ...,  E. Duzel, G. Ziegler<br>
> https://arxiv.org/pdf/

## Abstract
Neuroimaging markers based on Magnetic Resonance Imaging (MRI) combined
with various other measures (such as informative covariates, vascular risks, brain
activity, neuropsychological test etc.,) might provide useful predictions of clinical
outcomes during progression towards Alzheimer’s disease (AD). The Bayesian
approach aims to provide a trade-off by employing relevant features combina-
tions to build decision support systems in clinical settings where uncertainties
are relevant. We tested the approach in the MRI data across 959 subjects, aged
59-89 years and 453 subjects with available neuropsychological test scores and
CSF biomarker status (Aβ42/40 and pTau) from a large sample multi-centric
observational cohort (DELCODE). In order to explore the beneficial combina-
tions of information from different sources we presented an MRI-based predictive
modelling of memory performance and CSF biomarker status (positive or neg-
ative) in healthy ageing group as well as subjects at risk of Alzheimer’s disease
using a Gaussian process multikernel framework. Furthermore, we systemati-
cally evaluated predictive combinations of input feature sets and their model
variations, i.e. (A) combinations of tissue classes (GM, WM, CSF) and feature
type (modulated vs. unmodulated), choices of filter size of smoothing (ranging
from 0 to 15 mm FWHM), and image resolution (1mm, 2mm, 4mm and 8mm);
(B) incorporating demography and covariates (B) the impact of the size of the
training data set (i.e., number of subjects); (C) the influence of reducing the
dimensions of data and (D) choice of kernel types. Finally, the approach was
tested to reveal individual cognitive scores at follow-up (up to 4 years) using
the baseline features. The highest accuracy for memory performance prediction
was obtained for a combination of neuroimaging markers, demographics, ApoE4
and CSF-biomarkers explaining 57% of outcome variance in out of sample pre-
dictions. The best accuracy for Aβ42/40 status classification was achieved for
combination demographics, ApoE4 and memory score while usage of structural
MRI improved the classification of individual patient’s pTau status.

## Pre-Requisits

We tested our code on Python 3.8

The following packages should also be installed: <br>
certifi==2021.10.8 <br>
charset-normalizer==2.0.12 <br>
idna==3.3<br>
joblib==1.1.0<br>
nibabel==3.2.2<br>
nilearn==0.9.0<br>
numpy==1.22.2<br>
packaging==21.3<br>
pandas==1.4.1<br>
pyparsing==3.0.7<br>
python-dateutil==2.8.2<br>
pytz==2021.3<br>
requests==2.27.1<br>
scikit-learn==1.0.2<br>
scipy==1.8.0<br>
six==1.16.0<br>
threadpoolctl==3.1.0<br>
urllib3==1.26.8

If any of these packages are not installed on your computer, you can install them using the supplied `requirements.txt` file:<br>
```pip install -r requirements.txt```

## Implementation

We have included example scripts of the implementation in the examples folder 


## Citation
If you use this code for your research, please cite our paper.
```
Will be updated soon
```

[comment]: <> (```)

[comment]: <> (@article{DELCODEGPMKL,)

[comment]: <> (  title={Style{SDF}: {H}igh-{R}esolution {3D}-{C}onsistent {I}mage and {G}eometry {G}eneration},)

[comment]: <> (  author={Or-El, Roy and)

[comment]: <> (          Luo, Xuan and)

[comment]: <> (          Shan, Mengyi and)

[comment]: <> (          Shechtman, Eli and)

[comment]: <> (          Park, Jeong Joon and)

[comment]: <> (          Kemelmacher-Shlizerman, Ira},)

[comment]: <> (  journal={arXiv preprint arXiv:2112.11427},)

[comment]: <> (  year={2021})

[comment]: <> (})

[comment]: <> (```)
