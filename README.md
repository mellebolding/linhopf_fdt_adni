# linhopf_fdt_adni

This repository contains the code to produce the results presented in my [MSc thesis](MSc_Thesis.pdf) "An analytic application of the Fluctuation-Dissipation Theorem in Alzheimer’s Disease using the Linear Hopf Model". The thesis uses two approaches to compute the FDT violation metric: model-free and model-based analytical. The model used in this study is the linear Hopf model. Furthermore, a RF+SVM classifier is used in the analysis.

The repo contains 6 main scripts:
- compute_linhopf.py: implements the linear Hopf model (data from ADNI-B_DATA)
- compute_FDT_modelbased.py: takes output of Hopf model to compute analytical FDT violation (data from HOPF_DATA)
- compute_FDT_modelfree.py: uses the model-free approach to compute the FDT violation (data from ADNI-B_DATA)
- compute_analysis.py: plots the figures of the thesis and saves them in RESULT_PLOTS (data from FDT_DATA & abeta+tau from ADNI-B_DATA)
- compute_classifier.py: uses RF+SVM classifier to produce classification results (data from FDT_DATA & abeta+tau from ADNI-B_DATA)
- compute_classifier_analysis.py: plots the classifier figures of the thesis and saves them in teh RESULT_PLOTS (data from CLASSIFIER_DATA)

Other files included are:
- hyperparameters.json: model hyperparameters and RSN/parcel maps
- run_compute_linhopf.sh: bash file to run compute_linhopf.py on computing cluster
- src: includes helper functions, such as plotting functions (analysis), data loading functions (data_loaders), preprocessing functions (data_processing), and computation functions (functions_frameworks)
