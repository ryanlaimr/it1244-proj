# Data Description:
- classification dataset
- around 2100 training samples and around 1000 test samples
- first 350 columns (`length_51`, `length_52`, ..., `length_400`) are features: `length_51` denotes the max normalized frequency of DNA fragment `length 51`
- last column (`class_label`) is the sample class: healthy/screening stage/early stage/mid stage/late stage cancer
- It's a class imbalanced dataset

# Task:
- need to classify cancer vs healthy
- need to try and do well for (1) screening stage cancer vs healthy and (2) early stage cancer vs healthy
- need to use appropriate metrics (not accuracy) to account for positive class accuracy and negative class accuracy

# Suggestions:
- perform appropriate normalization and feature extraction through data visualization
- design appropriate loss function for your model
- perform innovation in modeling this task through introduction of multiple models