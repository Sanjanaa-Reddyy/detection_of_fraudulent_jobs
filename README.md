The project focuses on implementing binary classification to detect fraudulent job postings. The selected dataset includes job descriptions and their associated metadata, where the job postings are labeled as 0 or 1 in the "fraudulent" column which serves as an indicator of whether each of the respective job postings is fraudulent or not. 

Objectives:
● Cleaning and preprocessing the data for quality and model training.
● Selecting key features relevant for predicting fraud in job postings.
● Implementing and evaluating 5 classification models using cross-validation for accuracy.
● Comparing algorithm performance based on accuracy, precision, recall, F1-score, specificity, AUC score. ANOVA test is used for statistical comparison of the performance metrics of the model.
● Choosing the two best-performing models and implementing them from scratch.
● Design the GUI of the application for creating a user-friendly application interface for detecting fraud based on job details.
● Design the pipelines for streamlining the preprocessing and model predictions in the backend.
● Integrate the front-end and back-end by defining API endpoints, creating components and making API calls to provide a robust platform for users to detect if a job is fraudulent.
● Testing and optimising the application for performance and usability.
The five models used are:
- XGBoost
- Random Forest
- Support vector Machine
- Logistic Regression
- Decision Tree

Out of the five Random Forest and XGBoost consistently showed the best results, especially in F1 score of 0.61–0.76 and AUC of 0.92–0.94 with recall values up to 0.89.