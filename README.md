# Campus Noise Analyzer: Supervised ML Project

**1. Introduction**

In modern campus environments, monitoring sound levels is essential to ensure a comfortable and productive atmosphere for students and staff. This project analyzes sound patterns across various campus locations and classifies them into three categories: Quiet, Moderate, and Loud. Using machine learning techniques, we aim to provide a data-driven approach to understanding noise patterns, helping campus administrators make informed decisions about space utilization, noise mitigation, and overall campus planning.

**2. Data Collection**

Sound data was collected across five campus locations, totaling 267 measurements. Each measurement includes key attributes such as:

timestamp: Date and time of the measurement

location: Specific campus area

decibel_level: Measured sound intensity in decibels (dB)

area_size: Size of the location (Small, Medium, Large)

type: Whether the location is Indoor or Outdoor

zone_type: Categorized noise level (Quiet, Moderate, Loud)

Dataset Overview:
<img width="859" height="520" alt="image" src="https://github.com/user-attachments/assets/b48b66cd-0202-49fd-b74c-58b875ff51e3" />


Observations:

Outdoor locations tend to be noisier on average compared to indoor areas.

The Commons Cafeteria exhibits the highest sound levels, while Sondheim Hall is the quietest area.

University Center shows the highest variability in sound levels, indicating fluctuating activity patterns.

**3. Tools & Libraries Used**

The following tools and libraries were used for data processing, feature engineering, and machine learning:

Python 3

pandas, numpy

matplotlib, seaborn

scikit-learn (Decision Tree, Random Forest, Logistic Regression)

**4. Feature Engineering**

To improve the predictive power of our models, we engineered several features from the raw data:

hour_of_day: Extracted from the timestamp, representing the hour (0–23). Captures diurnal variations in sound levels.

day_of_week: Extracted from the timestamp, numerically representing the day (0 = Monday, 6 = Sunday). Helps capture weekly patterns.

is_indoor: Binary indicator (1 = Indoor, 0 = Outdoor). Distinguishes locations with different acoustics.

area_encoded: Numerical encoding of area size (Small = 0, Medium = 1, Large = 2). Accounts for how the size of a space may affect sound propagation.

decibel_level: The original measured sound intensity in dB, included as a key predictive variable.

By incorporating time, location, and physical characteristics, these features provided meaningful inputs to the machine learning models.

**5. Classification Algorithms Used**

The project used three popular supervised learning algorithms for multi-class classification:

Decision Tree – A tree-based model that splits the data based on feature thresholds. Easy to interpret and visualizable.

Random Forest – An ensemble of decision trees that improves generalization and reduces overfitting.

Logistic Regression – A linear model adapted for multi-class classification using a one-vs-rest strategy. Performs well for linearly separable data.

**6. Performance Comparison**

The models were evaluated using accuracy, precision, recall, and F1-score. A train-test split of 75%-25% was used, with feature scaling applied.

6.1 Decision Tree

Accuracy: 91.04%

Precision: 0.9104

Recall: 0.9104

F1-Score: 0.9104

Confusion Matrix:

[[12  1  0]
 [ 1 23  2]
 [ 0  2 26]]


6.2 Random Forest

Accuracy: 91.04%

Precision: 0.9104

Recall: 0.9104

F1-Score: 0.9104

Confusion Matrix:

[[12  1  0]
 [ 1 23  2]
 [ 0  2 26]]


6.3 Logistic Regression

Accuracy: 94.03%

Precision: 0.9412

Recall: 0.9403

F1-Score: 0.9398

Confusion Matrix:

[[13  0  0]
 [ 1 23  2]
 [ 0  1 27]]


**7. Performance Summary**

The following table summarizes the comparative performance of all models:

<img width="882" height="246" alt="image" src="https://github.com/user-attachments/assets/42dd4b6e-6bef-4273-bfe5-03aee0195c25" />

Observations:

Logistic Regression outperformed other models in all metrics, especially in F1-score, indicating robust performance across all classes.

Decision Tree and Random Forest performed equally, with slightly lower accuracy and F1-scores.

**8. Conclusion**

This study successfully demonstrated the use of machine learning to classify campus sound levels. Key takeaways:

Outdoor areas generally exhibit higher sound levels.

Time-based features like hour_of_day and day_of_week are significant predictors of noise patterns.

Logistic Regression emerged as the best-performing model, achieving an F1-score of 0.9398 and an accuracy of 94.03%.

These insights can help campus administrators monitor noise levels, optimize study spaces, and implement effective sound management strategies.


