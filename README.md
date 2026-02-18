# Credit Card Fraud Detection using SMOTE

## Project Overview
This project demonstrates how to handle highly imbalanced datasets using SMOTE (Synthetic Minority Oversampling Technique).

Credit card fraud datasets usually contain very few fraud cases compared to normal transactions. This imbalance can reduce model performance. In this project, SMOTE is applied to balance the dataset before training a machine learning model.

## Objective
- Perform data ingestion using pandas
- Preprocess and clean the dataset
- Handle class imbalance using SMOTE
- Train a Logistic Regression model
- Evaluate model performance

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)

## Workflow
1. Load dataset  
2. Data preprocessing  
   - Remove unnecessary columns  
   - Convert date column  
   - Encode categorical variables  
3. Split dataset into training and testing  
4. Apply SMOTE to balance fraud and non-fraud classes  
5. Train Logistic Regression model  
6. Evaluate using:
   - Confusion Matrix  
   - Precision  
   - Recall  
   - F1-Score  

## Project Structure
credit-card-fraud-detection-smote/
│
├── credit_card_fraud_dataset.csv
├── fraud_detection.py
├── requirements.txt
├── README.md
└── .gitignore

## How to Run the Project

Step 1: Clone Repository

git clone https://github.com/yourusername/credit-card-fraud-detection-smote.git

Step 2: Install Dependencies

pip install -r requirements.txt

Step 3: Run the Project

python fraud_detection.py

## Model Used
Logistic Regression

SMOTE is applied only on the training dataset to prevent data leakage.

## Key Learning Outcomes
- Understanding imbalanced datasets  
- Applying SMOTE for oversampling  
- Feature encoding techniques  
- Model evaluation using classification metrics  

## Conclusion
Using SMOTE significantly improves fraud detection performance by balancing minority and majority classes. This approach is useful in real-world financial fraud detection systems.

## Author
Anisha Todmal
