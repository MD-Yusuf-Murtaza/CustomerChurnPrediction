# 🏦 Customer Churn Prediction using Machine Learning  

## 📌 Project Description  
Customer churn prediction helps businesses identify customers who are likely to leave, allowing for proactive retention strategies. This project focuses on predicting churn for a bank’s customers based on attributes like credit score, balance, tenure, and activity status.  

Multiple machine learning models are implemented, including:  
✔ Logistic Regression  
✔ Random Forest  
✔ K-Means Clustering  
✔ Artificial Neural Networks (ANNs)  

The project aims to analyze churn behavior effectively and provide actionable insights.  

---  

## 💊 Dataset Overview  
The dataset (`Churn_Dataset.csv`) contains customer information from a bank with **10,000 entries** and **14 features**.  

### 🔹 Key Columns & Their Description  

| Feature | Description |
|---------|------------|
| `RowNumber`, `CustomerId`, `Surname` | Unique identifiers (Not useful for analysis) |
| `CreditScore` | Customer’s creditworthiness score |
| `Geography` | Country of residence (France, Spain, Germany) |
| `Gender` | Male/Female |
| `Age` | Customer’s age |
| `Tenure` | Years with the bank |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products used |
| `HasCrCard` | Has a credit card (1 = Yes, 0 = No) |
| `IsActiveMember` | Active membership status (1 = Yes, 0 = No) |
| `EstimatedSalary` | Annual estimated salary |
| `Exited` | Target variable (1 = Customer left, 0 = Customer stayed) |

---  

## ⚙️ Installation & Setup  

### 🛠️ Prerequisites  
Ensure you have the following installed:  
- Python 3.x  
- Jupyter Notebook or Google Colab  
- Required Python libraries (install using the command below)  

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
```

### 🚀 Running the Project  
1️⃣ Clone this repository or download the `.ipynb` file.  
2️⃣ Open Jupyter Notebook and load the `CustomerChurnPrediction.ipynb` file.  
3️⃣ Run each cell sequentially to execute the code.  

---  

## 📊 Data Visualization  
The dataset is explored using various visualization techniques:  

✔ **Credit Score, Balance, and Estimated Salary** → Distribution plots, box plots  
✔ **Has Credit Card & Active Member** → Bar charts  
✔ **Churn (Exited column)** → Count plots, comparisons with other features  

---  

## 🤖 Machine Learning Models Implemented  

### 🔹 **Data Preprocessing & Transformation**  
- Droping unnecessary Columns
- Feature encoding  
- Scaling  

### 🔹 **Classification Models Used**  
1️⃣ **Logistic Regression** → Baseline model for churn prediction  
2️⃣ **Random Forest** → Ensemble learning for improved accuracy  
3️⃣ **K-Means Clustering** → Customer segmentation  
4️⃣ **Artificial Neural Networks (ANNs)** → Deep learning approach  

---  

## 📊 Model Performance Comparison  

| Model | Accuracy | Precision | Recall |
|--------|---------|-----------|---------|
| Logistic Regression | 81% | 58% | 24% |
| Random Forest | 87% | 75% | 53% |
| K-Means Clustering | 57% | 25% | 55% |
| ANN | 86% | 73% | 50% |

---  

## 📌 Key Insights & Findings  
✔ **Factors influencing churn**: Age, Balance, and Activity status  
✔ **ANN achieved the highest accuracy**, making deep learning highly effective in churn prediction  
✔ **Inactive members** are more likely to leave the bank  
✔ **Customers with high balance tend to stay**  

---  

## 🛠️ Tools & Libraries Used  

| Category | Tools & Libraries |
|------------|----------------|
| Programming Language | Python 🐍 |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, TensorFlow/Keras |
| Development Environment | Jupyter Notebook / Google Colab |

---  

## 🙏 Acknowledgements  
Special thanks to open datasets and the machine learning community for valuable resources on churn prediction.  

---  

## 📢 Conclusion  
This project demonstrates how **machine learning and deep learning** can effectively predict customer churn. By leveraging different models, we gained insights into customer behavior, helping businesses take **proactive measures** to retain customers. 🚀  

