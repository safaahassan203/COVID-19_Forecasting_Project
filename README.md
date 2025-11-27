
---

## Data Description  
The dataset contains two tables:

- **Table 1:** COVID-19 case and housing-status related data for people experiencing homelessness (PEH).  
- **Table 2:** Patient-level counts broken down by PEH vs. non-PEH status.

These datasets include numerical features such as case counts, provider counts, referral logs, housing status categories, and other public health indicators.

> *The datasets themselves are not included in this repository but can be replaced with any similarly structured public dataset.*

---

## Methodology  

### 1. **Data Cleaning**
- Handled missing values by imputing column means.
- Ensured all numerical fields were usable for downstream analysis.

### 2. **Exploratory Analysis**
- Visualized feature distributions via histograms.
- Calculated and plotted correlation matrices to identify strong relationships.

### 3. **Feature Reduction**
To avoid redundancy and multicollinearity, features highly correlated with the target variable (correlation > 0.8) were dropped.

### 4. **Modeling**
Two regression approaches were used:
- **Decision Tree Regressor**  
- **Random Forest Regressor**

Model performance was evaluated using:
- Mean Squared Error (MSE)  
- R² Score  

### 5. **Feature Importance**
Random Forests were used to quantify feature importance, highlighting which variables were most predictive of COVID-19 case patterns.

---

## Key Findings  

### **Model Performance**
- The Table 1 model achieved strong predictive performance (R² ≈ 0.84), suggesting that the remaining features captured meaningful variation.
- The Table 2 model performed poorly due to limited feature variety.

### **Top Predictive Features**
From Table 1, the most important features were:
- **PEH Providers (%)**
- **QI Logs (%)**
- **Unknown Housing Status (%)**

These indicate that healthcare access and housing-status data play major roles in understanding case patterns among homeless communities.

---

## Technologies Used  
- **Python**  
- **Pandas, NumPy**  
- **Matplotlib, Seaborn**  
- **Scikit-learn**  
- **Random Forests, Decision Trees**  

