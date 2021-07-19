################################### problem1 ####################################
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
mdata = pd.read_csv("C:/Users/DELL/Downloads/mdata.csv")
mdata.head(10)
mdata.columns

mdata.honors.value_counts()
mdata.describe()
mdata.dtypes

#EDA
mdata = mdata.iloc[:,1:]

#Label_Encoding
label_encoder = preprocessing.LabelEncoder()
mdata['ses'] = label_encoder.fit_transform(mdata['ses'] )
mdata['schtyp'] = label_encoder.fit_transform(mdata['schtyp'] )
mdata['honors'] = label_encoder.fit_transform(mdata['honors'] )
mdata['female'] = label_encoder.fit_transform(mdata['female'] )

mdata.dtypes

# Boxplot of independent variable distribution for each category of choice 
sns.boxplot(x = "prog", y = "ses", data = mdata)
sns.boxplot(x = "prog", y = "schtyp", data = mdata)
sns.boxplot(x = "prog", y = "honors", data = mdata)
sns.boxplot(x = "prog", y = "read", data = mdata)
sns.boxplot(x = "prog", y = "write", data = mdata)
sns.boxplot(x = "prog", y = "math", data = mdata)
sns.boxplot(x = "prog", y = "science", data = mdata)
sns.boxplot(x = "prog", y = "female", data = mdata)


# Scatter plot for each categorical choice of car
sns.stripplot(x = "prog", y = "ses", jitter = True, data = mdata)
sns.stripplot(x = "prog", y = "schtyp", jitter = True, data = mdata)
sns.stripplot(x = "prog", y = "honors", jitter = True, data = mdata)
sns.stripplot(x = "prog", y = "read", jitter = True, data = mdata)
sns.stripplot(x = "prog", y = "write", jitter = True, data = mdata)
sns.stripplot(x = "prog", y = "math", jitter = True, data = mdata)
sns.stripplot(x = "prog", y = "science", jitter = True, data = mdata)
sns.stripplot(x = "prog", y = "female", jitter = True, data = mdata)

# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(mdata) # Normal
sns.pairplot(mdata, hue = "prog") # With showing type of the program students choosed  in the scatter plot

# Correlation values between each independent features
mdata.corr()

train, test = train_test_split(mdata, test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:,[0,1,2,3,5,6,7,8,9]], train.iloc[:, 4])
help(LogisticRegression)

test_predict = model.predict(test.iloc[:, [0,1,2,3,5,6,7,8,9]]) # Test predictions

# Test accuracy 
accuracy_score(test.iloc[:,4], test_predict)

train_predict = model.predict(train.iloc[:, [0,1,2,3,5,6,7,8,9]]) # Train predictions 
# Train accuracy 
accuracy_score(train.iloc[:,4], train_predict) 

####################################### problem2 #########################################3
#Multinomial Regression
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

loan = pd.read_csv("C:/Users/DELL/Downloads/loan.csv")

#subsetting thr dataset
loan = loan.iloc[:,:17]

#Dropping unwanted columns
loan.drop(loan.iloc[:, [0,1,10,11,15]] , axis = 1 , inplace = True)
loan.head(10)

#EDA
loan.describe()
loan.loan_status.value_counts()
loan.columns

#label encoding 
#converting into binary
lb = LabelEncoder()
loan["term"] = lb.fit_transform(loan["term"])
loan["grade"] = lb.fit_transform(loan["grade"])
loan["sub_grade"] = lb.fit_transform(loan["sub_grade"])
loan["home_ownership"] = lb.fit_transform(loan["home_ownership"])
loan["verification_status"] = lb.fit_transform(loan["verification_status"])

#converting percentage value to float for 'int_rate' column
loan['int_rate'] = loan['int_rate'].str.rstrip('%').astype(float)

# Boxplot of independent variable distribution for each category of choice 
sns.boxplot(x = "loan_status", y = "loan_amnt", data = loan)
sns.boxplot(x = "loan_status", y = "funded_amnt", data = loan)
sns.boxplot(x = "loan_status", y = "funded_amnt_inv", data = loan)
sns.boxplot(x = "loan_status", y = "term", data = loan)
sns.boxplot(x = "loan_status", y = "int_rate", data = loan)
sns.boxplot(x = "loan_status", y = "installment", data = loan)
sns.boxplot(x = "loan_status", y = "grade", data = loan)
sns.boxplot(x = "loan_status", y = "sub_grade", data = loan)
sns.boxplot(x = "loan_status", y = "home_ownership", data = loan)
sns.boxplot(x = "loan_status", y = "annual_inc", data = loan)
sns.boxplot(x = "loan_status", y = "verification_status", data = loan)

# Scatter plot for each categorical choice of car
sns.stripplot(x = "loan_status", y = "loan_amnt", jitter = True, data = loan)
sns.stripplot(x = "loan_status", y = "funded_amnt", jitter = True, data = loan)
sns.stripplot(x = "loan_status", y = "funded_amnt_inv", jitter = True, data = loan)
sns.stripplot(x = "loan_status", y = "term", jitter = True, data = loan)
sns.stripplot(x = "loan_status", y = "int_rate", jitter = True, data = loan)
sns.stripplot(x = "loan_status", y = "installment", jitter = True, data = loan)
sns.stripplot(x = "loan_status", y = "grade", jitter = True, data = loan)
sns.stripplot(x = "loan_status", y = "sub_grade", jitter = True, data = loan)
sns.stripplot(x = "loan_status", y = "home_ownership", jitter = True, data = loan)
sns.stripplot(x = "loan_status", y = "annual_inc", jitter = True, data = loan)
sns.stripplot(x = "loan_status", y = "verification_status", jitter = True, data = loan)

# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(loan) # Normal
sns.pairplot(loan, hue = "loan_status") # With showing the category of each car choice in the scatter plot

# Correlation values between each independent features
loan.corr()

train, test = train_test_split(loan, test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, :11], train.iloc[:, 11])

test_predict = model.predict(test.iloc[:, :11]) # Test predictions

# Test accuracy 
accuracy_score(test.iloc[:,11], test_predict)

train_predict = model.predict(train.iloc[:, :11]) # Train predictions 
# Train accuracy 
accuracy_score(train.iloc[:,11], train_predict)

#########################################END##########################################








