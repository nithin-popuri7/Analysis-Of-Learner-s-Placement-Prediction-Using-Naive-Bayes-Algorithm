# ANALYSING LEARNER'S PLACEMENT PREDICTION USING NAIVE BAYES ALGORITHM
## INTRODUCTION
The majority of students in higher education enroll in an educational curriculumprogram to improve their chances of landing a good job. As a result, making an informed professional decision about where to work after completing a course is criticalin a student's life. One of the variables examined in establishing the institution's excellence is placement. The admissions rate to an educational institution is based on placements, which is a well-known truth worldwide. As a result, every institution workshard to improve student placement.

In India, 1.5 million engineers graduate each year, according to data. The demand for qualified graduates in the IT business is increasing. However, most students are unaware of the importance of the IT business. The number of graduates who meet thecompany's requirements and quality standards is relatively low. An institute's placement cell and instructors should take the necessary procedures to generate a group of students that fit each company's needs. In each academic group, campus placement plays a critical role in assisting college students in achieving their objectives. Because of the enormous number of students, managing placement and education information in a large organization is difficult; differentiation and classification into distinct categories become tedious in this case. Before they leave college, every student hopes to find a job that they can do with their hands. 

In this project, the Naive Bayes algorithm predicts student placement using factors such as gender, ssc_p, ssc_b, hsc_p, hsc_b, hsc_s degree_p, degree_t, workex, etest_p, specialization, mba_p, status. A placement probability predictor gives collegestudents an idea of where they stand and what they need to do to get a good job. Theinformation gained from this can help students better identify their weak areas and how to improve them. The transition from academia to the professional realm is a significant milestone for students, and optimizing this transition requires strategic planning and informed decision-making. A mini project focused on placement prediction serves as a valuable tool for both students and educational institutions to streamline and enhance the placement process.. By leveraging data analytics and machine learning techniques, this project aims to create a predictive model that can assess and predict a student's placement success based onhistorical data, academic performance, skills, and other relevant metrics.


## PROBLEM STATEMENT
The algorithm is chosen for its simplicity, effectiveness, and ability to handle categorical and numerical attributes. By leveraging attributes such as age, gender, educational background, work experience, technical skills, communication skills, and quantitative aptitude, a predictive model will be built to determine the placement status of new learners.The proposed solution aims to streamline the learner placement process by automating the evaluation of learners attributes and predicting their placement outcomes. This will enable educational institutions and career placement agencies to make informed decisions in a timely manner. Additionally, the Naive Bayes algorithm's assumption of feature independence makes it suitable for this analysis, allowing for efficient model training and prediction.

## EXISTING SYSTEM
Online Job Portals: Platforms like LinkedIn, Indeed, and Glassdoor serve as online job portals that connect job seekers with employers. These platforms allow employers to post job listings, and students can apply for those positions. These systems often include features like resume building, job search filters, and application tracking.
Campus Placement Systems: Many educational institutions have their own placement systems specifically designed to streamline the placement process for their students. These systems enable employers to post job opportunities, schedule interviews, and interact with students. Students can access job listings, submit their resumes, and participate in recruitment events through these systems.
 ## FLOW DIAGRAM
![o](https://github.com/nithin-popuri7/Analysis-Of-Learner-s-Placement-Prediction-Using-Naive-Bayes-Algorithm/assets/94154780/90cb2e76-a804-4c02-8c8e-a5aa22b5a831)

## PROPOSED SYSTEM METHODOLOGY
### Data Collection: 
Gather a comprehensive dataset that includes relevant information about students and their placement outcomes. This data may include academic performance, skill sets,
internships or work experience, extracurricular activities, and any other relevant factors that could contribute to placement success.

### Data Preprocessing: 
Clean and preprocess the collected data. Handle missing values, remove irrelevant or redundant features, and convert categorical variables into numerical representations, if necessary. Additionally, perform exploratory data analysis to gain insights into the data and identify any patterns or correlations.

### Feature Engineering: 
Extract meaningful features from the available data that can contribute to placement predictions. This may involve transforming or combining existing features, creating new features based on domain knowledge, or utilizing feature selection techniques to identify the most relevant predictors.

### Splitting the Dataset: 
Divide the preprocessed data into training and testing/validation sets. The training set will be used for model training, while the testing/validation set will assess the model's performance and generalization ability.

### Model Selection and Training: 
Choose an appropriate machine learning algorithm for placement predictions. Commonly used algorithms for this task include logistic regression, decision trees, random forests, support vector machines (SVM), and gradient boosting methods. Train the selected model using the training data and tune its hyperparameters to optimize performance.

### Model Evaluation: 
Evaluate the trained model using the testing/validation set. Employ appropriate evaluation metrics such as accuracy, precision, recall, F1-score.
# LITERATURE SURVEY
### REFERENCE 01:
This PDF file discusses the challenges of student placement in educational institutions and proposes a mechanism to prioritize academic performance parameters relevant for student placement through developed classifiers using machine learning algorithms. The study concludes that percentage in Tenth is the top priority followed by percentage in Twelve and Backlog in B Tech. The PDF also includes a detailed explanation of the results obtained through the experiment conducted and appropriate discussions required elaborating the highlights of the results. The study uses various classifiers and analyzes their accuracy, impact of MSE and Log Loss on classifier accuracy, and AUC-ROC Curve to compare the performance of different classifiers. The findings of this study can be applied to other fields beyond engineering institutions.

### REFERENCE 02:
This PDF file is a research article that explores the use of data mining to forecast students' career placement probabilities and recommendations in the programming field. The study focuses on critical perspectives of educational data mining, highlighting the strengths and weaknesses of present literature and imparting a unique contribution to this field. The article provides a literature analysis and a summary of approaches to data mining, followed by a description of the data, pre-processing, and methodology used in the research. The results and discussion are presented, and the final section provides a conclusion and addresses emerging directions for future endorsements. Overall, this research has important implications for educators and students in the programming field, providing valuable insights into career guidance and decision-making.

### REFERENCE 03:
The PDF file titled "Prediction of student learning outcomes using the Naive Bayesian Algorithm (Case Study of Tama Jagakarsa University)" by Arini Aha Pekuwali presents a case study exploring the potential of the Naive Bayesian Algorithm in predicting student performance. The study analyzes the algorithm's ability to predict final grades of students in the future based on their final grade data in the previous semester. The results of the study indicate that the Naive Bayesian Classifier algorithm successfully classifies data with an accuracy of 94.2446%. The study is useful for students to improve their grades, according to their predicted weaknesses through this research (wake-up calling). The paper provides valuable insights into the potential of the Naive Bayesian Algorithm in predicting student performance and can be used as a literature review for further research in this area.

### REFERENCE 04:
This PDF file from the Australasian Journal of Engineering Education evaluates the use of educational data mining to predict graduation rates in higher education institutions. The authors analyze data from 441 computer science engineering students at a university from 2002 to 2015 using various algorithms, including J48 and random tree. The results of the study can be used to identify at-risk students and implement strategies to improve academic indicators. However, the authors note that ethical and safe use of data must be considered in future research. Overall, this study provides valuable insights into the potential of educational data mining to improve student outcomes in higher education.

### REFERENCE 05:
This PDF file is a research paper published in the International Journal of Scientific and Engineering Research in August 2011. The paper explores the accuracy of various data mining techniques in predicting the performance of undergraduate computer science students. The study was conducted using a sample of 365 student records from the distance learning stream of Hellenic Open University, Greece. The authors applied five classification algorithms, including Decision Trees, Perceptron-based Learning, Bayesian Nets, Instance-Based Learning, and Rule-learning, to predict student performance. The paper provides a detailed analysis of the results obtained from each algorithm and concludes that Decision Trees and Instance-Based Learning were the most accurate in predicting student performance. The paper also discusses the relevant attributes chosen from the collected data and provides a literature review of related studies in the field.

### REFERENCE 06:
This PDF file discusses the use of machine learning algorithms to predict the likelihood of a student being placed in the IT industry based on their academic performance. The study compares the effectiveness of three models - Decision Tree, Random Forest, and Naïve Bayes - and concludes that these are the best-suited models for classification problems. The paper also presents a recommendation framework that forecasts the scholars to have one of the five placement statuses, namely, Dream Company, Core Company, Mass Recruiters, Not Eligible, and Not Interested in Placements. The model benefits the placement cell within an institute to recognize the potential students and advance their technical as well as social abilities. Overall, this study highlights the importance of accurate placement prediction for both students and academic institutions.

### REFERENCE 07:
This PDF file explores the use of machine learning techniques to predict the probability of undergraduate students getting placed. The paper presents an overview of various algorithms such as MLP, LMT, SMO, simple logistic, and logistic classifiers, and analyzes their accuracy in predicting student placement performance. The study also examines different matrices to determine which algorithm performs better and provides guidelines for improving student placement performance in education. Overall, this literature review provides valuable insights into the use of machine learning techniques for predicting student placement performance and can be useful for future research in this area.
![R1](https://github.com/nithin-popuri7/Analysis-Of-Learner-s-Placement-Prediction-Using-Naive-Bayes-Algorithm/assets/94154780/0d0f5f0f-2663-4815-994c-0d6224368610)

![R2](https://github.com/nithin-popuri7/Analysis-Of-Learner-s-Placement-Prediction-Using-Naive-Bayes-Algorithm/assets/94154780/40e51ae8-20ea-42b4-887a-d28a95391b92)

# SYSTEM ANALYSIS AND DESIGN
### HARDWARE REQUIREMENTS
1.	Processor (CPU)
2.	Storage
3.	Network Connectivity

### SOFTWARE REQUIREMENTS
1.	Programming Language
2.	Integrated Development Environment (IDE)

# SYSTEM ARCHITECTURE
The system architecture for placement prediction typically involves multiple components and layers that work together to process data, train the prediction model, and make accurate predictions. Here's a high-level overview of a typical system architecture for placement prediction
 

1.	Data Collection and Storage: The system begins by collecting relevant data about students, including academic performance, skills, internships, and extracurricular activities. This data may come from various sources such as student databases, surveys, resumes, or online profiles. The collected data is then stored in a data storage system, such as a database or data warehouse.

2.	Data Preprocessing: The collected data often requires preprocessing to ensure it is clean, consistent, and ready for analysis. Preprocessing steps may include handling missing values, data normalization or scaling, feature encoding or transformation, and removing outliers. This processed data is then used for model training and prediction.

3.	Feature Engineering: Feature engineering involves extracting relevant features from the preprocessed data that can improve the predictive performance of the model. Domain knowledge and data analysis techniques are employed to create new features or transform existing ones. Feature engineering aims to capture the most informative aspects of the data that are likely to influence the placement outcome.

4.	Model Training and Evaluation: The preprocessed data with engineered features is split into training and testing sets. The training set is used to train a machine learning model using a chosen algorithm (e.g., logistic regression, decision tree, random forest, or neural networks). The model learns patterns and relationships in the training data to make predictions. The trained model is then evaluated using the testing set to assess its accuracy, precision, recall, or other relevant metrics.

5.	Model Deployment: Once the model is trained and evaluated, it is ready for deployment. The deployment phase involves integrating the model into the production environment, allowing it to make predictions on new, unseen data. This can be achieved by creating APIs or building a web application that interacts with the model.

6.	User Interface (UI): A user interface is typically provided to allow users (e.g., administrators, career counselors, or students) to interact with the system. The UI may include features such as data input forms, visualization of predictions or performance metrics, and the ability to explore the underlying data.

7.	Continuous Monitoring and Maintenance: The deployed model should be continuously monitored to ensure its performance remains optimal. Monitoring can involve tracking prediction accuracy, detecting concept drift, and periodically retraining the model with new data to maintain its relevance. Additionally, regular maintenance and updates may be required to address bug fixes, improve system performance, or incorporate new features.
 
8.	Integration and APIs: The placement prediction system may need to integrate with other existing systems, such as student information systems or job portals. This integration allows seamless data exchange and facilitates the flow of information between different components.

![image](https://github.com/nithin-popuri7/Analysis-Of-Learner-s-Placement-Prediction-Using-Naive-Bayes-Algorithm/assets/94154780/ea4a4391-fb39-4566-bdd3-fa733c3b2d81)
## MODULE DESCRIPTION

1.		Data Collection Module: Responsible for collecting relevant data about students from various sources, such as student databases, surveys, resumes, or online profiles.Handles data extraction and integration from different sources into a unified dataset.Ensures data quality and consistency.
   
2.	Data Preprocessing Module:Performs data cleaning by handling missing values, removing duplicates, and addressing inconsistencies.Conducts data transformation and normalization to make it suitable for analysis.Applies feature scaling or encoding techniques to prepare categorical or numerical features for model training.
   
3.	Feature Engineering Module:Extracts meaningful features from the preprocessed data that are likely to contribute to placement predictions.Conducts feature selection or dimensionality reduction techniques to identify the most informative features.Creates new features by combining or transforming existing ones based on domain knowledge or statistical analysis.
  
4.	Model Training Module:Utilizes machine learning algorithms (e.g., logistic regression, decision trees, random forests, or neural networks) to train a placement prediction model.Uses the preprocessed data with engineered features to train the model on historical placement outcomes.Incorporates techniques like cross-validation and hyperparameter tuning to optimize model performance.
  
5.	Model Evaluation Module:Assesses the performance of the trained model using evaluation metrics like accuracy, precision, recall, F1-score, or area under the receiver operating characteristic curve (AUC-ROC).Conducts validation techniques, such as train-test splits or k-fold cross-validation, to evaluate model generalization and identify potential issues like overfitting or underfitting.
   
6.	Prediction Module:Applies the trained model to new, unseen data to make predictions about student placement outcomes.Takes input from users, such as student profiles or attributes, and generates predictions using the trained model.Outputs the predicted placement labels or probabilities for decision-making.
   
7.	User Interface Module:Provides a user-friendly interface for users (e.g., administrators, career counselors, or students) to interact with the placement prediction system.Includes features such as data input forms, visualization of predictions or performance metrics, and options to explore the underlying data.
   
8.	Deployment and Integration Module:Handles the deployment of the placement prediction system into a production environment.Integrates with other systems or platforms, such as student information systems or job portals, for seamless data exchange or interaction.Manages APIs or web services to enable external access and facilitate system integration.
   
9.	Monitoring and Maintenance Module:Monitors the performance of the deployed model in real-time, tracking metrics like prediction accuracy or model drift.Conducts regular maintenance and updates to address bug fixes, performance improvements, or new feature incorporations.Supports model retraining with new data to maintain model relevance over time.
# IMPLEMENTATION
## CODE
```
import pandas as pd
import matplotlib.pyplot as plt import seaborn as sns
from sklearn.preprocessing import LabelEncoder from sklearn.preprocessing import StandardScaler from sklearn.model_selection import train_test_split from sklearn.linear_model import LogisticRegression from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dataset = pd.read_csv('./Placement_Data_Full_Class.csv') dataset.head()

# as salary and sl_no columns are not required for placement status prediction so we drop it dataset.drop(['salary','sl_no'], axis=1, inplace=True)

# missing values checking dataset.isnull().sum()

# checking column values data type dataset.info()

# label encoding needs to be done to ensure all values in the dataset is numeric
# hsc_s, degree_t columns needs to be splitted into columns (get_dummies needs to be applied) features_to_split = ['hsc_s','degree_t']
for feature in features_to_split:
dummy = pd.get_dummies(dataset[feature])
 
dataset = pd.concat([dataset, dummy], axis=1) dataset.drop(feature, axis=1, inplace=True)
dataset

dataset.rename(columns={"Others": "Other_Degree"},inplace=True) dataset

encoder = LabelEncoder() # to encode string to the values like 0,1,2 etc. columns_to_encode = ['gender','ssc_b', 'hsc_b','workex','specialisation','status'] for column in columns_to_encode:
dataset[column] = encoder.fit_transform(dataset[column]) dataset
dataset.describe()

fig, axs = plt.subplots(ncols=6,nrows=3,figsize=(20,10)) index = 0
axs = axs.flatten()
for k,v in dataset.items(): sns.boxplot(y=v, ax=axs[index]) index+=1

fig.delaxes(axs[index])
plt.tight_layout(pad=0.3, w_pad=0.5,h_pad = 4.5) # for styling by giving padding # deleting some outliers in 2 columns degree_p and hsc_p
dataset = dataset[~(dataset['degree_p']>=90)] dataset = dataset[~(dataset['hsc_p']>=95)]

dataset.corr()

# heatmap for checking correlation or linearity plt.figure(figsize=(20,10)) sns.heatmap(dataset.corr().abs(), annot=True) dataset.shape

# checking distributions of all features
fig, axs = plt.subplots(ncols=6,nrows=3,figsize=(20,10)) index = 0
axs = axs.flatten()
for k,v in dataset.items(): sns.distplot(v, ax=axs[index])
 
index+=1

fig.delaxes(axs[index]) # deleting the 18th figure plt.tight_layout(pad=0.3, w_pad=0.2,h_pad = 4.5)

x = dataset.loc[:,dataset.columns!='status'] # all features are used y = dataset.loc[:, 'status'] # label is status of placement
x y
sc= StandardScaler()
x_scaled = sc.fit_transform(x) # for standardising the features x_scaled = pd.DataFrame(x_scaled)
x_train,x_test, y_train, y_test = train_test_split(x_scaled,y,test_size=0.18, random_state=0)

nbclassifier = GaussianNB() nbclassifier.fit(x_train, y_train) y_pred_nb = nbclassifier.predict(x_test) accuracy_score(y_test, y_pred_nb) nbclassifier.score(x_train, y_train) confusion_matrix(y_test, y_pred_nb)
print(classification_report(y_test,y_pred_nb))

### Using Naive Bayes Classifier - Gaussian Naive Bayes

nbclassifier = GaussianNB() nbclassifier.fit(x_train, y_train) y_pred_nb = nbclassifier.predict(x_test) accuracy_score(y_test, y_pred_nb) nbclassifier.score(x_train, y_train) confusion_matrix(y_test, y_pred_nb)
print(classification_report(y_test,y_pred_nb))

### Using SVM Linear Kernel clf = svm.SVC(kernel="linear") clf.fit(x_train, y_train) y_pred_svm = clf.predict(x_test)

accuracy_score(y_test, y_pred_svm) clf.score(x_train, y_train) confusion_matrix(y_test, y_pred_svm) print(classification_report(y_test, y_pred_svm))

```
# OUTPUT
### HEAT MAP
![HM](https://github.com/nithin-popuri7/Analysis-Of-Learner-s-Placement-Prediction-Using-Naive-Bayes-Algorithm/assets/94154780/d4c07547-b82c-4258-ba49-bbebd4b346af)

### ACCURACY
![A](https://github.com/nithin-popuri7/Analysis-Of-Learner-s-Placement-Prediction-Using-Naive-Bayes-Algorithm/assets/94154780/2de7aafe-4c55-43b6-9f89-99f0badd2348)

### MATRIX
![MATRIX](https://github.com/nithin-popuri7/Analysis-Of-Learner-s-Placement-Prediction-Using-Naive-Bayes-Algorithm/assets/94154780/316a032a-702c-4607-b2c8-548551b9eb85)

### REPORT
![CR](https://github.com/nithin-popuri7/Analysis-Of-Learner-s-Placement-Prediction-Using-Naive-Bayes-Algorithm/assets/94154780/84609db4-0b76-479c-82fc-bc886e3d4ab0)

# CONCLUSION
The Naïve Bayes model is validated with different performance metrics in which we took
 
LogLoss as our primary metric to test performance. The loss function used in logistic regression and extensions such as neural networks is the negative log- likelihood of a logistic model that returns actual output predictions probabilities for its training data. After developing the model with better hyperparameters, we should validate it with other performance metrics. We can test model accuracy against training data with a performance metric accuracy score. Visualizing performance makes us understand the loose ends of the algorithm.
The Student Placement Prediction model for university student evaluation using the Naive Bayes ML algorithm has been successfully developed with 85.72% accuracy. This NB model can be used to predict the placement of the student.

# SUMMARY OF THE LITERATURE SURVEY
![image](https://github.com/nithin-popuri7/Analysis-Of-Learner-s-Placement-Prediction-Using-Naive-Bayes-Algorithm/assets/94154780/a4977fd3-4512-428e-b466-f4c1c97742bd)
![image](https://github.com/nithin-popuri7/Analysis-Of-Learner-s-Placement-Prediction-Using-Naive-Bayes-Algorithm/assets/94154780/2e1d487c-464e-4298-b248-ad968e8a8740)

# OUTCOME OF THE LITERATURE SURVEY
The outcome of this literature survey was to gain insights into the existing research and studies that utilized Naive Bayes for predicting learner's placement and to understand the outcomes of applying the algorithm.The survey begins by exploring various research articles, academic papers, and publications related to learner's placement prediction using the Naive Bayes algorithm. The researchers examine the methodologies, datasets, and evaluation techniques employed in these studies to assess the performance of the algorithm.The findings of the literature survey indicate that Naive Bayes has been widely used for learner's placement prediction due to its simplicity and effectiveness. The algorithm assumes independence between features, making it suitable for datasets with a large number of attributes.Many studies reported favorable outcomes when applying Naive Bayes for learner's placement prediction. The algorithm demonstrated good accuracy, precision, recall, and F1-score values, indicating its effectiveness in classifying learners into placement categories accurately.Additionally, Naive Bayes showcased robustness when dealing with noisy or incomplete data, which is often encountered in educational datasets.The algorithm's ability to handle missing values and imbalanced classes made it a popular choice for learner's placement prediction tasks.
In conclusion, the literature survey reveals that Naive Bayes is a popular and effective algorithm for predicting learner's placement. While it has limitations in capturing complex relationships, its simplicity and robustness make it a valuable tool in the educational domain. Proper maintenance and updates are vital for ensuring the continued accuracy of Naive Bayes models in learner's placement prediction. Future research can focus on enhancing the algorithm's ability to handle complex relationships and developing maintenance strategies to further improve its predictive performance.

# SCOPE OF THE SYSTEM
The ML model we are building is used by universities where there is a need for students placement prediction based on their academic performance. Many ML Algorithms can solve this problem, but we adopt the Naive Bayes algorithm to predict student placement in this project, one of the better performing ML algorithms. Using source code and compatible software for machines makes it possible to make any pc to predict student placement, which helps educational management concentrate and focus on students who are predicted as Not Placed by an algorithm.

# REFERENCE
[1]. Laxmi Shanker Maurya;Md Shadab Hussain;Sarita Singh; (2021). Developing Classifiers 
through Machine Learning Algorithms for Student Placement Prediction Based on Academic 
Performance . Applied Artificial Intelligence, (), – . doi:10.1080/08839514.2021.1901032 

[2]. Mahboob, K., Asif, R., & Haider, N. G. (2023). A data mining approach to forecast students’ 
career placement probabilities and recommendations in the programming field. Mehran 
University Research Journal Of Engineering & Technology, 42(2), 169–187. 
https://search.informit.org/doi/10.3316/informit.002615895193590 

[3]. Arini Aha Pekuwali 2020 IOP Conf. Ser.: Mater. Sci. Eng. 823 012056DOI
10.1088/1757899X/823/1/012056 

[4]. Moscoso-Zea, Oswaldo; Saa, Pablo; Luján-Mora, Sergio (2019). Evaluation of algorithms to 
predict graduation rate in higher education institutions by applying educational data mining. 
Australasian Journal of Engineering Education, (), 1–10. doi:10.1080/22054952.2019.1601063 

[5]. International Journal of Scientific & Engineering Research Volume 2, Issue 8, August-2011 1 
ISSN 2229-5518

[6]. Rai, Kajal. "Students Placement Prediction Using Machine Learning Algorithms." South
Asia Journal of Multidisciplinary Studies, vol. 8, no. 5, June 2022, pp. 57-63, ISSN: 2395-1079,.

[7]. SAMRIDDHI: A Journal of Physical Sciences, Engineering and Technology
Published Nov 30, 2020 DOI https://doi.org/10.18090/samriddhi.v12iS2.17

[8]. International Journal of Research Studies in Computer Science and Engineering (IJRSCSE) 
Volume 3, Issue 2, 2016, PP 10-14 ISSN 2349-4840 (Print) & ISSN 2349-4859 (Online) 
www.arcjournals.org.

[9]. Shukla, M. and Malviya, Anil Kumar, Modified Classification and Prediction Model for 
Improving Accuracy of Student Placement Prediction (March 12, 2019). Proceedings of 2nd 
International Conference on Advanced Computing and Software Engineering (ICACSE) 
2019, Available at SSRN: https://ssrn.com/abstract=3351006 or http://dx.doi.org/10.2139/ssrn.3351006

[10]. International Journal of Computer Applications (0975 – 8887) Volume 31– No.3, October 
2011 40 A Generalized Data mining Framework for Placement Chance Prediction Problems 
Sudheep Elayidom Associate Professor, CUSAT Kochi, 682022, India Sumam Mary Idikkula 
Professor, CUSAT, Kochi, 682022, India Joseph Alexander Project officer, NODAL Center CUSAT, 
Kochi, 682022, India.doi: available online at https://journals.edwin.co.in/index.php/esajms/issue/view/619
3https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=1a70b7f0cda389b8a8d2c6 d5f17c87b630ba2a71 















