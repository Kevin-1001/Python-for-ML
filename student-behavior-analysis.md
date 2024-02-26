# Title: Analyzing Student Engagement and Predictive Modeling in Online Learning Environments

## 1. Introduction

The advent of online education has revolutionized access to learning, transcending geographical boundaries and providing unprecedented opportunities for individuals worldwide. However, with the proliferation of Massive Open Online Courses (MOOCs) and other digital learning platforms, challenges such as student disengagement and high dropout rates have emerged. Addressing these challenges requires a deep understanding of student behavior and the ability to predict academic performance. This report delves into the analysis of student behavior data and proposes predictive modeling techniques to classify students based on their engagement levels and academic outcomes.

## 2. Data

The dataset used in this project is the "Students' Academic Performance Dataset" sourced from Kaggle, comprising information on over 480 students enrolled in the Kalboard360 online learning platform. The dataset includes a comprehensive range of features such as gender, nationality, educational stages, grade levels, classroom section, course topics, and various aspects of student behavior, including participation in classroom activities, parental involvement, and absenteeism.The DataSet from Kaggle is called "Students' Academic Performance Dataset". It can be obtained at https://www.kaggle.com/aljarah/xAPI-Edu-Data.

## 2.1 Data Structure

The dataset comprises 16 distinct columns, each providing essential insights into various aspects of student and parental engagement within an educational context. Here is an overview of the data structure:

- **Gender**: Indicates the gender of the student, categorized as 'Male' or 'Female'.
- **Nationality**: Represents the nationality of the student, offering a range of countries such as Kuwait, Lebanon, Egypt, etc.
- **Place of Birth**: Specifies the birthplace of the student, similarly encompassing various countries.
- **Educational Stages**: Classifies the educational level of the student into categories such as 'lowerlevel', 'MiddleSchool', or 'HighSchool'.
- **Grade Levels**: Identifies the grade level of the student, ranging from 'G-01' to 'G-12'.
- **Section ID**: Denotes the classroom section to which the student belongs, labeled as 'A', 'B', or 'C'.
- **Topic**: Indicates the subject or course topic the student is enrolled in, encompassing subjects like English, Math, Chemistry, etc.
- **Semester**: Specifies the school year semester as 'First' or 'Second'.
- **Parent Responsible for Student**: Specifies whether the mother or father is primarily responsible for the student.
- **Raised Hand**: Represents the frequency of the student raising their hand in the classroom, measured numerically from 0 to 100.
- **Visited Resources**: Indicates how often the student accesses course content resources, measured numerically from 0 to 100.
- **Viewing Announcements**: Measures the frequency of the student checking new announcements, quantified numerically from 0 to 100.
- **Discussion Groups**: Reflects the student's participation frequency in discussion groups, quantified numerically from 0 to 100.
- **Parent Answering Survey**: Specifies whether the parent has responded to surveys provided by the school, categorized as 'Yes' or 'No'.
- **Parent School Satisfaction**: Indicates the level of satisfaction of the parent with the school, categorized as 'Yes' or 'No'.
- **Student Absence Days**: Quantifies the number of absence days for each student, categorized as 'above-7' or 'under-7'.

An important aspect of this dataset is its inclusion of parental data, which provides a comprehensive approach to understanding the student's educational environment and engagement. This holistic perspective facilitates a deeper analysis of the factors influencing student performance and satisfaction within the educational system.

## 3. Methodology

The methodology adopted for this study encompasses several key stages aimed at comprehensively analyzing the dataset and deriving actionable insights to fulfill the research objectives.

### 3.1 Data Exploration and Insight Generation
The initial phase involves a thorough exploration of the dataset, focusing on understanding the underlying structure and characteristics of the data columns. This exploratory analysis aims to uncover latent features and establish relationships between various attributes. By delving into the data, we seek to gain valuable insights that will inform subsequent analytical approaches.

### 3.2 Descriptive Analysis and Clustering
Following the data exploration phase, a descriptive analysis is conducted to construct a dataset suitable for clustering algorithms. By leveraging clustering techniques, we aim to segment the student population based on their behavioral patterns and characteristics. This approach enhances our understanding of student behaviors and enables more targeted decision-making processes.

By systematically progressing through these methodological steps, we aim to extract valuable insights from the data and empower stakeholders with actionable information to support effective interventions and strategies in the educational domain.

## 4. Analysis

This section delves into the exploratory analysis of the dataset, aimed at constructing a clustering dataset effectively.

### 4.1 Exploratory Analysis

The exploration begins by importing necessary libraries and loading the dataset:

```python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('xAPI-Edu-Data.csv')
dataset.head(5)
```

#### 4.1.1 Setting Up Dataframe for Clustering

To focus on understanding student behavior, a refined dataframe with pertinent columns is created:

```python
df = dataset[['gender', 'PlaceofBirth', 'StageID', 'Topic', 'raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 'StudentAbsenceDays', 'Class']]
df.head()
```

#### 4.1.2 Analysis of Parental Satisfaction

The proportion of student classifications based on parental satisfaction is examined:

```python
df.groupby(['ParentschoolSatisfaction'])['Class'].value_counts(normalize=True)
```

Insights reveal potential correlations between parental survey involvement and school satisfaction.

#### 4.1.3 Relationship Between Parental Involvement and Student Classification

The impact of parental engagement, specifically in survey responses, on student classification is explored:

```python
df.groupby(['ParentAnsweringSurvey'])['Class'].value_counts(normalize=True)
```

The analysis indicates a significant influence of parental participation on student classification.

#### 4.1.4 Understanding Student Behavior

Various aspects of student behavior contributing to academic success are analyzed. Firstly, the correlation between raising hands and classification is investigated:

```python
df2 = dataset[['gender', 'raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion', 'StudentAbsenceDays', 'Class']]
df2['raisedhands'] = pd.cut(df2.raisedhands, bins=3, labels=np.arange(3), right=False)
df2.groupby(['raisedhands'])['Class'].value_counts(normalize=True)
```

Similarly, the exploration extends to other behavioral indicators like visiting course resources, viewing announcements, and participating in discussions.

## 5. Clustering DataSet

In this section, we proceed to construct a dataset suitable for clustering analysis using the K-Means algorithm, leveraging the insights gained from our previous exploratory analysis.

### 5.1 Dataset Construction

To facilitate comprehension, we begin by re-implementing the dataset building phases. 

```python
df2 = dataset[['gender','raisedhands','VisITedResources','AnnouncementsView','Discussion','StudentAbsenceDays', 'Class']]
df2.tail()
```

#### 5.2 Identifying Correlations

We examine the correlations between student actions:

```python
correlation = df2[['raisedhands','VisITedResources','AnnouncementsView','Discussion']].corr(method='pearson')
correlation
```

The analysis reveals that raised hands and visited resources are highly correlated, indicating their significance in our model dataset.

#### 5.3 One-Hot Encoding

To prepare the dataset, we perform one-hot encoding on gender, absence, and class columns:

```python
df2 = pd.concat([df2,pd.get_dummies(df2['gender'], prefix='gender_')], axis=1)
df2 = pd.concat([df2,pd.get_dummies(df2['StudentAbsenceDays'], prefix='absence_')], axis=1)
df2 = pd.concat([df2,pd.get_dummies(df2['Class'], prefix='class_')], axis=1)

df2.drop(['gender', 'StudentAbsenceDays', 'Class'], axis=1, inplace=True)
df2.head()
```

#### 5.4 Feature Selection

Based on our previous analysis, we select raised hands and visited resources as our features:

```python
X = df2[['raisedhands', 'VisITedResources']].values

# Normalize the array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X)

# Get the X axis
X = pd.DataFrame(x_scaled).values
X[:5]
```

#### 5.5 Determining Optimal K for K-Means

Using the Elbow Method, we determine the ideal number of clusters (K) for K-Means based on our data:

![image](https://github.com/Kevin-1001/Python-for-ML/assets/133469619/448a624d-82e2-46c5-aa46-1d53e302cb0f)


The Elbow Method suggests K = 3 as the optimal number of clusters.

#### 5.6 Building K-Means Model

Finally, we construct the K-Means model with K = 3:

```python
kmeans = KMeans(n_clusters = 3, init = 'k-means++')
kmeans.fit(X)

k_means_labels = kmeans.labels_
k_means_cluster_centers = kmeans.cluster_centers_
```

![image](https://github.com/Kevin-1001/Python-for-ML/assets/133469619/e79fa789-2782-43cf-a903-39ed5e957739)


The resulting plot illustrates three distinct clusters representing:

- High applied students
- Mid applied students
- Low applied students

## 6. Results and Discussion

This research delved into data analytics to comprehend student behavior in online learning courses, culminating in valuable insights:

* Parental Involvement and Student Satisfaction

Active participation and monitoring by parents emerged as crucial factors. The absence of parental engagement correlated with student absenteeism and heightened dissatisfaction with the school.

* Importance of Resource Utilization

Students who engaged with announcements and utilized course resources demonstrated a propensity for higher classification. This underscores the significance of proactive engagement with learning materials.

* Discussion Activity and Academic Performance

Contrary to expectations, participation in discussions showed minimal impact on student outcomes. This suggests that while discussions may foster engagement, they may not directly contribute to academic performance.

* Predictive Modeling for Decision Support

A predictive model was developed to aid online platforms in understanding student behavior. This model offers insights into student actions, facilitating informed decision-making without involving supervised learning techniques.

* Considerations and Limitations

The study omitted the analysis of location data pertaining to the student's birthplace. However, future research could explore the significance of student connectivity locations, particularly in regions with limited internet access.

### 7. Conclusion

Data analytics plays a vital role in enhancing the online learning experience. Tailoring content to individual students is essential for sustaining enrollment and motivation. Leveraging online learning has the potential to elevate societal education levels, contributing to economic growth. This research lays the groundwork for future studies aimed at refining online education practices.
