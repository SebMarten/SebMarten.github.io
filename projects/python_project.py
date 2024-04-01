#!/usr/bin/env python
# coding: utf-8

# In[1]:


### The data set on car prices and features can be found here:
#https://www.kaggle.com/datasets/rupindersinghrana/car-features-and-prices-dataset/data

### The formula for the visualisation were found online by another author working on this data set:
# https://www.kaggle.com/code/mustafacakir/car-price-prediction

### This Python project aims to underline the steps of:
# Data Exploration
# Data Cleaning
# Feature Engineering
# Supervised Learning
# Unsupervised Learning


# In[2]:


#Importing libraries
import numpy as np
import pandas as pd

#Importing Visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sb

# Configure default settings for plots
sb.set(style='ticks')
sb.set_palette('Paired')
plt.rcParams['axes.spines.top'] = False    # Remove top border
plt.rcParams['axes.spines.right'] = False  # Remove right border


# In[3]:


#Importing the data
input_path = r"C:\Users\sgayc\Desktop\Uni\7.Semester\BA for Marketing\Assignment 2\RawData\car_features.csv"
inpDf = pd.read_csv(input_path, sep=',', header = 0)
print(f'The dataset has {inpDf.shape[0]} rows and {inpDf.shape[1]} columns')
print(inpDf.head())


# In[4]:


#Creating a copy of the input data for further analysis
df = inpDf.copy()


# In[5]:


### BASIC DATA EXPLORATION ###
#Getting an overview of the data
df.head()


# In[6]:


#Getting an overview of the data
df.tail()


# In[7]:


#Getting an overview of the columns and data types
df.info()


# In[8]:


#Getting an overview of missing values. (Formula for fuction below can be found on top of page) 
def check_missing_values(data):
  '''
    Prints the number of features with missing values and it's percentage
    in the dataset.
    Displayes a heatmap plot for the missing values.
    ---
    ###Parameters
    data <dataframe>: tabular data in data frame format.
  '''
  count = data.isnull().sum()[data.isnull().sum() > 0]
  percentage = (count / data.shape[0]) * 100

  print(count.shape[0], 'columns has missing values')
  print('-'*50)
  print(pd.DataFrame({'Count':count, 'Percentage %':percentage}))

  plt.figure(figsize=(10, 6))
  sb.heatmap(data.isnull(), yticklabels=False, cmap='cividis', cbar=False)
  plt.show(block=False)

check_missing_values(df)
#Conclusion: Four columns have few missing values. The column Market Category sticks out with around 31% missing values.


# In[9]:


#Getting an overview of the the Duplicates:
print(f'There are {df.duplicated().sum()} missing values')


# In[10]:


#Getting an overview of all features:
df.describe(include='all').transpose()


# In[11]:


### DATA CLEANING AND PREPARATION ###
### Removing Duplicates ###
df.drop_duplicates(inplace=True)
print(df.shape)
print(df.duplicated().sum())


# In[12]:


### Focusing on top makes and models ###
df['Make'].value_counts()


# In[13]:


#Finding out how many different types of 'Makes' are common defined here as appearing 200 or more times in the dataset.
n = len(df['Make'].value_counts().loc[lambda x : x>=200])
n


# In[14]:


#Create a list of top makes based on n
top_makes = df['Make'].value_counts().nlargest(n).index.values
top_makes


# In[15]:


#Looking at the number models in the new top_makes dataframe
df[df['Make'].isin(list(top_makes))]['Model'].value_counts()


# In[16]:


#Looking at the number models in the new top_makes dataframe
df[df['Make'].isin(list(top_makes))].shape


# In[17]:


#Create a list of top models appearing 200 or more times in the dataset.
df['Model'].value_counts().loc[lambda x : x>=20]


# In[18]:


#Finding out how many different types of 'Models' are common defined here as appearing 20 or more times in the dataset.
nr_top_models = len(df['Model'].value_counts().loc[lambda x : x>=20])
nr_top_models


# In[19]:


#Create a list of top makes based on nr_top_models
top_models = df['Model'].value_counts().nlargest(nr_top_models).index.values
top_models


# In[20]:


#Looking at the number of models in the new top_makes dataframe
df_top_models = df[df['Model'].isin(list(top_models))]
df_top_models.shape


# In[21]:


#Looking at the number of makes in the new top_makes dataframe
df_top_models['Make'].value_counts().shape


# In[22]:


#Handling missing values (Formula for fuction below can be found on top of page) 
check_missing_values(df_top_models)
#Conclusion: Focusing on the top models and makes has led to a reduction of missing values


# In[23]:


# Handling missing values for EngineHP
df_top_models[df_top_models['Engine HP'].isna()]
#Conclusion: For the models Focus and MKZ the missing values for Engine HP may be researched online.


# In[24]:


#Researching the missing Engine HP values online
df_top_models[(df_top_models['Make'] == 'Ford') & (df_top_models['Model'] == 'Focus') 
              & (df_top_models['Year'] == 2015) 
              & (df_top_models['Engine Fuel Type'] == 'electric')]


# In[25]:


# Updating the Engine HP for all electric Ford Focus vehicles to 143
df_top_models.loc[(df_top_models['Make'] == 'Ford') & (df_top_models['Model'] == 'Focus') 
              & (df_top_models['Year'].isin([2015, 2016, 2017])) 
              & (df_top_models['Engine Fuel Type'] == 'electric'), 'Engine HP'] = 143
df_top_models[df_top_models['Engine HP'].isna()]


# In[26]:


#Looking at all Lincoln MKZ vehicles
df_top_models[(df_top_models['Make'] == 'Lincoln') & (df_top_models['Model'] == 'MKZ') 
              & (df_top_models['Year'] == 2017)]


# In[27]:


# Updating the Engine HP for all Lincoln MKZ vehicles
df_top_models.loc[(df_top_models['Make'] == 'Lincoln') & (df_top_models['Model'] == 'MKZ') 
              & (df_top_models['Year'] == 2017), 'Engine HP'] = 245
df_top_models[df_top_models['Engine HP'].isna()]


# In[28]:


#Handling missing values for Market Category
df_top_models[df_top_models['Market Category'].isna()].shape


# In[29]:


#Conclusion: Dropping the Market Category as it has too many missing values and no way to impute it
df_top_models.drop('Market Category', axis=1, inplace=True)


# In[30]:


#Check whether all missing values are imputed. (Formula for fuction below can be found on top of page) 
check_missing_values(df_top_models)
#Conclusion: All missing values are imputed.


# In[31]:


# Transform all column names to lower case and replace empty spaces with underscores
df_top_models.columns = df_top_models.columns.str.lower().str.replace(' ', '_',)

# Display the transformed column names
df_top_models.columns.tolist()


# In[32]:


### Outlier Detection ###
# Create lists for categorical and numeric columns (Formula for fuction below can be found on top of page) 
cat_cols = []
num_cols = []

for column in df_top_models.columns:
    if pd.api.types.is_numeric_dtype(df_top_models[column]):
        num_cols.append(column)
    elif pd.api.types.is_object_dtype(df_top_models[column]):
        cat_cols.append(column)
        
print(f'Categorical features: {len(cat_cols)}', cat_cols)
print(f'Numeric features: {len(num_cols)}', num_cols)


### For Categorical Data ###

# Function to create labeled barplots for categorical features
def plot_counts(data, feature, perc = False, n = None, hue = None):
  """
    This function takes the name of the feature and plots the distribution
    of the categorical values and saves the figure for future usage using
    countplot function of seaborn.

    ---
    ### Parameters:
    - data <dataframe>: tabular data in data frame format.
    - feature <string>: Name of categorical column in dataframe to be visualized.
    - perc <bool>: whether to display percentages instead of count (default is False)
    - n <int>: displays the top n category levels (default is None, i.e., display all levels)
  """

  total = len(data[feature])            # How many number of rows are there in the feature?
  num_unique = data[feature].nunique()  # How many unique number of category are there in the feature?

  # Set the size of the figure according to the number of categories to be displayed
  if n is None:
    plt.figure(figsize = (num_unique+1, 5))
  else:
    plt.figure(figsize = (n+1, 5))

  # Set the style of the ticks on x axis
  plt.xticks(rotation=90, fontsize=12)

  # Plot the counts for each category
  ax = sb.countplot(
      data=data,
      x = feature,
      palette = 'Paired',
      order = data[feature].value_counts().index[:n].sort_values(),
      hue = hue
  )

  # Display the percentage of each category on top of the bar
  for p in ax.patches:
    if perc == True:
      label = '{:.1f}%'.format(
          100 * p.get_height() / total  # Display percentage of each class of the category
      )
    else:
      label = p.get_height() # Display count of each level of the category

    x = p.get_x() + p.get_width() / 2 # Width of the plot
    y = p.get_height() # Height of the plot

    rotation = 0
    if hue is not None:
      rotation = 30  # Rotate annotations by 30 degrees if hue is not None


    ax.annotate(
        label,
        (x, y),
        ha = 'center',
        va = 'center',
        size = 12,
        xytext = (0, 5),
        textcoords = 'offset points',
        rotation = rotation
    )
  plt.show()
for i in cat_cols:
    print (f'This is the Bar Plot for {i}')
    plot_counts(df_top_models, i)


# In[33]:


# Researching 'UNKNOWN' values in transmission type
df_top_models[(df_top_models['transmission_type'] == 'UNKOWN')]


# In[34]:


df_top_models[(df_top_models['model'] == 'Jimmy')]
#After some online research the Transmission Types of the unkown vehicles is Manual


# In[35]:


#Repalce all 'Unkown' values with 'Manual' (Formula for fuction below can be found on top of page) 
df_top_models['transmission_type'] = df_top_models['transmission_type'].replace('UNKNOWN', 'MANUAL')


# In[36]:


### For Numeric Data ###

# A function to display the both the histogram and the boxplot of a numeric column
def dist_plot(data, feature):
  '''
        This function takes the name of the feature and
        plots the distribution of the numeric values
        both using boxplot and hisplot of seaborn.
        The purpose is to check if the numeric feature has
        normal distribution and detect outliers manualy.
        Skewness and kurtosis values of the data are also
        shown.

        ---
        ### Parameters
        - data <dataframe>: tabular data in data frame format.
        - feature <string>: Name of numeric column in dataframe to be visualized.
  '''

  # Create the canvas with 2 sub-plots in rows with 0.15 to 0.85 ratio
  fig, (ax_box, ax_hist) = plt.subplots(
      nrows = 2,
      sharex = True,
      gridspec_kw = {'height_ratios':(0.15, 0.84)},
      figsize = (12, 6)
  )

  # Add the boxblot to the canvas in the first row
  sb.boxplot(
      data = data,
      x = feature,
      color = 'lightblue',
      showmeans = True,
      ax = ax_box
  )
  ax_box.set_xlabel('') # Remove the label from the x axis of the boxplot

  sb.histplot(
      data = data,
      x = feature,
      kde = True,
      ax = ax_hist
  )
  ax_hist.set_xlabel('') # Remove the label from the x axis of the histogram

  # Calculate the skewness and kurtosis
  skewness = data[feature].skew()
  kurt = data[feature].kurt()
  # Add skewness and kurtosis as text on the histogram plot
  ax_hist.text(
      0.95, 0.85,
      f'Skewness: {skewness:.2f}\nKurtosis: {kurt:.2f}',
      transform = ax_hist.transAxes,
      verticalalignment = 'top',
      horizontalalignment = 'right',
      bbox = dict(facecolor='white', edgecolor='gray', boxstyle='round, pad=0.5')
  )

  # Calculcate mean and median values of the feature
  mean_value = data[feature].mean()
  median_value = data[feature].median()

  # Add these values as a vertical line to the histogram
  ax_hist.axvline(mean_value, color='green', linestyle='dotted', linewidth=2, label='Mean')
  ax_hist.axvline(median_value, color='purple', linestyle='dotted', linewidth=2, label='Median')

  # Add legends
  ax_hist.legend(loc='lower right')

  plt.suptitle(feature)
  plt.tight_layout()
  plt.ticklabel_format(style='plain', axis='x')
  plt.show()

for i in num_cols:
    print (f'This is the Distribution Plot for {i}')
    dist_plot(df_top_models, i)


# In[37]:


# Detecting outliers in highway_mpg
df_top_models[df_top_models['highway_mpg'] > 50].head(50)
#For the Ford Focus that is correct after research
#For the Audi A6 the correct highwayy mpg is 34


# In[38]:


#Correcting the highway mpg of the Audi A6
df_top_models.loc[1119, 'highway_mpg'] = 34


# In[39]:


#Looking at Outliers in city_mpg
df_top_models[df_top_models['city_mpg'] > 60].head(50)
df_top_models[df_top_models['engine_fuel_type'] == 'electric']
# It seems those are the only entriees for electric_fuel_type. 
# There are so little other entries for electrict cars, they are removed.
df_top_models = df_top_models[df_top_models['engine_fuel_type'] != 'electric']
df_top_models


# In[40]:


df_top_models[df_top_models['popularity'] > 4000].head(500)

# There is sadly no possibilty to check this. Therefore, this popularity is assumed correctly although it a appears striking that only Ford´s models perform this well


# In[41]:


#Checking outliers in engine_hp
df_top_models[df_top_models['engine_hp'] > 600].head(500)
#All the higher value are correct after checking

#Decision: No further outlier detection of this data set since the values are observable facts and true.


# In[42]:


### Feature Engineering ###
### Categorising door numbers as this is often rather a stylistic decision rather than something to be quanitfied ###
def categorize_doors(n):
    if n == 2:
        return "Two-door"
    elif n == 4:
        return "Four-door"
    else:
        return "Other"

df_top_models['door_category'] = df_top_models['number_of_doors'].apply(categorize_doors)

# Dropping the 'number_of_doors' column from the dataframe
df_top_models = df_top_models.drop('number_of_doors', axis=1)
df_top_models


# In[43]:


### Calculating Engine Efficiency ###
df_top_models['engine_efficiency'] = df_top_models['engine_hp'] / df_top_models['engine_cylinders']
df_top_models


# In[44]:


#Calculating the average_mpg (55% City, 45% Highway)
df_top_models['combined_mpg'] = (df_top_models['city_mpg'] * 0.55) + (df_top_models['highway_mpg'] * 0.45)


# In[45]:


### Getting an overview of the final Data ###
### Outlier Detection ###
# Create lists for categorical and numeric columns
cat_cols = []
num_cols = []

for column in df_top_models.columns:
    if pd.api.types.is_numeric_dtype(df_top_models[column]):
        num_cols.append(column)
    elif pd.api.types.is_object_dtype(df_top_models[column]):
        cat_cols.append(column)
        
print(f'Categorical features: {len(cat_cols)}', cat_cols)
print(f'Numeric features: {len(num_cols)}', num_cols)

for i in cat_cols:
    print (f'This is the Bar Plot for {i}')
    plot_counts(df_top_models, i)

for i in num_cols:
    print (f'This is the Distribution Plot for {i}')
    dist_plot(df_top_models, i)


# In[46]:


df_top_models.info()


# In[47]:


df_top_models.head()


# In[48]:


output_path = r"C:\Users\sgayc\Desktop\Uni\7.Semester\BA for Marketing\Assignment 2\RawData\car_features_cleaned.csv"
df_top_models.to_csv(output_path, sep=',', float_format='%.2f', header=True, index=False)


# In[49]:


### DATA PREPARATION FOR MACHINE LEARNING ###
#Creating Dummies for the categorical columns
# use for loop to go through categories and print the ones with type == 'object'
to_dummy_list = []
for col_name in df_top_models.columns:
    if df_top_models[col_name].dtypes == 'object':
        unique_cat = len(df_top_models[col_name].unique())
        print(f'Feature {col_name} has {unique_cat} unique categories')
        # Create a list of features to 'dummy' 
        to_dummy_list.append(col_name)


# In[50]:


# Function to dummy all the categorical variables used for modeling using a loop for
def dummy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df

inpDf = dummy_df(df_top_models, to_dummy_list)


# In[51]:


inpDf


# In[52]:


### Supervised Learning ###
# Import libraries
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Define X and Y Variable
xDf = inpDf.drop(columns='msrp')
yDf = inpDf['msrp']
print(xDf.shape)
print(yDf.shape)
# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)
print('Train set')
print(X_train.shape)
print(y_train.shape)
print('Test set')
print(X_test.shape)
print(y_test.shape)

# Simple Linear Regression
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
# Fit the data
clf.fit(X_train, y_train)
# Analyse the output's score
print(clf.score(X_test, y_test))
print(clf.predict(X_test))
from sklearn.metrics import explained_variance_score
y_pred = clf.predict(X_test)
explained_variance = explained_variance_score(y_test, y_pred)
print(f'Explained Variance: {explained_variance}')


# In[53]:


# The linear regression can explain 94.68 % of the variance which is a very good score.


# In[54]:


### Unsupervised Learning ###
# Dropping not wanted columns for clustering, in this case engine_hp because then, it can be compared how the clusters related to engine hp.
from sklearn.cluster import KMeans
xDf = inpDf.drop(columns='engine_hp')
# Preparing list to find the best fitting k value for elbow point
inertiaLst = []
for kVal in range(1, 16):
    kmeans = KMeans(n_clusters=kVal)
    kmeans.fit(xDf)
    inertiaLst.append([kVal, kmeans.inertia_])
# Finding the best fitting k value for elbow point via visualisation
inertiaArr = np.array(inertiaLst).transpose()
plt.plot(inertiaArr[0], inertiaArr[1])
plt.xlabel('k')
plt.ylabel('Inertia')
plt.show()
# -> Elbow point at k=4


# In[55]:


# Applying found elbow point k value to k means algorithm
kVal = 4
kmeans = KMeans(n_clusters=kVal)
kmeans.fit(xDf)
# Adding the cluster allocation to each row in new column ‘labels’ and creating output / Export
xDf['kmeans_label'] = kmeans.labels_
outDf = pd.concat((xDf, inpDf['engine_hp']), axis=1)
print(outDf)
outDf


# In[56]:


# Conclusion: The unsupervised learning algorithm k-Means has provided four unique clusters. 
# Now, I want to get a better understanding for the clusters. 
# Loop through each unique value in the 'kmeans_label' column
for label in outDf['kmeans_label'].unique():
    print(f'This is the analysis for ### CLUSTER {label} ###')
    # Filter the DataFrame for rows with the current label
    filtered_df = outDf[outDf['kmeans_label'] == label]
    # Now, analysing the 'hp_engine' values for this specific cluster
    unique_hp_values = filtered_df['engine_hp'].unique()
    # To get summary statistics of 'hp_engine' values in this cluster:
    hp_engine_summary = filtered_df['engine_hp'].describe()
    print(f"Cluster {label}:")
    print("Unique HP Engine values:", unique_hp_values)
    print("\nSummary Statistics of HP Engine:\n", hp_engine_summary)
    print("\n---\n")


# In[57]:


### Conclusion k_means:
# Cluster 1 seems to have many cars with high roughly the highes engine hp.
# Cluster 0 seems to have cars with roughly the lowest engine hp.
# Cluster 2 and 3 have a wide range of values, averaging between the first two clusters.

#This analysis could be further extended to a holistic anaylysis to other characteristics except hp. However, that is beyond the scope of this analysis.


# In[58]:


### Conclusion k_means:
# Cluster 1 seems to have many cars with high roughly the highes engine hp.
# Cluster 0 seems to have cars with roughly the lowest engine hp.
# Cluster 2 and 3 have a wide range of values, averaging between the first two clusters.

#This analysis could be further extended to a holistic anaylysis to other characteristics except hp. However, that is beyond the scope of this analysis.

