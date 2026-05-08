import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.inspection import permutation_importance


df = pd.read_csv("genz_social_media_usage_1M.csv")

print(df.head(10))
list(df.columns)

df.info()
df.describe()

num_cols = [
    'age',
    'daily_usage_hours',
    'num_platforms_used',
    'avg_session_minutes',
    'mental_health_score',
    'screen_time_before_sleep'
]

cat_cols = [
    'gender',
    'country',
    'primary_platform',
    'purpose',
    'addiction_level'
]

binary_cols = ['night_usage']

#Checking for missingness
df.isnull().sum ()

#Count plots for categorical values
df['addiction_level'].value_counts().plot(kind = 'bar')
plt.title('Addiction Levels')
plt.show()

df['gender'].value_counts().plot(kind = 'bar')
plt.title('Gender Distribution')
plt.show()

df['country'].value_counts().plot(kind = 'bar')
plt.title('Top Countries')
plt.show()

df['primary_platform'].value_counts().plot(kind = 'bar')
plt.title('Platform Popularity')
plt.show()

df['purpose'].value_counts().plot(kind = 'bar')
plt.title('Purpose of Usage')
plt.show()

#Correlation between Usage vs Mental Health
sample_df=df.sample(1000) #Due to the dataset having 1 million rows I sampled 1000 to give a better visualization
plt.scatter(sample_df['daily_usage_hours'],
            sample_df['mental_health_score'],
            alpha=0.1)
plt.xlabel('Daily Usage Hours')
plt.ylabel('Mental Health Score')
plt.title('Usage Vs Mental Health')
plt.show()

#Addiction level comparisons
sample_df.boxplot(column='mental_health_score',
                  by='addiction_level')
plt.xlabel('Addiction Level')
plt.ylabel('Mental Health Score')
plt.title('Addiction Level Comparison')
plt.show()

#Night usage impact
night_means = df.groupby('night_usage')['mental_health_score'].mean()
night_means.index = ['No Night Usage', 'Night Usage']
#Visualize night usage
night_means.plot(kind='bar', color=['steelblue','tomato'], edgecolor='black', figsize=(8,5))
plt.title('Average Mental Health Score by Night Usage', fontsize=14, fontweight='bold')
plt.xlabel('Night Usage', fontsize=12)
plt.ylabel('Avg Mental Health Score', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#Platform Comparisons
platform_scores = df.groupby('primary_platform')['mental_health_score'].mean()
#Visualize
platform_scores.sort_values().plot(kind='bar',color=['skyblue','tomato','orange', 'steelblue', 'magenta'], edgecolor='black', figsize=(8,5))
plt.show()

#Correlation Analysis
corr = df.corr(numeric_only=True)
print(corr)

#Heatmap
sns.heatmap(corr,
            annot=True,
            cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

#Age Group Analysis
df['age_group'] = pd.cut(
    df['age'],
    bins=[13,18,25,35],
    labels=['Teen', 'Young Adult', 'Adult']
)

age_means = df.groupby('age_group')['mental_health_score'].mean()
#Visualize
age_means.plot(kind='bar', color=['#4C72B0', '#DD8452', '#55A868'],
               edgecolor='black', figsize=(8,5))
plt.title('Average Mental Health Score by Age Group', fontsize=14, fontweight='bold')
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Avg Mental Health Score', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#Better correlation - include age_group as encoded
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['addiction_encoded'] = le.fit_transform(df['addiction_level'])

#Usage by platform AND addiction level together
platform_addiction = df.groupby(['primary_platform', 'addiction_level'])['mental_health_score'].mean().unstack()
platform_addiction.plot(kind='bar', figsize=(10,6))
plt.title('Mental Health Score by Platform and Addiction Level')
plt.tight_layout()
plt.show()

#Screen time before sleep vs mental health by addiction
sns.lmplot(data=df.sample(2000), x='screen_time_before_sleep', y='mental_health_score',
           hue='addiction_level', palette='RdYlGn_r', scatter_kws={'alpha':0.3})
plt.title('Screen Time Before Sleep vs Mental Health')
plt.show()
