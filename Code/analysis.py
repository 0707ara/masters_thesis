import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from datetime import datetime

df = pd.read_csv('../Data/Final_data/Final_data.csv', delimiter=';', encoding='utf-8')
df1 = pd.read_csv('../Data/Final_data/Final_data2.csv', delimiter=';', encoding='utf-8')

i=0
for student_hash in df['Student_hash'].unique():
    if len(df[(df['Student_hash']==student_hash) & (df['Academic_year']=='2021-2022')]['Class'].unique()) > 0:
        if len(df[(df['Student_hash']==student_hash) & (df['Academic_year']=='2022-2023')]['Class'].unique()) > 0: 
            df.loc[(df['Student_hash']==student_hash) & (df['Academic_year']=='2021-2022'), 'Class'] =  df[(df['Student_hash']==student_hash) & (df['Academic_year']=='2022-2023')]['Class'].unique()[0]-1
        else:
            pass
    if len(df[(df['Student_hash']==student_hash) & (df['Academic_year']=='2020-2021')]['Class'].unique()) > 0: 
        if len(df[(df['Student_hash']==student_hash) & (df['Academic_year']=='2021-2022')]['Class'].unique()) > 0:
            df.loc[(df['Student_hash']==student_hash) & (df['Academic_year']=='2020-2021'), 'Class'] = df[(df['Student_hash']==student_hash) & (df['Academic_year']=='2021-2022')]['Class'].unique()[0]-1
    i=i+1
    if i%50 ==0:
        print(i)
list_odd=pd.Series(df1[df1['Class']==0]['Student_hash'].unique())
df2 = df1[df1['Class'] != 0]

df.reset_index(drop=True)

df = pd.read_csv('../Data/UC_POISSAOLOT.csv', delimiter=';', encoding='utf-8')
sns.countplot(x=df['aika'].str[:2], palette = "Set2")
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Absent count by time')



hours = df['aika'].str[:2]
# Compute counts
counts = hours.value_counts()
# Convert counts to percentages
percentages = (counts / counts.sum() * 100).sort_index()
percentages = percentages.round(1)
# Plot using sns.barplot
ax = sns.barplot(x=percentages.index, y=percentages.values, palette="Set2")

# Add percentages on top of each bar
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 0.2,
            '{:1.0f}%'.format(height), ha="center") 

plt.xlabel('Time')
plt.ylabel('Percentage')
plt.title('Absent percentage by time')
plt.show()

plt.figure(figsize=(16,4))
sns.countplot(x=df['tapahtumapvm'].str[:7].sort_values(), palette = "Set2")
plt.xlabel('Month')
plt.ylabel('Count')
plt.title('Absent count by month')
plt.xticks(rotation=45)

plt.figure(figsize=(16,4))
sns.countplot(x=df['tapahtumapvm'].str[5:7].sort_values(), palette = "Set2")
plt.xlabel('Month')
plt.ylabel('Count')
plt.title('Absent count by month')
plt.xticks(rotation=45)

months = df['tapahtumapvm'].str[5:7]
counts = months.value_counts().sort_index()
percentages = (counts / counts.sum() * 100).sort_index()
percentages=percentages.round(0)
plt.figure(figsize=(16,4))
ax = sns.barplot(x=percentages.index, y=percentages.values, palette="Set2")

for p in ax.patches:
    height = p.get_height()
    ax.annotate('{:1.0f}%'.format(height), (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.xlabel('Month')
plt.ylabel('Percentage')
plt.title('Absent percentage by month')
plt.xticks(rotation=45)
plt.show()

day_of_the_week = pd.to_datetime(df['tapahtumapvm'],format='%Y-%m-%d %H.%M.%S,%f').dt.day_name()
order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
day_of_the_week = pd.Categorical(day_of_the_week, categories=order, ordered=True)
ax = sns.countplot(x=day_of_the_week, palette = "Set2")
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.xlabel('Weekday')
plt.ylabel('Count')
plt.title('Absent count by weekday')
plt.xticks(rotation=45)



day_of_the_week = pd.to_datetime(df['tapahtumapvm'], format='%Y-%m-%d %H.%M.%S,%f').dt.day_name()

# Order of the days
order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Convert day_of_the_week to categorical with the specified order
day_of_the_week = pd.Categorical(day_of_the_week, categories=order, ordered=True)

# Compute percentages for each day
counts = day_of_the_week.value_counts()
percentages = (counts / counts.sum() * 100).reindex(order)
percentages=percentages.round(0)
# Plot using sns.barplot
ax = sns.barplot(x=percentages.index, y=percentages.values, palette="Set2")

# Annotate each bar with its percentage value
for p in ax.patches:
    height = p.get_height()
    ax.annotate('{:1.0f}%'.format(height), (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.xlabel('Weekday')
plt.ylabel('Percentage')
plt.title('Absent percentage by weekday')
plt.xticks(rotation=45)
plt.show()


df = pd.read_csv('../Data/UC_OPPILAAT.csv', delimiter=';', encoding='utf-8')
mapping_category1 = {'Yleinen tuki': 'CommonSupport', 'Tehostettu tuki': 'IntensiveSupport', 'Erityinen tuki': 'SpecialSupport'}
mapping_category2 = {'Ei': 'No', 'Kyllä': 'Yes'}
mapping_category3 = {'Mies': 'Male', 'Nainen': 'Female'}
df['tuen_vaihe'] = df['tuen_vaihe'].replace(mapping_category1)
df['stk'] = df['stk'].replace(mapping_category2)
df['sukupuoli'] = df['sukupuoli'].replace(mapping_category3)
df.columns = ['StudentHash','School name', 'luokka', 'Support type', 'Finnish as a second language', 'Gender', 'Class','Birth quarter']
df.drop(['StudentHash','luokka'], axis=1,inplace=True)
sns.countplot(x=df['Birth quarter'], palette = "Set2")
plt.xlabel('School name')
plt.ylabel('Count')
plt.title('Number of student by school')
plt.xticks(rotation=45)


total = len(day_of_the_week)

# Add percentage labels
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width() / 2.
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')




# Show the plot
plt.show()
#################################################################################################


df = pd.read_csv('../Data/Final_data/data5.csv', delimiter=';', encoding='utf-8')

mapping_category1 = {'CommonSupport': 1, 'IntensiveSupport': 2, 'SpecialSupport': 3}
mapping_category2 = {'No': 0, 'Yes': 1}
mapping_category3 = {'Male': 0, 'Female': 1}


# Map the categorical values to numerical values using applymap()
df['Support_type'] = df['Support_type'].replace(mapping_category1)
df['Finnish_as_secondlanguage'] = df['Finnish_as_secondlanguage'].replace(mapping_category2)
df['Gender'] = df['Gender'].replace(mapping_category3)

# Define a list of column names to select
columns_to_select = ['Grade', 
                     'Total_absent_time', 
                     'Support_type', 'Finnish_as_secondlanguage', 'Gender',
                     'Class', 'Birth_quarter']
df.dropna(subset = ['Grade'], inplace=True)

# Select the columns using df.loc
selected_columns = df.loc[:, columns_to_select]

corr_matrix = selected_columns.corr()
# Calculate the p-values for each correlation coefficient
p_values = selected_columns.corr(method=lambda x, y: np.array(scipy.stats.pearsonr(x, y))[1])

# Filter the correlation matrix based on p-value < 0.05
corr_matrix_filtered = corr_matrix.mask(p_values >= 0.05)
corr_matrix_filtered = corr_matrix_filtered.round(2)
vmax_val = np.nanmax(corr_matrix_filtered.abs())

sns.heatmap(corr_matrix_filtered, annot=True, cmap='coolwarm', vmin=-vmax_val, vmax=vmax_val)
# sns.heatmap(corr_matrix_filtered, annot=True, cmap='coolwarm')
# plt.tight_layout()
plt.xlabel('Features')
plt.title('Corelation between features')

# Significant found
# Support and grade highly negatively corelated 
# Sex is positively corelated (women tend to have better grades than men)
# Total absent time, Finnish as second language, grade, birth quarter is negatively corelated. 
# The lower the total absent time, birthquarter, Finnish speaker the higher grade 
plt.show()
plt.close()

##############################################################################################################
# Plotting box plot
df = pd.read_csv('../Data/Final_data/Final_data3.csv', delimiter=';', encoding='utf-8')
mapping_category1 = {'Yleinen tuki': 'CommonSupport', 'Tehostettu tuki': 'IntensiveSupport', 'Erityinen tuki': 'SpecialSupport'}
mapping_category2 = {'Ei': 'No', 'Kyllä': 'Yes'}
mapping_category3 = {'Mies': 'Male', 'Nainen': 'Female'}

# Map the categorical values to numerical values using applymap()
df['Support_type'] = df['Support_type'].replace(mapping_category1)
df['Finnish_as_secondlanguage'] = df['Finnish_as_secondlanguage'].replace(mapping_category2)
df['Gender'] = df['Gender'].replace(mapping_category3)

# Define a list of column names to select
columns_to_select = ['Grade', 
                     'Total_absent_time', 'Number_of_absent',
                     'Support_type', 'Finnish_as_secondlanguage', 'Gender',
                     'Class', 'Birth_quarter']
df.dropna(subset = ['Grade'], inplace=True)

# Map the categorical values to numerical values using applymap()
df['Support_type'] = df['Support_type'].replace(mapping_category1)
df['Finnish_as_secondlanguage'] = df['Finnish_as_secondlanguage'].replace(mapping_category2)
df['Gender'] = df['Gender'].replace(mapping_category3)

# Class-Grade line chart by gender
df_grade = df.dropna(subset = ['Grade']).reset_index(drop=True)

props = dict(boxes="DarkGreen", whiskers="DarkOrange", medians="DarkBlue", caps="Gray")

boxprops = dict(linestyle='-', linewidth=2, color='k')
medianprops = dict(linestyle='-', linewidth=2, color='orange')
whiskerprops=dict(linestyle='--', linewidth=1.5, color='darkgreen')
capprops=dict(linestyle='-', linewidth=1.5, color='')

df_grade.boxplot(column='Grade', by=['Class', 'Gender'], showfliers=False,showmeans=True, 
                boxprops=boxprops, medianprops=medianprops, 
                whiskerprops=whiskerprops,
                # capprops=capprops,
                return_type='dict',grid=False)
plt.suptitle('')
plt.title('Box plot of Grade by Class for each Gender')
plt.xlabel('Class and Gender')
plt.ylabel('Grade')
plt.xticks(rotation=45)
plt.show()


df_grade.boxplot(column='Grade', by=['Class', 'Finnish_as_secondlanguage'], showfliers=False,showmeans=True, 
                boxprops=boxprops, medianprops=medianprops, 
                whiskerprops=whiskerprops,
                # capprops=capprops,
                return_type='dict',grid=False)
plt.suptitle('')
plt.title('Box plot of Grade by Class for each language minority group')
plt.xlabel('Class and Language minority')
plt.ylabel('Grade')
plt.xticks(rotation=45)
plt.show()

df_grade.boxplot(column='Grade', by=['Class', 'Birth_quarter'], showfliers=False,showmeans=True, 
                boxprops=boxprops, medianprops=medianprops, 
                whiskerprops=whiskerprops,
                # capprops=capprops,
                return_type='dict',grid=False)
plt.suptitle('')
plt.title('Box plot of Grade by Class for each birth quarter')
plt.xlabel('Class and Birth quarter')
plt.ylabel('Grade')
plt.xticks(rotation=45)
plt.show()

# box plot ended


############################# plot line chart #############################
grouped=df_grade.groupby(['Class', 'Subject', 'Gender'])['Grade'].mean().reset_index()
grouped = grouped.groupby(['Class', 'Gender'])['Grade'].mean().reset_index()
ax=sns.lineplot(data=grouped, x='Class', y='Grade', hue='Gender', marker='o', palette = "Set2")

# Pivot the DataFrame to have Class as the index and Gender as the columns
pivot_table = pd.pivot_table(grouped, values='Grade', index='Class', columns='Gender')
plt.title('Average Grade by Class and Gender')
plt.xlabel('Class')
plt.ylabel('Average Grade')

# Create a line chart
pivot_table.plot(marker='o')
for grade_level in pivot_table.index:
    for gender in pivot_table.columns:
        grade = pivot_table.loc[grade_level, gender].round(2)
        ax.annotate(str(grade), xy=(grade_level, grade), xytext=(3, 10),
                    textcoords='offset points', ha='left', va='center')
plt.show()


df_grade = df.dropna(subset = ['Grade']).reset_index(drop=True)
grouped=df_grade.groupby(['Class', 'Subject', 'Finnish_as_secondlanguage'])['Grade'].mean().reset_index()
grouped = grouped.groupby(['Class', 'Finnish_as_secondlanguage'])['Grade'].mean().reset_index()

ax=sns.lineplot(data=grouped, x='Class', y='Grade', hue='Finnish_as_secondlanguage', marker='o', palette = "Set2")

# Pivot the DataFrame to have Class as the index and Finnish_as_secondlanguage as the columns
pivot_table = pd.pivot_table(grouped, values='Grade', index='Class', columns='Finnish_as_secondlanguage')
plt.title('Average Grade by Class and Finnish as second language')
plt.xlabel('Class')
plt.ylabel('Average Grade')

# Create a line chart
pivot_table.plot(marker='o')
for grade_level in pivot_table.index:
    for gender in pivot_table.columns:
        grade = pivot_table.loc[grade_level, gender].round(2)
        ax.annotate(str(grade), xy=(grade_level, grade), xytext=(3, 5),
                    textcoords='offset points', ha='left', va='center')
plt.show()


df_grade = df.dropna(subset = ['Grade']).reset_index(drop=True)
grouped=df_grade.groupby(['Class', 'Subject', 'Birth_quarter'])['Grade'].mean().reset_index()
grouped = grouped.groupby(['Class', 'Birth_quarter'])['Grade'].mean().reset_index()

ax=sns.lineplot(data=grouped, x='Class', y='Grade', hue='Birth_quarter', marker='o', palette = "Set2")

# Pivot the DataFrame to have Class as the index and Finnish_as_secondlanguage as the columns
pivot_table = pd.pivot_table(grouped, values='Grade', index='Class', columns='Birth_quarter')
plt.title('Average Grade by Class and Birth quarter')
plt.xlabel('Class')
plt.ylabel('Average Grade')

# Create a line chart
pivot_table.plot(marker='o')
for grade_level in pivot_table.index:
    for gender in pivot_table.columns:
        grade = pivot_table.loc[grade_level, gender].round(2)
        ax.annotate(str(grade), xy=(grade_level, grade), xytext=(3, 5),
                    textcoords='offset points', ha='left', va='center')
plt.show()




df = pd.read_csv('../Data/Final_data/Final_data3.csv', delimiter=';', encoding='utf-8')
mapping_category1 = {'Yleinen tuki': 'CommonSupport', 'Tehostettu tuki': 'IntensiveSupport', 'Erityinen tuki': 'SpecialSupport'}
mapping_category2 = {'Ei': 'No', 'Kyllä': 'Yes'}
mapping_category3 = {'Mies': 'Male', 'Nainen': 'Female'}
mapping_category4 = {'Historia': 'History','Kuvataide': 'Art','Katsomusaine': 'Ethics','Käsityö' : 'Handicraft','Liikunta' : 'Physical education','Matematiikka' : 'Mathematics','Musiikki' : 'Music','Ympäristöoppi' : 'Environmental studies','Äidinkieli' : 'Finnish language','Biologia' : 'Biology','Kotitalous' : 'Cooking studies','Maantieto' : 'Geography','Kemia' : 'Chemistry','Terveystieto' : 'Health education','Yhteiskuntaoppi' : 'Social studies','Fysiikka' : 'Physics','Tietotekniikka' : 'Information technology'}



# Map the categorical values to numerical values using applymap()
df['Support_type'] = df['Support_type'].replace(mapping_category1)
df['Finnish_as_secondlanguage'] = df['Finnish_as_secondlanguage'].replace(mapping_category2)
df['Gender'] = df['Gender'].replace(mapping_category3)
df['Subject'] = df['Subject'].replace(mapping_category4)

# Define a list of column names to select
columns_to_select = ['Grade', 
                     'Total_absent_time', 'Number_of_absent',
                     'Support_type', 'Finnish_as_secondlanguage', 'Gender',
                     'Class', 'Birth_quarter']
df.dropna(subset = ['Grade'], inplace=True)

df_filtered = df[['Gender', 'Grade','Class', 'Subject']]
# df_filtered = df[['Gender', 'Grade','Class', 'Subject', 'Finnish_as_secondlanguage']]

# Group by Gender and Grade, calculate average grade per subject
grouped = df_filtered.groupby(['Gender', 'Class', 'Subject']).mean().reset_index()

# Pivot the DataFrame
pivot_table = pd.pivot_table(grouped, values='Grade', index='Gender',columns=['Class','Subject'])
pivot_table.dropna(axis=1, inplace=True)


# Plot radar chart for each grade by Gender
for grade in pivot_table.columns.levels[0]:
    df_grade = pivot_table[grade].reset_index()
    gender = df_grade['Gender']
    scores = df_grade.drop('Gender', axis=1).values[0]
    df_grade=df_grade.melt(id_vars=['Gender'], value_vars=list(df_grade.drop('Gender', axis=1).columns))
    plt.figure(figsize=(12,4))
    ax=sns.lineplot(data=df_grade, x='Subject', y='value', hue='Gender', marker='o')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.title('Average Grade by subject in grade' + str(grade)+' by Gender') 
    plt.xlabel('Subject')
    plt.ylabel('Average grade')

    for _, row in df_grade.iterrows():
        plt.annotate(round(row['value'],2), (row['Subject'], row['value']), textcoords="offset points", xytext=(3,0), ha='center')
    plt.tight_layout()
    filename = "Average grade - Subject by class"+str(grade) +".png"  # Specify the file name and extension
    plt.savefig(filename)
    plt.show()

df_filtered = df[['Grade','Class', 'Subject', 'Finnish_as_secondlanguage']]

# Group by Gender and Grade, calculate average grade per subject
grouped = df_filtered.groupby(['Finnish_as_secondlanguage', 'Class', 'Subject']).mean().reset_index()

# Pivot the DataFrame
pivot_table = pd.pivot_table(grouped, values='Grade', index='Finnish_as_secondlanguage',columns=['Class','Subject'])
pivot_table.dropna(axis=1, inplace=True)


# Plot radar chart for each grade by Finnish_as_secondlanguage
for grade in pivot_table.columns.levels[0]:
    df_grade = pivot_table[grade].reset_index()
    gender = df_grade['Finnish_as_secondlanguage']
    scores = df_grade.drop('Finnish_as_secondlanguage', axis=1).values[0]
    df_grade=df_grade.melt(id_vars=['Finnish_as_secondlanguage'], value_vars=list(df_grade.drop('Finnish_as_secondlanguage', axis=1).columns))
    plt.figure(figsize=(12,4))
    ax=sns.lineplot(data=df_grade, x='Subject', y='value', hue='Finnish_as_secondlanguage', marker='o')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.title('Average Grade by subject in grade' + str(grade) +' by Finnish as a second language')
    plt.xlabel('Subject')
    plt.ylabel('Average grade')

    for _, row in df_grade.iterrows():
        plt.annotate(round(row['value'],2), (row['Subject'], row['value']), textcoords="offset points", xytext=(5,0), ha='center')
    
    plt.tight_layout()
    filename = "Average grade - Subject by class"+str(grade) +"by_STK.png"  # Specify the file name and extension
    plt.savefig(filename)
    plt.show()


df_filtered = df[['Grade','Class', 'Subject', 'Birth_quarter']]

# Group by Gender and Grade, calculate average grade per subject
grouped = df_filtered.groupby(['Birth_quarter', 'Class', 'Subject']).mean().reset_index()

# Pivot the DataFrame
pivot_table = pd.pivot_table(grouped, values='Grade', index='Birth_quarter',columns=['Class','Subject'])
pivot_table.dropna(axis=1, inplace=True)


# Plot radar chart for each grade by Birth_quarter
for grade in pivot_table.columns.levels[0]:
    df_grade = pivot_table[grade].reset_index()
    gender = df_grade['Birth_quarter']
    scores = df_grade.drop('Birth_quarter', axis=1).values[0]
    df_grade=df_grade.melt(id_vars=['Birth_quarter'], value_vars=list(df_grade.drop('Birth_quarter', axis=1).columns))
    plt.figure(figsize=(12,4))
    ax=sns.lineplot(data=df_grade, x='Subject', y='value', hue='Birth_quarter', marker='o',palette=['red', 'orange', 'green', 'darkgrey'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.title('Average Grade by subject in grade ' + str(grade) + ' by Sum quarter')
    plt.xlabel('Subject')
    plt.ylabel('Average grade')

    # for _, row in df_grade.iterrows():
    #     plt.annotate(round(row['value'],2), (row['Subject'], row['value']), textcoords="offset points", xytext=(0,10), ha='center')

    filename = "Average grade - Subject by Birthquarter grade "+str(grade) +".png"  # Specify the file name and extension
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()