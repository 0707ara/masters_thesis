import pandas as pd
import numpy as np

df = pd.read_csv('../Data/Final_data/data4.csv', delimiter=';', encoding='utf-8')
df = df.dropna(subset = ['Grade']).reset_index(drop=True)

student_list = df['Student_hash'].unique()
final_result = pd.DataFrame()
result = pd.DataFrame()

for i in student_list:
    data = df[df['Student_hash']==i]
    if(len(data['Class'].unique()) == 3) & (len(data['Subject'].unique()) > 7):
        for j in np.sort(data['Class'].unique()):
            data_class = data[data['Class']==j]
            filtered_data = data_class[data_class['Subject'].isin(['History','Ethics','Finnish language','Social studies'])]
            filtered_data_1 = data_class[data_class['Subject'].isin(['Art','Music','Handicraft','Cooking studies','Physical education','Health education'])]
            filtered_data_2 = data_class[data_class['Subject'].isin(['Mathematics','Environmental studies','Biology','Geography','Chemistry','Physics','Information technology'])]
            new_data1 = filtered_data.groupby(['Student_hash']).agg({\
                'School_name':'first',\
                'Class_1':'first',\
                'Class':'first',\
                'Support_type':'first',\
                'Special_support':'first',\
                'Support_started_date':'first',\
                'Finnish_as_secondlanguage':'first',\
                'Gender':'first',\
                'Birth_quarter':'first',\
                'Academic_year':'first',\
                'Total_absent_time':'sum',\
                'Number_of_absent':'first',\
                'Grade':'mean'
                }).reset_index()
            new_data1['subject_category'] = 'Social_science_category'
            new_data2 = filtered_data_1.groupby(['Student_hash']).agg({\
                'School_name':'first',\
                'Class_1':'first',\
                'Class':'first',\
                'Support_type':'first',\
                'Special_support':'first',\
                'Support_started_date':'first',\
                'Finnish_as_secondlanguage':'first',\
                'Gender':'first',\
                'Birth_quarter':'first',\
                'Academic_year':'first',\
                'Total_absent_time':'sum',\
                'Number_of_absent':'first',\
                'Grade':'mean'
                }).reset_index()
            new_data2['subject_category'] = 'Art_physical_category'
            new_data3 = filtered_data_2.groupby(['Student_hash']).agg({\
                'School_name':'first',\
                'Class_1':'first',\
                'Class':'first',\
                'Support_type':'first',\
                'Special_support':'first',\
                'Support_started_date':'first',\
                'Finnish_as_secondlanguage':'first',\
                'Gender':'first',\
                'Birth_quarter':'first',\
                'Academic_year':'first',\
                'Total_absent_time':'sum',\
                'Number_of_absent':'first',\
                'Grade':'mean'
                }).reset_index()
            new_data3['subject_category'] = 'Science_category'
            result = pd.concat([new_data1, new_data2,new_data3], axis=0,ignore_index=True)
            final_result = pd.concat([final_result, result], axis=0,ignore_index=True)    
            new_data1 = None
            new_data2 = None
            new_data3 = None
            result = pd.DataFrame()
students_with_three_rows = final_result['Student_hash'].value_counts()
students_with_three_rows = students_with_three_rows[students_with_three_rows == 9].index.tolist()

# Filter the DataFrame
filtered_df = final_result[final_result['Student_hash'].isin(students_with_three_rows)]
filtered_df.to_csv('../Data/Final_data/subject_categorized.csv', sep=';', header=True, index=False)