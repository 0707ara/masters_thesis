import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler

df = pd.read_csv('../Data/Final_data/subject_categorized.csv', delimiter=';', encoding='utf-8')
mapping_col = ['Support_type','Finnish_as_secondlanguage','Gender','Academic_year','subject_category']
encoders = {}
for column in mapping_col:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

mapping_df = pd.DataFrame()
max_classes = max(len(encoder.classes_) for encoder in encoders.values())
for column, encoder in encoders.items():
    classes = pd.Series(index=encoder.transform(encoder.classes_), data=encoder.classes_)
    
    if len(classes) < max_classes:
        classes = classes.reindex(range(max_classes))
        
    mapping_df[column] = classes

df['Total_absent_time'] = df['Total_absent_time'].round(2)
df['Grade'] = df['Grade'].round(2)


def map_score(score):
    if 4 <= score <= 5:
        return 'low'
    elif 6 <= score <= 7:
        return 'medium'
    else:
        return 'high'
    
# Bin scores into categories

df['state'] = df['Grade'].apply(map_score)

states = df['state'].unique()
transition_matrix_art = pd.DataFrame(data=0, index=states, columns=states).fillna(0)
transition_matrix_sci = pd.DataFrame(data=0, index=states, columns=states).fillna(0)
transition_matrix_sc = pd.DataFrame(data=0, index=states, columns=states).fillna(0)
final_art = pd.DataFrame(data=0, index=states, columns=states).fillna(0)
final_sci = pd.DataFrame(data=0, index=states, columns=states).fillna(0)
final_sc = pd.DataFrame(data=0, index=states, columns=states).fillna(0)

for i in range(4,7):
    transition_matrix_art =transition_matrix_art.fillna(0)
    transition_matrix_sci =transition_matrix_sci.fillna(0)
    transition_matrix_sc =transition_matrix_sc.fillna(0)
    class0_students = df.loc[df['Class'] == i, 'Student_hash'].unique()
    class1_students = df.loc[df['Class'] == i+1, 'Student_hash'].unique()
    class2_students = df.loc[df['Class'] == i+2, 'Student_hash'].unique()

# Then, find the intersection of the three arrays
    student_list = np.sort(np.intersect1d(class0_students, np.intersect1d(class1_students, class2_students)))
#  for loop student/ and check the matrix. 
    for j in student_list:
        data = df[df['Student_hash']==j]
        for j in range(4,6):
            current_state_art = data[(data['subject_category']==0) & (data['Class']==j)]['state']
            current_state_sci = data[(data['subject_category']==1) & (data['Class']==j)]['state']
            current_state_pe = data[(data['subject_category']==2) & (data['Class']==j)]['state']
            next_state_art = data[(data['subject_category']==0) & (data['Class']==j+1)]['state']
            next_state_sci = data[(data['subject_category']==1) & (data['Class']==j+1)]['state']
            next_state_pe = data[(data['subject_category']==2) & (data['Class']==j+1)]['state']
            transition_matrix_art.loc[current_state_art,next_state_art] += 1
            transition_matrix_sci.loc[current_state_sci,next_state_sci] += 1
            transition_matrix_sc.loc[current_state_pe,next_state_pe] += 1
            # Normalize to get probabilities
    row_sums_art = transition_matrix_art.sum(axis=1).replace(0, 1)
    transition_matrix_art = transition_matrix_art.div(row_sums_art, axis=0)
    row_sums_sci = transition_matrix_sci.sum(axis=1).replace(0, 1)
    transition_matrix_sci = transition_matrix_sci.div(row_sums_sci, axis=0)
    row_sums_pe = transition_matrix_sc.sum(axis=1).replace(0, 1)
    transition_matrix_sc = transition_matrix_sc.div(row_sums_pe, axis=0)
    print(i)
    if i == 4:
        final_art = transition_matrix_art
        final_sci = transition_matrix_sci
        final_sc = transition_matrix_sc            
    else: 
        final_art = transition_matrix_art * final_art
        final_sci = transition_matrix_sci * final_sci
        final_sc = transition_matrix_sc * final_sc




