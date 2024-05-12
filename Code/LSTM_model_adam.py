import seaborn as sns
import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import LSTM, Dense,Concatenate,Input
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import tensorflow as tf


df = pd.read_csv('../Data/Final_data/subject_categorized.csv', delimiter=';', encoding='utf-8')
df = df.dropna(subset = ['Grade']).reset_index(drop=True)
df.drop(['Class_1','Number_of_absent','Special_support','Support_started_date','School_name'], axis=1,inplace=True)


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

scaler = MinMaxScaler()   
scale_columns= ['Total_absent_time','Class','subject_category','Grade','Birth_quarter','Academic_year','Support_type']
for i in scale_columns:
    df[i] = scaler.fit_transform(df[i].values.reshape(-1, 1))
    
df['Class'] = df['Class'].round(1)
df['Support_type'] = df['Support_type'].round(1)
df['Birth_quarter'] = df['Birth_quarter'].round(1)
df['Academic_year'] = df['Academic_year'].round(1)
df['subject_category'] = df['subject_category'].round(1)

x_input = []
y_output = []
x_input_1 = []
y_output_1 = []
class0_students = df.loc[df['Academic_year'] == 0, 'Student_hash'].unique()
class1_students = df.loc[df['Academic_year'] == 0.5, 'Student_hash'].unique()
class2_students = df.loc[df['Academic_year'] == 1, 'Student_hash'].unique()

# Then, find the intersection of the three arrays
student_list = np.sort(np.intersect1d(class0_students, np.intersect1d(class1_students, class2_students)))
order = 0

for i in student_list: 

    order += 1
    data = df[df['Student_hash']==i]
    x_input.append(data[data['Academic_year']==0].drop(['Student_hash'], axis=1).round(6).values.tolist())
    y_output.append(data[data['Academic_year'] == 0.5]['Grade'].mean())
    x_input_1.append(data[data['Academic_year']== 0.5].drop(['Student_hash'], axis=1).round(6).values.tolist())
    y_output_1.append(data[data['Academic_year'] == 1]['Grade'].mean())

print(len(x_input),len(y_output))
print(len(x_input_1),len(y_output_1))


x_input=np.array(x_input)
y_output=np.array(y_output)
x_input_1=np.array(x_input_1)
y_output_1=np.array(y_output_1)


X_train, X_test, y_train, y_test = train_test_split(x_input, y_output, test_size=0.2, random_state=42)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(x_input_1, y_output_1, test_size=0.2, random_state=42)

input_1 = Input(shape=(len(x_input[0]), 9))
input_2 = Input(shape=(len(x_input_1[0]), 9))


lstm_out1 = LSTM(32, input_shape=(len(x_input),len(x_input[0])), return_sequences=True)(input_1)
lstm_out2 = LSTM(32, input_shape=(len(x_input_1),len(x_input_1[0])), return_sequences=True)(input_2)

# Merge the LSTM outputs
merged = Concatenate(axis=-1)([lstm_out1, lstm_out2])

# Further dense layers can be added if required
lstm_out3 = LSTM(64, return_sequences=False)(merged)

# Output layer
output = Dense(1, activation='sigmoid', name="output")(lstm_out3)
# Something you can try
# model.compile(loss='mae', optimizer='adam')

model = Model(inputs=[input_1, input_2], outputs=output)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error')
# model.compile(optimizer=tf.keras.optimizers.Adagrad(), loss='mean_squared_error')
# model.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.7), loss='mean_squared_error')

# Print the model architecture
model.summary()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 9)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 9)

X_train_1 = X_train_1.reshape(X_train_1.shape[0], X_train_1.shape[1], 9)
X_test_1 = X_test_1.reshape(X_test_1.shape[0], X_test_1.shape[1], 9)


history = model.fit(
    [X_train, X_train_1],  # Input data for both branches
    y_train_1,  # Target data
    epochs=50,  # Number of epochs e.g., 100
    batch_size=40,  # Batch size e.g., 32
    validation_data=([X_test, X_test_1], y_test)  # Validation data
)

train_mse = history.history['loss']
val_mse = history.history['val_loss']

threshold = 0.1 

train_accuracy = np.mean(np.abs(model.predict([X_train, X_train_1]) - y_train_1) < threshold)
val_accuracy = np.mean(np.abs(model.predict([X_test, X_test_1]) - y_test) < threshold)

# Create a range of epochs
epochs = range(1, len(train_mse) + 1)

# Plot the training and validation MSE
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_mse, color='orange', label='Training MSE')
plt.plot(epochs, val_mse, 'b', label='Test MSE')
plt.title('Training and Validation MSE (Adam)')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()

plot_model(model, to_file='lstm_model.png', show_shapes=True, show_layer_names=True)

plt.figure(figsize=(12, 6))
plt.plot(epochs, [train_accuracy]*len(epochs), color='orange', label='Training Accuracy')
# plt.plot(epochs, [val_accuracy]*len(epochs), 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy (Adam)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

plot_model(model, to_file='lstm_model.png', show_shapes=True, show_layer_names=True)

# 2. Evaluate the model
loss = model.evaluate([X_test, X_test_1], y_test_1, batch_size=32) 
print(f"Test loss: {loss}")

# 3. Make predictions
predictions = model.predict([X_test, X_test_1])

mse = np.mean((predictions - y_test)**2)
print(f"Computed MSE: {mse}")