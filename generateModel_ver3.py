#使用lstm
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.init as init
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

csv_directory = "./dataset"
csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
csv_files_test_n = round(len(csv_files) * 0.2)

train_data = []
test_data = []


for index, csv_file in enumerate(csv_files):
    file_path = os.path.join(csv_directory, csv_file)
    df = pd.read_csv(file_path)  #  一篇csv的資料
    df['token'] = df['token'].apply(lambda x: [float(i) for i in eval(x)])  #  轉成list
    
    csv_data_collect = df['token'].tolist()
    for i in df.values.tolist():
        csv_data_collect.append(i[0])    

    #for test
    train_data.append(csv_data_collect)
    test_data.append(csv_data_collect)

    if index > csv_files_test_n:
        train_data.append(csv_data_collect)
    else:
        test_data.append(csv_data_collect)

#---------------------------------

data_tensor = torch.tensor(train_data[0], dtype=torch.float32).to(device)

X = data_tensor[:, :-4].unsqueeze(1)
Y = data_tensor[:, -4:]  

input_size = X.size(-1)  
print(input_size)
hidden_size = 64  
num_layers = 1  
lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
linear = nn.Linear(hidden_size, Y.size(-1)).to(device)  

lstm = lstm.to(device)
linear = linear.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(lstm.parameters()) + list(linear.parameters()), lr=0.001)

best_model = None
lowest_loss = float('inf')

batch_size = 64
num_epochs = 20

training_losses = []
testing_losses = []

for epoch in range(num_epochs):

    total_loss = 0
    lstm.train() 

    for data in train_data:

        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

        X = data_tensor[:, :-4].unsqueeze(1) 
        Y = data_tensor[:, -4:]  

        dataset = TensorDataset(X, Y)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        
        for input_seq, target_seq in dataloader:
            h0 = torch.zeros(num_layers, input_seq.size(0), hidden_size).to(device)
            c0 = torch.zeros(num_layers, input_seq.size(0), hidden_size).to(device)
            lstm_output, _ = lstm(input_seq, (h0, c0))
            # print(lstm_output)
            # print(lstm_output[0].shape)
            lstm_output_last = lstm_output[:, -1, :]
            
            predicted_output = linear(lstm_output_last)
            loss = criterion(predicted_output, target_seq)
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    avg_loss = total_loss / len(dataloader)
    training_losses.append(avg_loss)
    print(f"[Training] : Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    lstm.eval() 
    total_test_loss = 0
    with torch.inference_mode():
        for data in test_data:
            data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
            X = data_tensor[:, :-4].unsqueeze(1)
            Y = data_tensor[:, -4:]
            test_dataset = TensorDataset(X, Y)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

            for input_seq, target_seq in test_dataloader:
                h0 = torch.zeros(num_layers, input_seq.size(0), hidden_size).to(device)
                c0 = torch.zeros(num_layers, input_seq.size(0), hidden_size).to(device)
                lstm_output, _ = lstm(input_seq, (h0, c0))
                lstm_output_last = lstm_output[:, -1, :]
                predicted_output = linear(lstm_output_last)
                
                
                test_loss = criterion(predicted_output, target_seq)
                total_test_loss += test_loss.item()
        avg_test_loss = total_test_loss / len(test_dataloader)
        testing_losses.append(avg_test_loss)
        print(f"[Testing] : Epoch {epoch+1}/{num_epochs}, Average Test Loss: {avg_test_loss:.4f}")

        if avg_test_loss < lowest_loss:
            lowest_loss = avg_test_loss
            best_model = lstm.state_dict()

model_directory = './Supervised_Model/'

if not os.path.exists(model_directory):
    os.makedirs(model_directory)

torch.save(best_model, './Supervised_Model/best_model.pth')




