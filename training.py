import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import joblib

class GRUInverseModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=1, num_layers=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 3, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, sequence, desired_pos):
        _, h_n = self.gru(sequence)  
        h_n = h_n[-1]                # Get last layer's hidden state: [batch, hidden_dim]
        combined = torch.cat((h_n, desired_pos), dim=-1)
        return self.fc(combined)

if __name__ == "__main__":
    print(torch.cuda.is_available())

    # Load training data
    df = pd.read_csv('training_data3.csv')

    position_cols = ['pos_x', 'pos_y', 'pos_z']
    pressure_col = 'pressure'

    features = df[position_cols].values
    target = df[pressure_col].values.reshape(-1, 1)

    # Normalize features and target
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    features_scaled = feature_scaler.fit_transform(features)
    target_scaled = target_scaler.fit_transform(target)

    def create_sequences(features, target, seq_len):
        X_seq, X_desired, y = [], [], []
        for i in range(len(features) - seq_len):
            seq = features[i:i+seq_len]  
            desired_pos = features[i+seq_len][:3]  
            pressure = target[i+seq_len][0]  

            X_seq.append(seq[:, :3])  
            X_desired.append(desired_pos)
            y.append(pressure)

        return np.array(X_seq), np.array(X_desired), np.array(y)

    SEQ_LEN = 10
    X_seq, X_desired, y = create_sequences(features_scaled, target_scaled, SEQ_LEN)

    # Convert to PyTorch tensors
    X_seq = torch.tensor(X_seq, dtype=torch.float32)
    X_desired = torch.tensor(X_desired, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    model = GRUInverseModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_seq = X_seq.to(device)
    X_desired = X_desired.to(device)
    y = y.to(device)
    model = model.to(device)

    for epoch in range(2000):
        optimizer.zero_grad()
        output = model(X_seq, X_desired)
        loss = criterion(output, y.unsqueeze(1))
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
    # Save the model
    torch.save(model.state_dict(), 'gru_inverse_model.pth')
    joblib.dump(feature_scaler, "feature_scaler.pkl")
    joblib.dump(target_scaler, "target_scaler.pkl")