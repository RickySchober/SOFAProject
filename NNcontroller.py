import Sofa
import random
import csv
from datetime import datetime
from Sofa.Core import Controller
from Sofa.constants import *
import torch
from training import GRUInverseModel  
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

class NeuralNetController(Controller):
    def __init__(self, node, finger):
        super().__init__()
       #y: 10-39
       #x: -55 - -60
        self.desired_pos = np.array([0.0,0.0,0.0])  # Initialize desired position
        self.setstart = False  # Flag to set the start position
        self.SEQ_LEN = 10  
        # Load Model
        self.feature_scaler = joblib.load("feature_scaler.pkl")
        self.target_scaler = joblib.load("target_scaler.pkl")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GRUInverseModel()
        self.model.load_state_dict(torch.load('gru_inverse_model.pth'))
        self.model.eval()  # Set to inference mode
        self.model.to(self.device)
       
        self.node = node
        self.finger_nodes = finger
        self.pressure_values = [0.0 for _ in self.finger_nodes]
        self.log_file = f"test_data2.csv"

        self.modifier = 0.05
        self.direction = False
    
        self.seq_buffer = [[] for _ in self.finger_nodes]  # One sequence buffer per finger

        

        with open(self.log_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time, ""RMSE", "pos_x", "pos_y", "pos_z", "des_x", "des_y", "des_z"])

        self.time = 0.0
        self.dt = self.node.dt.value if hasattr(self.node, "dt") else 0.01

    def make_prediction(self, sequence, desired_pos):
        seq_input_scaled = self.feature_scaler.transform(sequence)[:, :3]
        desired_pos_scaled = self.feature_scaler.transform(desired_pos.reshape(1, -1))[:, :3].squeeze()

        # Convert to tensor
        seq_input_tensor = torch.tensor(seq_input_scaled, dtype=torch.float32).unsqueeze(0)      # shape: [1, SEQ_LEN, 3]
        desired_pos_tensor = torch.tensor(desired_pos_scaled, dtype=torch.float32).unsqueeze(0)  # shape: [1, 3]

        # Send to GPU if applicable
        seq_input_tensor = seq_input_tensor.to(self.device)
        desired_pos_tensor = desired_pos_tensor.to(self.device)

        # Predict
        with torch.no_grad():
            predicted_pressure = self.model(seq_input_tensor, desired_pos_tensor)

        # Denormalize pressure
        predicted_pressure_value = self.target_scaler.inverse_transform(predicted_pressure.cpu().numpy())[0, 0]
        print(f"Predicted pressure: {predicted_pressure_value}")
        return predicted_pressure_value

    def onAnimateBeginEvent(self, event):
        
        self.time += self.dt
        print(f"Desired position: {self.desired_pos}")

        for idx, finger in enumerate(self.finger_nodes):
            # Get position and velocity
            dofs = finger.getObject("tetras")
            com_pos = list(dofs.position.array().mean(axis=0))
            print(f"error: {com_pos[1] - self.desired_pos[1]}")
            self.seq_buffer[idx].append(com_pos)

            # Only predict when we have enough data
            if len(self.seq_buffer[idx]) >= self.SEQ_LEN + 1:
                # Prepare sequence and desired position
                sequence = np.array(self.seq_buffer[idx][-self.SEQ_LEN-1:-1])  # SEQ_LEN past positions
                if not self.setstart:
                    self.desired_pos = np.array(self.seq_buffer[idx][-1])
                    self.setstart = True
                    print(f"Setting desired position to: {self.desired_pos}")
                else:
                    if(self.desired_pos[1] < 10.0):
                        self.direction = True
                        print("Changing direction up")
                    elif(self.desired_pos[1] > 39.0):
                        self.direction = False    
                        print("Changing direction down")   
                    if self.direction:
                        self.desired_pos[1] += self.modifier
                        self.desired_pos[0] -= self.modifier/10        
                    else:
                        self.desired_pos[1] -= self.modifier
                        self.desired_pos[0] += self.modifier/10 
                self.pressure_values[idx] = self.make_prediction(sequence, self.desired_pos)

            # Set pressure
            cavity = finger.getChild("Cavity").getObject("SurfacePressureConstraint")
            cavity.value = [self.pressure_values[idx]]

            rmse = np.sqrt(np.mean((com_pos - self.desired_pos) ** 2))
            # Log to file
            with open(self.log_file, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.time, rmse,  *com_pos, *self.desired_pos
                ])

   