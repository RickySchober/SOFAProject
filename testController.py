import Sofa
import random
import csv
from datetime import datetime
from Sofa.Core import Controller

class DataCollectionController(Controller):
    def __init__(self, node, finger):
        super().__init__()
        self.node = node
        self.finger_nodes = finger
        self.pressure_values = [0.0 for _ in finger]
        self.log_file = f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.modifier = 0.0
        self.direction = False
        with open(self.log_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time", "finger", "pressure", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"])

        self.time = 0.0
        self.dt = self.node.dt.value if hasattr(self.node, "dt") else 0.01

    def onAnimateBeginEvent(self, event):
        self.time += self.dt
        if(self.modifier > 1.75):
            self.direction = False
        elif(self.modifier < -0.5):
            self.direction = True
        if self.direction:
            self.modifier += 0.005
        else:
            self.modifier -= 0.005
        for idx, finger in enumerate(self.finger_nodes):
            # Apply small random pressure change
            delta = random.uniform(-0.5, 0.5)
            self.pressure_values[idx] = self.modifier + delta

            # Set pressure
            cavity = finger.getChild("Cavity").getObject("SurfacePressureConstraint")
            cavity.value = [self.pressure_values[idx]]

            # Get position and velocity
            dofs = finger.getObject("tetras")
            com_pos = list(dofs.position.array().mean(axis=0))
            com_vel = list(dofs.velocity.array().mean(axis=0))

            # Log to file
            with open(self.log_file, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.time, idx + 1, self.pressure_values[idx],
                    *com_pos, *com_vel
                ])
