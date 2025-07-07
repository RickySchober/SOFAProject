import numpy as np
import pandas as pd
import random
import Sofa
import SofaRuntime
import time

# Create the SOFA simulation environment
def createScene(rootNode):
    # Set up the root node (Scene)
    rootNode.addObject('RequiredPlugin', name="SoftRobots")
    rootNode.addObject('VisualStyle', displayFlags="showAll")

    # Create the actuator's geometry
    actuator = rootNode.addChild('Actuator')
    actuator.addObject('MechanicalObject', name='actuatorMesh', template='Vec3d', position='0 0 0, 0.1 0.1 0, 0.2 0 0')
    actuator.addObject('TetrahedronSetTopologyContainer', name='topo', files='meshfile')
    actuator.addObject('TetrahedronSetTopologyModifier')

    # Create the pneumatic pressure actuator
    actuator.addObject('CavityPressureConstraint', name='pressure', cavity_pressure=0.0, stiffness=1000.0, damping=1.0)

    # Set up the force field for deformation (Soft robot material properties)
    actuator.addObject('LinearSolver', name="linearSolver")
    actuator.addObject('RestShapeSpringsForceField', name="restShape", stiffness=1.0)

    # Add an external force (in case we need to simulate other external forces)
    actuator.addObject('ConstantForceField', name="gravityForce", gravity="0 -9.81 0")

    # Add a mechanical solver
    actuator.addObject('MechanicalObject', name='actuatorMechanical', template='Vec3d', position='0 0 0, 0.1 0.1 0, 0.2 0 0')
    actuator.addObject('GenericConstraintSolver', name="solver")

    # Set up the logger for data collection
    actuator.addObject('PythonScriptController', name='logger', script='random_pressure_input_and_log_data.py')

    return rootNode


# Function to apply random pressure changes and log data
def random_pressure_input_and_log_data(scene):
    # Data collection
    pressure_data = []
    deformation_data = []

    # Define the simulation time step and duration
    duration = 10  # seconds
    timestep = 0.1  # seconds

    # Start simulation
    start_time = time.time()
    while time.time() - start_time < duration:
        # Apply a random pressure between 0 and 1000 Pa
        pressure = random.uniform(0, 1000)  # Random pressure (Pa)
        scene.getChild('Actuator').getObject('pressure').cavity_pressure = pressure

        # Collect deformation data (e.g., displacement of the actuator)
        deformation = scene.getChild('Actuator').getObject('actuatorMesh').position
        deformation_data.append(deformation)

        # Store pressure data
        pressure_data.append(pressure)

        # Simulate for a timestep
        time.sleep(timestep)

    # Save the collected data as a CSV
    df = pd.DataFrame({
        'time': np.linspace(0, duration, len(pressure_data)),
        'pressure': pressure_data,
        'deformation': deformation_data
    })

    # Output data to CSV
    df.to_csv("actuator_data.csv", index=False)
    print("Data saved to actuator_data.csv")

# Initialize the scene and run it
if __name__ == "__main__":
    scene = Sofa.Core.init(rootNode=Sofa.Core.Node())
    createScene(scene)
    random_pressure_input_and_log_data(scene)
