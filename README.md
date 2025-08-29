# Soft Robotics: RNN-Based Control of a Pneumatic Actuator in SOFA

This project explores **learning-based control for soft robots** using a recurrent neural network (RNN).  
A simple pneumatic actuator was modeled in the [SOFA Framework](https://www.sofa-framework.org/), and an **inverse model** was trained to directly control the actuator’s position.

---

## Demo
This demo show the movements of the simple pnuematic actuator in simulation during the dataset generation process.
<p align="center">
  <img src="SOFASImulationGIF.gif" width="600" alt="SOFA Pneumatic Actuator RNN Control Demo"/>
</p>

---

## Project Overview

- **Simulation Environment**: SOFA Framework was used to model a pneumatic actuator.  
- **Dataset Generation**:  
  - Semi-Random pressure inputs were applied to the actuator to ensure data covers the entire joint space.  
  - Resulting actuator states (pressure, positions, velocities) were recorded at a regular interval to build a training and testing datasets.  
- **Model Training**:  
  - A **Recurrent Neural Network (RNN)** was trained as an **inverse dynamics model**.  
  - The RNN maps a desired position to the actuator’s required input pressure.  
- **Control**:  
  - The trained model allows direct position control of the actuator using keyboard input. 

---

## Repository Structure
  - Data:
    - Training: training data generated in simulation.
    - Test: test data used to verify accuracy of trained model.
    - Mesh: the mesh for the simple pneumatic actuator
  - Results: contains the graphs of actuators position and error over time using the trained model to control the actuator.
  
