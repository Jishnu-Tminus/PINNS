
# Physics-Informed Neural Networks for Solving PDEs

This repository contains the implementation of a Physics-Informed Neural Network (PINN) for solving a system of partial differential equations (PDEs). Specifically, the system consists of two PDEs that govern the behavior of streamfunction (`ψ`) and temperature (`θ`) fields in fluid dynamics, along with boundary conditions. The PINN is trained to approximate the solutions for both `ψ` and `θ` by minimizing the residuals of the PDEs and the boundary conditions.

## Overview

The code uses a deep learning approach to solve a system of coupled PDEs using a fully connected feed-forward neural network (PINN). This model learns both the streamfunction (`ψ`) and temperature field (`θ`) for a given Rayleigh number (`Ra`).

### Governing Equations

1. **Streamfunction equation** (`ψ`):
   \[
   \frac{\partial^2 \psi}{\partial x^2} + \frac{\partial^2 \psi}{\partial y^2} = - Ra \frac{\partial \theta}{\partial x}
   \]
   
2. **Temperature equation** (`θ`):
   \[
   \frac{\partial \theta}{\partial \tau} + \left(\frac{\partial \psi}{\partial y} \frac{\partial \theta}{\partial x} - \frac{\partial \psi}{\partial x} \frac{\partial \theta}{\partial y}\right) = \frac{\partial^2 \theta}{\partial x^2} + \frac{\partial^2 \theta}{\partial y^2}
   \]

### Boundary Conditions

The boundary conditions applied in the problem are:

- For temperature (`θ`), the boundary condition is applied at `Y = 0` and `Y = 1`.
- Placeholder boundary conditions are applied for the streamfunction (`ψ`).

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:
   ```bash
   pip install torch numpy matplotlib
   ```

## Code Explanation

### Model Architecture

The neural network used is a fully connected feed-forward network (PINN) with `tanh` activations between layers. The architecture is defined as:

- Input: `[X, Y, tau]` (3 input features for spatial and temporal dimensions)
- Hidden layers: 3 fully connected layers with 50 neurons each.
- Output: A single scalar value representing `ψ` or `θ` depending on the model.

### Loss Functions

- **PDE Loss**: The loss function measures the residuals of the two PDEs governing `ψ` and `θ`. It is computed using automatic differentiation.
- **Boundary Loss**: The loss associated with boundary conditions for `θ` and `ψ`. For `θ`, boundary conditions are imposed at `Y = 0` and `Y = 1`.

### Training

The model is trained using the Adam optimizer. The total loss is a combination of the PDE loss and the boundary loss, which the optimizer tries to minimize over several epochs.

### Visualization

After training, the code generates contour plots for both `ψ` and `θ` at a specific time (`τ = 0.08`). The contours are labeled using the `clabel` function in `matplotlib` for better visualization.

## Running the Code

1. Ensure all dependencies are installed.
2. Run the script:
   ```bash
   python pinn_pde_solver.py
   ```

The training process will run for `5000` epochs, and after completion, it will generate two contour plots for `ψ` and `θ`.

## Output

The code will output the following during training:

- **Epoch number** and **loss** values at regular intervals (every 500 epochs).
- **Contour plots** for `ψ` and `θ` after the training is complete. The contours are labeled with numerical values for better interpretation.

### Example Contour Plots

- **Psi Contour Plot**: A contour plot of the streamfunction `ψ` over the extended domain.
- **Theta Contour Plot**: A contour plot of the temperature field `θ` over the original domain.

## Customization

- **Layers**: You can modify the `layers` list in the code to adjust the number of neurons or hidden layers.
- **Rayleigh Number (Ra)**: The Rayleigh number can be adjusted by changing the value of `Ra`.
- **Boundary Conditions**: Modify the `boundary_loss` function to impose more specific boundary conditions for `ψ` if needed.

## Dependencies

- `Python >= 3.6`
- `PyTorch >= 1.7`
- `NumPy >= 1.19`
- `Matplotlib >= 3.3`

