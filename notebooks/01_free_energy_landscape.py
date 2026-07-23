# 01_free_energy_landscape.py 
#%%
import numpy as np
import matplotlib.pyplot as plt

def F(y1,y2):
    '''Function to calculate the free energy landscape, eq.2 in the paper'''
    L1=L2=300
    L12=100
    T1=T2=250
    N1=N2=100
    beta=1

    F = -L1*y1**2 + T1*y1
    F+= -L2*y2**2 + T2*y2
    F+= L12*y1*y2

    F+= (N1/beta) * (y1*np.log(y1) + (1-y1)*np.log(1-y1))
    F+= (N2/beta) * (y2*np.log(y2) + (1-y2)*np.log(1-y2))
    #F_control = -L1*y1**2 + T1*y1 - L2*y2**2 + T2*y2 + L12*y1*y2 + (N1/beta)*(y1*np.log(y1) + (1-y1)*np.log(1-y1)) + (N2/beta)*(y2*np.log(y2) + (1-y2)*np.log(1-y2))
    #assert np.allclose(F, F_control), "The two calculations of F do not match!"
    return F

#F(0.5, 0.2)
#%%
import time
start_time = time.time()
F_values = np.zeros((100, 100))
for i, y1 in enumerate(np.linspace(0.01, 0.99, 100)):
    for j, y2 in enumerate(np.linspace(0.01, 0.99, 100)):
        F_value = F(y1, y2)
        F_values[i, j] = F_value
print("Time taken for nested loops: {:.4f} seconds".format(time.time() - start_time))
#%%
plt.imshow(F_values, extent=(0.01, 0.99, 0.01, 0.99), origin='lower', aspect='auto')
plt.colorbar(label='F(y1, y2)')
plt.xlabel('y1')
plt.ylabel('y2')
plt.show()
# %%
#let's time this
import time
start_time = time.time()
y = np.linspace(0.01, 0.99, 100)
Y1, Y2 = np.meshgrid(y, y)
Z = F(Y1, Y2)
print("Time taken for meshgrid and F calculation: {:.4f} seconds".format(time.time() - start_time))
# asset F_values and Z are the same
assert np.allclose(F_values, Z), "F_values and Z do not match!"
plt.contourf(Y1, Y2, Z, levels=30)
plt.colorbar(label='F(y1, y2)')
plt.xlabel("Positive affect $y_1$")
plt.ylabel("Negative affect $y_2$")
plt.title("AIM free-energy landscape")
plt.show()
# %%
# gradiente 

def grad_F(y1, y2):
    '''Function to calculate the gradient of the free energy landscape, eq.3 in the paper'''
    L1 = L2 = 300
    L12 = 100
    T1 = T2 = 250
    N1 = N2 = 100
    beta = 1
    dF_dy1 = (
        -2 * L1 * y1
        + T1
        + L12 * y2
        + (N1 / beta) * np.log(y1 / (1 - y1))
    )

    dF_dy2 = (
        -2 * L2 * y2
        + T2
        + L12 * y1
        + (N2 / beta) * np.log(y2 / (1 - y2))
    )
 
    return dF_dy1, dF_dy2
# %%
def validate_gradient():
    """Validate the analytical gradient using finite differences."""
    y1_test = 0.3
    y2_test = 0.7
    h = 1e-6

    numerical_dy1 = (
        F(y1_test + h, y2_test)
        - F(y1_test - h, y2_test)
    ) / (2 * h)

    numerical_dy2 = (
        F(y1_test, y2_test + h)
        - F(y1_test, y2_test - h)
    ) / (2 * h)

    analytical_dy1, analytical_dy2 = grad_F(y1_test, y2_test)

    print("dy1:", numerical_dy1, analytical_dy1)
    print("dy2:", numerical_dy2, analytical_dy2)

    assert np.allclose(
        numerical_dy1, analytical_dy1, rtol=1e-5
    )
    assert np.allclose(
        numerical_dy2, analytical_dy2, rtol=1e-5
    )

    print("Gradient validation passed") 

# %%
