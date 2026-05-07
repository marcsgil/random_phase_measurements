import numpy as np

MFD = 400e-6
wavelength = 780e-9
D = 0.1
f1 = 0.405

D1 = 5e-3
f2 = f1 * D1 / D

f5 = 5e-2

D2 = 4.48 * wavelength * f5 / np.pi / D

print(f"D2: {D2}")

# factor = 4.48 * wavelength / np.pi / MFD / D * f1 / f2

# print(factor * 18.4e-3)