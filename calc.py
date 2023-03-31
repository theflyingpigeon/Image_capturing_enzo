import numpy as np

# 1920 × 1048

pixelsize: float = 3.45 * 0.001
dv: float = 1080 * pixelsize
dh: float = 1920 * pixelsize
d = 8.7586
# d = np.sqrt(dh ** 2 * dv ** 2)

f: int = 6
a1 = 180 / np.pi
a2 = np.arctan(d / (2 * f))

a = a1 * 2 * a2

print(a)



vert_photo = 1024 / (800 / 24)
afstand_tot_muntje_cm = 22.5 * 10
hoek = a1 * np.arcsin(vert_photo / afstand_tot_muntje_cm)

print(hoek)