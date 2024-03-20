import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.optimize import curve_fit

tempnet_Bc = np.load("data/tempnet_Bc.npy")
meanfield_Bc = np.load("data/meanfield_Bc.npy")
N_values = np.load("data/N_values.npy")

# Using a standardized style set that largely holds to Nature's guidelines.
plt.style.use('science')
plt.style.use(['science','nature'])

def log_function(x, a, b):
    return (a + b * np.log(x)) # 3.99431 is the theoretical critical buffer value.

bounds = ([-100, 1.3], [100, 1.7])

BC_THEORY = 3.99431
# Calculate fit on temp net

popt, pcov = curve_fit(log_function, N_values, (BC_THEORY-tempnet_Bc)**-0.5)
a_opt, c_opt= popt
x_curve_tempnet = np.linspace(min(N_values), max(N_values), int(1E5))
y_curve_tempnet = log_function(x_curve_tempnet, a_opt, c_opt)
# Calculate fit on mean-field
popt2, pcov2 = curve_fit(log_function, N_values, (BC_THEORY-meanfield_Bc)**-0.5)
a_opt2, c_opt2= popt2
x_curve_meanfield = np.linspace(min(N_values), max(N_values), int(1E5))
y_curve_meanfield = log_function(x_curve_meanfield, a_opt2, c_opt2)

# Plot fits + data.
fig, ax = plt.subplots()
ax.plot(x_curve_tempnet, y_curve_tempnet, alpha=0.4, c="r", label=r"numerical\ fit\ STN")#, $\alpha = {str(a_opt)[:5]}$, $\gamma={str(c_opt)[:4]}$, $\delta = {str(d_opt)[:4]}$")
ax.plot(x_curve_meanfield, y_curve_meanfield, alpha=0.4, c="b", label=r"numerical\ fit\ MF")#, $\alpha = {str(a_opt2)[:5]}$, $\gamma={str(c_opt2)[:4]}$, $\delta = {str(d_opt2)[:4]}$", c="b")
ax.set_xscale("log")
#plt.yscale("log")
ax.set_ylim(0.7, 2.3)
ax.set_xlim(8, 1.5E5)
arrow_x = 110 # x-coordinate of the arrow
arrow_y = 4  # y-coordinate of the arrow
text_x = 20  # x-coordinate of the text box
text_y = 3.7  # y-coordinate of the text box

ax.set_xlabel(r"$N$")
ax.set_ylabel(r"$[B^{*}_{\mathrm{c}} -B_\mathrm{c}(N)]^{-1/2}$", rotation=90,labelpad=5)
ax.tick_params(direction="in")
ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=(1.0,)))
ax.scatter(N_values, (BC_THEORY-tempnet_Bc)**-0.5, label=r"data\ STN", c="r", s=10, alpha=0.5)
ax.scatter(N_values, (BC_THEORY-meanfield_Bc)**-0.5, label=r"data\ MF", c="b", s=10, alpha=0.5)
ax.legend(frameon=False)
plt.savefig("finite_size.pdf", bbox_inches="tight")
plt.show()

