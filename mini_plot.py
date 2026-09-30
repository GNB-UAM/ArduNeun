import pandas as pd
import matplotlib.pyplot as plt 


data = pd.read_csv("single_neuron.csv", skiprows=10, skipfooter=1,header=None, names=["t","v"], delimiter=' ')

print(data)
plt.plot(data["t"], data["v"], color='navy')
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (ms)")
plt.savefig("single_neuron.png", dpi=200)
plt.show()



data = pd.read_csv("mini_circuit.csv", skiprows=11, skipfooter=1,header=None, names=["t","v1","v2"], delimiter=' ')

print(data)
plt.plot(data["t"], data["v1"], color='navy', alpha=0.8)
plt.plot(data["t"], data["v2"], color='purple', alpha=0.6)
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (ms)")
plt.savefig("mini_circuit.png", dpi=200)
plt.show()