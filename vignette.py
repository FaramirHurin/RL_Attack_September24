"""
Build a payment transaction dataset using cardsim
"""

from importlib import resources
from cardsim import Cardsim

dcpc_path = resources.files("cardsim.dcpc")
simulator = Cardsim(dcpc_folder=dcpc_path)

df = simulator.simulate()
if df is None:
    raise ValueError("No transactions were generated")
print(len(df))
print(df.head())

print(simulator.payers.head())
# simulator.export_transaction_data(df, folder="data")

# params = simulator.export_run_parameters(df, folder="data", file_name="cardsim_runs")
