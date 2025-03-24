# Cardsim

A flexible, scalable Bayesian simulator for payment card fraud detection research.

## Setup 

### Package installation

After cloning the repository, you can build the package from within the directory
in which the project lives in editable mode. This is recommended if you want to 
make modifications to the code. 

```shell
pip install -e .
```

Alternatively, if you prefer to use `cardsim` in another project without editing 
the source code, you can install it after pulling the repository. The final 
forward slash is generally optional.

```shell
pip install /path/to/cardsim/
```

### Dependencies

`requirments.txt` lays out the core dependencies and some optional installs 
depending on how you want to use the package. Installing `pandas` and `scipy`
should take care of all core dependencies, including `numpy`, which is used 
extensively. 

## Basic usage

`vignette.py` shows how to use `cardsim`. The simulator generally works as 
follows: 

```python
from cardsim import Cardsim

# Instantiate the class
simulator = Cardsim()

# Run the simulator and store the transaction data
df = simulator.simulate()

# Export the transaction data
simulator.export_transaction_data(df)
```

There are a significant number of parameters for the `Cardsim()` constructor 
and a few options in the `.simulate()` method. Check out the docstrings for more
details. 

## Output fields

The simulator output fields are defined below: 

| Field         | Definition |
|---------------|------------|
| date_time | The transaction datetime; transactions are sorted chronologically |
| payer_id | A unique identifier for the payer | 
| payee_id | A unique identifier for the payee | 
| amount | The payment amount | 
| credit_card | 1 if the payment is made using a credit card and 0 if using a debit card | 
| remote | 1 if the payment is made remotely and 0 if in-person | 
| distance | The Euclidean distance between the payer and payee | 
| fraud | 1 if the payment is fraudulent and 0 if it is legitimate | 

The output also includes several fields related to `date_time`, such as the hour
of day and the date, as well as a `run_id`, which is constructed as: 
`S{seed}P{n_payers}D{n_days}`. For example, a `run_id` could be: S42P500D180. 

## Survey and diary of consumer payment choice data

The package ships with the 2022 and 2023 [Survey and Diary of Consumer Payment Choice](https://www.atlantafed.org/banking-and-payments/consumer-payments/survey-and-diary-of-consumer-payment-choice)
public use datasets, published by the Federal Reserve Bank of Atlanta (Foster, Green, and Stavins, 2024). 
The data are housed in the `cardsim/dcpc` folder. The vignette shows how to 
load the files. Storing them locally speeds up the simulator. Otherwise, the 
simulator will attempt to fetch them from the Atlanta Fed's website. 

> Foster, Kevin, Claire Greene, and Joanna Stavins (2024). "2023 Survey and Diary of Consumer Payment Choice." Federal Reserve Bank of Atlanta, https://doi.org/10.29338/rdr2024-01.