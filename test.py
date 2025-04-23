import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Example cards and colors
card_colors = {
    "Visa": "blue",
    "MasterCard": "green",
    "Amex": "red",
}

# Sample transaction list: [(datetime, amount, card)]
transactions = [
    (datetime(2025, 4, 20, 10), 25.5, "Visa"),
    (datetime(2025, 4, 21, 14), 40.0, "Amex"),
    (datetime(2025, 4, 23, 9), 15.75, "MasterCard"),
    # Add more transactions dynamically...
]

# Initialize the figure


# Initialize an empty scatter plot
sc = ax.scatter([], [], s=50)

# Store data for updates
x_data = []
y_data = []
colors = []


# Function to add a new transaction to the plot
def add_transaction(transaction):
    dt, amount, card = transaction
    x_data.append(dt)
    y_data.append(amount)
    colors.append(card_colors.get(card, "gray"))  # default to gray if unknown
    sc.set_offsets(list(zip(mdates.date2num(x_data), y_data)))
    sc.set_color(colors)
    plt.draw()
    plt.pause(0.1)


# Show the plot in interactive mode
plt.ion()
plt.show()

# Iteratively add transactions
for tx in transactions:
    add_transaction(tx)

# Keep the plot open
plt.ioff()
plt.show()
