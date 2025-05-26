from parameters import Parameters, PPOParameters, ClassificationParameters, CardSimParameters
from datetime import timedelta
from banksys import Banksys
from copy import deepcopy
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

if __name__ == "__main__":
    p = Parameters(PPOParameters())

    start = datetime.now()
    bs = p.create_env().system
    end = datetime.now()
    print(f"Creating Banksys took {end - start} seconds")

    start = datetime.now()
    bs2 = deepcopy(bs)
    end = datetime.now()
    print(f"Deepcopy took {end - start} seconds")
