from parameters import Parameters, CardSimParameters


def test_dataset_dir_different():
    p1 = Parameters(cardsim=CardSimParameters(n_days=50))
    p2 = Parameters(cardsim=CardSimParameters(n_days=51))
    assert p1.dataset_dir != p2.dataset_dir


def test_banksys_dir():
    assert False
