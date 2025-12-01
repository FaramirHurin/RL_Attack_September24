from parameters import Parameters, CardSimParameters, PPOParameters, VAEParameters, ClassificationParameters
import os
from datetime import datetime
import shutil

CACHE_ROOT = os.path.join("cache")


def test_cardsim_dir():
    p1 = CardSimParameters(n_days=100)
    p2 = CardSimParameters(n_days=200)
    p3 = CardSimParameters(n_days=100)
    dir1 = p1.cache_dir(CACHE_ROOT)
    dir2 = p2.cache_dir(CACHE_ROOT)
    dir3 = p3.cache_dir(CACHE_ROOT)
    assert dir1 != dir2
    assert dir1 == dir3


def test_dataset_dir():
    p1 = Parameters(cardsim=CardSimParameters(n_days=50), agent=PPOParameters())
    p2 = Parameters(cardsim=CardSimParameters(n_days=51))
    p3 = Parameters(cardsim=CardSimParameters(n_days=50), agent=VAEParameters())
    assert p1.dataset_dir != p2.dataset_dir
    assert p1.dataset_dir == p3.dataset_dir


def test_banksys_dir():
    p1 = ClassificationParameters(fn_rate=0.02)
    p2 = ClassificationParameters(fn_rate=0.01)
    p3 = ClassificationParameters(fn_rate=0.02)
    dir1 = p1.cache_dir(CACHE_ROOT)
    dir2 = p2.cache_dir(CACHE_ROOT)
    dir3 = p3.cache_dir(CACHE_ROOT)
    assert dir1 != dir2
    assert dir1 == dir3


def test_cache_dir():
    p1 = Parameters(seed=1)
    p2 = Parameters(seed=2)
    p3 = Parameters(seed=1)
    assert p1.cache_dir != p2.cache_dir
    assert p1._cache_root == p3._cache_root


def test_parameters_repeated():
    CACHE_ROOT = f"{datetime.now().timestamp()}"
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    params = Parameters(cache_root=CACHE_ROOT, cardsim=CardSimParameters(n_days=90, n_payers=20))
    dataset_dirs = set()
    banksys_files = set()
    for p in params.repeat(3):
        p.seed_random()
        p.make_env()
        dataset_dirs.add(p.dataset_dir)
        banksys_files.add(p.banksys_file)

    assert len(dataset_dirs) == 3
    assert len(banksys_files) == 3
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)


def test_repeat_different_dataset_dir_for_differnet_seeds():
    CACHE_ROOT = f"{datetime.now().timestamp()}"
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    params = Parameters(cache_root=CACHE_ROOT)
    for p in params.repeat(30):
        if p.seed == params.seed:
            continue
        assert p.dataset_dir != params.dataset_dir
        assert p.banksys_file != params.banksys_file
        assert p._cache_root == params._cache_root
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)


def test_directories():
    CACHE_ROOT = f"{datetime.now().timestamp()}"
    params = Parameters(cache_root=CACHE_ROOT)
    for p in params.repeat(10):
        assert p.cache_dir.startswith(p._cache_root)
        assert p.dataset_dir.startswith(p.cache_dir)
        assert p.banksys_dir.startswith(p.dataset_dir)
        assert p.banksys_file.startswith(p.banksys_dir)
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
