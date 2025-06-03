from datetime import timedelta
from parameters import Parameters

if __name__ == "__main__":
    p = Parameters()
    # b = p.create_banksys(use_cache=True)
    env = p.create_env()
    b = env.system
    b.simulate_until(b.attack_start + timedelta(days=20))
