import os

class ShieldSelector:
    def __init__(self, env_name, n_actions=None, n_sensors=None, filename = ""):
        
        self.base_dir = "shields"
        self.env_name = env_name
        self.n_actions = n_actions
        self.n_sensors = n_sensors

        assert filename or (n_actions and n_sensors), "Either filename or (n_actions and n_sensors) must be provided."

        if filename:
            print("Using provided filename as shield program.")
            self.file = filename

        else:
            self.file = f"{env_name}/shield_a{n_actions}_s{n_sensors}.pl"

        self.verify()

    def verify(self):
        full_path = os.path.join(self.base_dir, self.file)
        assert os.path.exists(full_path), f"Shield program not found at {full_path}."
