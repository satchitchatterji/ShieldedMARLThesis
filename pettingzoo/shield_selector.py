import os

class ShieldSelector:
    def __init__(self, env_name, n_actions=None, n_sensors=None, filename = ""):
        
        self.base_dir = "shields"
        self.env_name = env_name
        self.n_actions = n_actions
        self.n_sensors = n_sensors

        assert filename or (n_actions and n_sensors), "Either filename or (n_actions and n_sensors) must be provided."

        if filename != "" and filename != "default":
            print("Using provided filename as shield program.")
            self.file = filename
        
        if filename == "default":
            if env_name == "simple_stag_v0":
                self.file = "simple_stag_v0/shield_v0.pl"

            elif env_name == "simple_pd_v0":
                self.file = "simple_pd_v0/shield_v0.pl"

            elif env_name == "markov_stag_hunt":
                self.file = "markov_stag_hunt/shield_v0.pl"

            elif env_name == "waterworld":
                raise NotImplementedError("Shield program for Waterworld not implemented yet.")

        self.verify()

    def verify(self):
        full_path = os.path.join(self.base_dir, self.file)
        assert os.path.exists(full_path), f"Shield program not found at {full_path}."
