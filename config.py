import pprint

class config:
    
    # env = 'CartPole-v1'
    env = 'PrisonersDilemmaMA'
    render_mode = False

    observation_space = None
    action_space = None

    discretization = 30
    num_states = None

    num_episodes = 100
    n_runs = 3

    def update_observation_space(observation_space):
        config.observation_space = observation_space
        config.num_states = config.discretization**config.observation_space
    
    def update_discretization(discretization):
        config.discretization = discretization
        config.num_states = discretization**config.observation_space

    def update_action_space(action_space):
        config.action_space = action_space

    def ready():
        return config.observation_space is not None and config.action_space is not None
    
    def print():
        variables = {k: v for k, v in vars(config).items() if not callable(v) and not k.startswith('__')}
        pprint.pprint(variables)