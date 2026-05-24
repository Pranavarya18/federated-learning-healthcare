import copy
import torch


class Server:
    def __init__(self, global_model):
        self.global_model = global_model

    def aggregate(self, client_weights):
        avg_weights = copy.deepcopy(client_weights[0])

        for key in avg_weights.keys():
            for i in range(1, len(client_weights)):
                avg_weights[key] += client_weights[i][key]

            avg_weights[key] = torch.div(avg_weights[key], len(client_weights))

        self.global_model.load_state_dict(avg_weights)
        return self.global_model.state_dict()

    def get_global_weights(self):
        return self.global_model.state_dict()