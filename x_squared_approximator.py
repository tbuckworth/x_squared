import numpy as np
# import pandas as pd
import torch.nn as nn
import torch
import threading
from torch import Tensor

from image_generator import generate_results_gif


# from matplotlib import pyplot as plt
# from filelock import FileLock, Timeout

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)


class XSquaredApproximator(nn.Module):
    def __init__(self, epochs, learning_rate, time, fps):
        super(XSquaredApproximator, self).__init__()
        self.results = {}
        self.losses = []
        self.test_loss = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.elapsed = 0
        self.batch_size = 256
        self.checkpoint = max(epochs // (fps * time),1)
        self.nb_epoch = epochs
        self.input_size = 1
        self.mid_weight = 8
        self.weight_size = 8
        self.learning_rate = learning_rate

        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.mid_weight),
            nn.ReLU(),
            nn.Linear(self.mid_weight, self.mid_weight),
            nn.ReLU(),
            # nn.Linear(self.mid_weight, self.mid_weight),
            # nn.ReLU(),
            # nn.Linear(self.mid_weight, self.mid_weight),
            # nn.ReLU(),
            # nn.Linear(self.mid_weight, self.mid_weight),
            # nn.ReLU(),
            # nn.Linear(self.mid_weight, self.mid_weight),
            # nn.ReLU(),
            # nn.Linear(self.mid_weight, self.mid_weight),
            # nn.ReLU(),
            # nn.Linear(self.mid_weight, self.mid_weight),
            # nn.ReLU(),
            # nn.Linear(self.mid_weight, self.mid_weight),
            # nn.ReLU(),
            # nn.Linear(self.mid_weight, self.mid_weight),
            # nn.ReLU(),
            # nn.Linear(self.mid_weight, self.mid_weight),
            # nn.ReLU(),
            # nn.Linear(self.mid_weight, self.mid_weight),
            # nn.ReLU(),
            # nn.Linear(self.mid_weight, self.mid_weight),
            # nn.ReLU(),
            # nn.Linear(self.mid_weight, self.mid_weight),
            # nn.ReLU(),
            # nn.Linear(self.mid_weight, self.mid_weight),
            # nn.ReLU(),
            # nn.Linear(self.mid_weight, self.mid_weight),
            # nn.ReLU(),
            # nn.Linear(self.mid_weight, self.mid_weight),
            # nn.ReLU(),
            # nn.Linear(self.mid_weight, self.mid_weight),
            # nn.ReLU(),
            # nn.Linear(self.mid_weight, self.mid_weight),
            # nn.ReLU(),
            # nn.Linear(self.mid_weight, self.mid_weight),
            # nn.ReLU(),
            # nn.Linear(self.mid_weight, self.mid_weight),
            # nn.ReLU(),
            nn.Linear(self.mid_weight, self.input_size),
        )

        self.model.apply(init_weights)
        self.model.double()

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=0.001, lr=learning_rate)  # , lr=self.learning_rate)#, weight_decay=1e-4)
        self.to(self.device)

    def fit(self, x, y, x_test, y_test, gif_info):
        self.results = {}
        x = torch.from_numpy(x).to(self.device)
        y = torch.from_numpy(y).to(self.device)
        x_test = torch.from_numpy(x_test).to(self.device)
        y_test = torch.from_numpy(y_test).to(self.device)
        batch_size = self.batch_size
        for epoch in range(self.nb_epoch):
            permutation = torch.randperm(x.size()[0])
            for i in range(0, x.size()[0], batch_size):
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = x[indices], y[indices]
                outputs = self.forward(batch_x)
                self.optimizer.zero_grad()
                loss = self.loss_fn(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
            self.losses.append(float(loss))
            print(str(round(float(loss), 2)) + ": loss at epoch " + str(epoch), flush=True)

            # test section
            if epoch % self.checkpoint == 0 or epoch < 100:
                outputs = self.forward(x_test)
                loss = self.loss_fn(outputs, y_test)
                self.results[epoch] = outputs.detach().numpy().squeeze()
                self.test_loss.append(float(loss))

            if epoch % 100 == 0 and epoch > 0:
                generate_results_gif(x_test, y_test, self.results, gif_info)

    def forward(self, batch_x):
        batch_x = batch_x.reshape(batch_x.shape[0], 1)
        return self.model(batch_x)

    def predict(self, state):
        return self.model(state)
