import re

import numpy as np
import matplotlib.pyplot as plt
import imageio

from x_squared_approximator import XSquaredApproximator


def generated_x_squared_data(low, high, size):
    x = np.random.uniform(low, high, size=size)
    y = x ** 2
    return x, y


def create_frame(t, x, y, results, low, high):
    fig = plt.figure(figsize=(6, 6))
    plt.axvspan(low, high, color='blue', alpha=0.3, label="Training Region")
    plt.scatter(x, y, color='grey', label=r"$y=x^2$")
    plt.scatter(x, results[t], color='black', label=r"Neural Network")
    plt.legend()  # loc="best")#outside lower left")
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title(fr'Plot of $y = x^2$ vs network at epoch {t}',
              fontsize=14)
    plt.savefig(f'./img/img_{t}.png',
                transparent=False,
                facecolor='white'
                )
    plt.close()


def generate_results_gif(x_test, y_test, results, low, high, fps, name):
    res_arr = np.array([res.detach().numpy().squeeze() for res in results])
    time = range(len(res_arr))
    frames = []
    for t in time:
        create_frame(t, x_test, y_test, res_arr, low, high)
        image = imageio.v2.imread(f'./img/img_{t}.png')
        frames.append(image)
    imageio.mimsave(f'{name}.gif', frames, fps=fps)



def main():
    low = 0
    high = 30
    fps = 5
    time = 20
    epochs = 15000
    x, y = generated_x_squared_data(low, high, 10000)
    x_test, y_test = generated_x_squared_data(-50, 50, 10000)

    model = XSquaredApproximator(epochs=epochs, learning_rate=1e-3, time=time, fps=fps)
    n = sum([bool(re.search(r"\bLinear\b", str(x))) for x in model.model])
    name = f"{n}_layer"
    model.fit(x, y, x_test, y_test)

    generate_results_gif(x_test, y_test, model.results, low, high, fps, name)





if __name__ == '__main__':
    main()
