import re
import numpy as np

from image_generator import generate_results_gif
from x_squared_approximator import XSquaredApproximator


def generated_x_squared_data(low, high, size):
    x = np.random.uniform(low, high, size=size)
    y = x ** 2
    return x, y

def main():
    low = -20
    high = 20
    fps = 5
    time = 20
    epochs = 15000

    x, y = generated_x_squared_data(low, high, 10000)
    x_test, y_test = generated_x_squared_data(-50, 50, 10000)

    model = XSquaredApproximator(epochs=epochs, learning_rate=1e-3, time=time, fps=fps)
    n = sum([bool(re.search(r"\bLinear\b", str(x))) for x in model.model])
    name = f"{n}_layer"
    gif_info = [low, high, fps, "_" + name]
    model.fit(x, y, x_test, y_test, gif_info)

    gif_info[-1] = name
    generate_results_gif(x_test, y_test, model.results, gif_info)





if __name__ == '__main__':
    main()
