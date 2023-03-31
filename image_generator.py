import matplotlib.pyplot as plt
import imageio
import numpy as np

def create_frame(t, x, y, y_pred, low, high, epoch, out_file=None):
    if out_file is None:
        out_file = f'./img/img_{t}.png'
    fig = plt.figure(figsize=(6, 6))
    plt.axvspan(low, high, color='blue', alpha=0.3, label="Training Region")
    plt.scatter(x, y, color='grey', label=r"$y=x^2$")
    plt.scatter(x, y_pred, color='black', label=r"Neural Network")
    plt.legend()  # loc="best")#outside lower left")
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title(fr'Plot of $y = x^2$ vs network at epoch {epoch}',
              fontsize=14)
    plt.savefig(out_file,
                transparent=False,
                facecolor='white'
                )
    plt.close()


def generate_results_gif(x_test, y_test, results, gif_info):
    low, high, fps, name = tuple(gif_info)
    # res_arr = np.array([res.detach().numpy().squeeze() for res in results])
    # time = range(len(res_arr))
    frames = []
    for t, epoch in enumerate(results.keys()):
        create_frame(t, x_test, y_test, results[epoch], low, high, epoch)
        image = imageio.v2.imread(f'./img/img_{t}.png')
        frames.append(image)
    imageio.mimsave(f'{name}.gif', frames, fps=fps)