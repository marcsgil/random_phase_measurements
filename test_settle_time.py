import slmcontrol
import numpy as np
from tqdm import trange
from cameras.ImagingSourceNew import ImagingSourceCamera
import matplotlib.pyplot as plt
from functools import partial


def _prepare(xs, ys, two_pi_modulation, xperiod, yperiod, n):
    max = 5
    mode = slmcontrol.hg(xs, ys, m=n // max, n=n % max, w=30)
    return slmcontrol.generate_hologram(np.concatenate([mode, mode], axis = 1), two_pi_modulation, xperiod, yperiod)

def _measure(dataset, camera, n):
    dataset[n] = camera.capture()[:,:,0]

if __name__ == "__main__":
    slm = slmcontrol.SLMDisplay(host="localhost")
    camera = ImagingSourceCamera()
    camera.set_exposure(100)

    _xs = np.arange(slm.width // 2) - slm.width // 4
    _ys = np.arange(slm.height) - slm.height // 2
    xs, ys = np.meshgrid(_xs, _ys)

    n = 5

    prepare = partial(_prepare, xs, ys, 192, -3, 19)
    test_img = camera.capture()[:,:,0]
    images = np.empty((n**2, *test_img.shape), test_img.dtype)

    measure = partial(_measure, images, camera)

    slmcontrol.prepare_and_measure(prepare, measure, slm, 0.2, n**2)

    fig, axs = plt.subplots(n, n)
    for n, ax in enumerate(axs.flatten()):
        ax.imshow(images[n], cmap="hot")
    plt.savefig("plots/test_settle_time.png")

    camera.close()
    slm.close()