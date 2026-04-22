import jax
import jax.numpy as jnp
from jax.lax import scan
import matplotlib.pyplot as plt
from utils import set_phase_reference
import slmcontrol
from phase_screens import fourier_phase_screen


def fidelity(u1, u2):
    _u1 = jnp.ravel(u1)
    _u2 = jnp.ravel(u2)
    return jnp.real(
        jnp.abs(jnp.vdot(_u1, _u2)) ** 2 / jnp.vdot(_u1, _u1) / jnp.vdot(_u2, _u2)
    )


def gs_step(A0, A1, f, inv_f, phase):
    field0 = A0 * jnp.exp(1j * phase)
    field1 = A1 * jnp.exp(1j * jnp.angle(f(field0)))
    return jnp.angle(A0 * jnp.exp(1j * jnp.angle(inv_f(field1))))


def augmented_gs_step(A0, A1, A2, f1, inv_f1, f2, inv_f2, phase):
    phase = gs_step(A0, A1, f1, inv_f1, phase)
    return gs_step(A0, A2, f2, inv_f2, phase)


def select_best_phase(phases, direct_amplitude, fourier_amplitude):
    # phases: [B, N, N]; returns the phase [N, N] with lowest Fourier amplitude MSE
    predicted_fourier_amplitude = jnp.abs(
        jnp.fft.fft2(direct_amplitude * jnp.exp(1j * phases))
    )
    mses = jnp.mean(
        (predicted_fourier_amplitude - fourier_amplitude) ** 2, axis=(-2, -1)
    )
    return phases[jnp.argmin(mses)]


def augmented_gs(
    direct_amplitude,
    fourier_amplitude,
    phase_fourier_amplitude,
    phase_transformation,
    initial_phase,
    nsteps,
):
    f1 = jnp.fft.fft2
    inv_f1 = jnp.fft.ifft2

    def f2(x):
        return f1(phase_transformation * x)

    def inv_f2(x):
        return jnp.conj(phase_transformation) * inv_f1(x)

    def scan_step(phase, _):
        return augmented_gs_step(
            direct_amplitude,
            fourier_amplitude,
            phase_fourier_amplitude,
            f1,
            inv_f1,
            f2,
            inv_f2,
            phase,
        ), None

    final_phase, _ = scan(scan_step, initial_phase, None, length=nsteps)
    best_phase = select_best_phase(final_phase, direct_amplitude, fourier_amplitude)
    return jnp.mod(best_phase, 2 * jnp.pi)


def simulated_augmented_gs(u, phase_transformation, initial_phase, nsteps):
    u_fourier = jnp.fft.fft2(u)
    u_phase_fourier = jnp.fft.fft2(phase_transformation * u)

    predicted_phase = augmented_gs(
        jnp.abs(u),
        jnp.abs(u_fourier),
        jnp.abs(u_phase_fourier),
        phase_transformation,
        initial_phase,
        nsteps,
    )

    return predicted_phase, u_fourier, u_phase_fourier


if __name__ == "__main__":
    L = 1
    N = 128
    d = L / N

    xs = jnp.arange(-L / 2, L / 2, d)
    ys = jnp.arange(-L / 2, L / 2, d)
    xs, ys = jnp.meshgrid(xs, ys, sparse=True)

    amplitude = 2**16
    sigma = 2 / 2 / jnp.pi

    phase_mask = fourier_phase_screen(N, N, amplitude=amplitude, sigma=sigma)

    phase_transformation = jnp.exp(1j * phase_mask)

    initial_phases = jax.random.uniform(jax.random.key(0), (1, N, N))

    u = slmcontrol.hg(xs, ys, m=10, n=10, w=0.1)  # type: ignore

    predicted_phase, u_fourier, u_phase_fourier = simulated_augmented_gs(
        u, phase_transformation, initial_phases, 10
    )

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
    axs[0, 0].imshow(jnp.abs(u) ** 2, cmap="hot")
    axs[0, 0].set_title("Direct")

    axs[0, 1].imshow(jnp.fft.fftshift(jnp.abs(u_fourier) ** 2), cmap="hot")
    axs[0, 1].set_title("Fourier")

    axs[0, 2].imshow(jnp.fft.fftshift(jnp.abs(u_phase_fourier) ** 2), cmap="hot")
    axs[0, 2].set_title("Phase Fourier")

    axs[1, 0].imshow(
        set_phase_reference(jnp.angle(u), posY=0.45),
        cmap="twilight",
        vmin=-jnp.pi,
        vmax=jnp.pi,
    )
    axs[1, 0].set_title("True Phase")

    axs[1, 1].imshow(
        set_phase_reference(predicted_phase, posY=0.45),
        cmap="twilight",
        vmin=-jnp.pi,
        vmax=jnp.pi,
    )
    axs[1, 1].set_title("Predicted Phase")

    axs[1, 2].imshow(phase_mask, cmap="twilight", vmin=-jnp.pi, vmax=jnp.pi)
    axs[1, 2].set_title("Transformation Phase")

    for m in range(axs.shape[0]):
        for n in range(axs.shape[1]):
            axs[m, n].set_axis_off()

    plt.tight_layout()
    plt.show()
