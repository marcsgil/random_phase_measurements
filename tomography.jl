ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "@venv"

using PythonCall, QuantumMeasurements, CairoMakie, LinearAlgebra

scipy = pyimport("scipy")
h5py = pyimport("h5py")
np = pyimport("numpy")
pyimport("sys").path.append(pwd())
utils = pyimport("utils")

function remove_backgorund(img, bg)
    map(x -> x > bg ? x-bg : zero(x), img)
end
##
mode_idx = 2
phase_idx = 0
sigma_idx = 2

f = h5py.File("results/up_to_order_1/calibration/calibration.h5")
A_direct = np.array(f["A_direct"])
t_direct = np.array(f["t_direct"])
A_fourier = np.array(f["A_fourier"])
t_fourier = np.array(f["t_fourier"])
f.close()

f = h5py.File("results/up_to_order_1/modes.h5")
basis = np.array(f["basis"])
coefficients = PyArray(f["coefficients"][mode_idx])
f.close()

f = h5py.File("results/phases.h5")
phase = f["phases"][sigma_idx, phase_idx]
f.close()

f = h5py.File("results/up_to_order_1/data.h5")
image = PyArray(f["images_phase_fourier"][sigma_idx, phase_idx, mode_idx])
f.close()
##
function match_fourier_basis_to_measurement(u, A_fourier, t_fourier)
    u_fourier = utils.fourier_transform(u)
    scipy.ndimage.affine_transform(u_fourier, scipy.linalg.inv(A_fourier), - np.matmul(scipy.linalg.inv(A_fourier), t_fourier))
end

function match_fourier_basis_to_measurement(u, A_fourier, t_fourier, phase)
    match_fourier_basis_to_measurement(np.exp(pycomplex(im) * phase) * u, A_fourier, t_fourier)
end

phase_fourier_basis = PyArray(np.array(pylist([match_fourier_basis_to_measurement(u, A_fourier, t_fourier, phase) for u in basis])))
normalization = sum(abs2, phase_fourier_basis, dims=(2,3))
phase_fourier_basis ./= sqrt.(normalization)

mode = dropdims(sum(phase_fourier_basis .* coefficients, dims=1), dims=1)
theo_image = abs2.(mode)


with_theme(theme_latexfonts()) do
    fig = Figure(size = (800, 400))
    ax1 = Axis(fig[1,1], aspect = 1)
    ax2 = Axis(fig[1,2], aspect = 1)
    hidedecorations!(ax1)
    hidedecorations!(ax2)
    plot!(ax1, theo_image)
    plot!(ax2, image)
    fig
end

μ = assemble_measurement_matrix(conj.(c) for c in eachslice(phase_fourier_basis, dims=(2,3)))

tr(inv(fisher(μ, traceless_vectorization(coefficients))))
##
method = MaximumLikelihood()
ρ = estimate_state(remove_backgorund(image, 5), μ, method)[1]
fidelity(project2pure(ρ), coefficients)