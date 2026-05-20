ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "@venv"

using PythonCall, QuantumMeasurements, CairoMakie, LinearAlgebra, ProgressMeter, Statistics, HDF5, FFTW, ProgressMeter, PoissonPhaseRetrieval

scipy = pyimport("scipy")
np = pyimport("numpy")
pyimport("sys").path.append(pwd())
utils = pyimport("utils")
h5py = pyimport("h5py")

function remove_backgorund(img, bg::Number)
    broadcast(x -> x > bg ? x - bg : zero(x), img)
end

function remove_backgorund(img, bg)
    broadcast((x, bg) -> x > bg ? x - bg : zero(x), img, bg)
end

function match_fourier_basis_to_measurement(u, A_fourier, t_fourier)
    u_fourier = utils.fourier_transform(u)
    scipy.ndimage.affine_transform(u_fourier, scipy.linalg.inv(A_fourier), -np.matmul(scipy.linalg.inv(A_fourier), t_fourier))
end

function match_fourier_basis_to_measurement(u, A_fourier, t_fourier, phase)
    match_fourier_basis_to_measurement(np.exp(pycomplex(im) * phase) * u, A_fourier, t_fourier)
end

function amplitude_fidelity(image, coeffs, basis)
    mode = PyArray(np.sum(coeffs.reshape(-1, 1, 1) * basis, axis=0))
    amplitude_fidelity(image, abs2.(mode))
end

function amplitude_fidelity(image1, image2)
    amplitude1 = sqrt.(image1 / sum(image1))
    amplitude2 = sqrt.(image2 / sum(image2))
    amplitude1 ⋅ amplitude2
end

function linear_combination!(dest, coefficients, basis)
    dest .= 0
    for (c, u) in zip(coefficients, basis)
        dest .+= c * u
    end
end

function linear_combination(coefficients, basis)
    dest = complex(similar(first(basis)))
    linear_combination!(dest, coefficients, basis)
    dest
end

function transform_basis(basis, A, t)
    mapslices(basis, dims=(1, 2)) do u
        scipy.ndimage.affine_transform(u, inv(A), -inv(A) * t) |> PyArray
    end
end

function fourier_transform(u)
    ifftshift(fft(fftshift(u, (1, 2)), (1, 2)), (1, 2))
end

base_dir = "results/fixed_basis_normalization/"

A_direct, t_direct, A_fourier, t_fourier, = h5open(joinpath(base_dir, "calibration/calibration.h5")) do f
    reverse(read(f["A_direct"]) |> transpose),
    reverse(read(f["t_direct"])),
    reverse(read(f["A_fourier"]) |> transpose),
    reverse(read(f["t_fourier"]))
    # read(f["bg_direct"]),
    # read(f["bg_fourier"])
end
##
mode_idx = 5
phase_idx = 2
sigma_idx = 1
order = 4

phase = h5open(joinpath(base_dir, "phases.h5")) do f_phases
    f_phases["phases"][:, :, phase_idx, sigma_idx]
end

coefficients, _direct_basis = h5open(joinpath(base_dir, "up_to_order_$order/modes.h5")) do f_modes
    f_modes["coefficients"][:, mode_idx], read(f_modes["basis"])
end

image = h5open(joinpath(base_dir, "up_to_order_$order/data.h5")) do f_data
    f_data["images_phase_fourier"][:, :, mode_idx, phase_idx, sigma_idx]
end

method = MaximumLikelihood()

_phase_fourier_basis = fourier_transform(_direct_basis .* cis.(phase))
phase_fourier_basis = transform_basis(_phase_fourier_basis, A_fourier, t_fourier)
normalization = sum(abs2, phase_fourier_basis, dims=(1, 2))
phase_fourier_basis ./= sqrt.(normalization)
μ = assemble_measurement_matrix(conj.(c) for c in eachslice(phase_fourier_basis, dims=(1, 2)))

ρ = estimate_state(remove_backgorund(image, 5), μ, method)[1]
fidelity(project2pure(ρ), normalize(coefficients .* vec(sqrt.(normalization))))


probs = reshape(get_probabilities(μ, traceless_vectorization(normalize(coefficients .* vec(sqrt.(normalization))))), size(image))
##
display(plot(probs, axis=(; aspect=1)))
display(plot(remove_backgorund(image, 5), axis=(; aspect=1)))
##
sensing_operator = reshape(phase_fourier_basis, :, size(phase_fourier_basis, 3))
y = reshape(image, :)
b = fill(5, length(y))
x0 = optimal_initialization(sensing_operator, y, b)

abs2(normalize(x0) ⋅ coefficients)

ψ, loss = poisson_phase_retrieval(sensing_operator, x0, y, b, 500, Val(true))

abs2(normalize(ψ) ⋅ normalize(coefficients .* vec(sqrt.(normalization))))

compute_loss(ψ, sensing_operator, y, b)

display(plot(reshape(get_probabilities(μ, traceless_vectorization(normalize(ψ))), size(image)), axis=(; aspect=1)))

compute_loss(normalize(coefficients .* vec(sqrt.(normalization))), sensing_operator, y, b)

predicted_probs_pr = reshape(get_probabilities(μ, traceless_vectorization(normalize(ψ))), size(image))
display(plot(predicted_probs_pr, axis=(; aspect=1)))
##
using CoordinateTransformations, Interpolations, Rotations
from_points, to_points = h5open(joinpath(base_dir, "calibration/calibration.h5")) do f
    read(f["slm_points_fourier"]), read(f["cam_points_fourier"])
end

from_points
to_points


rigid = AffineMap(to_points => mapslices(LinearMap(RotMatrix{2}(0.3)), from_points, dims=1))
coords_x = 0:size(_direct_basis, 1)-1
coords_y = 0:size(_direct_basis, 1)-1


new_coords = map(rigid, collect(Iterators.product(coords_x, coords_y)))

phase_fourier_basis = mapslices(_phase_fourier_basis, dims=(1, 2)) do slice
    interp_cubic = cubic_spline_interpolation((coords_x, coords_y), slice, extrapolation_bc=zero(eltype(slice)))
    map(x -> interp_cubic(x[1], x[2]), new_coords)
end

μ = assemble_measurement_matrix(conj.(c) for c in eachslice(phase_fourier_basis, dims=(1, 2)))
probs = reshape(get_probabilities(μ, traceless_vectorization(coefficients)), size(image))

display(plot(probs, axis=(; aspect=1)))
display(plot(remove_backgorund(image, bg_fourier), axis=(; aspect=1)))


plot(map(x -> abs2.(interp_cubic(x[1], x[2])), new_coords))
plot(abs2.(_direct_basis[:, :, 1]))