ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "@venv"

using PythonCall, CairoMakie, LinearAlgebra, ProgressMeter, Statistics, HDF5, FFTW, ProgressMeter, PoissonPhaseRetrieval, LinearMaps

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

A_direct, t_direct, A_fourier, t_fourier, _bg_direct, _bg_fourier = h5open(joinpath(base_dir, "calibration/calibration.h5")) do f
    reverse(read(f["A_direct"]) |> transpose),
    reverse(read(f["t_direct"])),
    reverse(read(f["A_fourier"]) |> transpose),
    reverse(read(f["t_fourier"])),
    read(f["bg_direct"]),
    read(f["bg_fourier"])
end

bg_direct .+= 1
bg_fourier .+= 1
##
function direct!(dest, src, phase_transformation, plan)
    N2 = length(src)
    N = isqrt(N2)

    dest[1:N2] .= src

    src_image = reshape(src, N, N)
    dest_fourier = reshape(view(dest, N2+1:2N2), N, N)

    mul!(dest_fourier, plan, src_image)
    dest_fourier ./ N

    src_image .*= phase_transformation
    dest_phase_fourier = reshape(view(dest, 2N2+1:3N2), N, N)
    mul!(dest_phase_fourier, plan, src_image)
    dest_phase_fourier ./= N
end

function adjoint!(dest, src, phase_transformation, plan, buffer)
    N2 = length(dest)
    N = isqrt(N2)

    dest .= view(src, 1:N2)

    src_fourier = reshape(view(src, N2+1:2N2), N, N)
    src_phase_fourier = reshape(view(src, 2N2+1:3N2), N, N)

    mul!(buffer, plan, src_fourier)
    dest .+= reshape(buffer, :) ./ N

    mul!(buffer, plan, src_phase_fourier)
    buffer .*= conj.(phase_transformation)
    dest .+= reshape(buffer, :) ./ N
end

N = 640
src = ones(ComplexF32, N^2)
dest = Array{ComplexF32}(undef, 3N^2)
phase_transformation = cis.(Float32.(2π .* rand(N, N)))
buffer = similar(phase_transformation)

plan = plan_fft(Array{ComplexF32}(undef, N, N))
bplan = plan_bfft(Array{ComplexF32}(undef, N, N))

direct!(dest, src, phase_transformation, plan)
adjoint!(src, dest, phase_transformation, bplan, buffer)
##
src ≈ ones(ComplexF32, 3N^2)

@benchmark direct!($dest, $src, $phase_transformation, $plan)
@benchmark adjoint!($src, $dest, $phase_transformation, $bplan, $buffer)
##
mode_idx = 2
phase_idx = 2
sigma_idx = 1
order = 1

phase = h5open(joinpath(base_dir, "phases.h5")) do f_phases
    f_phases["phases"][:, :, phase_idx, sigma_idx]
end

coefficients, direct_basis = h5open(joinpath(base_dir, "up_to_order_$order/modes.h5")) do f_modes
    f_modes["coefficients"][:, mode_idx], read(f_modes["basis"])
end

_image_direct, _image_fourier, _image_phase_fourier = h5open(joinpath(base_dir, "up_to_order_$order/data.h5")) do f_data
    f_data["images_direct"][:, :, mode_idx],
    f_data["images_fourier"][:, :, mode_idx],
    f_data["images_phase_fourier"][:, :, mode_idx, phase_idx, sigma_idx]
end


fourier_basis = fourier_transform(direct_basis)
phase_fourier_basis = fourier_transform(direct_basis .* cis.(phase))

for slice ∈ eachslice(fourier_basis, dims=3)
    normalize!(slice)
end

for slice ∈ eachslice(phase_fourier_basis, dims=3)
    normalize!(slice)
end

image_direct = scipy.ndimage.affine_transform(_image_direct, A_direct, t_direct) |> PyArray
image_fourier = scipy.ndimage.affine_transform(_image_fourier, A_fourier, t_fourier) |> PyArray
image_phase_fourier = scipy.ndimage.affine_transform(_image_phase_fourier, A_fourier, t_fourier) |> PyArray

bg_direct = scipy.ndimage.affine_transform(_bg_direct, A_direct, t_direct) |> PyArray
bg_fourier = scipy.ndimage.affine_transform(_bg_fourier, A_fourier, t_fourier) |> PyArray

plot(image_phase_fourier)
plot(abs2.(linear_combination(coefficients, eachslice(phase_fourier_basis, dims=3))))
##
N_direct = Float32(sqrt(sum(image_direct)))
N_fourier = Float32(sqrt(sum(image_fourier)))
N_phase_fourier = Float32(sqrt(sum(image_phase_fourier)))


y = reshape(cat(image_direct / N_direct, fftshift(image_fourier) / N_fourier, fftshift(image_phase_fourier) / N_phase_fourier, dims=3), :)
b = reshape(cat(bg_direct / N_direct, fftshift(bg_fourier) / N_fourier, fftshift(bg_fourier) / N_phase_fourier, dims=3), :)

plan = plan_fft(Array{ComplexF32}(undef, size(image_direct)))
bplan = plan_bfft(Array{ComplexF32}(undef, size(image_direct)))
phase_transformation = cis.(phase)
buffer = similar(phase_transformation)


direct!(dest, src, phase_transformation, plan)
adjoint!(src, dest, phase_transformation, bplan, buffer)

sensing_operator = LinearMap{ComplexF32}((x, y) -> direct!(x, y, phase_transformation, plan), (x, y) -> adjoint!(x, y, phase_transformation, bplan, buffer), length(y), length(y) ÷ 3, ismutating=true)
# x0 = optimal_initialization(sensing_operator, y, b)

# sensing_operator' * complex(y)

x0 = ComplexF32.(sqrt.(reshape(image_direct, :)))

x, loss = poisson_phase_retrieval(sensing_operator, x0, y, b, 600, Val(true))

pred_u = reshape(x, size(image_direct))

u = linear_combination(coefficients, eachslice(direct_basis, dims=3))

@show abs2(normalize(pred_u) ⋅ normalize(u))

with_theme(theme_latexfonts()) do
    fig = Figure()
    axs = [Axis(fig[m, n], aspect = 1) for m ∈ 1:2, n ∈ 1:2]
    plot!(axs[1, 1], abs2.(u))
    plot!(axs[1, 2], abs2.(pred_u))
    plot!(axs[2, 1], angle.(u), colormap=:twilight)
    plot!(axs[2, 2], angle.(pred_u), colormap=:twilight)
    fig
end