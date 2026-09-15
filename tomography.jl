ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "@venv"

using PythonCall, CairoMakie, HDF5, FFTW

scipy = pyimport("scipy")
np = pyimport("numpy")
slm_camera_calibration = pyimport("slm_camera_calibration")
os = pyimport("os")
h5py = pyimport("h5py")
pyimport("sys").path.append(pwd())
utils = pyimport("utils")

parent_folder = "results/phase_direct_flip_1"

calib_res_direct = slm_camera_calibration.CalibrationResult.load(os.path.join(parent_folder, "calibration_data", "calibration_direct.h5"))
calib_res_fourier = slm_camera_calibration.CalibrationResult.load(os.path.join(parent_folder, "calibration_data", "calibration_fourier.h5"))

direct_image = utils.load_data("$parent_folder/up_to_order_1/data.h5", "images_direct", background=2, calibration_result=calib_res_direct, index=0) |> PyArray |> transpose
phase_fourier_image = utils.load_data("$parent_folder/up_to_order_1/data.h5", "images_phase_fourier", background=2, calibration_result=calib_res_fourier, index=(0, 0, 0)) |> PyArray |> transpose

heatmap(direct_image)
heatmap(abs2.(mode))
##

coefficients, basis = h5open("$parent_folder/up_to_order_1//modes.h5") do f
    read(f["coefficients"]), read(f["basis"])
end

phases = h5open("$parent_folder/phases.h5") do f
    read(f["phases"])
end

mode = sum(prod, zip(eachslice(basis, dims=3), view(coefficients, :, 1)))

phase_fourier_mode = fftshift(fft(ifftshift(mode * cis.(phases[:, :, 1, 1]'))))

heatmap(abs2.(phase_fourier_mode)[128-16:128+16, 128-16:128+16])
heatmap(phase_fourier_image[128-16:128+16, 128-16:128+16])