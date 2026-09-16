# Preparing the direct basis on the camera grid

The Fourier calibration relates coordinates on the computational Fourier
array to coordinates on the camera array. The purpose of the preparation step
is to move this registration from the measured images to the theoretical
fields, so that Julia can compare raw camera images with a Fourier transform
already expressed on the camera grid.

## Calibration convention

The [calibration package](https://github.com/marcsgil/slm-camera-calibration)
fits a similarity transform in NumPy `(y, x)` order:

$$
    Y = A X + b,
$$

where `X` is a coordinate in the computational Fourier array and `Y` is the
corresponding camera coordinate. The existing image-side calibration uses this
map as a pull transform:

$$
    I_{\rm computational}(X)
    = I_{\rm camera}(A X + b).
$$

For the reverse direction, the theoretical Fourier field must therefore be
sampled at

$$
    U_{\rm camera}(Y)
    = U_{\rm computational}\bigl(A^{-1}(Y-b)\bigr).
$$

## Normalized Fourier coordinates

The source Fourier grid and camera grid do not necessarily have the same
shape. Let their shapes be $N_s$ and $N_c$, interpreted as two-component
vectors in `(y, x)` order, and let

$$
    c_s = \left\lfloor\frac{N_s}{2}\right\rfloor,
    \qquad
    c_c = \left\lfloor\frac{N_c}{2}\right\rfloor
$$

be their centered-array coordinates. The normalized frequency coordinates are

$$
    \nu_s = D_s^{-1}(X-c_s),
    \qquad
    \nu_c = D_c^{-1}(Y-c_c),
$$

with $D_s=\operatorname{diag}(N_s)$ and
$D_c=\operatorname{diag}(N_c)$. Substituting the calibration relation gives

$$
    \nu_c = C\nu_s+d,
$$

where

$$
    C = D_c^{-1} A D_s,
    \qquad
    d = D_c^{-1}(A c_s+b-c_c).
$$

Using $A$ directly in the direct-plane transform would be incorrect when,
for example, the computational grid is 256×256 and the camera grid is
384×384. The normalized matrix $C$ accounts for that change in sampling.

## Fourier-dual direct-plane transform

The desired camera-grid Fourier field is

$$
    V(\nu_c) = U\bigl(C^{-1}(\nu_c-d)\bigr).
$$

Let $\bar y=y-c_c$ denote a centered coordinate on the camera-sized
direct grid. By the affine Fourier-transform relation, the corresponding
direct-plane field is, up to the normalization required by the chosen
discrete FFT,

$$
    v(y)
    = \alpha\,u(C^{\mathsf T}\bar y+c_s)\,
      \exp\!\left(2\pi i\,d^{\mathsf T}\bar y\right),
$$

where the global factor used here for orthonormal discrete FFTs is

$$
    \alpha
    = |\det C|\sqrt{\frac{\prod_i (N_c)_i}{\prod_i (N_s)_i}}.
$$

The matrix is transposed—not inverted—in the direct-plane spatial pull
transform. The Fourier-plane translation `b` becomes the linear phase ramp
$\exp(2\pi i d^{\mathsf T}\bar y)$.

Because `scipy.ndimage.affine_transform` is itself a pull transform, the
direct basis is sampled with

$$
    M_{\rm direct}=C^{\mathsf T},
    \qquad
    t_{\rm direct}=c_s-C^{\mathsf T}c_c.
$$

The resulting output shape is the camera image shape.

## Phase screens

The phase screen is applied before the Fourier transform, so it must undergo
the same direct-plane coordinate change. Writing

$$
    P_{s,p}(x)=\exp\bigl(i\phi_{s,p}(x)\bigr),
$$

the prepared phase factor is

$$
    P'_{s,p}(y)
    = P_{s,p}(C^{\mathsf T}\bar y+c_s)
      \exp\!\left(2\pi i\,d^{\mathsf T}\bar y\right).
$$

The prepared direct basis for mode $j$ is

$$
    B'_j(y)=\alpha\,B_j(C^{\mathsf T}\bar y+c_s).
$$

Consequently, Julia can form

$$
    V_{j,s,p}
    = \mathcal{F}_{\rm camera}\!\left[B'_j P'_{s,p}\right],
$$

which is equivalent to Fourier transforming the original phase-modulated
mode and then applying the inverse calibration map to the camera grid.

Complex phase factors are stored rather than phase angles. This avoids
interpolating across the $-\pi/\pi$ branch cut and retains the translation
ramp without any additional convention in Julia.

## Prepared HDF5 layout

`prepare_images.py` writes one file at the result-directory root:

```text
prepared.h5
├── phase_factors                         (sigma, phase, y, x), ComplexF32
└── orders/
    ├── <order-directory-name>/basis       (mode, y, x)
    └── ...
```

The raw experimental images remain in each order’s `data.h5`; coefficients
and the original basis remain in `modes.h5`. `single_image_tomography.jl`
reads the prepared direct fields, Fourier transforms them in memory, and uses
the raw camera images for estimation.

## Scope and assumptions

This construction relies on the calibration package’s similarity model: one
uniform scale, an orthogonal rotation or reflection, and a translation. It
assumes that all order directories in one result run use the same camera ROI
and camera shape, and that the Fourier arrays use the centered
`fftshift(fft2(ifftshift(...)))` convention.

The calibration describes geometric registration. It does not model camera
pixel integration, point-spread functions, saturation, or nonlinear detector
response. Those effects must be handled separately when interpreting the
experimental intensities.
