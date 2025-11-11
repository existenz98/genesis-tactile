
# Photometric stereo + Marker grid (e.g. GelSight) image generator  

Here is Blueprint of what the image should look like, and how synthesize is implemented.



## 1) What the Photometric stereo + Marker grid image looks like

- **Neutral gray at rest**
  With the gel surface flat (normal ≈ +z), the three oblique RGB lights contribute equally → the camera sees a near-uniform mid-gray field.  

- **Color encodes local surface normal**  
  Under indentation, each pixel’s color shifts toward the light whose direction it faces.  
  - Face **R-light** → redder; **G-light** → greener; **B-light** → bluer.  
  - Mixed orientations produce mixed hues.  
  - Steeper slopes darken against lights they turn away from (self-shadow).

- **Marker grid on the reflective skin**  
  A regular grid of dark micro-dots printed on the thin top reflective layer:  
  - Dots move with the skin → from the camera they appear slightly scaled/warped by foreshortening.  
  - They are high-contrast (nearly black) over the shaded background, aiding sub-pixel localization.  

- **Typical artifacts we should mimic (TODO)**  
  - Slight *non-Lambertian* sheen (a gentle highlight near a light direction).  
  - *Illumination non-uniformity* and *color cross-talk* (so it’s not too perfect).  
  - Vignetting, camera PSF blur, quantization, and sensor noise...  
  - A subtle dark ring at contact boundary due to shadowing ($\mathbf{n}\cdot\mathbf{l} < 0$).  

## 2) Inputs & frames (assumptions)  

- **FEM top surface**: deformed displacement field  
  $$
  \mathbf{u}(x,y,0) = (u_x, u_y, u_z)
  $$  
  on the gel’s top surface mesh.  

- **Camera**: same pinhole model you already use (from viewbox).  

- **Lights**: three directional lights (R,G,B) with known unit vectors  
  $$
  \ell_r, \;\ell_g, \;\ell_b
  $$  
  in the **camera frame**, plus per-channel intensities.  

- **Reflective layer**: simple diffuse model with small specular lobe, albedo $\rho$ (possibly spatially uniform).  

- **Markers**: regular (or jittered) dot grid: spacing, radius, color (dark), dropout rate.

## 3) Geometry → normals (ground truth first)

We’ll derive **per-pixel normals** and **depth** from the FEM surface:

1. **Parameterize the rest top surface** as a 2D domain $(x, y)$ in mm over the viewbox.  

2. **Sample deformed 3D positions** on a regular image grid using barycentric (or tri-linear) interpolation on the FEM mesh:  

   $$
   \mathbf{p}(x,y) =
   \begin{bmatrix}
   x + u_x(x,y) \\
   y + u_y(x,y) \\
   u_z(x,y)\;\; \text{(down is negative)}
   \end{bmatrix}
   $$

3. **Compute tangents and normals** by finite differences:  

   $$
   t_x = \frac{\partial \mathbf{p}}{\partial x}, \quad
   t_y = \frac{\partial \mathbf{p}}{\partial y}, \quad
   \mathbf{n} = \frac{t_x \times t_y}{\|t_x \times t_y\|}
   $$

   (Or directly from the deformed surface triangles if available.)

4. **Depth map**:  
   $$
   z(x,y) = p_z(x,y)
   $$  
   relative to the rest plane (for saving as GT).  

*Note*: For small slopes, the simple gradient-of-height formula is equivalent and fast.

$$
\mathbf{n} \propto \left(-\frac{\partial z}{\partial x},\; -\frac{\partial z}{\partial y},\; 1\right)
$$  

## 4) Photometric shading (physics-light but robust)

At each pixel with normal **n**:

1. **Diffuse (Oren–Nayar / Lambert hybrid)**  
   Use Lambert as default:  

   $$
   d_k = \max(0, \mathbf{n} \cdot \ell_k) \quad \text{for } k \in \{r,g,b\}
   $$

   Optionally include a roughness parameter $\sigma$ (Oren–Nayar) to widen lobes.

2. **Small specular lobe** (optional, gentle Blinn–Phong):  

   $$
   s_k = \left( \max(0, \mathbf{h}_k \cdot \mathbf{n}) \right)^\alpha,
   \quad
   \mathbf{h}_k = \frac{\ell_k + \mathbf{v}}{\|\ell_k + \mathbf{v}\|}
   $$

   with view vector $\mathbf{v} \approx (0,0,1)$, shininess $\alpha \in [8,64]$, and small weight $\eta \ll 1$.

3. **Per-channel signal before mixing**  

   $$
   I_k^{raw} = \rho L_k d_k + \eta S_k s_k + A_k
   $$

   where $L_k$ is light intensity, $S_k$ is specular intensity (optional), and $A_k$ is ambient baseline per channel.

4. **Color cross-talk & white balance**  

   Apply a $3 \times 3$ mixing matrix **C** (close to identity) to model camera spectral response and LED leakage:

   $$
   \mathbf{I} = \mathbf{C} 
   \begin{bmatrix}
   I_r^{raw} \\
   I_g^{raw} \\
   I_b^{raw}
   \end{bmatrix}
   $$

5. **Flat-field & vignetting** (optional realism)  
   Multiply by a smooth per-pixel gain map $F(x,y)$ per channel.

6. **Gamma & quantization**  
   Apply camera gamma (or tone curve), add mild shot+read noise, clamp to $[0,1]$, convert to **8-bit**.  
   *(We’ll output BGR for OpenCV.)*

**Neutral gray calibration**: choose $\{L_k, A_k\}$ so that for the flat normal  

$$
\mathbf{n}_0 = (0,0,1)
$$  

the three channels are equal (mid-gray).  This guarantees the **“unpressed = gray”** look.

## 5) Marker grid synthesis

- **Rest grid generation**: regular spacing (e.g., 0.5–1.0 mm), radius $r_m$, optional jitter and dropout.  

- **Deformed 3D positions**: sample FEM at marker $(x,y)$ to get $\mathbf{p}_i^{def}$.  

- **Projection to pixels**: using the same pinhole camera → `markers_rest_px` & `markers_def_px`.  

- **Foreshortening / shape**:  
  - Simplest: draw as circles with radius scaled by $1/n_z$ (cap at a factor to avoid extremes).  
  - Advanced (optional): ellipses using the local Jacobian  
    $$
    J = \frac{\partial (u,v)}{\partial (x,y)}.
    $$  

- **Compositing**: draw markers as **dark overlays** on top of the shaded image (alpha or replace).  
  Add tiny blur to mimic printing and defocus.  

**save** both rest/def **pixel** and **3D** coordinates as 'GT' for downstream iFEM/CNN.

## 6) Dense optical flow from markers (optional)

- **Compute sparse 2D displacements**  
  $$
  \Delta \mathbf{p}_i = \text{def\_px}_i - \text{rest\_px}_i
  $$

- **Interpolate to a dense field** on the image grid (with a mask):  
  - Preferred: **Thin-Plate Spline (TPS)** with robust weighting and outside-hull damping.  
  - Alternatives: **RBF**, **Delaunay barycentric**, or **griddat**a.  

- **Output**:  
  - `flow_from_markers_dense.npy` (H×W×2)  
  - `flow_from_markers_mask.png`  

## 7) Camera & rasterization details

- **Supersampling** (as you already use): render at $s \times$ resolution and downsample (box or Gaussian) to reduce aliasing in both shading and marker edges.  

- **PSF blur**: apply a small Gaussian after downsampling to mimic optics.  

- **Background**: keep `background_bgr` outside the gel area.

## 8) Outputs (modalities) to save per frame

- `render.png` *(final rendered image)*  
- `geom/normal_map.npy` (H×W×3, float32)  
- `geom/depth_map.npy` (H×W, float32, mm)  
- `markers/rest_px.npy`, `markers/def_px.npy` (N×2, float32)  
- `markers/rest_xyz.npy`, `markers/def_xyz.npy` (N×3, float32, mm)  
- `flow/dense.npy`, `flow/mask.png` *(from markers, optional, good for quick eval)*  
- `calib/` *(shared per dataset: light directions, intensities, mixing matrix, gamma, flat-field, camera intrinsics)*  

