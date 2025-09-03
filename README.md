# Genesis.tactile — Vision‑Tactile Platform

**Design EDA → AI Model → Real‑Time Runtime** • Unify different types of camera‑based “vision‑tactile” sensors  
**Goals:** *Simulate • Benchmark • AI • Deploy • Scale*

---

## TL;DR

Vision-Tactile Sensor’s goal is to reconstruct **dense 3D force fields** (pressure + shear).

**Genesis.tactile** is a **Vision-Tactile Platform** that takes you all the way from **sensor design & simulation**, to **AI model training**, to a **production-ready runtime**, and a **developer SDK**.  

The Runtime algorithm uses the camera observations of soft tactile sensors **in real time**, and to provide a unified interface that works across **different vision-tactile sensor types**, including GelSight-style sensors (e.g., GelSight, 千觉), random-particle sensors, and Tac3D-style sensors (e.g., Acorn Robotics Tac3D).


<p align="center">
  <img src="docs/assets/genesis_tactile_overview.png" width="500" alt="Genesis.tactile overview">
  <br><em>Figure 1 — Platform overview.</em>
</p>

---

## What’s inside (4 components)

## Platform Components

| Component | Illustration | Description |
|-----------|--------------|-------------|
| **1. EDA / Simulation (offline)** | <img src="docs/assets/panel_eda.png" width="700" alt="EDA / Simulation"> | Generative **force-field** → **FEM** → **iFEM** (verification) → **synthetic camera image** supports several types of sensor design. |
| **2. Dataset & AI Training** | <img src="docs/assets/panel_training.png" width="700" alt="Dataset & Training"> | Sample generators, dataset builders, ai model (CNN) with PINN physics-guided residual losses, and training pipelines. |
| **3. Runtime (online)** | <img src="docs/assets/panel_runtime.png" width="700" alt="Runtime"> | Sensor HAL → preprocessing → optical flow → **solvers** (physics, iFEM, CNN) → **3D force field** at interactive rates. |
| **4. SDK & Visualizer** | <img src="docs/assets/panel_sdk.png" width="700" alt="SDK & Visualizer"> | Multi-client SDK to subscribe to force fields, plus **2D/3D visualizer** for inspection, debugging, and demos. |


---

## Quick links

- **Design&Simulation**: [`src/`](src/) → 3D force field generators, FEM, iFEM, synthetic camera observations
- **Runtime (service)**: [`runtime/`](runtime/) → configs, deployment, metrics
- **SDK (client API)**: [`sdk/`](sdk/) → Python API, examples
- **Apps & Visualizer**: [`apps/`](apps/) → 2D+3D viewer

---

## Highlights

- **End‑to‑end**: design & simulate → auto‑generate datasets → train AI → deploy.  
- **Physics‑aware AI**: CNN/PINN heads for **pressure** and **shear**, with physical residuals.  
- **Unified platform**: common runtime + SDK across vision‑tactile sensor types.  
- **Realtime**: camera → pre‑proc → flow → solver → **3D force field** stream.  
- **Dev‑friendly**: simple SDK; reference **2D/3D visualizer** out of the box.  

---

## Repository layout & stability

```text
.
├── src/       # research & EDA: force-field gen, FEM/iFEM, synthetic camera, dataset generation, ai model training
├── runtime/   # product-ready real-time service
├── sdk/       # developer SDKs and stable APIs
└── app/       # 2D/3D visualizer reference applications
```

---

## License

### Dual Licensing (Commercial Use)
If you wish to use `/runtime` or `/src` in a **commercial or production** setting, or you require terms different from AGPL/LGPL/Research‑NC, please contact:

**Yue Fei** \<feiyuefy@gmail.com\>

We offer commercial licenses (OEM, Enterprise, Cloud) that allow closed-source integration, production deployment, SLA support, and brand/compatibility certification.

## License

This repository uses **directory-specific licensing**:

| Path       | License                                                        |
|-----------:|:----------------------------------------------------------------|
| `/runtime` | **AGPL-3.0-or-later** + **Device Interface Exception** (see `LICENSES/AGPL-3.0-or-later.txt` and `LICENSES/LicenseRef-YF-Device-Interface-Exception.txt`) |
| `/sdk`     | **LGPL-3.0-or-later** (see `LICENSES/LGPL-3.0-or-later.txt`)   |
| `/src`     | **YF Research & Non-Commercial License v1.0** (see `LICENSES/LicenseRef-YF-Research-NC-1.0.txt`) |
| `/apps`    | **MIT** (see `LICENSES/MIT.txt`)                               |

**Commercial Use**  
To use `/runtime` or `/src` in closed-source or production settings, or under terms
different from AGPL/LGPL/Research-NC, please contact **Yue Fei** <feiyuefy@gmail.com>.
Commercial licenses (OEM/Enterprise/Cloud) are available.

**Patents**  
Except as granted under applicable open-source licenses, no patent license is granted
or implied. You are responsible for third-party patent clearance (e.g., tactile hardware/methods).

### Trademarks
Project names and logos are trademarks of their respective owners. No trademark license is granted.

## FAQ

- **Can I build a closed-source app using the SDK?**  
  Yes. The SDK is under LGPL-3.0-or-later. You may link it from proprietary apps
  under LGPL terms. If you modify the SDK itself and distribute those changes,
  you must release those modifications under the same license.

- **Can I ship a proprietary device driver for my tactile sensor?**  
  Yes. If your driver communicates with the Runtime **only** through the
  documented TSI/plugin/IPC interfaces, the **Device Interface Exception**
  allows you to license that driver under your own terms. If you modify the
  Runtime itself, those modifications must follow AGPL.

- **Can I use `/src` (design/simulation/training) in a company project?**  
  For **internal research/evaluation** only, yes (non-commercial). Any
  **production or commercial** use requires a separate commercial license.

- **Can I host the Runtime as a network service?**  
  The Runtime is AGPL-licensed, which applies to network use. If you provide it
  as a service or embed it into a closed system, you must comply with AGPL by
  releasing source code or instead obtain a **commercial license**.

- **Do I need to license `/apps` (like the 3D Visualizer) if I modify and
  redistribute it?**  
  No special license is required beyond MIT. The code is under MIT, so you may
  modify and redistribute it freely. However, the **brand name, logo, and UI
  themes** are *not* under MIT. Forks and redistributions must remove or replace
  the original branding unless you have written permission. See
  `LICENSE-APPS-ASSETS` and `TRADEMARKS.md`.

- **Are patents included in the open-source licenses?**  
  For AGPL/LGPL/MIT components, only the standard patent clauses (if any) apply.
  The custom Research-NC and Exception licenses **do not** grant patent rights.
  Users are responsible for ensuring they do not infringe third-party patents.

- **What if I want to contribute code?**  
  Contributions are welcome! By submitting, you agree to the Contributor License
  Agreement (CLA), which allows us to keep dual-licensing (community + commercial)
  consistent. See `CONTRIBUTING.md` for details.

- **How do I get a commercial license?**  
  Contact **Yue Fei** <feiyuefy@gmail.com>. We offer OEM, Enterprise, and Cloud
  licenses that allow closed-source integration, production deployment, SLA
  support, and certified branding.


---

## Citation

If you use Genesis.touch in your research, please cite:

```bibtex
@software{genesis_touch_2025,
  title        = {Genesis.tactile — Vision–Tactile Platform},
  author       = {Yue Fei},
  year         = {2025},
  version      = {0.9.0},
  url          = {https://github.com/existenz98/genesis-tactile.git}
}
```

