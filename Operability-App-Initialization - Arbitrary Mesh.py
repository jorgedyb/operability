import numpy as np
import xarray as xr
import capytaine as cpt
from capytaine.meshes.predefined.rectangles import mesh_parallelepiped
from capytaine.post_pro import rao
import matplotlib.pyplot as plt
import os
import joblib

# Quieten Capytaine logging
cpt.set_logging('ERROR')

import numpy as np
import capytaine as cpt
import copy

def scale_mesh_to_dims(base_mesh, L_target, B_target, T_target):
    """Return a fresh copy of *base_mesh* scaled to L, B, T.
    Assumptions
    -----------
    * x axis  : length (FP at x=0, AP at x=L)
    * y axis  : transverse, symmetric about y=0
    * z axis  : vertical, z = 0 at still-water plane,
                z < 0 in the water (draft), z > 0 freeboard
    """

    # ---- 1. reference dimensions ------------------------------------------
    x_min, x_max, y_min, y_max, z_min, z_max = base_mesh.axis_aligned_bbox
    L0 = x_max - x_min
    B0 = y_max - y_min
    T0 = -z_min                     # draft is a positive number

    # ---- 2. scale factors --------------------------------------------------
    s_x = L_target / L0
    s_y = B_target / B0
    s_z = T_target / T0             # keeps the same freeboard / draft ratio

    # ---- 3. build scaled copy ---------------------------------------------
    mesh = copy.deepcopy(base_mesh)         # don't mutate the template
    V = mesh.vertices
    # a) move origin to the chosen reference points
    V[:, 0] -= x_min          # FP → 0
    V[:, 1] -= (y_min + y_max)/2      # centre-line → 0
    # z already has z=0 at water-line
    # b) anisotropic scaling
    V *= np.array([s_x, s_y, s_z])
    mesh.vertices = V
    return mesh


def create_barge_mesh(length, beam, draft,
                      n_length=20, n_beam=8, n_draft=4,
                      freeboard=2.0):
    """
    Generate a subdivided barge mesh (quads) and return a Capytaine-triangulated Mesh.
    The waterline is at z=0: bottom at -draft, deck at +freeboard.
    """
    total_height = draft + freeboard
    center_z = (freeboard - draft) / 2
    quad_mesh = mesh_parallelepiped(
        size=(length, beam, total_height),
        resolution=(n_length, n_beam, n_draft),
        center=(length/2, 0.0, center_z),
        name="barge_quad_mesh"
    )
    print(f"Quad mesh: {quad_mesh.faces.shape[0]} faces, "
          f"max quad radius = {quad_mesh.faces_radiuses.max():.6f}")
    # convert quads to degenerate-quad triangles
    V = quad_mesh.vertices
    Q = quad_mesh.faces  # (n_quads, 4)
    F = np.vstack([
        [[a, b, c, a], [a, c, d, a]]
        for a, b, c, d in Q
    ])
    mesh = cpt.Mesh(vertices=V, faces=F)
    print(f"Triangulated mesh: {mesh.faces.shape[0]} panels, "
          f"max tri radius = {mesh.faces_radiuses.max():.6f}")
    #mesh.show()
    return mesh


def create_floating_body(mesh, length, draft):
    """
    Wrap a Capytaine Mesh into a FloatingBody with correct center of mass and rotation.
    """
    body = cpt.FloatingBody(
        mesh=mesh,
        name="OSV",
        center_of_mass=[length/2, 0.0, -draft/2]
    )
    body.add_all_rigid_body_dofs()
    body.rotation_center = [length/2, 0.0, -draft/2]
    #print(f"Created body: {body.name}, "
    #      f"mass={body.mass}, center_of_mass={body.center_of_mass}, "
    #      f"rotation_center={body.rotation_center}")
    return body


def compute_hydrodynamics(body, omegas, wave_directions):
    """
    Use fill_dataset to compute radiation and diffraction for given omegas and wave_directions.
    Returns an xarray Dataset with hydrodynamic coefficients.
    """
    test_matrix = xr.Dataset(coords={
        'omega':          omegas,
        'wave_direction': wave_directions,
        'radiating_dof':  list(body.dofs),
        'water_depth':    [np.inf]
    })
    #print(f"Computing hydrodynamics for {len(omegas)} frequencies and {len(wave_directions)} directions.")
    #print(f"Radiating DOFs: {list(body.dofs)}")
    dataset = cpt.BEMSolver().fill_dataset(test_matrix, body, n_jobs=20)
    return dataset


def add_hydrostatic_and_inertia(dataset, body):
    """
    Compute hydrostatic stiffness and rigid-body inertia, add them to the dataset.
    """
    immersed = body.immersed_part()
    hydro = immersed.compute_hydrostatics()
    inertia = immersed.compute_rigid_body_inertia()
    dataset['hydrostatic_stiffness'] = hydro['hydrostatic_stiffness']
    dataset['inertia_matrix'] = inertia
    return dataset


def postprocess_raos(dataset, output_prefix="OSV"):
    """
    Compute RAOs from the dataset and plot/save results.
    """
    raos = rao(dataset)
    #print("Computed RAOs:")
    #print(raos)
    # Plot RAOs
    #for dof in raos.radiating_dof.values:
        #plt.figure()
        #plt.plot(raos.omega, np.abs(raos.sel(radiating_dof=dof)))
        #plt.title(f"{dof} RAO Magnitude vs Frequency")
        #plt.xlabel('Omega [rad/s]')
        #plt.ylabel('RAO Magnitude')
        #plt.grid(True)
        #plt.tight_layout()
        #plt.show()
    
    # Export
    rao_df = raos.to_dataframe(name="RAO").reset_index()
    csv_name = f"{output_prefix}_rao.csv"
    rao_df.to_csv(csv_name, index=False)
    print(f"RAO results saved to {csv_name}")


def main():
    # Parameter ranges
    lengths   = np.linspace(70,90,21)  # Lengths in meters
    beams     = np.linspace(15, 20, 11)
    drafts    = np.linspace(4, 7, 4)
    omegas    = np.linspace(0.1, 4, 100)
    wave_dirs = np.linspace(0, 2*np.pi, 17)

    # 1) make sure the folder exists
    output_dir = "rao_outputs_fine"
    os.makedirs(output_dir, exist_ok=True)

    import time, logging
    from tqdm.auto import tqdm

    logging.basicConfig(level=logging.INFO)   # or DEBUG for even more chatter

    template_mesh = cpt.load_mesh("OSV.stl", file_format='stl')
    template_mesh = template_mesh.translate_z(-4)
    template_mesh.show()


    for L in tqdm(lengths, desc="Lengths"):
        for B in tqdm(beams, desc="Beams", leave=False):
            for T in tqdm(drafts, desc="Drafts", leave=False):
                prefix = f"{output_dir}/OSV_L{int(L)}_B{int(B)}_T{int(T)}"
                if prefix + "_rao.csv" in os.listdir(output_dir):
                    continue
                else:
                    tic = time.perf_counter()  
                    mesh = scale_mesh_to_dims(template_mesh, L, B, T)
                    body = create_floating_body(mesh, L, T)
                    logging.debug("prep %.2fs", time.perf_counter() - tic)
                    ds = compute_hydrodynamics(body, omegas, wave_dirs)
                    ds = add_hydrostatic_and_inertia(ds, body)
                    postprocess_raos(ds, output_prefix=prefix)


if __name__ == '__main__':
    main()

#RAO results saved to rao_outputs_fine/barge_L89_B10_T5_rao.csv