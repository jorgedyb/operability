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
        name="barge",
        center_of_mass=[length/2, 0.0, -draft/2]
    )
    body.add_all_rigid_body_dofs()
    body.rotation_center = [length/2, 0.0, -draft/2]
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


def postprocess_raos(dataset, output_prefix="barge"):
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
    lengths   = np.linspace(70, 90, 31)
    beams     = np.linspace(15, 20, 11)
    drafts    = np.linspace(6, 7, 4)
    omegas    = np.linspace(0.1, 4, 1000)
    wave_dirs = np.linspace(0, 2*np.pi, 17)

    # 1) make sure the folder exists
    output_dir = "rao_outputs_fine"
    os.makedirs(output_dir, exist_ok=True)

    for L in lengths:
        for B in beams:
            for T in drafts:
                mesh = create_barge_mesh(L, B, T)
                mesh.show()
                body = create_floating_body(mesh, L, T)
                ds = compute_hydrodynamics(body, omegas, wave_dirs)
                ds = add_hydrostatic_and_inertia(ds, body)

                # 2) build a filename inside that folder
                prefix = f"{output_dir}/barge_L{int(L)}_B{int(B)}_T{int(T)}"
                postprocess_raos(ds, output_prefix=prefix)


if __name__ == '__main__':
    main()

#RAO results saved to rao_outputs_fine/barge_L89_B10_T5_rao.csv