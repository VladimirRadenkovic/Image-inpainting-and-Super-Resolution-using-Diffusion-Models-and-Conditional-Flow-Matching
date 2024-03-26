from typing import Tuple, Union

import biotite.structure as struc
import numpy as np
import ammolite

from src.utils.biotite_utils import atom_array_from_numpy




def convert_ss_annotations(ss_array: np.ndarray) -> np.ndarray:
    # Define mapping from biotite annotations to PyMOL annotations
    ss_mapping = {"c": "L", "a": "H", "b": "S"}

    # Create a vectorized function to apply the mapping to each element of ss_array
    vectorized_mapping = np.vectorize(ss_mapping.get)

    # Apply the mapping and return the result
    return vectorized_mapping(ss_array)


def superimpose(vec1: np.ndarray, vec2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the rotation matrix and translation vector
    that superimpose vec2 onto vec1.

    NOTE: You can use the rotation R without the translation t if you
        want to rotate a vector in the same coordinate system as vec1.
    NOTE: To superimpose, apply: vec2 = np.dot(vec2, R) + t

    Args:
        vec1 (np.ndarray): The reference vector, of shape (n, 3)
        vec2 (np.ndarray): The vector to be superimposed, of shape (n, 3)

    Returns:
        R (np.ndarray): The rotation matrix that superimposes vec2 onto vec1
        t (np.ndarray): The translation vector that superimposes vec2 onto vec1
    """
    assert vec1.shape == vec2.shape
    n = vec1.shape[0]  # total points

    centroid_vec1 = np.mean(vec1, axis=0)
    centroid_vec2 = np.mean(vec2, axis=0)

    # center the points
    _vec1 = vec1 - centroid_vec1
    _vec2 = vec2 - centroid_vec2

    # singular value decomposition
    H = np.dot(_vec1.T, _vec2)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_vec2 - np.dot(centroid_vec1, R)

    return R, t


def apply_publication_settings() -> None:
    """
    Apply settings that are suitable for publication.
    """
    ammolite.cmd.set("cartoon_discrete_colors", 1)
    ammolite.cmd.set("cartoon_oval_length", 1.0)
    ammolite.cmd.set("ray_trace_mode", 1)
    ammolite.cmd.set("ambient", 0.1)


def visualise_structure(ca, show_secondary_structure: bool = True):
    # Load the C-alpha structure
    pymol_object = ammolite.PyMOLObject.from_structure(ca)

    if show_secondary_structure:
        # Compute secondary structure with biotite
        sse = struc.annotate_sse(ca)
        sse = convert_ss_annotations(sse)

        # 1) Color each secondary structure element
        # Create a selection for each secondary structure element
        selections = {"L": [], "H": [], "S": []}
        for resi, ss in zip(ca.res_id, sse):
            selections[ss].append(resi)

        # Color each selection based on its secondary structure
        # You can use any colors you like for these
        colors = {"L": "white", "H": "salmon", "S": "lightblue"}
        for ss, color in colors.items():
            # Create a selection string for this secondary structure
            selection_string = (
                f'resi {"+".join(map(str, selections[ss]))} and model {pymol_object.name}'
            )
            # Color this selection
            ammolite.cmd.color(color, selection_string)

        # 2) Show secondary structure as cartoon
        for resi, ss in zip(ca.res_id, sse):
            ammolite.cmd.alter(f"resi {resi}", f'ss="{ss}"')

    return pymol_object


def add_spheres_at_residues(pymol_object, res_ids):
    # Create a selection string for these residues
    selection_string = (
        f'resi {"+".join(map(str, res_ids))} and name CA and model {pymol_object.name}'
    )

    # Show these residues as dots
    ammolite.cmd.show("spheres", selection_string)

    # Adjust the sphere_scale to make the dots smaller, if necessary
    ammolite.cmd.set("sphere_scale", 0.2, selection_string)
    return selection_string


def quick_save(save_path: str):
    ammolite.cmd.png(str(save_path), width=1000, height=660, dpi=200, ray=0)


def publish_save(save_path: str):
    ammolite.cmd.png(str(save_path), width=3000, height=2000, dpi=300, ray=1)


def quick_vis(
    coords: np.ndarray,
    cond_res: np.ndarray = None,
    current_evec: np.ndarray = None,
    target_evec: np.ndarray = None,
    zoom_out: bool = True,
    save_path: str = None,
    save_for_publication: bool = False,
    amplitude=1.0,
):
    """
    Visualize a protein structure with optional annotations and save the image.

    Args:
        coords: A numpy array of shape (n, 3) containing the coordinates of the backbone CA atoms.
        cond_res: A numpy array of shape (m,) containing the indices of the conditioned residues to annotate.
        current_evec: A numpy array of shape (m, 3) containing the eigenvector components of the current structure to annotate.
        target_evec: A numpy array of shape (m, 3) containing the target eigenvector to annotate.
        zoom_out: A boolean indicating whether to zoom out to show the full structure.
        save_path: A string containing the path to save the image. If None, the image is not saved.
        save_for_publication: A boolean indicating whether to apply publication settings to the image.
        amplitude: A float indicating the amplitude of the eigenvectors.

    Returns:
        If save_path is None, returns an ammolite ObjectMolecule containing the visualization.
        If save_path is not None, saves the image and returns the path to the saved image.
    """


def quick_vis(
    coords: np.ndarray,
    cond_res: np.ndarray = None,
    current_evec: np.ndarray = None,
    target_evec: np.ndarray = None,
    current_displacement: np.ndarray = None,
    target_displacement: np.ndarray = None,
    zoom_out: bool = True,
    save_path: str = None,
    save_for_publication: bool = False,
    amplitude=1.0,
):
    """
    Visualize a protein structure with optional annotations and save the image.
    Specify either evecs or displacements."""
    ammolite.reset()

    ca = atom_array_from_numpy(coords)
    pymol_object = visualise_structure(ca)

    if cond_res is not None:
        selection = add_spheres_at_residues(
            pymol_object, cond_res + 1
        )  # +1 for 1-indexing in biotite
        ammolite.cmd.color("purple", selection)

        if current_evec is not None:
            # Show current eigenvector
            ammolite.draw_arrows(
                ca.coord[cond_res],
                ca.coord[cond_res] + current_evec * amplitude,
                radius=0.2,
                head_radius=0.4,
                head_length=0.7,
                # purple color in RGB
                color=(0.5, 0.0, 0.5),
            )
        elif current_displacement is not None:
            # Show current displacement
            ammolite.draw_arrows(
                ca.coord[cond_res],
                ca.coord[cond_res] + current_displacement,
                radius=0.2,
                head_radius=0.4,
                head_length=0.7,
                # purple color in RGB
                color=(0.5, 0.0, 0.5),
            )

            if target_evec is not None:
                # Show target eigenvector
                R, _ = superimpose(current_evec, target_evec)
                target_evec = np.dot(target_evec, R)
                ammolite.draw_arrows(
                    ca.coord[cond_res],
                    ca.coord[cond_res] + target_evec * amplitude,
                    radius=0.2,
                    head_radius=0.4,
                    head_length=0.7,
                    color=(0.52, 0.75, 0.00),
                )
            elif target_displacement is not None:
                # Show target displacement
                R, _ = superimpose(current_displacement, target_displacement)
                target_displacement = np.dot(target_displacement, R)
                ammolite.draw_arrows(
                    ca.coord[cond_res],
                    ca.coord[cond_res] + target_displacement,
                    radius=0.2,
                    head_radius=0.4,
                    head_length=0.7,
                    color=(0.52, 0.75, 0.00),
                )

        # Zoom on condition
        ammolite.cmd.zoom(f"resi {max(min(cond_res) - 2, 0)}-{min(max(cond_res)+2, len(ca))}")

    if zoom_out:
        # Zoom on full structure
        ammolite.cmd.zoom("all")

    # Make the cartoon transparent to reveal potentially hidden arrows
    ammolite.cmd.set("cartoon_transparency", 0.6, pymol_object.name)

    if save_path is None:
        # return ammolite.show()
        return pymol_object
    else:
        if save_for_publication:
            apply_publication_settings()
            sess = ammolite.cmd.get_session(names='')
            ammolite.cmd.save('dyn_mot.pse')  
            # publish_save(save_path)
        else:
            quick_save(save_path)
        return save_path
    
def vis_pt_sample(sample):
    pos = (sample.pos*15).cpu().numpy()
    motif_inds = sample.motif_inds.cpu().numpy()
    target_displacements = sample.target_displacements.cpu().numpy()*15
    current_displacements = sample.final_displacements.squeeze().cpu().numpy()*15

    # Visualize the sample
    quick_vis(
        pos,
        motif_inds,
        current_displacement=current_displacements,
        target_displacement=target_displacements,
        save_path="sample.png",
        save_for_publication=True,
    )

if __name__ == "__main__":
    # Load the data
    import torch
    sample = torch.load("/home/ked48/rds/hpc-work/protein-diffusion/data/test/jloss_0.080_s1.pt", map_location=torch.device('cpu'))

    vis_pt_sample(sample)