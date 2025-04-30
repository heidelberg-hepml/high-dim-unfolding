import torch
import fastjet as fj
import numpy as np
import awkward as ak


def convert_to_ak(batched_particles, ptr):
    """
    Convert a batch of particles to awkward ragged array.
    Args:
        batched_particles: 2D array of shape (n_particles, 4) representing the 4-momentum of particles.
        ptr: 1D array of shape (n_jets + 1,) indexing the start of each jet in the batch.
    Returns:
        particles: 3D array of shape (n_jets, n_particles_in_jet, 4) representing the 4-momentum of
            particles in each jet and where n_particles_in_jet is variable.
    """
    count = ptr[1:] - ptr[:-1]
    nested_array = ak.unflatten(batched_particles, count)
    E, px, py, pz = (
        nested_array[..., 0],
        nested_array[..., 1],
        nested_array[..., 2],
        nested_array[..., 3],
    )
    particles = ak.zip({"px": px, "py": py, "pz": pz, "E": E})
    return particles


def calculate_nsubjettiness(jet_particles, n_subjets=2, beta=1, R0=0.4):
    """
    Compute N-subjettiness for a jet given a list of jet particles (PseudoJets).
    Args:
        jet_particles: List of fastjet PseudoJets.
        n_subjets: Number of subjets for N-subjettiness computation.
    Returns:
        tau_N: List of N-subjettiness for 1, 2, and 3 subjets.
    """
    # Ensure that jet_particles is a list of fastjet PseudoJets
    if not isinstance(jet_particles[0], fj.PseudoJet):
        raise TypeError("The input must be a list of fastjet.PseudoJet objects.")

    # Compute the N-subjettiness for N = 1, 2, 3 using the fastjet clustering
    jet_clusterer = fj.ClusterSequence(
        jet_particles, fj.JetDefinition(fj.antikt_algorithm, R0)
    )  # R=0.4
    all_jets = jet_clusterer.inclusive_jets()

    tau_N = []
    for N in range(1, n_subjets + 1):
        # For each N-subjettiness, calculate the ratio of the transverse momentum alignment
        tau = 0
        d0 = np.sum([jet.pt() for jet in all_jets[:N]]) * (
            R0**beta
        )  # Normalization factor
        for jet in all_jets[:N]:
            for particle in jet_particles:
                # Calculate the delta R between the jet and the particle
                delta_r = jet.delta_r(particle)
                if delta_r < min_delta_r:
                    min_delta_r = delta_r
                tau += particle.pt() * (min_delta_r**beta)
        tau /= d0  # Normalize by total pt
        tau_N.append(tau)

    return tau_N
