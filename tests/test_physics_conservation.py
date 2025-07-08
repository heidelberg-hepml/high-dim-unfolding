import pytest
import torch
import numpy as np

from experiments.utils import get_pt, get_phi, get_eta, get_mass
from experiments.coordinates import (
    fourmomenta_to_jetmomenta,
    jetmomenta_to_fourmomenta
)
from experiments.transforms import (
    FourmomentaToPtPhiEtaM2,
    FourmomentaToLogPtPhiEtaLogM2
)


class TestFourMomentumConservation:
    """Test conservation of four-momentum in transformations"""
    
    @pytest.fixture
    def sample_events(self):
        """Create sample particle physics events"""
        batch_size, max_particles = 10, 20
        
        # Generate realistic fourmomenta
        fourmomenta = torch.randn(batch_size, max_particles, 4)
        
        # Ensure physical constraints: E >= |p|
        fourmomenta[:, :, 0] = torch.abs(fourmomenta[:, :, 0]) + 2.0  # E > 0
        
        # Adjust energies to satisfy E^2 >= p^2
        p_squared = (fourmomenta[:, :, 1:] ** 2).sum(dim=-1)
        E_squared = fourmomenta[:, :, 0] ** 2
        
        mask = E_squared < p_squared
        fourmomenta[mask, 0] = torch.sqrt(p_squared[mask]) + 0.1
        
        # Create multiplicities (number of particles per event)
        mults = torch.randint(5, max_particles, (batch_size,))
        
        return fourmomenta, mults
    
    def test_energy_momentum_conservation_in_transforms(self, sample_events):
        """Test that coordinate transforms preserve total four-momentum"""
        fourmomenta, mults = sample_events
        
        transform = FourmomentaToPtPhiEtaM2()
        
        for i, mult in enumerate(mults):
            # Get particles for this event
            particles = fourmomenta[i, :mult]
            
            # Calculate total four-momentum before transform
            total_before = particles.sum(dim=0)
            
            # Transform to cylindrical coordinates
            cylindrical = transform.forward(particles)
            
            # Transform back to fourmomenta
            reconstructed = transform.inverse(cylindrical)
            
            # Calculate total four-momentum after round-trip
            total_after = reconstructed.sum(dim=0)
            
            # Check conservation
            torch.testing.assert_close(
                total_before, total_after,
                rtol=1e-4, atol=1e-5,
                msg=f"Four-momentum not conserved in event {i}"
            )
    
    def test_invariant_mass_conservation(self, sample_events):
        """Test that invariant mass is preserved in transforms"""
        fourmomenta, mults = sample_events
        
        transform = FourmomentaToPtPhiEtaM2()
        
        for i, mult in enumerate(mults):
            particles = fourmomenta[i, :mult]
            
            # Calculate invariant mass before transform
            total_4mom = particles.sum(dim=0)
            mass_before = torch.sqrt(
                total_4mom[0]**2 - (total_4mom[1:]**2).sum()
            )
            
            # Transform and back
            cylindrical = transform.forward(particles)
            reconstructed = transform.inverse(cylindrical)
            
            # Calculate invariant mass after
            total_4mom_after = reconstructed.sum(dim=0) 
            mass_after = torch.sqrt(
                total_4mom_after[0]**2 - (total_4mom_after[1:]**2).sum()
            )
            
            torch.testing.assert_close(
                mass_before, mass_after,
                rtol=1e-4, atol=1e-5,
                msg=f"Invariant mass not conserved in event {i}"
            )
    
    def test_individual_particle_mass_shell(self, sample_events):
        """Test that individual particles satisfy mass-shell constraint E² - p² = m²"""
        fourmomenta, mults = sample_events
        
        for i, mult in enumerate(mults):
            particles = fourmomenta[i, :mult]
            
            E = particles[:, 0]
            p_squared = (particles[:, 1:] ** 2).sum(dim=-1)
            
            # Mass-shell constraint: E² - p² = m² >= 0
            mass_squared = E**2 - p_squared
            
            assert torch.all(mass_squared >= -1e-6), \
                f"Mass-shell constraint violated in event {i}: min m² = {mass_squared.min()}"
    
    def test_lorentz_invariance_under_boosts(self):
        """Test that physics quantities remain invariant under Lorentz boosts"""
        # Create a simple two-particle system
        particles = torch.tensor([
            [5.0, 3.0, 4.0, 0.0],  # Particle 1
            [4.0, 1.0, 2.0, 3.0]   # Particle 2
        ])
        
        # Calculate invariant mass in lab frame
        total = particles.sum(dim=0)
        invariant_mass_lab = torch.sqrt(total[0]**2 - (total[1:]**2).sum())
        
        # Apply a boost in z-direction (β = 0.6, γ = 1.25)
        beta = 0.6
        gamma = 1.0 / torch.sqrt(1 - beta**2)
        
        boosted_particles = particles.clone()
        # Lorentz boost: E' = γ(E - βpz), pz' = γ(pz - βE)
        boosted_particles[:, 0] = gamma * (particles[:, 0] - beta * particles[:, 3])
        boosted_particles[:, 3] = gamma * (particles[:, 3] - beta * particles[:, 0])
        
        # Calculate invariant mass in boosted frame
        total_boosted = boosted_particles.sum(dim=0)
        invariant_mass_boosted = torch.sqrt(
            total_boosted[0]**2 - (total_boosted[1:]**2).sum()
        )
        
        # Invariant mass should be the same
        torch.testing.assert_close(
            invariant_mass_lab, invariant_mass_boosted,
            rtol=1e-6, atol=1e-7,
            msg="Invariant mass not preserved under Lorentz boost"
        )


class TestAngularQuantities:
    """Test conservation and properties of angular quantities"""
    
    def test_rapidity_pseudorapidity_relation(self):
        """Test relationship between rapidity and pseudorapidity for massless particles"""
        # Create massless particles (E = |p|)
        particles = torch.randn(5, 4)
        p_mag = torch.sqrt((particles[:, 1:] ** 2).sum(dim=-1))
        particles[:, 0] = p_mag  # Set E = |p| for massless particles
        
        # Calculate pseudorapidity η
        pt = torch.sqrt(particles[:, 1]**2 + particles[:, 2]**2)
        pz = particles[:, 3]
        eta = 0.5 * torch.log((p_mag + pz) / (p_mag - pz + 1e-8))
        
        # Calculate rapidity y
        E = particles[:, 0]
        rapidity = 0.5 * torch.log((E + pz) / (E - pz + 1e-8))
        
        # For massless particles, η ≈ y
        torch.testing.assert_close(
            eta, rapidity,
            rtol=1e-5, atol=1e-6,
            msg="Pseudorapidity and rapidity should be equal for massless particles"
        )
    
    def test_phi_periodicity(self):
        """Test that φ angle is properly handled with 2π periodicity"""
        # Create particles with φ near boundaries
        particles = torch.tensor([
            [5.0, 1.0, 0.1, 0.0],    # φ ≈ 0
            [5.0, -1.0, 0.1, 0.0],   # φ ≈ π
            [5.0, 0.1, 1.0, 0.0],    # φ ≈ π/2
            [5.0, 0.1, -1.0, 0.0]    # φ ≈ -π/2
        ])
        
        phi = get_phi(particles)
        
        # Check that φ is in [-π, π]
        assert torch.all(phi >= -np.pi), "φ should be >= -π"
        assert torch.all(phi <= np.pi), "φ should be <= π"
        
        # Test specific values
        expected_phi = torch.tensor([
            np.arctan2(0.1, 1.0),
            np.arctan2(0.1, -1.0),
            np.arctan2(1.0, 0.1),
            np.arctan2(-1.0, 0.1)
        ])
        
        torch.testing.assert_close(phi, expected_phi, rtol=1e-5, atol=1e-6)
    
    def test_angular_separation_symmetry(self):
        """Test that angular separation ΔR is symmetric"""
        particles1 = torch.tensor([[5.0, 2.0, 1.0, 0.5]])
        particles2 = torch.tensor([[4.0, 1.0, 2.0, -0.5]])
        
        eta1 = get_eta(particles1)
        phi1 = get_phi(particles1) 
        eta2 = get_eta(particles2)
        phi2 = get_phi(particles2)
        
        # Calculate ΔR
        delta_eta = eta1 - eta2
        delta_phi = phi1 - phi2
        
        # Handle φ wraparound
        delta_phi = torch.where(
            delta_phi > np.pi, delta_phi - 2*np.pi,
            torch.where(delta_phi < -np.pi, delta_phi + 2*np.pi, delta_phi)
        )
        
        delta_r_12 = torch.sqrt(delta_eta**2 + delta_phi**2)
        
        # Reverse calculation
        delta_eta_21 = eta2 - eta1
        delta_phi_21 = phi2 - phi1
        delta_phi_21 = torch.where(
            delta_phi_21 > np.pi, delta_phi_21 - 2*np.pi,
            torch.where(delta_phi_21 < -np.pi, delta_phi_21 + 2*np.pi, delta_phi_21)
        )
        
        delta_r_21 = torch.sqrt(delta_eta_21**2 + delta_phi_21**2)
        
        # Should be symmetric
        torch.testing.assert_close(delta_r_12, delta_r_21, rtol=1e-6, atol=1e-7)


class TestJetPhysics:
    """Test jet-related physics constraints"""
    
    def test_jet_momentum_conservation(self):
        """Test that jet momentum equals sum of constituents"""
        # Create constituent particles
        constituents = torch.tensor([
            [10.0, 6.0, 8.0, 0.0],
            [8.0, 3.0, 4.0, 6.0],
            [5.0, 1.0, 2.0, 4.0]
        ])
        
        # Calculate jet momentum as sum
        jet_momentum = constituents.sum(dim=0)
        
        # Use the fourmomenta_to_jetmomenta function
        jet_from_function = fourmomenta_to_jetmomenta(constituents.sum(dim=0, keepdim=True))
        
        # Should be consistent
        torch.testing.assert_close(
            jet_momentum.unsqueeze(0), jet_from_function,
            rtol=1e-6, atol=1e-7
        )
    
    def test_jet_mass_calculation(self):
        """Test jet mass calculation"""
        jet_momentum = torch.tensor([[50.0, 30.0, 40.0, 0.0]])
        
        # Calculate mass: m = √(E² - p²)
        E = jet_momentum[:, 0]
        p_squared = (jet_momentum[:, 1:] ** 2).sum(dim=-1)
        expected_mass = torch.sqrt(E**2 - p_squared)
        
        # Use utility function
        calculated_mass = get_mass(jet_momentum)
        
        torch.testing.assert_close(expected_mass, calculated_mass, rtol=1e-6, atol=1e-7)
    
    def test_jet_pt_calculation(self):
        """Test jet pT calculation"""
        jet_momentum = torch.tensor([
            [10.0, 6.0, 8.0, 0.0],
            [15.0, 9.0, 12.0, 0.0]
        ])
        
        # Calculate pT: pt = √(px² + py²)
        expected_pt = torch.sqrt(jet_momentum[:, 1]**2 + jet_momentum[:, 2]**2)
        
        # Use utility function
        calculated_pt = get_pt(jet_momentum)
        
        torch.testing.assert_close(expected_pt, calculated_pt, rtol=1e-6, atol=1e-7)


class TestCoordinateSystemPhysics:
    """Test physics properties specific to coordinate systems"""
    
    def test_cylindrical_coordinate_ranges(self):
        """Test that cylindrical coordinates have physical ranges"""
        # Create sample fourmomenta
        fourmomenta = torch.randn(10, 4)
        fourmomenta[:, 0] = torch.abs(fourmomenta[:, 0]) + 2.0
        
        # Ensure mass-shell constraint
        p_squared = (fourmomenta[:, 1:] ** 2).sum(dim=-1)
        E_squared = fourmomenta[:, 0] ** 2
        mask = E_squared < p_squared
        fourmomenta[mask, 0] = torch.sqrt(p_squared[mask]) + 0.1
        
        transform = FourmomentaToPtPhiEtaM2()
        cylindrical = transform.forward(fourmomenta)
        
        pt = cylindrical[:, 0]
        phi = cylindrical[:, 1]
        eta = cylindrical[:, 2]
        m2 = cylindrical[:, 3]
        
        # Check ranges
        assert torch.all(pt >= 0), "pT should be non-negative"
        assert torch.all(phi >= -np.pi), "φ should be >= -π"
        assert torch.all(phi <= np.pi), "φ should be <= π"
        assert torch.all(torch.isfinite(eta)), "η should be finite"
        assert torch.all(m2 >= -1e-6), "m² should be non-negative (within tolerance)"
    
    def test_log_coordinates_monotonicity(self):
        """Test that log transforms preserve ordering"""
        # Create particles with different pT values
        fourmomenta = torch.zeros(3, 4)
        fourmomenta[:, 0] = torch.tensor([5.0, 4.0, 3.0])  # Decreasing energy
        fourmomenta[:, 1] = torch.tensor([3.0, 2.0, 1.0])  # Decreasing px
        fourmomenta[:, 2] = torch.tensor([1.0, 1.0, 1.0])  # Same py
        fourmomenta[:, 3] = torch.tensor([0.0, 0.0, 0.0])  # Zero pz
        
        transform = FourmomentaToLogPtPhiEtaLogM2(pt_min=0.1, mass_scale=1.0)
        log_coords = transform.forward(fourmomenta)
        
        log_pt = log_coords[:, 0]
        
        # Log pT should preserve ordering (higher pT → higher log pT)
        original_pt = torch.sqrt(fourmomenta[:, 1]**2 + fourmomenta[:, 2]**2)
        
        # Check that ordering is preserved
        for i in range(len(original_pt) - 1):
            if original_pt[i] > original_pt[i+1]:
                assert log_pt[i] > log_pt[i+1], "Log transform should preserve pT ordering"


class TestNumericalStability:
    """Test numerical stability of physics calculations"""
    
    def test_nearly_massless_particles(self):
        """Test handling of nearly massless particles"""
        # Create particles very close to mass shell
        fourmomenta = torch.tensor([
            [10.0, 6.0, 8.0, 0.0],     # Exactly massless
            [10.001, 6.0, 8.0, 0.0]   # Tiny mass
        ])
        
        transform = FourmomentaToPtPhiEtaM2()
        cylindrical = transform.forward(fourmomenta)
        reconstructed = transform.inverse(cylindrical)
        
        # Should handle without numerical issues
        torch.testing.assert_close(fourmomenta, reconstructed, rtol=1e-4, atol=1e-5)
    
    def test_extreme_rapidity_particles(self):
        """Test particles with extreme rapidity (very forward/backward)"""
        fourmomenta = torch.tensor([
            [100.0, 1.0, 1.0, 99.9],    # Very forward (large +η)
            [100.0, 1.0, 1.0, -99.9]   # Very backward (large -η)
        ])
        
        # Calculate η manually
        pt = torch.sqrt(fourmomenta[:, 1]**2 + fourmomenta[:, 2]**2)
        pz = fourmomenta[:, 3]
        p = torch.sqrt(pt**2 + pz**2)
        
        eta = 0.5 * torch.log((p + pz) / (p - pz + 1e-8))
        
        # η should be finite and have expected signs
        assert torch.all(torch.isfinite(eta)), "η should be finite for extreme particles"
        assert eta[0] > 0, "Forward particle should have positive η"
        assert eta[1] < 0, "Backward particle should have negative η"
    
    def test_conservation_with_rounding_errors(self):
        """Test that small numerical errors don't break conservation"""
        # Create fourmomenta with small rounding errors
        fourmomenta = torch.tensor([
            [5.0000001, 3.0, 4.0, 0.0],
            [4.9999999, 2.0, 3.0, 1.0]
        ])
        
        total_before = fourmomenta.sum(dim=0)
        
        transform = FourmomentaToPtPhiEtaM2()
        cylindrical = transform.forward(fourmomenta)
        reconstructed = transform.inverse(cylindrical)
        
        total_after = reconstructed.sum(dim=0)
        
        # Should conserve four-momentum despite small errors
        torch.testing.assert_close(
            total_before, total_after,
            rtol=1e-5, atol=1e-6
        )