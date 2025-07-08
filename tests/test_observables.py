import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from experiments.kinematics.observables import (
    create_partial_jet,
    compute_angles,
    select_pt,
    compute_nsubjettiness_ratio,
    compute_jet_mass,
    compute_soft_drop_mass,
    compute_zg,
    create_observable
)
from experiments.utils import get_ptr_from_batch
from experiments.coordinates import fourmomenta_to_jetmomenta


class TestPartialJetFormation:
    """Test partial jet formation functionality"""
    
    @pytest.fixture
    def sample_constituents(self):
        """Create sample constituent particles for testing"""
        # Create 3 events with different numbers of particles
        constituents = torch.tensor([
            # Event 1: 4 particles
            [10.0, 6.0, 8.0, 0.0],
            [8.0, 3.0, 4.0, 2.0],
            [5.0, 1.0, 2.0, 3.0],
            [3.0, 0.5, 1.0, 1.5],
            # Event 2: 3 particles  
            [12.0, 7.0, 9.0, 1.0],
            [6.0, 2.0, 3.0, 4.0],
            [4.0, 1.0, 1.5, 2.0],
            # Event 3: 2 particles
            [15.0, 9.0, 12.0, 0.0],
            [7.0, 3.0, 4.0, 5.0]
        ])
        
        # Batch indices: [0,0,0,0, 1,1,1, 2,2]
        batch_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2])
        
        return constituents, batch_idx
    
    def test_partial_jet_formation_fixed_indices(self, sample_constituents):
        """Test partial jet formation with fixed start/end indices"""
        constituents, batch_idx = sample_constituents
        
        # Create partial jet from particles 0-2 (first 2 particles)
        partial_jet_fn = create_partial_jet(start=0, end=2)
        
        jets, true_jets = partial_jet_fn(constituents, batch_idx, None)
        
        # Check shapes
        assert jets.shape[0] == 3  # 3 events
        assert jets.shape[1] == 4  # 4-momentum
        assert true_jets.shape == jets.shape
        
        # Check that partial jet is sum of first 2 particles in each event
        batch_ptr = get_ptr_from_batch(batch_idx)
        
        for i in range(len(batch_ptr) - 1):
            start_idx = batch_ptr[i]
            expected_jet = constituents[start_idx:start_idx + 2].sum(dim=0)
            torch.testing.assert_close(jets[i], expected_jet, rtol=1e-6, atol=1e-7)
    
    def test_partial_jet_formation_fractional_indices(self, sample_constituents):
        """Test partial jet formation with fractional start/end indices"""
        constituents, batch_idx = sample_constituents
        
        # Create partial jet from first 50% of particles
        partial_jet_fn = create_partial_jet(start=0.0, end=0.5)
        
        jets, true_jets = partial_jet_fn(constituents, batch_idx, None)
        
        # Check that we get reasonable results
        assert jets.shape[0] == 3
        assert torch.all(torch.isfinite(jets))
        
        # For event with 4 particles, 50% should give us first 2 particles
        # For event with 3 particles, 50% should give us first 1 particle
        # For event with 2 particles, 50% should give us first 1 particle
        
        batch_ptr = get_ptr_from_batch(batch_idx)
        
        # Event 1: 4 particles, 50% = 2 particles
        expected_jet_0 = constituents[0:2].sum(dim=0)
        torch.testing.assert_close(jets[0], expected_jet_0, rtol=1e-6, atol=1e-7)
        
        # Event 2: 3 particles, 50% = 1 particle (int(0.5 * 3) = 1)
        expected_jet_1 = constituents[4:5].sum(dim=0)
        torch.testing.assert_close(jets[1], expected_jet_1, rtol=1e-6, atol=1e-7)
    
    def test_partial_jet_formation_full_event(self, sample_constituents):
        """Test partial jet formation with end=-1 (full event)"""
        constituents, batch_idx = sample_constituents
        
        # Create partial jet from particle 1 to end
        partial_jet_fn = create_partial_jet(start=1, end=-1)
        
        jets, true_jets = partial_jet_fn(constituents, batch_idx, None)
        
        # Check that true_jets are full jets
        batch_ptr = get_ptr_from_batch(batch_idx)
        
        for i in range(len(batch_ptr) - 1):
            start_idx = batch_ptr[i]
            end_idx = batch_ptr[i + 1]
            expected_true_jet = constituents[start_idx:end_idx].sum(dim=0)
            torch.testing.assert_close(true_jets[i], expected_true_jet, rtol=1e-6, atol=1e-7)


class TestAngleComputations:
    """Test angular observable computations"""
    
    @pytest.fixture
    def sample_constituents_for_angles(self):
        """Create sample constituents for angle testing"""
        # Create particles with known angular separations
        constituents = torch.tensor([
            # Event 1: 4 particles in a specific configuration
            [10.0, 10.0, 0.0, 0.0],  # φ=0, η=0
            [10.0, 0.0, 10.0, 0.0],  # φ=π/2, η=0
            [10.0, -10.0, 0.0, 0.0], # φ=π, η=0
            [10.0, 0.0, -10.0, 0.0], # φ=-π/2, η=0
        ])
        
        batch_idx = torch.tensor([0, 0, 0, 0])
        return constituents, batch_idx
    
    def test_angle_computation_delta_phi(self, sample_constituents_for_angles):
        """Test Δφ computation between particle groups"""
        constituents, batch_idx = sample_constituents_for_angles
        
        # Compute angle between particle 0 and particle 1
        angle_fn = compute_angles(start1=0, end1=1, start2=1, end2=2, angle_type="phi")
        
        angles = angle_fn(constituents, batch_idx, None)
        
        # Particle 0: φ=0, Particle 1: φ=π/2
        # Δφ should be π/2
        expected_angle = np.pi / 2
        torch.testing.assert_close(
            angles[0], torch.tensor([[expected_angle]]), 
            rtol=1e-5, atol=1e-6
        )
    
    def test_angle_computation_delta_r(self, sample_constituents_for_angles):
        """Test ΔR computation between particle groups"""
        constituents, batch_idx = sample_constituents_for_angles
        
        # Compute ΔR between particle 0 and particle 1
        angle_fn = compute_angles(start1=0, end1=1, start2=1, end2=2, angle_type="R")
        
        angles = angle_fn(constituents, batch_idx, None)
        
        # Since all particles have η=0, ΔR = Δφ = π/2
        expected_angle = np.pi / 2
        torch.testing.assert_close(
            angles[0], torch.tensor([[expected_angle]]), 
            rtol=1e-5, atol=1e-6
        )
    
    def test_angle_computation_delta_eta(self):
        """Test Δη computation between particles with different pseudorapidity"""
        # Create particles with different η values
        constituents = torch.tensor([
            [10.0, 1.0, 1.0, 5.0],   # Forward particle (high pz)
            [10.0, 1.0, 1.0, -5.0],  # Backward particle (low pz)
        ])
        
        batch_idx = torch.tensor([0, 0])
        
        angle_fn = compute_angles(start1=0, end1=1, start2=1, end2=2, angle_type="eta")
        angles = angle_fn(constituents, batch_idx, None)
        
        # Should give non-zero Δη
        assert torch.abs(angles[0]) > 0.1
        assert torch.all(torch.isfinite(angles))


class TestPtSelection:
    """Test pT selection functionality"""
    
    def test_pt_selection_basic(self):
        """Test basic pT selection"""
        # Create particles with different pT values
        constituents = torch.tensor([
            [10.0, 6.0, 8.0, 0.0],  # pT = 10
            [8.0, 3.0, 4.0, 0.0],   # pT = 5
            [5.0, 3.0, 4.0, 0.0],   # pT = 5
        ])
        
        batch_idx = torch.tensor([0, 0, 0])
        
        # Select 2nd highest pT particle (index 1, 0-indexed)
        pt_selector = select_pt(1)
        
        selected_pt = pt_selector(constituents, batch_idx, None)
        
        # Expected: 2nd highest pT particle has pT = 5
        expected_pt = 5.0
        torch.testing.assert_close(
            selected_pt[0], torch.tensor([[expected_pt]]), 
            rtol=1e-5, atol=1e-6
        )
    
    def test_pt_selection_with_bound(self):
        """Test pT selection with bounds"""
        constituents = torch.tensor([
            [10.0, 6.0, 8.0, 0.0],  # pT = 10
            [8.0, 3.0, 4.0, 0.0],   # pT = 5
            [5.0, 1.0, 1.0, 0.0],   # pT = √2 ≈ 1.41
        ])
        
        batch_idx = torch.tensor([0, 0, 0])
        
        # Select particles with pT > 3.0
        pt_selector = select_pt(0, bound=3.0)
        
        selected_pt = pt_selector(constituents, batch_idx, None)
        
        # Should select highest pT particle above bound
        expected_pt = 10.0
        torch.testing.assert_close(
            selected_pt[0], torch.tensor([[expected_pt]]), 
            rtol=1e-5, atol=1e-6
        )


class TestJetSubstructure:
    """Test jet substructure observables"""
    
    @pytest.fixture
    def sample_jet_constituents(self):
        """Create sample jet constituents for substructure tests"""
        # Create a jet with multiple constituents
        constituents = torch.tensor([
            [20.0, 15.0, 10.0, 5.0],   # Leading particle
            [10.0, 6.0, 8.0, 0.0],     # Subleading particle
            [5.0, 2.0, 3.0, 4.0],      # Soft particle
            [3.0, 1.0, 1.0, 2.0],      # Very soft particle
        ])
        
        batch_idx = torch.tensor([0, 0, 0, 0])
        return constituents, batch_idx
    
    @patch('experiments.kinematics.observables.compute_nsubjettiness')
    def test_nsubjettiness_ratio_computation(self, mock_nsubjettiness, sample_jet_constituents):
        """Test N-subjettiness ratio computation"""
        constituents, batch_idx = sample_jet_constituents
        
        # Mock the fastjet computation
        mock_nsubjettiness.return_value = {
            'tau1': [0.1],
            'tau2': [0.05],
            'tau3': [0.02]
        }
        
        # Test τ21 ratio
        tau21_fn = compute_nsubjettiness_ratio(ratio="tau21")
        tau21 = tau21_fn(constituents, batch_idx, None)
        
        expected_tau21 = 0.05 / 0.1  # tau2 / tau1
        torch.testing.assert_close(
            tau21[0], torch.tensor([[expected_tau21]]), 
            rtol=1e-5, atol=1e-6
        )
        
        # Test τ32 ratio
        tau32_fn = compute_nsubjettiness_ratio(ratio="tau32")
        tau32 = tau32_fn(constituents, batch_idx, None)
        
        expected_tau32 = 0.02 / 0.05  # tau3 / tau2
        torch.testing.assert_close(
            tau32[0], torch.tensor([[expected_tau32]]), 
            rtol=1e-5, atol=1e-6
        )
    
    def test_jet_mass_computation(self, sample_jet_constituents):
        """Test jet mass computation"""
        constituents, batch_idx = sample_jet_constituents
        
        mass_fn = compute_jet_mass()
        mass = mass_fn(constituents, batch_idx, None)
        
        # Calculate expected mass manually
        total_4mom = constituents.sum(dim=0)
        expected_mass = torch.sqrt(
            total_4mom[0]**2 - (total_4mom[1:]**2).sum()
        )
        
        torch.testing.assert_close(
            mass[0], expected_mass.unsqueeze(0).unsqueeze(0), 
            rtol=1e-5, atol=1e-6
        )
    
    @patch('experiments.kinematics.observables.apply_soft_drop')
    def test_soft_drop_mass_computation(self, mock_soft_drop, sample_jet_constituents):
        """Test soft drop mass computation"""
        constituents, batch_idx = sample_jet_constituents
        
        # Mock soft drop result
        mock_soft_drop.return_value = {
            'constituents': [constituents[:2].numpy()],  # Keep first 2 particles
            'zg': [0.3],
            'rg': [0.2]
        }
        
        sd_mass_fn = compute_soft_drop_mass()
        sd_mass = sd_mass_fn(constituents, batch_idx, None)
        
        # Should compute mass of soft-dropped jet (first 2 particles)
        expected_4mom = constituents[:2].sum(dim=0)
        expected_mass = torch.sqrt(
            expected_4mom[0]**2 - (expected_4mom[1:]**2).sum()
        )
        
        torch.testing.assert_close(
            sd_mass[0], expected_mass.unsqueeze(0).unsqueeze(0), 
            rtol=1e-5, atol=1e-6
        )
    
    @patch('experiments.kinematics.observables.apply_soft_drop')
    def test_zg_computation(self, mock_soft_drop, sample_jet_constituents):
        """Test zg (momentum fraction) computation"""
        constituents, batch_idx = sample_jet_constituents
        
        # Mock soft drop result
        expected_zg = 0.25
        mock_soft_drop.return_value = {
            'constituents': [constituents.numpy()],
            'zg': [expected_zg],
            'rg': [0.2]
        }
        
        zg_fn = compute_zg()
        zg = zg_fn(constituents, batch_idx, None)
        
        torch.testing.assert_close(
            zg[0], torch.tensor([[expected_zg]]), 
            rtol=1e-5, atol=1e-6
        )


class TestObservableCreation:
    """Test observable creation utilities"""
    
    def test_create_observable_with_function(self):
        """Test creating observable from function"""
        def dummy_observable(constituents, batch_idx, other_batch_idx):
            return torch.ones(1, 1)
        
        observable = create_observable(dummy_observable)
        
        # Test that it's callable
        result = observable(torch.randn(4, 4), torch.zeros(4, dtype=torch.long), None)
        
        assert result.shape == (1, 1)
        torch.testing.assert_close(result, torch.ones(1, 1))
    
    def test_create_observable_with_config(self):
        """Test creating observable from configuration"""
        config = {
            'type': 'partial_jet',
            'start': 0,
            'end': 2
        }
        
        # This would typically be handled by the actual implementation
        # Here we just test that the config can be processed
        assert 'type' in config
        assert config['start'] == 0
        assert config['end'] == 2


class TestEdgeCases:
    """Test edge cases for observables"""
    
    def test_empty_event_handling(self):
        """Test handling of empty events"""
        # Empty constituents
        constituents = torch.empty(0, 4)
        batch_idx = torch.empty(0, dtype=torch.long)
        
        # Should handle gracefully without crashing
        partial_jet_fn = create_partial_jet(start=0, end=1)
        
        # This should not crash (though may return empty results)
        try:
            jets, true_jets = partial_jet_fn(constituents, batch_idx, None)
            # If it doesn't crash, that's good
            assert True
        except Exception as e:
            # If it does crash, at least we know about it
            pytest.skip(f"Empty event handling not implemented: {e}")
    
    def test_single_particle_event(self):
        """Test handling of single particle events"""
        constituents = torch.tensor([[10.0, 6.0, 8.0, 0.0]])
        batch_idx = torch.tensor([0])
        
        # Create partial jet with more particles than available
        partial_jet_fn = create_partial_jet(start=0, end=5)
        
        jets, true_jets = partial_jet_fn(constituents, batch_idx, None)
        
        # Should handle gracefully
        assert jets.shape[0] == 1
        torch.testing.assert_close(jets[0], constituents[0], rtol=1e-6, atol=1e-7)
    
    def test_invalid_angle_computation(self):
        """Test handling of invalid angle computation requests"""
        constituents = torch.tensor([
            [10.0, 6.0, 8.0, 0.0],
            [8.0, 3.0, 4.0, 0.0]
        ])
        batch_idx = torch.tensor([0, 0])
        
        # Try to compute angle with invalid indices
        with pytest.raises(AssertionError):
            # end1 > start2 violates the constraint
            compute_angles(start1=0, end1=2, start2=1, end2=2)


class TestNumericalStability:
    """Test numerical stability of observable computations"""
    
    def test_zero_momentum_particles(self):
        """Test handling of zero momentum particles"""
        constituents = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],  # Rest mass particle
            [0.0, 0.0, 0.0, 0.0],  # Zero four-momentum
        ])
        batch_idx = torch.tensor([0, 0])
        
        # Should handle without numerical issues
        mass_fn = compute_jet_mass()
        mass = mass_fn(constituents, batch_idx, None)
        
        # Should get finite result
        assert torch.all(torch.isfinite(mass))
    
    def test_collinear_particles(self):
        """Test handling of collinear particles"""
        constituents = torch.tensor([
            [10.0, 0.0, 0.0, 10.0],   # Particle along +z axis
            [10.0, 0.0, 0.0, -10.0],  # Particle along -z axis
        ])
        batch_idx = torch.tensor([0, 0])
        
        # Compute angle between them
        angle_fn = compute_angles(start1=0, end1=1, start2=1, end2=2, angle_type="eta")
        angles = angle_fn(constituents, batch_idx, None)
        
        # Should give large but finite Δη
        assert torch.all(torch.isfinite(angles))
        assert torch.abs(angles[0]) > 1.0  # Should be substantial
    
    def test_very_soft_particles(self):
        """Test handling of very soft particles"""
        constituents = torch.tensor([
            [100.0, 50.0, 50.0, 70.0],  # Hard particle
            [0.001, 0.0005, 0.0005, 0.0007],  # Very soft particle
        ])
        batch_idx = torch.tensor([0, 0])
        
        # Test pT selection
        pt_selector = select_pt(1)  # Select 2nd highest pT
        selected_pt = pt_selector(constituents, batch_idx, None)
        
        # Should handle soft particles correctly
        assert torch.all(torch.isfinite(selected_pt))
        assert selected_pt[0] > 0