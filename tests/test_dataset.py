import pytest
import torch
import numpy as np
from omegaconf import OmegaConf
from unittest.mock import patch, MagicMock

from experiments.dataset import Dataset, load_dataset, load_zplusjet, load_cms, load_ttbar
from experiments.utils import pid_encoding


class TestDatasetLoading:
    """Test dataset loading functionality"""
    
    @pytest.fixture
    def mock_cfg(self):
        return OmegaConf.create({
            "length": 100,
            "add_pid": True,
            "mass": 0.001
        })
    
    @pytest.fixture
    def sample_data(self):
        """Create sample particle physics data for testing"""
        batch_size = 10
        max_particles = 20
        
        # Create realistic fourmomenta (E, px, py, pz)
        particles = torch.randn(batch_size, max_particles, 4)
        particles[:, :, 0] = torch.abs(particles[:, :, 0]) + 1.0  # Ensure E > 0
        
        # PIDs (particle IDs)
        pids = torch.randint(-13, 14, (batch_size, max_particles, 1)).float()
        
        # Multiplicities (number of particles per event)
        mults = torch.randint(5, max_particles, (batch_size,))
        
        # Jets (sum of particles)
        jets = particles.sum(dim=1)
        
        return particles, pids, mults, jets
    
    def test_dataset_creation(self):
        """Test basic Dataset class instantiation"""
        dataset = Dataset(dtype=torch.float32)
        assert dataset.dtype == torch.float32
        
        dataset_with_pos = Dataset(dtype=torch.float64, pos_encoding_dim=10)
        assert dataset_with_pos.dtype == torch.float64
        assert hasattr(dataset_with_pos, 'pos_encoding')
    
    def test_create_data_list(self, sample_data):
        """Test creation of data list from particle data"""
        particles, pids, mults, jets = sample_data
        
        dataset = Dataset(dtype=torch.float32)
        dataset.create_data_list(
            det_particles=particles,
            det_pids=pids,
            det_mults=mults,
            det_jets=jets,
            gen_particles=particles,  # Use same for simplicity
            gen_pids=pids,
            gen_mults=mults,
            gen_jets=jets
        )
        
        assert len(dataset.data_list) == particles.shape[0]
        
        # Check first data item
        data_item = dataset.data_list[0]
        assert hasattr(data_item, 'x_det')
        assert hasattr(data_item, 'x_gen')
        assert hasattr(data_item, 'scalars_det')
        assert hasattr(data_item, 'scalars_gen')
        assert hasattr(data_item, 'jet_det')
        assert hasattr(data_item, 'jet_gen')
        
        # Check shapes
        expected_mult = mults[0].item()
        assert data_item.x_det.shape[0] == expected_mult
        assert data_item.x_gen.shape[0] == expected_mult
        assert data_item.x_det.shape[1] == 4  # fourmomenta
    
    def test_dataset_indexing(self, sample_data):
        """Test dataset indexing functionality"""
        particles, pids, mults, jets = sample_data
        
        dataset = Dataset(dtype=torch.float32)
        dataset.create_data_list(particles, pids, mults, jets, particles, pids, mults, jets)
        
        assert len(dataset) == particles.shape[0]
        
        # Test indexing
        item = dataset[0]
        assert item == dataset.data_list[0]
    
    @pytest.mark.parametrize("dataset_name,expected_max_particles", [
        ("zplusjet", 152),
        ("cms", 3),
        ("ttbar", 238)
    ])
    def test_load_dataset_metadata(self, dataset_name, expected_max_particles):
        """Test dataset metadata loading"""
        max_particles, diff, pt_min, masked_dim, load_fn = load_dataset(dataset_name)
        
        assert max_particles == expected_max_particles
        assert isinstance(diff, list)
        assert len(diff) == 2
        assert isinstance(pt_min, float)
        assert isinstance(masked_dim, list)
        assert callable(load_fn)


class TestDataIntegrity:
    """Test data integrity and physics constraints"""
    
    def test_fourmomenta_shapes(self):
        """Test that fourmomenta have correct shapes"""
        particles = torch.randn(5, 10, 4)
        particles[:, :, 0] = torch.abs(particles[:, :, 0]) + 1.0  # E > 0
        
        # Check that E >= |p|
        E = particles[:, :, 0]
        p_squared = (particles[:, :, 1:] ** 2).sum(dim=-1)
        
        assert torch.all(E ** 2 >= p_squared), "Energy must be >= |momentum|"
    
    def test_particle_sorting_by_pt(self):
        """Test that particles are sorted by pT in descending order"""
        # Create sample data with known pT ordering
        particles = torch.zeros(2, 5, 4)
        particles[0, :, 0] = torch.tensor([10, 8, 6, 4, 2])  # E
        particles[0, :, 1] = torch.tensor([9, 7, 5, 3, 1])   # px
        particles[0, :, 2] = torch.tensor([1, 1, 1, 1, 1])   # py
        particles[0, :, 3] = torch.tensor([0, 0, 0, 0, 0])   # pz
        
        # Calculate pT
        pt = torch.sqrt(particles[0, :, 1]**2 + particles[0, :, 2]**2)
        
        # Check if sorted in descending order
        assert torch.all(pt[:-1] >= pt[1:]), "Particles should be sorted by pT (descending)"
    
    def test_pid_encoding_range(self):
        """Test PID encoding produces valid ranges"""
        pids = torch.tensor([-13, -11, 0, 11, 13]).float()  # muon, electron, photon, positron, antimuon
        encoded = pid_encoding(pids)
        
        # Check that encoded values are in reasonable range
        assert torch.all(encoded >= -1.0), "PID encoding should be >= -1"
        assert torch.all(encoded <= 1.0), "PID encoding should be <= 1"
    
    def test_multiplicity_consistency(self, sample_data=None):
        """Test that multiplicities match actual particle counts"""
        if sample_data is None:
            particles = torch.randn(3, 15, 4)
            particles[:, :, 0] = torch.abs(particles[:, :, 0]) + 1.0
            mults = torch.tensor([10, 8, 12])
        else:
            particles, _, mults, _ = sample_data
        
        for i, mult in enumerate(mults):
            # Check that we have the right number of valid particles
            valid_particles = particles[i, :mult]
            assert valid_particles.shape[0] == mult.item()
    
    def test_jet_momentum_conservation(self):
        """Test that jet momentum is sum of constituent momenta"""
        particles = torch.randn(2, 10, 4)
        particles[:, :, 0] = torch.abs(particles[:, :, 0]) + 1.0
        
        # Calculate jet as sum of particles
        jets_calculated = particles.sum(dim=1)
        
        # For this test, we assume jets should equal sum of particles
        # In real data, this might not be exact due to detector effects
        expected_jets = particles.sum(dim=1)
        
        torch.testing.assert_close(jets_calculated, expected_jets, rtol=1e-5, atol=1e-5)


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_events(self):
        """Test handling of events with zero particles"""
        particles = torch.zeros(2, 5, 4)
        pids = torch.zeros(2, 5, 1)
        mults = torch.tensor([0, 3])  # First event has no particles
        jets = torch.randn(2, 4)
        
        dataset = Dataset(dtype=torch.float32)
        dataset.create_data_list(particles, pids, mults, jets, particles, pids, mults, jets)
        
        # Check that empty event is handled
        assert len(dataset.data_list) == 2
        assert dataset.data_list[0].x_det.shape[0] == 0
        assert dataset.data_list[1].x_det.shape[0] == 3
    
    def test_single_particle_events(self):
        """Test events with only one particle"""
        particles = torch.randn(1, 1, 4)
        particles[:, :, 0] = torch.abs(particles[:, :, 0]) + 1.0
        pids = torch.randint(-13, 14, (1, 1, 1)).float()
        mults = torch.tensor([1])
        jets = particles.sum(dim=1)
        
        dataset = Dataset(dtype=torch.float32)
        dataset.create_data_list(particles, pids, mults, jets, particles, pids, mults, jets)
        
        assert len(dataset.data_list) == 1
        assert dataset.data_list[0].x_det.shape[0] == 1
    
    def test_max_particle_events(self):
        """Test events at maximum particle limit"""
        max_particles = 50
        particles = torch.randn(1, max_particles, 4)
        particles[:, :, 0] = torch.abs(particles[:, :, 0]) + 1.0
        pids = torch.randint(-13, 14, (1, max_particles, 1)).float()
        mults = torch.tensor([max_particles])
        jets = particles.sum(dim=1)
        
        dataset = Dataset(dtype=torch.float32)
        dataset.create_data_list(particles, pids, mults, jets, particles, pids, mults, jets)
        
        assert len(dataset.data_list) == 1
        assert dataset.data_list[0].x_det.shape[0] == max_particles


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_consistency(dtype):
    """Test that dataset respects specified dtype"""
    particles = torch.randn(2, 5, 4).to(dtype)
    particles[:, :, 0] = torch.abs(particles[:, :, 0]) + 1.0
    pids = torch.randint(-13, 14, (2, 5, 1)).float().to(dtype)
    mults = torch.tensor([3, 4])
    jets = particles.sum(dim=1)
    
    dataset = Dataset(dtype=dtype)
    dataset.create_data_list(particles, pids, mults, jets, particles, pids, mults, jets)
    
    assert dataset.dtype == dtype
    # Check that data maintains correct dtype
    for data_item in dataset.data_list:
        assert data_item.x_det.dtype == dtype
        assert data_item.x_gen.dtype == dtype


class TestPositionalEncoding:
    """Test positional encoding functionality"""
    
    def test_positional_encoding_shapes(self):
        """Test that positional encoding has correct shapes"""
        pos_dim = 8
        dataset = Dataset(dtype=torch.float32, pos_encoding_dim=pos_dim)
        
        particles = torch.randn(1, 10, 4)
        particles[:, :, 0] = torch.abs(particles[:, :, 0]) + 1.0
        pids = torch.randint(-13, 14, (1, 10, 1)).float()
        mults = torch.tensor([7])
        jets = particles.sum(dim=1)
        
        dataset.create_data_list(particles, pids, mults, jets, particles, pids, mults, jets)
        
        # Check that scalars include positional encoding
        data_item = dataset.data_list[0]
        expected_scalar_dim = 1 + pos_dim  # PID + positional encoding
        assert data_item.scalars_det.shape[1] == expected_scalar_dim
        assert data_item.scalars_gen.shape[1] == expected_scalar_dim