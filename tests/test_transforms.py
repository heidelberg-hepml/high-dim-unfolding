import pytest
import torch
import numpy as np
from unittest.mock import patch

from experiments.transforms import (
    BaseTransform,
    EmptyTransform,
    FourmomentaToPtPhiEtaM2,
    FourmomentaToLogPtPhiEtaLogM2,
    StandardNormal,
    IndividualNormal,
    LogTransform,
    compose_transforms
)
from experiments.utils import EPS1, EPS2, CUTOFF


class TestBaseTransform:
    """Test BaseTransform abstract class functionality"""
    
    def test_base_transform_abstract_methods(self):
        """Test that BaseTransform cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BaseTransform()
    
    def test_finite_check_assertions(self):
        """Test that transforms check for finite values"""
        
        class TestTransform(BaseTransform):
            def _forward(self, x, **kwargs):
                return torch.tensor([float('inf')])
            
            def _inverse(self, x, **kwargs):
                return x
            
            def _jac_forward(self, x, y, **kwargs):
                return torch.eye(x.shape[-1])
            
            def _jac_inverse(self, y, x, **kwargs):
                return torch.eye(x.shape[-1])
            
            def _detjac_forward(self, x, y, **kwargs):
                return torch.ones(x.shape[:-1])
        
        transform = TestTransform()
        x = torch.tensor([1.0, 2.0])
        
        with pytest.raises(AssertionError):
            transform.forward(x)


class TestEmptyTransform:
    """Test EmptyTransform (identity transform)"""
    
    def test_empty_transform_identity(self):
        """Test that EmptyTransform is identity"""
        transform = EmptyTransform()
        x = torch.randn(10, 4)
        
        assert torch.equal(transform.forward(x), x)
        assert torch.equal(transform.inverse(x), x)
    
    def test_empty_transform_velocity(self):
        """Test EmptyTransform velocity operations"""
        transform = EmptyTransform()
        x = torch.randn(5, 4)
        v = torch.randn(5, 4)
        
        v_forward = transform.velocity_forward(v, x, x)
        v_inverse = transform.velocity_inverse(v, x, x)
        
        assert torch.equal(v_forward, v)
        assert torch.equal(v_inverse, v)
    
    def test_empty_transform_jacobian(self):
        """Test EmptyTransform Jacobian operations"""
        transform = EmptyTransform()
        x = torch.randn(3, 4)
        
        logdet_forward = transform.logdetjac_forward(x, x)
        logdet_inverse = transform.logdetjac_inverse(x, x)
        
        assert torch.allclose(logdet_forward, torch.zeros_like(logdet_forward))
        assert torch.allclose(logdet_inverse, torch.zeros_like(logdet_inverse))


class TestFourmomentaTransforms:
    """Test coordinate transformations for fourmomenta"""
    
    @pytest.fixture
    def sample_fourmomenta(self):
        """Create sample fourmomenta data"""
        # Create realistic particle data
        batch_size, n_particles = 5, 10
        
        # Generate fourmomenta (E, px, py, pz)
        fourmomenta = torch.randn(batch_size, n_particles, 4)
        fourmomenta[:, :, 0] = torch.abs(fourmomenta[:, :, 0]) + 2.0  # E > 0
        fourmomenta[:, :, 3] = fourmomenta[:, :, 3] * 0.5  # Reduce pz
        
        # Ensure physical constraint: E^2 >= p^2
        p_squared = (fourmomenta[:, :, 1:] ** 2).sum(dim=-1)
        E_squared = fourmomenta[:, :, 0] ** 2
        
        # Adjust E if needed
        mask = E_squared < p_squared
        fourmomenta[mask, 0] = torch.sqrt(p_squared[mask]) + 0.1
        
        return fourmomenta
    
    def test_fourmomenta_to_cylindrical(self, sample_fourmomenta):
        """Test fourmomenta to (pt, phi, eta, m²) transformation"""
        transform = FourmomentaToPtPhiEtaM2()
        
        # Forward transform
        cylindrical = transform.forward(sample_fourmomenta)
        
        # Check shapes
        assert cylindrical.shape == sample_fourmomenta.shape
        
        # Check pt > 0
        pt = cylindrical[:, :, 0]
        assert torch.all(pt >= 0), "pT should be non-negative"
        
        # Check phi in [-π, π]
        phi = cylindrical[:, :, 1]
        assert torch.all(phi >= -np.pi), "φ should be >= -π"
        assert torch.all(phi <= np.pi), "φ should be <= π"
        
        # Check eta finite
        eta = cylindrical[:, :, 2]
        assert torch.all(torch.isfinite(eta)), "η should be finite"
        
        # Check m² >= 0 for physical particles
        m2 = cylindrical[:, :, 3]
        assert torch.all(m2 >= -EPS1), "m² should be >= 0 (within tolerance)"
    
    def test_fourmomenta_cylindrical_invertibility(self, sample_fourmomenta):
        """Test invertibility of fourmomenta ↔ cylindrical transform"""
        transform = FourmomentaToPtPhiEtaM2()
        
        # Forward and inverse
        cylindrical = transform.forward(sample_fourmomenta)
        reconstructed = transform.inverse(cylindrical)
        
        # Check invertibility
        torch.testing.assert_close(
            sample_fourmomenta, reconstructed, 
            rtol=1e-4, atol=1e-5
        )
    
    def test_log_transform_properties(self, sample_fourmomenta):
        """Test logarithmic transform properties"""
        transform = FourmomentaToLogPtPhiEtaLogM2(pt_min=0.1, mass_scale=1.0)
        
        # Forward transform
        log_coords = transform.forward(sample_fourmomenta)
        
        # Check shapes
        assert log_coords.shape == sample_fourmomenta.shape
        
        # Check log(pt) values are reasonable
        log_pt = log_coords[:, :, 0]
        assert torch.all(torch.isfinite(log_pt)), "log(pT) should be finite"
        
        # Check phi unchanged
        phi_original = torch.atan2(sample_fourmomenta[:, :, 2], sample_fourmomenta[:, :, 1])
        phi_transformed = log_coords[:, :, 1]
        torch.testing.assert_close(
            phi_original, phi_transformed, 
            rtol=1e-4, atol=1e-5
        )
    
    def test_log_transform_invertibility(self, sample_fourmomenta):
        """Test invertibility of log transform"""
        transform = FourmomentaToLogPtPhiEtaLogM2(pt_min=0.1, mass_scale=1.0)
        
        # Filter out very low pT particles that might cause issues
        pt = torch.sqrt(sample_fourmomenta[:, :, 1]**2 + sample_fourmomenta[:, :, 2]**2)
        mask = pt > 0.2
        
        if mask.any():
            filtered_fourmomenta = sample_fourmomenta[mask]
            
            log_coords = transform.forward(filtered_fourmomenta)
            reconstructed = transform.inverse(log_coords)
            
            torch.testing.assert_close(
                filtered_fourmomenta, reconstructed,
                rtol=1e-3, atol=1e-4
            )


class TestNormalizationTransforms:
    """Test normalization transforms"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for normalization tests"""
        return torch.randn(100, 10, 4) * 5.0 + 2.0  # Mean ≈ 2, std ≈ 5
    
    def test_standard_normal_transform(self, sample_data):
        """Test StandardNormal transform"""
        transform = StandardNormal()
        
        # Fit transform
        transform.init_fit(sample_data)
        
        # Transform
        normalized = transform.forward(sample_data)
        
        # Check normalization (should have mean ≈ 0, std ≈ 1)
        mean = normalized.mean(dim=(0, 1))
        std = normalized.std(dim=(0, 1))
        
        torch.testing.assert_close(mean, torch.zeros_like(mean), rtol=0.1, atol=0.1)
        torch.testing.assert_close(std, torch.ones_like(std), rtol=0.1, atol=0.1)
    
    def test_standard_normal_invertibility(self, sample_data):
        """Test StandardNormal invertibility"""
        transform = StandardNormal()
        transform.init_fit(sample_data)
        
        normalized = transform.forward(sample_data)
        reconstructed = transform.inverse(normalized)
        
        torch.testing.assert_close(sample_data, reconstructed, rtol=1e-5, atol=1e-6)
    
    def test_individual_normal_transform(self, sample_data):
        """Test IndividualNormal transform (per-feature normalization)"""
        transform = IndividualNormal()
        transform.init_fit(sample_data)
        
        normalized = transform.forward(sample_data)
        
        # Check per-feature normalization
        for i in range(sample_data.shape[-1]):
            feature_mean = normalized[:, :, i].mean()
            feature_std = normalized[:, :, i].std()
            
            assert abs(feature_mean.item()) < 0.1, f"Feature {i} mean should be ~0"
            assert abs(feature_std.item() - 1.0) < 0.1, f"Feature {i} std should be ~1"
    
    def test_individual_normal_invertibility(self, sample_data):
        """Test IndividualNormal invertibility"""
        transform = IndividualNormal()
        transform.init_fit(sample_data)
        
        normalized = transform.forward(sample_data)
        reconstructed = transform.inverse(normalized)
        
        torch.testing.assert_close(sample_data, reconstructed, rtol=1e-5, atol=1e-6)


class TestLogTransform:
    """Test logarithmic transform"""
    
    def test_log_transform_basic(self):
        """Test basic log transform functionality"""
        transform = LogTransform()
        
        # Positive values
        x = torch.tensor([1.0, 2.0, 10.0, 100.0])
        log_x = transform.forward(x)
        reconstructed = transform.inverse(log_x)
        
        expected_log = torch.log(x)
        torch.testing.assert_close(log_x, expected_log, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(x, reconstructed, rtol=1e-5, atol=1e-6)
    
    def test_log_transform_with_offset(self):
        """Test log transform with offset for handling zeros/negatives"""
        offset = 1.0
        transform = LogTransform(offset=offset)
        
        x = torch.tensor([0.0, 1.0, 2.0])
        log_x = transform.forward(x)
        reconstructed = transform.inverse(log_x)
        
        expected_log = torch.log(x + offset)
        torch.testing.assert_close(log_x, expected_log, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(x, reconstructed, rtol=1e-5, atol=1e-6)
    
    def test_log_transform_jacobian(self):
        """Test log transform Jacobian computation"""
        transform = LogTransform()
        x = torch.tensor([1.0, 2.0, 5.0])
        x.requires_grad_(True)
        
        y = transform.forward(x)
        logdet = transform.logdetjac_forward(x, y)
        
        # For log transform, Jacobian determinant is 1/x, so log(det) = -log(x)
        expected_logdet = -torch.log(x).sum(dim=-1, keepdim=True)
        torch.testing.assert_close(logdet, expected_logdet, rtol=1e-5, atol=1e-6)


class TestTransformComposition:
    """Test composition of multiple transforms"""
    
    def test_compose_transforms(self):
        """Test composition of multiple transforms"""
        # Create a chain: log -> standardize
        transform1 = LogTransform(offset=1.0)
        transform2 = StandardNormal()
        
        # Sample data
        x = torch.abs(torch.randn(50, 4)) + 0.1  # Ensure positive
        
        # Fit the second transform on the output of the first
        y1 = transform1.forward(x)
        transform2.init_fit(y1)
        
        # Compose transforms
        composed = compose_transforms([transform1, transform2])
        
        # Apply composed transform
        y_composed = composed.forward(x)
        
        # Apply transforms separately
        y_separate = transform2.forward(transform1.forward(x))
        
        torch.testing.assert_close(y_composed, y_separate, rtol=1e-5, atol=1e-6)
    
    def test_composed_transform_invertibility(self):
        """Test invertibility of composed transforms"""
        transform1 = LogTransform(offset=1.0)
        transform2 = StandardNormal()
        
        x = torch.abs(torch.randn(20, 3)) + 0.1
        
        # Fit
        y1 = transform1.forward(x)
        transform2.init_fit(y1)
        
        # Compose
        composed = compose_transforms([transform1, transform2])
        
        # Test invertibility
        y = composed.forward(x)
        x_reconstructed = composed.inverse(y)
        
        torch.testing.assert_close(x, x_reconstructed, rtol=1e-4, atol=1e-5)


class TestEdgeCases:
    """Test edge cases and numerical stability"""
    
    def test_zero_momentum_handling(self):
        """Test handling of zero momentum particles"""
        # Create fourmomenta with zero momentum
        fourmomenta = torch.zeros(2, 3, 4)
        fourmomenta[:, :, 0] = 1.0  # Set energy to 1
        
        transform = FourmomentaToPtPhiEtaM2()
        
        # Should handle zero pT gracefully
        cylindrical = transform.forward(fourmomenta)
        
        # Check that pT is zero
        assert torch.allclose(cylindrical[:, :, 0], torch.zeros_like(cylindrical[:, :, 0]))
        
        # Check that phi is well-defined (could be any value for zero pT)
        assert torch.all(torch.isfinite(cylindrical[:, :, 1]))
    
    def test_collinear_particles(self):
        """Test handling of collinear particles (η → ±∞)"""
        fourmomenta = torch.zeros(1, 2, 4)
        
        # Particle 1: high pz
        fourmomenta[0, 0] = torch.tensor([10.0, 0.1, 0.1, 9.9])  # Very forward
        
        # Particle 2: high -pz  
        fourmomenta[0, 1] = torch.tensor([10.0, 0.1, 0.1, -9.9])  # Very backward
        
        transform = FourmomentaToPtPhiEtaM2()
        cylindrical = transform.forward(fourmomenta)
        
        # Check that η values are large but finite
        eta = cylindrical[:, :, 2]
        assert torch.all(torch.isfinite(eta)), "η should remain finite"
        assert torch.all(torch.abs(eta) > 3.0), "η should be large for collinear particles"
    
    def test_very_small_masses(self):
        """Test handling of very small/zero masses"""
        # Create nearly massless particles
        fourmomenta = torch.randn(3, 5, 4)
        fourmomenta[:, :, 0] = torch.sqrt((fourmomenta[:, :, 1:] ** 2).sum(dim=-1)) + 1e-6
        
        transform = FourmomentaToLogPtPhiEtaLogM2(mass_scale=1e-3)
        
        # Should handle small masses without overflow
        log_coords = transform.forward(fourmomenta)
        
        # Check that log(m²) doesn't explode
        log_m2 = log_coords[:, :, 3]
        assert torch.all(torch.isfinite(log_m2)), "log(m²) should be finite"
    
    def test_extreme_values(self):
        """Test with extreme coordinate values"""
        # Very high energy particles
        fourmomenta = torch.tensor([[[1000.0, 100.0, 100.0, 990.0]]])
        
        transform = FourmomentaToPtPhiEtaM2()
        cylindrical = transform.forward(fourmomenta)
        reconstructed = transform.inverse(cylindrical)
        
        # Should handle extreme values
        torch.testing.assert_close(fourmomenta, reconstructed, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("batch_size,n_particles", [(1, 1), (5, 10), (2, 100)])
def test_transforms_different_shapes(batch_size, n_particles):
    """Test transforms work with different tensor shapes"""
    fourmomenta = torch.randn(batch_size, n_particles, 4)
    fourmomenta[:, :, 0] = torch.abs(fourmomenta[:, :, 0]) + 1.0
    
    transform = FourmomentaToPtPhiEtaM2()
    
    cylindrical = transform.forward(fourmomenta)
    reconstructed = transform.inverse(cylindrical)
    
    assert cylindrical.shape == fourmomenta.shape
    assert reconstructed.shape == fourmomenta.shape
    torch.testing.assert_close(fourmomenta, reconstructed, rtol=1e-4, atol=1e-5)


def test_velocity_transforms_consistency():
    """Test that velocity transforms are consistent with finite differences"""
    transform = FourmomentaToPtPhiEtaM2()
    
    x = torch.randn(3, 4)
    x[:, 0] = torch.abs(x[:, 0]) + 2.0  # Ensure E > |p|
    x.requires_grad_(True)
    
    v_x = torch.randn_like(x)
    
    # Compute velocity transform
    y = transform.forward(x)
    v_y = transform.velocity_forward(v_x, x, y)
    
    # Check with finite differences
    eps = 1e-6
    x_plus = x + eps * v_x
    y_plus = transform.forward(x_plus)
    
    v_y_fd = (y_plus - y) / eps
    
    # Should be approximately equal
    torch.testing.assert_close(v_y, v_y_fd, rtol=1e-3, atol=1e-4)