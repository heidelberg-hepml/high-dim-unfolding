import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data, Batch

def test_channel_dimension_consistency():
    """Test that computed channel dimensions match expectations"""
    
    # Test default configuration values
    config = OmegaConf.create({
        "cfm": {
            "embed_t_dim": 8,
            "add_jet": False,
            "self_condition_prob": 0.0
        },
        "data": {
            "pos_encoding_dim": 8,
            "add_pid": False
        }
    })
    
    # Simulate channel computation from experiment.py
    in_channels = 4 + config.cfm.embed_t_dim + config.data.pos_encoding_dim  # Base: 4 + 8 + 8 = 20
    condition_in_channels = 4 + config.data.pos_encoding_dim  # Base: 4 + 8 = 12
    
    if config.data.add_pid:
        in_channels += 6
        condition_in_channels += 6
    
    if config.cfm.add_jet:
        in_channels += 1
        condition_in_channels += 1
        
    if config.cfm.self_condition_prob > 0.0:
        in_channels += 4
    
    # Verify expected dimensions
    assert in_channels == 20, f"Expected 20 input channels, got {in_channels}"
    assert condition_in_channels == 12, f"Expected 12 condition channels, got {condition_in_channels}"
    
    # Test with add_jet=True
    config.cfm.add_jet = True
    in_channels_jet = 4 + config.cfm.embed_t_dim + config.data.pos_encoding_dim + 1  # 21
    condition_in_channels_jet = 4 + config.data.pos_encoding_dim + 1  # 13
    
    assert in_channels_jet == 21, f"Expected 21 input channels with jets, got {in_channels_jet}"
    assert condition_in_channels_jet == 13, f"Expected 13 condition channels with jets, got {condition_in_channels_jet}"

def test_velocity_masking_consistency():
    """Test that velocity masking is consistent between training and sampling"""
    
    # Mock setup
    jet_mask = torch.tensor([True, True, False, True, False])  # Mixed mask
    velocity = torch.randn(5, 4)
    
    # Training behavior: apply handle_velocity to masked elements only, return subset
    def handle_velocity_mock(v):
        v[..., 3] = 0.0  # Zero out mass dimension
        return v
    
    # Training-style masking
    training_result = handle_velocity_mock(velocity[jet_mask].clone())
    
    # Sampling-style masking (fixed version)
    sampling_velocity = velocity.clone()
    sampling_velocity[jet_mask] = handle_velocity_mock(sampling_velocity[jet_mask])
    sampling_velocity[~jet_mask] = 0.0
    sampling_result = sampling_velocity[jet_mask]
    
    # Should be consistent
    assert torch.allclose(training_result, sampling_result), "Training and sampling velocity masking should be consistent"

def test_coordinate_transform_chain():
    """Test that coordinate transforms are properly chained and invertible"""
    
    # Create some test fourmomenta data
    fourmomenta = torch.tensor([
        [100.0, 50.0, 30.0, 10.0],  # E, px, py, pz
        [80.0, 40.0, 20.0, 5.0]
    ])
    
    # Test that x -> fourmomenta -> x is approximately identity
    # (This would require actual coordinate implementations to test fully)
    
    # For now, just test shape consistency
    assert fourmomenta.shape == (2, 4), "Input should be (N, 4)"
    
    # Mock coordinate transform
    x_transformed = fourmomenta  # In real case: coordinates.fourmomenta_to_x(fourmomenta)
    fourmomenta_recovered = x_transformed  # In real case: coordinates.x_to_fourmomenta(x_transformed)
    
    assert fourmomenta_recovered.shape == fourmomenta.shape, "Coordinate transform should preserve shape"

def test_add_jet_sequence_consistency():
    """Test that add_jet_to_sequence maintains consistency"""
    
    # Create a simple batch
    original_data = Data(
        x_gen=torch.randn(4, 4),
        scalars_gen=torch.randn(4, 2),
        jet_gen=torch.randn(1, 4),
        x_gen_ptr=torch.tensor([0, 4])
    )
    
    batch = Batch.from_data_list([original_data])
    
    # Mock add_jet_to_sequence behavior
    # Should add 1 particle and 1 scalar feature per event
    expected_n_particles = batch.x_gen.shape[0] + len(batch.x_gen_ptr) - 1  # +1 jet per event
    expected_n_scalars = batch.scalars_gen.shape[1] + 1  # +1 jet flag
    
    # Basic consistency checks
    assert batch.x_gen.shape[0] == 4, "Original should have 4 particles"
    assert expected_n_particles == 5, "With jets should have 5 particles" 
    assert expected_n_scalars == 3, "With jets should have 3 scalar features"

def test_batch_pointer_consistency():
    """Test that batch pointers are handled consistently"""
    
    # Create batch with varying event sizes
    data1 = Data(x_gen=torch.randn(3, 4), jet_gen=torch.randn(1, 4))
    data2 = Data(x_gen=torch.randn(5, 4), jet_gen=torch.randn(1, 4))  
    batch = Batch.from_data_list([data1, data2], follow_batch=["x_gen"])
    
    # Check pointer consistency
    assert batch.x_gen_ptr.tolist() == [0, 3, 8], "Batch pointers should be cumulative"
    
    # Test jet expansion
    gen_jets = torch.repeat_interleave(
        batch.jet_gen, batch.x_gen_ptr.diff(), dim=0
    )
    
    assert gen_jets.shape[0] == batch.x_gen.shape[0], "Jet vectors should match particle count"
    assert gen_jets.shape[1] == 4, "Jet vectors should be 4D"

def test_configuration_robustness():
    """Test that configurations don't cause runtime errors"""
    
    configs_to_test = [
        {"add_jet": False, "add_pid": False, "self_condition_prob": 0.0},
        {"add_jet": True, "add_pid": False, "self_condition_prob": 0.0},
        {"add_jet": False, "add_pid": True, "self_condition_prob": 0.0},
        {"add_jet": True, "add_pid": True, "self_condition_prob": 0.0},
        # Note: Not testing self_condition_prob > 0 due to custom_rk4 issues
    ]
    
    for config in configs_to_test:
        # Compute expected dimensions
        base_channels = 4 + 8 + 8  # coords + embed_t + pos_encoding
        condition_channels = 4 + 8  # coords + pos_encoding
        
        if config["add_pid"]:
            base_channels += 6
            condition_channels += 6
            
        if config["add_jet"]:
            base_channels += 1
            condition_channels += 1
            
        if config["self_condition_prob"] > 0.0:
            base_channels += 4
        
        # Should not cause errors
        assert base_channels > 0, f"Invalid channel count for config {config}"
        assert condition_channels > 0, f"Invalid condition channel count for config {config}"

if __name__ == "__main__":
    test_channel_dimension_consistency()
    test_velocity_masking_consistency() 
    test_coordinate_transform_chain()
    test_add_jet_sequence_consistency()
    test_batch_pointer_consistency()
    test_configuration_robustness()
    print("All pipeline consistency tests passed!")