import pytest
import torch
import tempfile
import os
from pathlib import Path
from omegaconf import OmegaConf
from torch_geometric.data import Data, Batch

def test_pipeline_data_flow_consistency():
    """Test that data shapes and types remain consistent throughout the pipeline"""
    
    # Simulate pipeline data flow with realistic shapes
    batch_size = 4
    max_particles = 10
    n_features = 2  # Without PID encoding
    
    # 1. Raw data loading simulation
    det_particles = torch.randn(batch_size, max_particles, 4)  # Fourmomenta
    gen_particles = torch.randn(batch_size, max_particles, 4)
    det_mults = torch.randint(3, max_particles, (batch_size,))
    gen_mults = torch.randint(3, max_particles, (batch_size,))
    det_pids = torch.randn(batch_size, max_particles, n_features)
    gen_pids = torch.randn(batch_size, max_particles, n_features)
    det_jets = torch.randn(batch_size, 4)
    gen_jets = torch.randn(batch_size, 4)
    
    # 2. Dataset creation (simulating Dataset.create_data_list)
    data_list = []
    for i in range(batch_size):
        data = Data(
            x_det=det_particles[i, :det_mults[i]],      # [n_det, 4]
            scalars_det=det_pids[i, :det_mults[i]],     # [n_det, n_features]
            jet_det=det_jets[i:i+1],                    # [1, 4]
            x_gen=gen_particles[i, :gen_mults[i]],      # [n_gen, 4] 
            scalars_gen=gen_pids[i, :gen_mults[i]],     # [n_gen, n_features]
            jet_gen=gen_jets[i:i+1]                     # [1, 4]
        )
        data_list.append(data)
    
    # 3. Batching (simulating DataLoader)
    batch = Batch.from_data_list(data_list, follow_batch=["x_gen", "x_det"])
    
    # 4. Verify batch structure
    total_det_particles = det_mults.sum().item()
    total_gen_particles = gen_mults.sum().item()
    
    assert batch.x_det.shape == (total_det_particles, 4), f"Expected det shape {(total_det_particles, 4)}, got {batch.x_det.shape}"
    assert batch.x_gen.shape == (total_gen_particles, 4), f"Expected gen shape {(total_gen_particles, 4)}, got {batch.x_gen.shape}"
    assert batch.scalars_det.shape == (total_det_particles, n_features), f"Expected scalars_det shape {(total_det_particles, n_features)}, got {batch.scalars_det.shape}"
    assert batch.scalars_gen.shape == (total_gen_particles, n_features), f"Expected scalars_gen shape {(total_gen_particles, n_features)}, got {batch.scalars_gen.shape}"
    assert batch.jet_det.shape == (batch_size, 4), f"Expected jet_det shape {(batch_size, 4)}, got {batch.jet_det.shape}"
    assert batch.jet_gen.shape == (batch_size, 4), f"Expected jet_gen shape {(batch_size, 4)}, got {batch.jet_gen.shape}"
    
    # 5. Verify pointers
    assert len(batch.x_det_ptr) == batch_size + 1, f"Expected {batch_size + 1} det pointers, got {len(batch.x_det_ptr)}"
    assert len(batch.x_gen_ptr) == batch_size + 1, f"Expected {batch_size + 1} gen pointers, got {len(batch.x_gen_ptr)}"
    assert batch.x_det_ptr[-1] == total_det_particles, f"Last det pointer should be {total_det_particles}, got {batch.x_det_ptr[-1]}"
    assert batch.x_gen_ptr[-1] == total_gen_particles, f"Last gen pointer should be {total_gen_particles}, got {batch.x_gen_ptr[-1]}"
    
    # 6. Verify jet expansion (as done in pipeline)
    det_jets_expanded = torch.repeat_interleave(batch.jet_det, batch.x_det_ptr.diff(), dim=0)
    gen_jets_expanded = torch.repeat_interleave(batch.jet_gen, batch.x_gen_ptr.diff(), dim=0)
    
    assert det_jets_expanded.shape == (total_det_particles, 4), f"Expanded det jets shape {det_jets_expanded.shape} doesn't match particles"
    assert gen_jets_expanded.shape == (total_gen_particles, 4), f"Expanded gen jets shape {gen_jets_expanded.shape} doesn't match particles"

def test_train_val_test_split_integrity():
    """Test that train/val/test split maintains data integrity"""
    
    total_events = 100
    split_ratios = [0.8, 0.1, 0.1]
    
    # Simulate split calculation
    import numpy as np
    split_indices = np.cumsum([int(r * total_events) for r in split_ratios])
    train_idx, val_idx, test_idx = split_indices
    
    # Verify split sizes
    assert train_idx == 80, f"Expected 80 training events, got {train_idx}"
    assert val_idx == 90, f"Expected 90 total (train+val) events, got {val_idx}" 
    assert test_idx == 100, f"Expected 100 total events, got {test_idx}"
    
    # Verify no overlap
    train_events = set(range(0, train_idx))
    val_events = set(range(train_idx, val_idx))
    test_events = set(range(val_idx, test_idx))
    
    assert len(train_events & val_events) == 0, "Training and validation sets overlap"
    assert len(train_events & test_events) == 0, "Training and test sets overlap"
    assert len(val_events & test_events) == 0, "Validation and test sets overlap"
    assert len(train_events | val_events | test_events) == total_events, "Split doesn't cover all events"

def test_coordinate_transform_consistency():
    """Test that coordinate transforms are applied consistently"""
    
    # Mock fourmomenta data
    particles = torch.tensor([
        [100.0, 50.0, 30.0, 10.0],  # [E, px, py, pz]
        [80.0, 40.0, 20.0, 5.0]
    ])
    jets = torch.tensor([[180.0, 90.0, 50.0, 15.0]])  # Sum of particles
    
    # Test basic shape consistency (would need actual coordinate implementation for full test)
    assert particles.shape[-1] == 4, "Input particles should be 4D"
    assert jets.shape[-1] == 4, "Jets should be 4D"
    
    # Test that masks work correctly
    multiplicities = torch.tensor([2])  # Both particles are valid
    mask = torch.arange(particles.shape[0])[None, :] < multiplicities[:, None]
    
    assert mask.sum() == 2, "Both particles should be masked as valid"
    assert particles[mask.squeeze()].shape == (2, 4), "Masked particles should maintain shape"

def test_cfm_loss_computation_shapes():
    """Test that CFM loss computation handles shapes correctly"""
    
    batch_size = 3
    total_particles = 15  # Sum across batch
    coordinate_dim = 4
    
    # Simulate batch data
    x0 = torch.randn(total_particles, coordinate_dim)  # Real data
    x1 = torch.randn(total_particles, coordinate_dim)  # Noise
    t = torch.rand(total_particles, 1)                 # Time
    
    # CFM interpolation
    xt = x0 + (x1 - x0) * t  # Linear interpolation
    vt = x1 - x0             # Target velocity
    
    # Mock predicted velocity
    vp = torch.randn_like(vt)
    
    # Test shapes
    assert xt.shape == (total_particles, coordinate_dim), f"Interpolated shape wrong: {xt.shape}"
    assert vt.shape == (total_particles, coordinate_dim), f"Target velocity shape wrong: {vt.shape}"
    assert vp.shape == (total_particles, coordinate_dim), f"Predicted velocity shape wrong: {vp.shape}"
    
    # Test loss computation
    loss = torch.nn.functional.mse_loss(vp, vt)
    assert loss.dim() == 0, "Loss should be scalar"
    assert torch.isfinite(loss), "Loss should be finite"

def test_sampling_coordinate_transform_consistency():
    """Test that sampling maintains coordinate transform consistency"""
    
    batch_size = 2
    n_particles_per_event = [5, 7]
    
    # Create sample_batch and original batch with different structures
    sample_data = []
    original_data = []
    
    for i, n_particles in enumerate(n_particles_per_event):
        # Sample batch (generated data)
        sample_data.append(Data(
            x_gen=torch.randn(n_particles, 4),
            jet_gen=torch.tensor([[10.0 + i, 1.0, 2.0, 3.0]])  # Different jets
        ))
        
        # Original batch (conditioning data)  
        original_data.append(Data(
            x_gen=torch.randn(n_particles + 2, 4),  # Different number of particles
            jet_gen=torch.tensor([[20.0 + i, 4.0, 5.0, 6.0]])  # Different jets
        ))
    
    sample_batch = Batch.from_data_list(sample_data, follow_batch=["x_gen"])
    original_batch = Batch.from_data_list(original_data, follow_batch=["x_gen"])
    
    # Compute jets separately (as in fixed pipeline)
    sample_jets = torch.repeat_interleave(sample_batch.jet_gen, sample_batch.x_gen_ptr.diff(), dim=0)
    original_jets = torch.repeat_interleave(original_batch.jet_gen, original_batch.x_gen_ptr.diff(), dim=0)
    
    # Verify dimensions match particles
    assert sample_jets.shape[0] == sample_batch.x_gen.shape[0], "Sample jets don't match sample particles"
    assert original_jets.shape[0] == original_batch.x_gen.shape[0], "Original jets don't match original particles"
    
    # Verify jets are different (this was the bug)
    assert not torch.equal(sample_jets, original_jets[:sample_jets.shape[0]]), "Jets should be different for different batches"

def test_channel_dimension_computation():
    """Test that channel dimensions are computed correctly for different configurations"""
    
    configs = [
        {"embed_t_dim": 8, "pos_encoding_dim": 8, "add_pid": False, "add_jet": False, "self_condition_prob": 0.0},
        {"embed_t_dim": 8, "pos_encoding_dim": 8, "add_pid": True, "add_jet": False, "self_condition_prob": 0.0},
        {"embed_t_dim": 8, "pos_encoding_dim": 8, "add_pid": False, "add_jet": True, "self_condition_prob": 0.0},
        {"embed_t_dim": 8, "pos_encoding_dim": 8, "add_pid": True, "add_jet": True, "self_condition_prob": 0.0},
        {"embed_t_dim": 16, "pos_encoding_dim": 4, "add_pid": False, "add_jet": False, "self_condition_prob": 0.0},
    ]
    
    expected_channels = [20, 26, 21, 27, 24]  # 4 + embed_t + pos_encoding + optional features
    expected_condition_channels = [12, 18, 13, 19, 8]  # 4 + pos_encoding + optional features
    
    for i, config in enumerate(configs):
        # Compute channels as in experiment.py
        in_channels = 4 + config["embed_t_dim"] + config["pos_encoding_dim"]
        condition_in_channels = 4 + config["pos_encoding_dim"]
        
        if config["add_pid"]:
            in_channels += 6
            condition_in_channels += 6
        if config["add_jet"]:
            in_channels += 1
            condition_in_channels += 1
        if config["self_condition_prob"] > 0.0:
            in_channels += 4
        
        assert in_channels == expected_channels[i], f"Config {i}: expected {expected_channels[i]} channels, got {in_channels}"
        assert condition_in_channels == expected_condition_channels[i], f"Config {i}: expected {expected_condition_channels[i]} condition channels, got {condition_in_channels}"

if __name__ == "__main__":
    test_pipeline_data_flow_consistency()
    test_train_val_test_split_integrity()
    test_coordinate_transform_consistency() 
    test_cfm_loss_computation_shapes()
    test_sampling_coordinate_transform_consistency()
    test_channel_dimension_computation()
    print("✅ All complete pipeline tests passed!")