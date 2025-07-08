import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from experiments.baselines.mlp import MLP
from experiments.baselines.transformer import (
    BaselineLayerNorm,
    ContextualLayerNorm,
    MultiHeadAttention,
    TransformerBlock,
    Transformer
)
from experiments.baselines.conditional_transformer import (
    ConditionalTransformerBlock,
    ConditionalTransformer
)


class TestMLP:
    """Test MLP baseline model"""
    
    def test_mlp_initialization(self):
        """Test MLP initialization with different configurations"""
        in_shape = (10, 4)
        out_shape = (10, 4)
        hidden_channels = 64
        hidden_layers = 3
        
        mlp = MLP(in_shape, out_shape, hidden_channels, hidden_layers)
        
        assert mlp.in_shape == in_shape
        assert mlp.out_shape == out_shape
        assert isinstance(mlp.mlp, torch.nn.Sequential)
        
        # Check that the model has correct input/output dimensions
        input_size = np.product(in_shape)
        output_size = np.product(out_shape)
        
        # First layer should go from input to hidden
        first_layer = mlp.mlp[0]
        assert isinstance(first_layer, torch.nn.Linear)
        assert first_layer.in_features == input_size
        assert first_layer.out_features == hidden_channels
        
        # Last layer should go from hidden to output
        last_layer = mlp.mlp[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        assert last_layer.in_features == hidden_channels
        assert last_layer.out_features == output_size
    
    def test_mlp_initialization_with_dropout(self):
        """Test MLP initialization with dropout"""
        in_shape = (5, 4)
        out_shape = (5, 4)
        hidden_channels = 32
        hidden_layers = 2
        dropout_prob = 0.1
        
        mlp = MLP(in_shape, out_shape, hidden_channels, hidden_layers, dropout_prob)
        
        # Check that dropout layers are present
        dropout_layers = [layer for layer in mlp.mlp if isinstance(layer, torch.nn.Dropout)]
        assert len(dropout_layers) > 0
        
        # Check dropout probability
        for dropout_layer in dropout_layers:
            assert dropout_layer.p == dropout_prob
    
    def test_mlp_forward_pass(self):
        """Test MLP forward pass"""
        batch_size = 8
        in_shape = (10, 4)
        out_shape = (10, 4)
        hidden_channels = 64
        hidden_layers = 2
        
        mlp = MLP(in_shape, out_shape, hidden_channels, hidden_layers)
        
        # Create input tensor - needs to be flattened for MLP
        input_tensor = torch.randn(batch_size, np.product(in_shape))
        
        # Forward pass
        output = mlp.forward(input_tensor)
        
        # Check output shape
        expected_output_shape = (batch_size, np.product(out_shape))
        assert output.shape == expected_output_shape
        
        # Check that output is finite
        assert torch.all(torch.isfinite(output))
    
    def test_mlp_different_input_output_shapes(self):
        """Test MLP with different input and output shapes"""
        in_shape = (20, 4)
        out_shape = (15, 3)
        hidden_channels = 128
        hidden_layers = 3
        
        mlp = MLP(in_shape, out_shape, hidden_channels, hidden_layers)
        
        batch_size = 4
        input_tensor = torch.randn(batch_size, np.product(in_shape))
        output = mlp.forward(input_tensor)
        
        expected_output_shape = (batch_size, np.product(out_shape))
        assert output.shape == expected_output_shape
    
    def test_mlp_invalid_hidden_layers(self):
        """Test that MLP raises error for invalid hidden layers"""
        with pytest.raises(NotImplementedError):
            MLP((5, 4), (5, 4), 64, 0)  # 0 hidden layers not supported
    
    def test_mlp_gradient_flow(self):
        """Test that gradients flow through MLP"""
        mlp = MLP((5, 4), (5, 4), 32, 2)
        
        input_tensor = torch.randn(2, 20, requires_grad=True)
        output = mlp.forward(input_tensor)
        
        # Create a dummy loss
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert input_tensor.grad is not None
        assert torch.any(input_tensor.grad != 0)
        
        # Check that model parameters have gradients
        for param in mlp.parameters():
            assert param.grad is not None
            assert torch.any(param.grad != 0)


class TestLayerNorm:
    """Test layer normalization modules"""
    
    def test_baseline_layer_norm(self):
        """Test BaselineLayerNorm functionality"""
        layer_norm = BaselineLayerNorm()
        
        # Create input with known statistics
        input_tensor = torch.randn(4, 8, 16) * 5.0 + 2.0
        
        # Apply layer norm
        output = layer_norm.forward(input_tensor)
        
        # Check that normalization works (mean ≈ 0, std ≈ 1 over last dimension)
        mean = output.mean(dim=-1)
        std = output.std(dim=-1)
        
        torch.testing.assert_close(mean, torch.zeros_like(mean), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(std, torch.ones_like(std), rtol=1e-5, atol=1e-6)
        
        # Check output shape
        assert output.shape == input_tensor.shape
    
    def test_contextual_layer_norm(self):
        """Test ContextualLayerNorm functionality"""
        in_channels = 64
        context_channels = 32
        
        layer_norm = ContextualLayerNorm(in_channels, context_channels)
        
        batch_size = 4
        seq_len = 10
        
        inputs = torch.randn(batch_size, seq_len, in_channels)
        condition = torch.randn(batch_size, context_channels)
        
        # Apply contextual layer norm
        output = layer_norm.forward(inputs, condition)
        
        # Check output shape
        assert output.shape == inputs.shape
        
        # Check that output is finite
        assert torch.all(torch.isfinite(output))
    
    def test_contextual_layer_norm_parameter_count(self):
        """Test ContextualLayerNorm has correct number of parameters"""
        in_channels = 64
        context_channels = 32
        
        layer_norm = ContextualLayerNorm(in_channels, context_channels)
        
        # Should have projection layer: context_channels -> in_channels * 2
        expected_params = context_channels * in_channels * 2 + in_channels * 2  # weights + bias
        
        total_params = sum(p.numel() for p in layer_norm.parameters())
        assert total_params == expected_params


class TestMultiHeadAttention:
    """Test multi-head attention module"""
    
    def test_multihead_attention_initialization(self):
        """Test MultiHeadAttention initialization"""
        embed_dim = 128
        num_heads = 8
        
        attention = MultiHeadAttention(embed_dim, num_heads)
        
        assert attention.embed_dim == embed_dim
        assert attention.num_heads == num_heads
        assert attention.head_dim == embed_dim // num_heads
    
    def test_multihead_attention_forward(self):
        """Test MultiHeadAttention forward pass"""
        embed_dim = 64
        num_heads = 4
        batch_size = 2
        seq_len = 10
        
        attention = MultiHeadAttention(embed_dim, num_heads)
        
        # Self-attention case
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        output = attention(x, x, x)
        
        # Check output shape
        assert output.shape == x.shape
        
        # Check that output is finite
        assert torch.all(torch.isfinite(output))
    
    def test_multihead_attention_with_mask(self):
        """Test MultiHeadAttention with attention mask"""
        embed_dim = 64
        num_heads = 4
        batch_size = 2
        seq_len = 8
        
        attention = MultiHeadAttention(embed_dim, num_heads)
        
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        output = attention(x, x, x, attn_mask=mask)
        
        assert output.shape == x.shape
        assert torch.all(torch.isfinite(output))
    
    def test_multihead_attention_cross_attention(self):
        """Test MultiHeadAttention for cross-attention"""
        embed_dim = 64
        num_heads = 4
        batch_size = 2
        seq_len_q = 8
        seq_len_kv = 12
        
        attention = MultiHeadAttention(embed_dim, num_heads)
        
        query = torch.randn(batch_size, seq_len_q, embed_dim)
        key = torch.randn(batch_size, seq_len_kv, embed_dim)
        value = torch.randn(batch_size, seq_len_kv, embed_dim)
        
        output = attention(query, key, value)
        
        # Output should have query sequence length
        assert output.shape == (batch_size, seq_len_q, embed_dim)
        assert torch.all(torch.isfinite(output))


class TestTransformerBlock:
    """Test Transformer block"""
    
    def test_transformer_block_initialization(self):
        """Test TransformerBlock initialization"""
        embed_dim = 128
        num_heads = 8
        mlp_dim = 512
        
        block = TransformerBlock(embed_dim, num_heads, mlp_dim)
        
        assert hasattr(block, 'self_attn')
        assert hasattr(block, 'mlp')
        assert hasattr(block, 'norm1')
        assert hasattr(block, 'norm2')
    
    def test_transformer_block_forward(self):
        """Test TransformerBlock forward pass"""
        embed_dim = 64
        num_heads = 4
        mlp_dim = 256
        batch_size = 2
        seq_len = 10
        
        block = TransformerBlock(embed_dim, num_heads, mlp_dim)
        
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        output = block(x)
        
        assert output.shape == x.shape
        assert torch.all(torch.isfinite(output))
    
    def test_transformer_block_with_checkpointing(self):
        """Test TransformerBlock with gradient checkpointing"""
        embed_dim = 64
        num_heads = 4
        mlp_dim = 256
        batch_size = 2
        seq_len = 10
        
        block = TransformerBlock(embed_dim, num_heads, mlp_dim, use_checkpoint=True)
        
        x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
        
        output = block(x)
        
        # Create dummy loss for gradient test
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert torch.any(x.grad != 0)


class TestTransformer:
    """Test full Transformer model"""
    
    def test_transformer_initialization(self):
        """Test Transformer initialization"""
        embed_dim = 128
        num_heads = 8
        num_layers = 6
        mlp_dim = 512
        
        transformer = Transformer(embed_dim, num_heads, num_layers, mlp_dim)
        
        assert len(transformer.layers) == num_layers
        assert hasattr(transformer, 'norm')
    
    def test_transformer_forward(self):
        """Test Transformer forward pass"""
        embed_dim = 64
        num_heads = 4
        num_layers = 3
        mlp_dim = 256
        batch_size = 2
        seq_len = 12
        
        transformer = Transformer(embed_dim, num_heads, num_layers, mlp_dim)
        
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        output = transformer(x)
        
        assert output.shape == x.shape
        assert torch.all(torch.isfinite(output))
    
    def test_transformer_with_mask(self):
        """Test Transformer with attention mask"""
        embed_dim = 64
        num_heads = 4
        num_layers = 2
        mlp_dim = 256
        batch_size = 2
        seq_len = 8
        
        transformer = Transformer(embed_dim, num_heads, num_layers, mlp_dim)
        
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        # Create padding mask (last 2 positions are padding)
        mask = torch.ones(batch_size, seq_len).bool()
        mask[:, -2:] = False
        
        output = transformer(x, src_key_padding_mask=mask)
        
        assert output.shape == x.shape
        assert torch.all(torch.isfinite(output))


class TestConditionalTransformer:
    """Test conditional transformer for conditional generation"""
    
    def test_conditional_transformer_block_initialization(self):
        """Test ConditionalTransformerBlock initialization"""
        embed_dim = 128
        num_heads = 8
        mlp_dim = 512
        
        block = ConditionalTransformerBlock(embed_dim, num_heads, mlp_dim)
        
        assert hasattr(block, 'self_attn')
        assert hasattr(block, 'cross_attn')
        assert hasattr(block, 'mlp')
    
    def test_conditional_transformer_block_forward(self):
        """Test ConditionalTransformerBlock forward pass"""
        embed_dim = 64
        num_heads = 4
        mlp_dim = 256
        batch_size = 2
        seq_len_target = 8
        seq_len_condition = 12
        
        block = ConditionalTransformerBlock(embed_dim, num_heads, mlp_dim)
        
        target = torch.randn(batch_size, seq_len_target, embed_dim)
        condition = torch.randn(batch_size, seq_len_condition, embed_dim)
        
        output = block(target, condition)
        
        assert output.shape == target.shape
        assert torch.all(torch.isfinite(output))
    
    def test_conditional_transformer_initialization(self):
        """Test ConditionalTransformer initialization"""
        embed_dim = 128
        num_heads = 8
        num_layers = 4
        mlp_dim = 512
        
        transformer = ConditionalTransformer(embed_dim, num_heads, num_layers, mlp_dim)
        
        assert len(transformer.layers) == num_layers
    
    def test_conditional_transformer_forward(self):
        """Test ConditionalTransformer forward pass"""
        embed_dim = 64
        num_heads = 4
        num_layers = 2
        mlp_dim = 256
        batch_size = 2
        seq_len_target = 10
        seq_len_condition = 15
        
        transformer = ConditionalTransformer(embed_dim, num_heads, num_layers, mlp_dim)
        
        target = torch.randn(batch_size, seq_len_target, embed_dim)
        condition = torch.randn(batch_size, seq_len_condition, embed_dim)
        
        output = transformer(target, condition)
        
        assert output.shape == target.shape
        assert torch.all(torch.isfinite(output))


class TestModelProperties:
    """Test general model properties"""
    
    def test_parameter_count(self):
        """Test parameter counting for different models"""
        embed_dim = 64
        num_heads = 4
        num_layers = 2
        mlp_dim = 256
        
        # Standard transformer
        transformer = Transformer(embed_dim, num_heads, num_layers, mlp_dim)
        transformer_params = sum(p.numel() for p in transformer.parameters())
        
        # Conditional transformer
        cond_transformer = ConditionalTransformer(embed_dim, num_heads, num_layers, mlp_dim)
        cond_transformer_params = sum(p.numel() for p in cond_transformer.parameters())
        
        # Conditional transformer should have more parameters due to cross-attention
        assert cond_transformer_params > transformer_params
        
        # MLP
        mlp = MLP((10, 4), (10, 4), 64, 2)
        mlp_params = sum(p.numel() for p in mlp.parameters())
        
        # All should have reasonable number of parameters
        assert transformer_params > 1000
        assert cond_transformer_params > 1000
        assert mlp_params > 100
    
    def test_model_modes(self):
        """Test that models can switch between train/eval modes"""
        models = [
            MLP((5, 4), (5, 4), 32, 2, dropout_prob=0.1),
            Transformer(64, 4, 2, 256),
            ConditionalTransformer(64, 4, 2, 256)
        ]
        
        for model in models:
            # Test train mode
            model.train()
            assert model.training
            
            # Test eval mode
            model.eval()
            assert not model.training
    
    def test_device_movement(self):
        """Test that models can be moved between devices"""
        models = [
            MLP((5, 4), (5, 4), 32, 2),
            Transformer(64, 4, 2, 256),
            ConditionalTransformer(64, 4, 2, 256)
        ]
        
        for model in models:
            # Start on CPU
            assert next(model.parameters()).device.type == 'cpu'
            
            # Move to CPU explicitly (should work)
            model.to('cpu')
            assert next(model.parameters()).device.type == 'cpu'
            
            # Test CUDA if available
            if torch.cuda.is_available():
                model.to('cuda')
                assert next(model.parameters()).device.type == 'cuda'
                model.to('cpu')  # Move back for other tests


class TestNumericalStability:
    """Test numerical stability of models"""
    
    def test_large_inputs(self):
        """Test models with large input values"""
        models = [
            MLP((5, 4), (5, 4), 32, 2),
            Transformer(64, 4, 2, 256)
        ]
        
        # Very large inputs
        large_input_mlp = torch.randn(2, 20) * 1000
        large_input_transformer = torch.randn(2, 8, 64) * 1000
        
        inputs = [large_input_mlp, large_input_transformer]
        
        for model, large_input in zip(models, inputs):
            model.eval()
            with torch.no_grad():
                output = model(large_input)
                
                # Should produce finite outputs
                assert torch.all(torch.isfinite(output))
    
    def test_small_inputs(self):
        """Test models with very small input values"""
        models = [
            MLP((5, 4), (5, 4), 32, 2),
            Transformer(64, 4, 2, 256)
        ]
        
        # Very small inputs
        small_input_mlp = torch.randn(2, 20) * 1e-6
        small_input_transformer = torch.randn(2, 8, 64) * 1e-6
        
        inputs = [small_input_mlp, small_input_transformer]
        
        for model, small_input in zip(models, inputs):
            model.eval()
            with torch.no_grad():
                output = model(small_input)
                
                # Should produce finite outputs
                assert torch.all(torch.isfinite(output))
    
    def test_zero_inputs(self):
        """Test models with zero inputs"""
        models = [
            MLP((5, 4), (5, 4), 32, 2),
            Transformer(64, 4, 2, 256)
        ]
        
        # Zero inputs
        zero_input_mlp = torch.zeros(2, 20)
        zero_input_transformer = torch.zeros(2, 8, 64)
        
        inputs = [zero_input_mlp, zero_input_transformer]
        
        for model, zero_input in zip(models, inputs):
            model.eval()
            with torch.no_grad():
                output = model(zero_input)
                
                # Should produce finite outputs
                assert torch.all(torch.isfinite(output))