"""
Unit tests for compute_topk_mse_loss function.

Tests cover:
1. Basic functionality with correct shapes
2. Numerical stability (FP16 inputs)
3. NaN handling
4. Edge cases
5. Device compatibility
"""
import torch
import torch.nn.functional as F
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantize.utils import compute_topk_kl_loss, compute_topk_mse_loss


class TestComputeTopkMseLoss:
    """Tests for compute_topk_mse_loss function."""
    
    def test_basic_functionality(self):
        """Test basic functionality with simple inputs."""
        batch_size, seq_len, num_experts = 2, 4, 8
        topk = 3
        
        # Create student logits
        student_logits = torch.randn(batch_size, seq_len, num_experts)
        
        # Create teacher logits and indices (simulating cached raw logit labels)
        teacher_logits = torch.randn(batch_size, seq_len, topk)
        teacher_indices = torch.randint(0, num_experts, (batch_size, seq_len, topk))
        
        # Compute loss
        loss = compute_topk_mse_loss(student_logits, teacher_logits, teacher_indices)
        
        # Verify output
        assert loss.shape == torch.Size([]), "Loss should be a scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be inf"
        assert loss >= 0, "MSE loss should be non-negative"
    
    def test_perfect_match(self):
        """Test when student perfectly matches teacher."""
        batch_size, seq_len, num_experts = 2, 4, 8
        topk = 3
        
        # Create deterministic student logits
        student_logits = torch.zeros(batch_size, seq_len, num_experts)
        # Make specific experts have high logits
        teacher_indices = torch.tensor([[[0, 1, 2], [3, 4, 5], [0, 2, 4], [1, 3, 5]],
                                        [[0, 1, 2], [3, 4, 5], [0, 2, 4], [1, 3, 5]]])
        
        # Set high logits for teacher's selected experts
        for b in range(batch_size):
            for s in range(seq_len):
                for k in range(topk):
                    idx = teacher_indices[b, s, k].item()
                    student_logits[b, s, idx] = 10.0 - k  # Decreasing logits
        
        # Use raw logits gathered at teacher indices as teacher values
        teacher_logits = torch.gather(student_logits, dim=-1, index=teacher_indices)
        
        loss = compute_topk_mse_loss(student_logits, teacher_logits, teacher_indices)
        
        # Loss should be very close to zero
        assert loss.item() < 1e-5, f"Loss should be near zero for perfect match, got {loss.item()}"
    
    def test_fp16_inputs(self):
        """Test numerical stability with FP16 inputs."""
        batch_size, seq_len, num_experts = 4, 128, 60
        topk = 20
        
        # Create FP16 tensors
        student_logits = torch.randn(batch_size, seq_len, num_experts, dtype=torch.float16)
        teacher_logits = torch.randn(batch_size, seq_len, topk, dtype=torch.float16)
        teacher_indices = torch.randint(0, num_experts, (batch_size, seq_len, topk))
        
        loss = compute_topk_mse_loss(student_logits, teacher_logits, teacher_indices)
        
        assert not torch.isnan(loss), "Loss should not be NaN with FP16 inputs"
        assert not torch.isinf(loss), "Loss should not be inf with FP16 inputs"
    
    def test_extreme_logits(self):
        """Test handling of extreme logit values."""
        batch_size, seq_len, num_experts = 2, 4, 8
        topk = 3
        
        # Create extreme logits (large positive and negative values)
        student_logits = torch.randn(batch_size, seq_len, num_experts)
        student_logits[0, 0, 0] = 100.0  # Very large positive
        student_logits[0, 0, 1] = -100.0  # Very large negative
        
        teacher_logits = torch.randn(batch_size, seq_len, topk)
        teacher_indices = torch.randint(0, num_experts, (batch_size, seq_len, topk))
        
        loss = compute_topk_mse_loss(student_logits, teacher_logits, teacher_indices)
        
        assert not torch.isnan(loss), "Loss should not be NaN with extreme logits"
        assert not torch.isinf(loss), "Loss should not be inf with extreme logits"
    
    def test_nan_input_handling(self):
        """Test that NaN inputs are handled gracefully."""
        batch_size, seq_len, num_experts = 2, 4, 8
        topk = 3
        
        # Create inputs with NaN
        student_logits = torch.randn(batch_size, seq_len, num_experts)
        student_logits[0, 0, 0] = float('nan')
        
        teacher_logits = torch.randn(batch_size, seq_len, topk)
        teacher_indices = torch.randint(0, num_experts, (batch_size, seq_len, topk))
        
        loss = compute_topk_mse_loss(student_logits, teacher_logits, teacher_indices)
        
        # Function should return zero loss when NaN is detected
        assert loss.item() == 0.0, "Loss should be 0 when NaN is detected in inputs"
    
    def test_nan_teacher_logits_handling(self):
        """Test that NaN in teacher logits is handled gracefully."""
        batch_size, seq_len, num_experts = 2, 4, 8
        topk = 3
        
        student_logits = torch.randn(batch_size, seq_len, num_experts)
        teacher_logits = torch.randn(batch_size, seq_len, topk)
        teacher_logits[0, 0, 0] = float('nan')
        teacher_indices = torch.randint(0, num_experts, (batch_size, seq_len, topk))
        
        loss = compute_topk_mse_loss(student_logits, teacher_logits, teacher_indices)
        
        # Function should return zero loss when NaN is detected
        assert loss.item() == 0.0, "Loss should be 0 when NaN is detected in teacher_logits"
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through the loss."""
        batch_size, seq_len, num_experts = 2, 4, 8
        topk = 3
        
        student_logits = torch.randn(batch_size, seq_len, num_experts, requires_grad=True)
        teacher_logits = torch.randn(batch_size, seq_len, topk)
        teacher_indices = torch.randint(0, num_experts, (batch_size, seq_len, topk))
        
        loss = compute_topk_mse_loss(student_logits, teacher_logits, teacher_indices)
        loss.backward()
        
        assert student_logits.grad is not None, "Gradients should flow to student_logits"
        assert not torch.isnan(student_logits.grad).any(), "Gradients should not contain NaN"
    
    def test_debug_mode(self):
        """Test debug mode prints information without errors."""
        batch_size, seq_len, num_experts = 2, 4, 8
        topk = 3
        
        student_logits = torch.randn(batch_size, seq_len, num_experts)
        teacher_logits = torch.randn(batch_size, seq_len, topk)
        teacher_indices = torch.randint(0, num_experts, (batch_size, seq_len, topk))
        
        # Should not raise any exceptions
        loss = compute_topk_mse_loss(student_logits, teacher_logits, teacher_indices, debug=True)
        
        assert not torch.isnan(loss), "Loss should not be NaN in debug mode"
    
    def test_device_consistency(self):
        """Test that function handles different devices correctly."""
        if not torch.cuda.is_available():
            print("  (skipped - CUDA not available)")
            return
        
        batch_size, seq_len, num_experts = 2, 4, 8
        topk = 3
        
        # Create tensors on different devices
        student_logits = torch.randn(batch_size, seq_len, num_experts, device='cuda')
        teacher_logits = torch.randn(batch_size, seq_len, topk)  # CPU
        teacher_indices = torch.randint(0, num_experts, (batch_size, seq_len, topk))  # CPU
        
        loss = compute_topk_mse_loss(student_logits, teacher_logits, teacher_indices)
        
        assert loss.device.type == 'cuda', "Loss should be on same device as student_logits"
        assert not torch.isnan(loss), "Loss should not be NaN with cross-device inputs"
    
    def test_large_batch(self):
        """Test with larger batch sizes similar to real usage."""
        batch_size, seq_len, num_experts = 128, 2048, 60
        topk = 20
        
        student_logits = torch.randn(batch_size, seq_len, num_experts)
        teacher_logits = torch.randn(batch_size, seq_len, topk)
        teacher_indices = torch.randint(0, num_experts, (batch_size, seq_len, topk))
        
        loss = compute_topk_mse_loss(student_logits, teacher_logits, teacher_indices)
        
        assert not torch.isnan(loss), "Loss should not be NaN with large batch"
        assert not torch.isinf(loss), "Loss should not be inf with large batch"
    
    def test_index_dtype(self):
        """Test that function handles different index dtypes correctly."""
        batch_size, seq_len, num_experts = 2, 4, 8
        topk = 3
        
        student_logits = torch.randn(batch_size, seq_len, num_experts)
        teacher_logits = torch.randn(batch_size, seq_len, topk)
        
        # Test with int32 indices
        teacher_indices_int32 = torch.randint(0, num_experts, (batch_size, seq_len, topk), dtype=torch.int32)
        loss_int32 = compute_topk_mse_loss(student_logits, teacher_logits, teacher_indices_int32)
        assert not torch.isnan(loss_int32), "Loss should work with int32 indices"
        
        # Test with int64 indices
        teacher_indices_int64 = torch.randint(0, num_experts, (batch_size, seq_len, topk), dtype=torch.int64)
        loss_int64 = compute_topk_mse_loss(student_logits, teacher_logits, teacher_indices_int64)
        assert not torch.isnan(loss_int64), "Loss should work with int64 indices"

    def test_shape_mismatch_returns_none_when_requested(self):
        """Invalid leading dims should return None instead of raising when requested."""
        student_logits = torch.randn(1, 10, 16)
        teacher_logits = torch.randn(2, 5, 4)
        teacher_indices = torch.randint(0, 16, (2, 5, 4))

        mse_loss = compute_topk_mse_loss(
            student_logits,
            teacher_logits,
            teacher_indices,
            return_none_on_nonfinite=True,
        )
        kl_loss = compute_topk_kl_loss(
            student_logits,
            teacher_logits,
            teacher_indices,
            return_none_on_nonfinite=True,
        )

        assert mse_loss is None, "MSE loss should return None on invalid leading dims when requested"
        assert kl_loss is None, "KL loss should return None on invalid leading dims when requested"


def run_tests():
    """Run all tests with verbose output."""
    test_instance = TestComputeTopkMseLoss()
    
    test_methods = [
        'test_basic_functionality',
        'test_perfect_match',
        'test_fp16_inputs',
        'test_extreme_logits',
        'test_nan_input_handling',
        'test_nan_teacher_logits_handling',
        'test_gradient_flow',
        'test_debug_mode',
        'test_large_batch',
        'test_index_dtype',
    ]
    
    # Add CUDA test only if available
    if torch.cuda.is_available():
        test_methods.append('test_device_consistency')
    
    passed = 0
    failed = 0
    
    for method_name in test_methods:
        try:
            method = getattr(test_instance, method_name)
            method()
            print(f"✓ {method_name}")
            passed += 1
        except Exception as e:
            print(f"✗ {method_name}: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
