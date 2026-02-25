# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import random
import torch
import torch.nn.functional as F
from transformer_engine.pytorch import parallel_cross_entropy

from utils import dtype_tols


class TestParallelCrossEntropy:

    def generate_iters(self, iters: int):
        self.iters = iters

    def generate_infra(self, reduce_loss: bool, label_smoothing: float):
        self.test_loss_func = parallel_cross_entropy
        self.ref_loss_func = torch.nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, reduction="mean" if reduce_loss else "none"
        )

    def generate_input(
        self,
        dtype: torch.dtype,
        swap_dim: bool,
        ignore_idx: bool,
        device: torch.device = "cuda",
    ):
        SQ = random.choice([64, 128])
        batch = random.choice([1, 2])
        vocab = random.choice([64000, 128000])
        ignore = random.sample(range(0, SQ - 1), 5)

        # Generate random data
        if swap_dim:
            self.input_test = torch.rand((SQ, batch, vocab), dtype=dtype, device=device)
            self.tar_test = torch.randint(0, vocab, (SQ, batch), device=device)
        else:
            self.input_test = torch.rand((batch, SQ, vocab), dtype=dtype, device=device)
            self.tar_test = torch.randint(0, vocab, (batch, SQ), device=device)

        if ignore_idx:
            for i in ignore:
                # Ignore 5 indices
                if swap_dim:
                    self.tar_test[i][0] = -100
                else:
                    self.tar_test[0][i] = -100

        # Make copy of data for reference implementation
        self.input_ref = torch.reshape(self.input_test.clone().detach(), (batch * SQ, vocab))
        self.tar_ref = torch.reshape(self.tar_test.clone().detach(), (batch * SQ,))

        # Enable autograd
        self.input_test.requires_grad_()
        self.input_ref.requires_grad_()

    def one_iteration_test(
        self,
        dtype: torch.dtype,
        swap_dim: bool,
        label_smoothing: float,
        reduce_loss: bool,
        ignore_idx: bool = False,
    ):

        # Random data
        self.generate_input(dtype, swap_dim, ignore_idx)

        # Forward pass
        test_loss = self.test_loss_func(
            self.input_test, self.tar_test, label_smoothing, reduce_loss, None
        )
        ref_loss = self.ref_loss_func(self.input_ref, self.tar_ref)

        # Compute square to avoid trivial backward pass
        test_loss = torch.square(test_loss)
        ref_loss = torch.square(ref_loss)

        # Backward pass
        if reduce_loss:
            test_loss.backward()
            ref_loss.backward()
        else:
            test_loss.sum().backward()
            ref_loss.sum().backward()

        # Check that loss and grad input match
        tols = dtype_tols(dtype)
        test_loss = test_loss.to(dtype=torch.float64, device="cpu")
        ref_loss = ref_loss.to(dtype=torch.float64, device="cpu")
        ref_loss = ref_loss.reshape(test_loss.size())
        test_grad_input = self.input_test.grad.to(dtype=torch.float64, device="cpu")
        ref_grad_input = self.input_ref.grad.to(dtype=torch.float64, device="cpu")
        ref_grad_input = ref_grad_input.reshape(test_grad_input.size())
        torch.testing.assert_close(test_loss, ref_loss, **tols)
        torch.testing.assert_close(test_grad_input, ref_grad_input, **tols)

        # Reset data
        self.input_test = None
        self.input_ref = None
        self.tar_test = None
        self.tar_ref = None

    def test_float32_input(self):
        self.generate_iters(5)
        self.generate_infra(True, 0)
        for i in range(self.iters):
            self.one_iteration_test(
                dtype=torch.float32, swap_dim=False, label_smoothing=0, reduce_loss=True
            )

    def test_bfloat16_input(self):
        self.generate_iters(5)
        self.generate_infra(True, 0)
        for i in range(self.iters):
            self.one_iteration_test(
                dtype=torch.bfloat16, swap_dim=False, label_smoothing=0, reduce_loss=True
            )

    def test_swapped_input(self):
        self.generate_iters(5)
        self.generate_infra(True, 0)
        for i in range(self.iters):
            self.one_iteration_test(
                dtype=torch.float32, swap_dim=True, label_smoothing=0, reduce_loss=True
            )

    def test_label_smoothing(self):
        self.generate_iters(3)
        self.generate_infra(True, 0.1)
        for i in range(self.iters):
            self.one_iteration_test(
                dtype=torch.float32, swap_dim=False, label_smoothing=0.1, reduce_loss=True
            )

    def test_non_reduced_loss(self):
        self.generate_iters(1)
        self.generate_infra(False, 0)
        for i in range(self.iters):
            self.one_iteration_test(
                dtype=torch.float32, swap_dim=False, label_smoothing=0, reduce_loss=False
            )

    def test_ignore_idx(self):
        self.generate_iters(5)
        self.generate_infra(False, 0)
        for i in range(self.iters):
            self.one_iteration_test(
                dtype=torch.float32,
                swap_dim=random.choice([True, False]),
                label_smoothing=0,
                reduce_loss=False,
                ignore_idx=True,
            )

    def test_ignore_idx_reduced_loss(self):
        """Test ignore_idx with reduce_loss=True"""
        self.generate_iters(5)
        self.generate_infra(True, 0)  # reduce_loss=True
        for i in range(self.iters):
            self.one_iteration_test(
                dtype=torch.float32,
                swap_dim=random.choice([True, False]),
                label_smoothing=0,
                reduce_loss=True,
                ignore_idx=True,
            )

    def test_z_loss(self):
        """Test basic z-loss correctness: CE(mean) + z_loss_weight * mean(lse^2)"""
        z_loss_weight = 1e-4
        tols = dtype_tols(torch.float32)

        for _ in range(3):
            B, SQ = 2, 64
            V = random.choice([64000, 128000])

            inp_test = torch.rand((B, SQ, V), dtype=torch.float32, device="cuda")
            tar_test = torch.randint(0, V, (B, SQ), device="cuda")

            inp_ref = inp_test.clone().detach().reshape(B * SQ, V)
            tar_ref = tar_test.clone().detach().reshape(B * SQ)

            inp_test.requires_grad_()
            inp_ref.requires_grad_()

            # Test: Triton with z-loss
            test_loss = parallel_cross_entropy(
                inp_test, tar_test, reduce_loss=True, z_loss_weight=z_loss_weight,
            )

            # Reference: PyTorch CE + manual z-loss
            ce_loss = F.cross_entropy(inp_ref, tar_ref, reduction="mean")
            lse = torch.logsumexp(inp_ref, dim=-1)
            ref_z_loss = z_loss_weight * (lse ** 2).mean()
            ref_loss = ce_loss + ref_z_loss

            torch.testing.assert_close(
                test_loss.to(torch.float64, device="cpu"),
                ref_loss.to(torch.float64, device="cpu"),
                **tols,
            )

            test_loss.backward()
            ref_loss.backward()

            test_grad = inp_test.grad.to(torch.float64, device="cpu")
            ref_grad = inp_ref.grad.to(torch.float64, device="cpu").reshape(test_grad.shape)
            torch.testing.assert_close(test_grad, ref_grad, **tols)

    def test_z_loss_with_label_smoothing(self):
        """Test z-loss + label smoothing composition"""
        z_loss_weight = 1e-4
        label_smoothing = 0.1
        tols = dtype_tols(torch.float32)

        for _ in range(3):
            B, SQ = 2, 64
            V = random.choice([64000, 128000])

            inp_test = torch.rand((B, SQ, V), dtype=torch.float32, device="cuda")
            tar_test = torch.randint(0, V, (B, SQ), device="cuda")

            inp_ref = inp_test.clone().detach().reshape(B * SQ, V)
            tar_ref = tar_test.clone().detach().reshape(B * SQ)

            inp_test.requires_grad_()
            inp_ref.requires_grad_()

            # Test
            test_loss = parallel_cross_entropy(
                inp_test, tar_test,
                label_smoothing=label_smoothing,
                reduce_loss=True,
                z_loss_weight=z_loss_weight,
            )

            # Reference
            ce_loss = F.cross_entropy(
                inp_ref, tar_ref, reduction="mean", label_smoothing=label_smoothing,
            )
            lse = torch.logsumexp(inp_ref, dim=-1)
            ref_loss = ce_loss + z_loss_weight * (lse ** 2).mean()

            torch.testing.assert_close(
                test_loss.to(torch.float64, device="cpu"),
                ref_loss.to(torch.float64, device="cpu"),
                **tols,
            )

            test_loss.backward()
            ref_loss.backward()

            test_grad = inp_test.grad.to(torch.float64, device="cpu")
            ref_grad = inp_ref.grad.to(torch.float64, device="cpu").reshape(test_grad.shape)
            torch.testing.assert_close(test_grad, ref_grad, **tols)

    def test_z_loss_with_ignore_idx(self):
        """Test z-loss with ignored tokens: ignored tokens get zero gradients"""
        z_loss_weight = 1e-4
        tols = dtype_tols(torch.float32)

        for _ in range(3):
            B, SQ = 1, 128
            V = 64000
            ignore_positions = random.sample(range(SQ), 5)

            inp_test = torch.rand((B, SQ, V), dtype=torch.float32, device="cuda")
            tar_test = torch.randint(0, V, (B, SQ), device="cuda")
            for pos in ignore_positions:
                tar_test[0, pos] = -100

            inp_ref = inp_test.clone().detach().reshape(B * SQ, V)
            tar_ref = tar_test.clone().detach().reshape(B * SQ)

            inp_test.requires_grad_()
            inp_ref.requires_grad_()

            # Test
            test_loss = parallel_cross_entropy(
                inp_test, tar_test, reduce_loss=True,
                ignore_idx=-100, z_loss_weight=z_loss_weight,
            )

            # Reference: manually compute masked CE + masked z-loss
            mask = tar_ref != -100
            n_valid = mask.sum().float()
            ce_loss = F.cross_entropy(inp_ref, tar_ref, ignore_index=-100, reduction="mean")
            lse = torch.logsumexp(inp_ref, dim=-1)
            ref_z_loss = z_loss_weight * (lse[mask] ** 2).sum() / n_valid
            ref_loss = ce_loss + ref_z_loss

            torch.testing.assert_close(
                test_loss.to(torch.float64, device="cpu"),
                ref_loss.to(torch.float64, device="cpu"),
                **tols,
            )

            test_loss.backward()
            ref_loss.backward()

            test_grad = inp_test.grad.to(torch.float64, device="cpu")
            ref_grad = inp_ref.grad.to(torch.float64, device="cpu").reshape(test_grad.shape)
            torch.testing.assert_close(test_grad, ref_grad, **tols)

            # Verify ignored tokens have zero gradients
            for pos in ignore_positions:
                assert torch.all(
                    inp_test.grad[0, pos] == 0
                ), f"Ignored position {pos} should have zero gradients"

    def test_z_loss_zero_weight(self):
        """Confirm z_loss_weight=0.0 is identical to baseline (no numerical drift)"""
        for _ in range(3):
            B, SQ = 2, 64
            V = random.choice([64000, 128000])

            inp_data = torch.rand((B, SQ, V), dtype=torch.float32, device="cuda")
            tar = torch.randint(0, V, (B, SQ), device="cuda")

            inp_baseline = inp_data.clone().detach().requires_grad_()
            inp_zloss = inp_data.clone().detach().requires_grad_()

            # Baseline: no z_loss_weight argument
            loss_baseline = parallel_cross_entropy(
                inp_baseline, tar, reduce_loss=True,
            )
            # With z_loss_weight=0.0 explicitly
            loss_zloss = parallel_cross_entropy(
                inp_zloss, tar, reduce_loss=True, z_loss_weight=0.0,
            )

            # Loss must be bit-identical
            torch.testing.assert_close(loss_baseline, loss_zloss, rtol=0, atol=0)

            loss_baseline.backward()
            loss_zloss.backward()

            # Gradients must be bit-identical
            torch.testing.assert_close(
                inp_baseline.grad, inp_zloss.grad, rtol=0, atol=0,
            )
