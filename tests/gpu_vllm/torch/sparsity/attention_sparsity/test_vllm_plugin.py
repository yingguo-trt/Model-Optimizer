# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU tests for the vLLM sparse attention plugin (ModelOptSparseAttentionImpl).

Covers the integration-critical metadata translation done in
``ModelOptSparseAttentionImpl.forward``:

* ``query_start_loc``       -> ``b_start_loc`` / ``b_seq_len``
* ``seq_lens``              -> ``b_seq_len_k``
* backend-declared K/V axis -> key_cache / value_cache
* ``k_cache.shape[1]``      -> ``page_size``

Asserted against a contiguous reference call to the underlying Triton kernel.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
from vllm.v1.attention.backends.flashinfer import FlashInferBackend, FlashInferImpl

from modelopt.torch.kernels.common.attention import IS_AVAILABLE as TRITON_KERNEL_AVAILABLE
from modelopt.torch.sparsity.attention_sparsity.plugins import vllm as vllm_plugin

if TRITON_KERNEL_AVAILABLE:
    from modelopt.torch.kernels.common.attention import attention as triton_attention
    from modelopt.torch.kernels.sparsity.attention.calibrate import attention_calibrate

_ACTIVE_PREFILL_SPARSE_KW = {
    "sparsity_n": 2,
    "sparsity_m": 4,
    "dense_sink_tokens": 0,
    "dense_recent_tokens": 0,
}


def _make_backend_paged_cache(k_cache, v_cache):
    layout = vllm_plugin._flash_attention_kv_cache_layout()
    if layout == "kv-first":
        return torch.stack([k_cache, v_cache], dim=0)
    if layout == "blocks-first":
        return torch.stack([k_cache, v_cache], dim=1)
    return torch.cat([k_cache, v_cache], dim=-1).transpose(1, 2)


def _make_paged_cache(k, v, b_start_loc, b_seq_len, num_kv_heads, head_dim, page_size):
    """Scatter contiguous K/V into the installed vLLM paged-cache layout.

    Returns a single ``kv_cache`` tensor with the backend-declared K/V axis.
    """
    batch = b_seq_len.shape[0]
    device, dtype = k.device, k.dtype

    blocks_per_seq = [(int(b_seq_len[b].item()) + page_size - 1) // page_size for b in range(batch)]
    num_blocks = sum(blocks_per_seq)
    max_blocks = max(blocks_per_seq)

    k_cache = torch.zeros(num_blocks, page_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.zeros(batch, max_blocks, device=device, dtype=torch.int32)

    g = 0
    for b in range(batch):
        start = int(b_start_loc[b].item())
        slen = int(b_seq_len[b].item())
        for blk in range(blocks_per_seq[b]):
            block_table[b, blk] = g
            ts = blk * page_size
            te = min(ts + page_size, slen)
            n = te - ts
            k_cache[g, :n] = k[start + ts : start + te]
            v_cache[g, :n] = v[start + ts : start + te]
            g += 1

    kv_cache = _make_backend_paged_cache(k_cache, v_cache)
    return kv_cache, block_table


def _make_impl(num_heads, head_dim, num_kv_heads):
    """Construct ModelOptSparseAttentionImpl with minimal valid kwargs."""
    return vllm_plugin.ModelOptSparseAttentionImpl(
        num_heads=num_heads,
        head_size=head_dim,
        scale=1.0 / (head_dim**0.5),
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
    )


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("q_len", [17, 1], ids=["prefill", "decode"])
@pytest.mark.parametrize("calibrate", [True, False], ids=["calibration", "serving"])
def test_flashinfer_packed_cache_matches_contiguous(dtype, q_len, calibrate):
    """Native packed writes feed real calibration and serving kernels correctly."""
    if getattr(FlashInferBackend, "forward_includes_kv_cache_update", True):
        pytest.skip("Packed caches require native pre-forward KV updates")

    torch.manual_seed(7)
    seq_len, num_heads, num_kv_heads, head_dim, page_size = 300, 4, 2, 64, 16
    num_pages = (seq_len + page_size - 1) // page_size
    num_blocks = num_pages + 2
    q = torch.ones(q_len + 2, num_heads, head_dim, device="cuda", dtype=dtype)
    q += torch.randn_like(q) * 0.01
    k = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=dtype) * 0.01
    k[:128] += 0.5
    v = torch.randn_like(k) * 0.01
    v[:128] += 0.25
    v[128:] += 2.0
    trials = [1e-3, 1e-1, 5e-1]
    scale = 1.0 / (head_dim**0.5)
    locs = torch.zeros(1, device="cuda", dtype=torch.int32)
    q_lens = torch.tensor([q_len], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    block_table = (torch.arange(num_pages, device="cuda", dtype=torch.int32).flip(0) + 2)[None, :]
    positions = torch.arange(seq_len, device="cuda", dtype=torch.int64)
    slots = block_table[0, positions // page_size].to(torch.int64) * page_size
    slots += positions % page_size
    cache_shape = (num_blocks, page_size, num_kv_heads, head_dim)
    key_cache = k.new_zeros(cache_shape)
    value_cache = v.new_zeros(cache_shape)
    key_cache.view(-1, num_kv_heads, head_dim)[slots] = k
    value_cache.view(-1, num_kv_heads, head_dim)[slots] = v

    packed = k.new_zeros(num_blocks, num_kv_heads, page_size, 2 * head_dim)
    layer = SimpleNamespace(
        _k_scale=torch.ones((), device="cuda"),
        _v_scale=torch.ones((), device="cuda"),
    )
    writer = object.__new__(FlashInferImpl)
    writer.__dict__.update(
        head_size=head_dim,
        num_kv_heads=num_kv_heads,
        cache_dtype="auto",
        is_kvcache_nvfp4=False,
        kv_sharing_target_layer_name=None,
    )
    writer.do_kv_cache_update(layer, k, v, packed, slots)
    packed_key, packed_value = packed.transpose(1, 2).split(head_dim, dim=-1)
    torch.testing.assert_close(packed_key, key_cache, rtol=0, atol=0)
    torch.testing.assert_close(packed_value, value_cache, rtol=0, atol=0)

    ref_kw = {
        "b_start_loc": locs,
        "b_seq_len": q_lens,
        "max_input_len": q_len,
        "is_causal": q_len > 1,
        "softmax_scale": scale,
        "b_start_loc_k": locs,
        "b_seq_len_k": seq_lens,
        "max_input_len_k": seq_len,
    }
    dense_ref, counters_ref = attention_calibrate(
        q[:q_len], k, v, threshold_trials=trials, **ref_kw
    )
    assert counters_ref[:, 0].tolist() == [num_heads * 3] * len(trials)
    assert counters_ref[1, 1].item() > 0
    expected = (
        dense_ref
        if calibrate
        else triton_attention(q[:q_len], k, v, skip_softmax_threshold=trials[1], **ref_kw)
    )
    if not calibrate:
        assert not torch.allclose(expected, dense_ref, rtol=1e-3, atol=1e-3)

    metadata = SimpleNamespace(
        use_cascade=False,
        _modelopt_block_table=block_table,
        _modelopt_seq_lens=seq_lens,
        _modelopt_query_start_loc=torch.tensor([0, q_len], device="cuda", dtype=torch.int32),
        _modelopt_num_actual_tokens=q_len,
        _modelopt_max_query_len=q_len,
        _modelopt_max_seq_len=seq_len,
        _modelopt_causal=q_len > 1,
        slot_mapping=slots[-q_len:],
    )
    legacy = torch.stack((key_cache, value_cache), dim=1)
    hnd = legacy.permute(0, 1, 3, 2, 4).contiguous().permute(0, 1, 3, 2, 4)

    def native_forward(*_args, **_kwargs):
        raise AssertionError("Active ModelOpt attention must not use native fallback")

    for kv_cache in (legacy, hnd, packed):
        impl = SimpleNamespace(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            scale=scale,
            sparse_kw={"skip_softmax_threshold": trials[1]},
            quant_kw={"p_qdq": None, "p_qdq_amax": 1.0, "v_qdq": None, "v_qdq_amax": None},
            _calibrate=calibrate,
            _calib_threshold_trials=trials,
            _calib_records=[],
        )
        output = torch.full_like(q, -123)
        actual = vllm_plugin._flashinfer_forward(
            impl,
            native_forward,
            layer,
            q,
            k[-q_len:],
            v[-q_len:],
            kv_cache,
            metadata,
            output=output,
        )
        assert actual is output
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual[:q_len], expected, rtol=1e-3, atol=1e-3)
        assert torch.equal(actual[q_len:], torch.full_like(actual[q_len:], -123))
        if calibrate:
            assert impl._calib_records == [
                {
                    "phase": "prefill" if q_len > 1 else "decode",
                    "sample_length": seq_len,
                    "total_tiles": counters_ref[:, 0].tolist(),
                    "skipped_tiles": counters_ref[:, 1].tolist(),
                }
            ]


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestModelOptSparseAttentionImpl:
    """Verify forward() metadata translation matches a contiguous reference."""

    def test_prefill_matches_contiguous(self):
        """Prefill: paged forward == contiguous Triton call on the same K/V."""
        batch = 2
        seq_len = 64
        num_heads, num_kv_heads, head_dim = 4, 2, 64
        page_size = 16
        total = batch * seq_len
        dtype = torch.float16

        torch.manual_seed(0)
        q = torch.randn(total, num_heads, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(total, num_kv_heads, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(total, num_kv_heads, head_dim, device="cuda", dtype=dtype)

        # Per-sequence offsets / lengths.
        seq_lens = torch.tensor([seq_len, seq_len], device="cuda", dtype=torch.int32)
        # vLLM-style cumulative query_start_loc has shape [batch + 1].
        query_start_loc = torch.tensor([0, seq_len, 2 * seq_len], device="cuda", dtype=torch.int32)
        b_start_loc = query_start_loc[:batch]
        b_seq_len = seq_lens

        # Contiguous reference output (what the kernel would return without paging).
        out_ref = triton_attention(
            q,
            k,
            v,
            b_start_loc,
            b_seq_len,
            seq_len,
            softmax_scale=1.0 / (head_dim**0.5),
            **_ACTIVE_PREFILL_SPARSE_KW,
        )

        # Build the paged cache using the installed backend's K/V axis.
        kv_cache, block_table = _make_paged_cache(
            k, v, b_start_loc, b_seq_len, num_kv_heads, head_dim, page_size
        )

        attn_metadata = SimpleNamespace(
            num_actual_tokens=total,
            max_query_len=seq_len,
            max_seq_len=seq_len,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=block_table,
        )

        impl = _make_impl(num_heads, head_dim, num_kv_heads)
        impl.sparse_kw = _ACTIVE_PREFILL_SPARSE_KW
        output = torch.empty_like(q)
        out_paged = impl.forward(
            layer=None,
            query=q,
            key=k,
            value=v,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
        )

        torch.testing.assert_close(out_paged, out_ref, rtol=1e-2, atol=1e-2)

    def test_chunked_prefill_is_forwarded_to_kernel(self):
        """Chunked prefill metadata is handled by the suffix-aware causal mask."""
        impl = _make_impl(num_heads=2, head_dim=64, num_kv_heads=2)
        impl.sparse_kw = _ACTIVE_PREFILL_SPARSE_KW
        attn_metadata = SimpleNamespace(
            num_actual_tokens=4,
            max_query_len=4,  # chunk length
            max_seq_len=16,  # full sequence length > chunk
            query_start_loc=torch.tensor([0, 4], device="cuda", dtype=torch.int32),
            seq_lens=torch.tensor([16], device="cuda", dtype=torch.int32),
            block_table=torch.zeros(1, 1, device="cuda", dtype=torch.int32),
        )
        q = torch.zeros(4, 2, 64, device="cuda", dtype=torch.float16)
        k_cache = torch.zeros(1, 16, 2, 64, device="cuda", dtype=torch.float16)
        kv_cache = _make_backend_paged_cache(k_cache, torch.zeros_like(k_cache))
        out = impl.forward(
            layer=None,
            query=q,
            key=q,
            value=q,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=torch.empty_like(q),
        )

        assert torch.isfinite(out).all()

    def test_mixed_prefill_decode_is_forwarded_to_kernel(self):
        """Mixed prefill/decode batches are valid with suffix-aware causal masking."""
        prefill_len = 64
        decode_q_len = 1
        total_q = prefill_len + decode_q_len
        num_heads, num_kv_heads, head_dim = 2, 2, 64
        page_size = 16
        dtype = torch.float16

        q = torch.zeros(total_q, num_heads, head_dim, device="cuda", dtype=dtype)

        # Sequence 0 is a full prefill. Sequence 1 is decode with one query
        # token but a longer KV cache. max_query_len == max_seq_len, so the
        # older max-only guard would not catch this mixed batch.
        seq_lens = torch.tensor([prefill_len, prefill_len], device="cuda", dtype=torch.int32)
        query_start_loc = torch.tensor([0, prefill_len, total_q], device="cuda", dtype=torch.int32)
        b_start_loc_k = torch.tensor([0, prefill_len], device="cuda", dtype=torch.int32)
        k = torch.zeros(prefill_len * 2, num_kv_heads, head_dim, device="cuda", dtype=dtype)
        v = torch.zeros_like(k)
        kv_cache, block_table = _make_paged_cache(
            k, v, b_start_loc_k, seq_lens, num_kv_heads, head_dim, page_size
        )

        attn_metadata = SimpleNamespace(
            num_actual_tokens=total_q,
            max_query_len=prefill_len,
            max_seq_len=prefill_len,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=block_table,
        )

        impl = _make_impl(num_heads, head_dim, num_kv_heads)
        impl.sparse_kw = _ACTIVE_PREFILL_SPARSE_KW
        out = impl.forward(
            layer=None,
            query=q,
            key=q,
            value=q,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=torch.empty_like(q),
        )

        assert torch.isfinite(out).all()

    def test_decode_delegates_to_vllm(self, monkeypatch):
        """Decode-only sparse work is not routed through the ModelOpt paged kernel."""
        batch = 2
        q_len = 1
        kv_lens = torch.tensor([17, 33], device="cuda", dtype=torch.int32)
        total_q = batch * q_len
        num_heads, num_kv_heads, head_dim = 4, 2, 64
        page_size = 16
        dtype = torch.float16

        torch.manual_seed(2)
        q = torch.randn(total_q, num_heads, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(
            int(kv_lens.sum().item()), num_kv_heads, head_dim, device="cuda", dtype=dtype
        )
        v = torch.randn_like(k)

        query_start_loc = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
        kv_start_loc = torch.tensor([0, int(kv_lens[0].item())], device="cuda", dtype=torch.int32)

        kv_cache, block_table = _make_paged_cache(
            k, v, kv_start_loc, kv_lens, num_kv_heads, head_dim, page_size
        )
        attn_metadata = SimpleNamespace(
            num_actual_tokens=total_q,
            max_query_len=q_len,
            max_seq_len=int(kv_lens.max().item()),
            query_start_loc=query_start_loc,
            seq_lens=kv_lens,
            block_table=block_table,
        )

        impl = _make_impl(num_heads, head_dim, num_kv_heads)
        impl.sparse_kw = _ACTIVE_PREFILL_SPARSE_KW
        output = torch.empty_like(q)
        called = {}

        def fake_forward(
            self,
            layer,
            query,
            key,
            value,
            kv_cache_arg,
            attn_metadata_arg,
            output_arg=None,
            output_scale=None,
            output_block_scale=None,
        ):
            called["attn_metadata"] = attn_metadata_arg
            output_arg.fill_(9)
            return output_arg

        monkeypatch.setattr(FlashAttentionImpl, "forward", fake_forward)

        result = impl.forward(
            layer=None,
            query=q,
            key=q,
            value=q,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
        )
        assert called["attn_metadata"] is attn_metadata
        assert result is output
        assert torch.all(result == 9)

    def test_profiling_run_returns_zeros(self):
        """attn_metadata=None (vLLM profiling pass) must zero-fill output and return."""
        impl = _make_impl(num_heads=2, head_dim=64, num_kv_heads=2)
        output = torch.full((4, 2, 64), 7.0, device="cuda", dtype=torch.float16)
        result = impl.forward(
            layer=None,
            query=output,
            key=output,
            value=output,
            kv_cache=torch.empty(0),
            attn_metadata=None,
            output=output,
        )
        assert torch.all(result == 0)

    def test_page_size_inferred_from_k_cache(self):
        """page_size passed to the kernel must equal k_cache.shape[1]."""
        # Use the smallest valid power-of-two page_size to confirm it's not hardcoded.
        seq_len = 32
        num_heads, num_kv_heads, head_dim = 2, 2, 64
        page_size = 8  # deliberately != default 16
        dtype = torch.float16

        torch.manual_seed(1)
        q = torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=dtype)
        b_start_loc = torch.tensor([0], device="cuda", dtype=torch.int32)
        b_seq_len = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
        query_start_loc = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)

        out_ref = triton_attention(
            q,
            k,
            v,
            b_start_loc,
            b_seq_len,
            seq_len,
            softmax_scale=1.0 / (head_dim**0.5),
            **_ACTIVE_PREFILL_SPARSE_KW,
        )

        kv_cache, block_table = _make_paged_cache(
            k, v, b_start_loc, b_seq_len, num_kv_heads, head_dim, page_size
        )
        assert kv_cache.shape[2] == page_size

        attn_metadata = SimpleNamespace(
            num_actual_tokens=seq_len,
            max_query_len=seq_len,
            max_seq_len=seq_len,
            query_start_loc=query_start_loc,
            seq_lens=b_seq_len,
            block_table=block_table,
        )

        impl = _make_impl(num_heads, head_dim, num_kv_heads)
        impl.sparse_kw = _ACTIVE_PREFILL_SPARSE_KW
        output = torch.empty_like(q)
        out_paged = impl.forward(
            layer=None,
            query=q,
            key=k,
            value=v,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
        )
        torch.testing.assert_close(out_paged, out_ref, rtol=1e-2, atol=1e-2)
