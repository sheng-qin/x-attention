from xattn.src.utils import *
import torch
import math
import torch.nn.functional as F
from xattn.src.kernels import (
    flat_group_gemm,
    softmax_fuse_block_sum,
    flat_group_gemm_fuse_reshape,
)
from block_sparse_attn import block_sparse_attn_func


def finalize_prefill_simple_masks(
    simple_masks: torch.Tensor,
    q_block_num: int,
    num_kv_head: int,
    causal: bool,
    keep_sink: bool,
    keep_recent: bool,
) -> torch.Tensor:
    if causal:
        simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
            torch.tril(
                torch.ones(
                    q_block_num, q_block_num, dtype=bool, device=simple_masks.device
                ),
                diagonal=0,
            ),
            simple_masks[:, :, -q_block_num:, -q_block_num:],
            False,
        )
    if keep_sink:
        simple_masks[:, :, 0, :] = True
    if keep_recent:
        eye_matrix = torch.eye(q_block_num, device=simple_masks.device, dtype=bool)
        eye_matrix_expanded = (
            eye_matrix.unsqueeze(0)
            .unsqueeze(0)
            .expand(1, num_kv_head, q_block_num, q_block_num)
        )
        simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
            eye_matrix_expanded, True, simple_masks[:, :, -q_block_num:, -q_block_num:]
        )
    return simple_masks


def mean_pool_by_block(hidden_states: torch.Tensor, block_size: int) -> torch.Tensor:
    batch_size, head_num, seq_len, head_dim = hidden_states.shape
    block_num = (seq_len + block_size - 1) // block_size
    num_to_pad = block_num * block_size - seq_len
    if num_to_pad > 0:
        padded_states = F.pad(hidden_states, (0, 0, 0, num_to_pad), value=0)
    else:
        padded_states = hidden_states

    valid_mask = hidden_states.new_ones((1, 1, seq_len, 1))
    if num_to_pad > 0:
        valid_mask = F.pad(valid_mask, (0, 0, 0, num_to_pad), value=0)

    padded_states = padded_states.view(batch_size, head_num, block_num, block_size, head_dim)
    valid_mask = valid_mask.view(1, 1, block_num, block_size, 1)
    pooled_states = (padded_states * valid_mask).sum(dim=3)
    pooled_states = pooled_states / valid_mask.sum(dim=3).clamp_min(1)
    return pooled_states


def block_mean_estimate(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    block_size,
    norm=1,
    threshold=0.9,
    chunk_size=16384,
    causal=True,
    keep_sink=False,
    keep_recent=False,
    retention_policy="threshold",
    retention_ratio=None,
    retention_topk=None,
) -> torch.Tensor:
    batch_size, num_kv_head, k_len, head_dim = key_states.shape
    batch_size, num_q_head, q_len, head_dim = query_states.shape
    assert num_q_head == num_kv_head

    q_block_num = (q_len + block_size - 1) // block_size
    k_block_num = (k_len + block_size - 1) // block_size
    assert k_block_num >= q_block_num

    num_blocks_per_chunk = max(chunk_size // block_size, 1)
    q_chunk_num = (q_block_num + num_blocks_per_chunk - 1) // num_blocks_per_chunk
    offset_block_num = k_block_num - q_block_num

    pooled_query_states = mean_pool_by_block(query_states, block_size)
    pooled_key_states = mean_pool_by_block(key_states, block_size)

    key_positions = torch.arange(k_block_num, device=query_states.device)
    attn_sum_list = []
    simple_mask_list = []

    for chunk_idx in range(q_chunk_num):
        q_start = chunk_idx * num_blocks_per_chunk
        q_end = min(q_start + num_blocks_per_chunk, q_block_num)
        chunked_query = pooled_query_states[:, :, q_start:q_end, :]
        attn_sum = torch.matmul(
            chunked_query,
            pooled_key_states.transpose(2, 3),
        )
        attn_sum = attn_sum / math.sqrt(head_dim) / norm

        if causal:
            query_positions = offset_block_num + torch.arange(
                q_start, q_end, device=query_states.device
            )
            causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
            attn_sum = attn_sum.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0),
                float("-inf"),
            )

        attn_sum = F.softmax(attn_sum, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_sum = F.dropout(attn_sum, p=0, training=False)
        current_index = offset_block_num + q_start
        simple_mask = find_blocks_chunked(
            attn_sum,
            current_index,
            threshold if retention_policy == "threshold" else None,
            retention_topk if retention_policy == "topk" else None,
            decoding=False,
            mode="prefill",
            causal=causal,
            retention_ratio=retention_ratio if retention_policy == "ratio" else None,
        )
        attn_sum_list.append(attn_sum)
        simple_mask_list.append(simple_mask)

    attn_sums = torch.cat(attn_sum_list, dim=-2)
    simple_masks = torch.cat(simple_mask_list, dim=-2)
    simple_masks = finalize_prefill_simple_masks(
        simple_masks,
        q_block_num=q_block_num,
        num_kv_head=num_kv_head,
        causal=causal,
        keep_sink=keep_sink,
        keep_recent=keep_recent,
    )
    return attn_sums, simple_masks


def xattn_estimate(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    block_size,
    stride,
    norm=1,
    softmax=True,
    threshold=0.9,
    chunk_size=16384,
    select_mode="inverse",
    use_triton=True,
    causal=True,
    kdb: int = 1,
    keep_sink=False,
    keep_recent=False,
    block_mean_score=False,
    retention_policy="threshold",
    retention_ratio=None,
    retention_topk=None,
) -> torch.Tensor:
    batch_size, num_kv_head, k_len, head_dim = key_states.shape
    batch_size, num_q_head, q_len, head_dim = query_states.shape
    assert num_q_head == num_kv_head

    if block_mean_score:
        return block_mean_estimate(
            query_states,
            key_states,
            block_size=block_size,
            norm=norm,
            threshold=threshold,
            chunk_size=chunk_size,
            causal=causal,
            keep_sink=keep_sink,
            keep_recent=keep_recent,
            retention_policy=retention_policy,
            retention_ratio=retention_ratio,
            retention_topk=retention_topk,
        )

    k_num_to_pad = ((k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
    q_num_to_pad = ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len
    k_chunk_num = (k_len + k_num_to_pad) // chunk_size
    k_block_num = (k_len + k_num_to_pad) // block_size
    q_chunk_num = (q_len + q_num_to_pad) // chunk_size
    q_block_num = (q_len + q_num_to_pad) // block_size
    assert k_chunk_num >= q_chunk_num
    offset_token_chunk_num = k_chunk_num - q_chunk_num

    if k_num_to_pad > 0:
        pad_key_states = F.pad(key_states, (0, 0, 0, k_num_to_pad), value=0).to("cuda")
    else:
        pad_key_states = key_states
    if q_num_to_pad > 0:
        pad_query_states = F.pad(query_states, (0, 0, 0, q_num_to_pad), value=0).to(
            "cuda"
        )
    else:
        pad_query_states = query_states

    assert num_kv_head == num_q_head
    attn_sum_list = []
    simple_mask_list = []

    if use_triton and (
        "100" not in torch.cuda.get_device_properties(torch.cuda.current_device()).name
    ):
        use_triton = False
        print(
            "setting use triton to false. Triton kernel not surpported on this device"
        )

    reshaped_chunk_size = chunk_size // stride
    reshaped_block_size = block_size // stride
    k_reshaped_num_to_pad = k_num_to_pad // stride
    k_reshaped_seq_len = (k_len + k_num_to_pad) // stride
    q_reshaped_num_to_pad = q_num_to_pad // stride
    num_blocks_per_chunk = reshaped_chunk_size // reshaped_block_size
    if not use_triton:
        if select_mode == "random":
            perm_idx = torch.randperm(stride)
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_query = torch.cat(
                [
                    pad_query_states[:, :, perm_idx[i] :: stride, :]
                    for i in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "inverse" or select_mode == "":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_query = torch.cat(
                [
                    (pad_query_states[:, :, (stride - 1 - q) :: (stride * kdb), :])
                    for q in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "slash":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_query = torch.cat(
                [(pad_query_states[:, :, q::stride, :]) for q in range(stride)], dim=-1
            )
        elif select_mode == "double":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_key = reshaped_key + torch.cat(
                [reshaped_key[:, :, :, head_dim:], reshaped_key[:, :, :, 0:head_dim]],
                dim=-1,
            )
            reshaped_query = torch.cat(
                [
                    (pad_query_states[:, :, (stride - 1 - q) :: stride, :])
                    for q in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "triple":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_key = reshaped_key + torch.cat(
                [reshaped_key[:, :, :, head_dim:], reshaped_key[:, :, :, 0:head_dim]],
                dim=-1,
            )
            reshaped_key = reshaped_key + torch.cat(
                [reshaped_key[:, :, :, -head_dim:], reshaped_key[:, :, :, 0:-head_dim]],
                dim=-1,
            )
            reshaped_query = torch.cat(
                [
                    (pad_query_states[:, :, (stride - 1 - q) :: stride, :])
                    for q in range(stride)
                ],
                dim=-1,
            )
        assert reshaped_key.shape[-2] == k_reshaped_seq_len

    for chunk_idx in range(q_chunk_num):
        if use_triton:
            if kdb != 1:
                raise ValueError("use_triton and kdb cannot be used together")
            attn_weights_slice = flat_group_gemm_fuse_reshape(
                pad_query_states[
                    :,
                    :,
                    (chunk_idx * reshaped_chunk_size)
                    * stride : (chunk_idx * reshaped_chunk_size + reshaped_chunk_size)
                    * stride,
                    :,
                ],
                pad_key_states,
                stride,
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size,
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size
                + reshaped_chunk_size,
                is_causal=causal,
            )
            attn_sum = softmax_fuse_block_sum(
                attn_weights_slice,
                reshaped_block_size,
                min(4096, reshaped_block_size),
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size,
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size
                + reshaped_chunk_size,
                k_reshaped_seq_len - k_reshaped_num_to_pad,
                1.4426950408889634 / math.sqrt(head_dim) / stride / norm,
                is_causal=causal,
            )
        else:
            chunked_query = reshaped_query[
                :,
                :,
                (chunk_idx * reshaped_chunk_size)
                // kdb : (chunk_idx * reshaped_chunk_size + reshaped_chunk_size)
                // kdb,
                :,
            ]
            attn_weights_slice = torch.matmul(
                chunked_query,
                reshaped_key.transpose(2, 3),
            ).to("cuda")

            attn_weights_slice = (
                attn_weights_slice / math.sqrt(head_dim) / stride / norm
            )

            if causal:
                causal_mask = torch.zeros(
                    (
                        batch_size,
                        num_q_head,
                        reshaped_chunk_size,
                        reshaped_chunk_size * k_chunk_num,
                    ),
                    device=key_states.device,
                )
                causal_mask[:, :, :, (-k_reshaped_num_to_pad):] = float("-inf")
                chunk_start = (chunk_idx + offset_token_chunk_num) * reshaped_chunk_size
                chunk_end = chunk_start + reshaped_chunk_size
                causal_mask[:, :, :, chunk_start:chunk_end] = torch.triu(
                    torch.ones(
                        1,
                        num_q_head,
                        reshaped_chunk_size,
                        reshaped_chunk_size,
                        device=key_states.device,
                    )
                    * float("-inf"),
                    diagonal=1,
                )

                if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
                    causal_mask[:, :, (-(q_reshaped_num_to_pad // kdb)) :, :] = float(
                        "-inf"
                    )

                causal_mask[:, :, :, chunk_end:] = float("-inf")
                causal_mask = causal_mask[:, :, kdb - 1 :: kdb, :]
                attn_weights_slice = attn_weights_slice + causal_mask.to(
                    attn_weights_slice.device
                )

            if softmax:
                attn_weights_slice = F.softmax(
                    attn_weights_slice, dim=-1, dtype=torch.float32
                ).to(pad_query_states.dtype)
            else:
                attn_weights_slice = torch.exp(attn_weights_slice).to(
                    pad_query_states.dtype
                )
            attn_weights_slice = F.dropout(attn_weights_slice, p=0, training=False)

            if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
                attn_weights_slice[:, :, (-(q_reshaped_num_to_pad // kdb)) :, :] = 0

            attn_sum = (
                attn_weights_slice.view(
                    batch_size,
                    num_kv_head,
                    num_blocks_per_chunk,
                    reshaped_block_size // kdb,
                    -1,
                    reshaped_block_size,
                )
                .sum(dim=-1)
                .sum(dim=-2)
                .to("cuda")
            )
            del chunked_query
        
        simple_mask = find_blocks_chunked(
            attn_sum,
            k_block_num - q_block_num + chunk_idx * num_blocks_per_chunk,
            threshold if retention_policy == "threshold" else None,
            retention_topk if retention_policy == "topk" else None,
            decoding=False,
            mode="prefill",
            causal=causal,
            retention_ratio=retention_ratio if retention_policy == "ratio" else None,
        )

        attn_sum_list.append(attn_sum)
        simple_mask_list.append(simple_mask)

        del attn_weights_slice

    if not use_triton:
        del reshaped_query, reshaped_key
    attn_sums = torch.cat(attn_sum_list, dim=-2)
    simple_masks = torch.cat(simple_mask_list, dim=-2)
    simple_masks = finalize_prefill_simple_masks(
        simple_masks,
        q_block_num=q_block_num,
        num_kv_head=num_kv_head,
        causal=causal,
        keep_sink=keep_sink,
        keep_recent=keep_recent,
    )

    return attn_sums, simple_masks


def Xattention_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    stride,
    norm=1,
    threshold=0.8,
    block_size=128,
    use_triton=True,
    causal=True,
    kdb=1,
    chunk_size=None,
    keep_sink=False,
    keep_recent=False,
    block_mean_score=False,
    retention_policy="threshold",
    retention_ratio=None,
    retention_topk=None,
):
    batch_size, num_heads, k_len, head_dim = key_states.shape
    _, _, q_len, _ = query_states.shape

    q_block_num = (q_len + block_size - 1) // block_size
    k_block_num = (k_len + block_size - 1) // block_size
    if chunk_size is None:
        chunk_size = int(
            max(
                min(
                    max(2048, 1 << (k_len - 1).bit_length()),
                    128 * 1024 * 2048 // (1 << (k_len - 1).bit_length()),
                ),
                2048,
            )
        )
    attn_sums, approx_simple_mask = xattn_estimate(
        query_states,
        key_states,
        block_size=block_size,
        stride=stride,
        norm=norm,
        threshold=threshold,
        select_mode="inverse",
        use_triton=use_triton,
        causal=causal,
        chunk_size=chunk_size,
        kdb=kdb,
        keep_sink=keep_sink,
        keep_recent=keep_recent,
        block_mean_score=block_mean_score,
        retention_policy=retention_policy,
        retention_ratio=retention_ratio,
        retention_topk=retention_topk,
    )

    if query_states.device != key_states.device:
        key_states = key_states.to(query_states.device)
    if query_states.device != value_states.device:
        value_states = value_states.to(query_states.device)
    if approx_simple_mask.device != query_states.device:
        approx_simple_mask = approx_simple_mask.to(query_states.device)

    ####################
    assert block_size == 128
    assert batch_size == 1
    query_states = query_states.transpose(1, 2).view(q_len, num_heads, head_dim)
    key_states = key_states.transpose(1, 2).view(k_len, num_heads, head_dim)
    value_states = value_states.transpose(1, 2).view(k_len, num_heads, head_dim)
    q_cu_seq_lens = torch.tensor(
        [0, q_len], dtype=torch.int32, device=query_states.device
    )
    k_cu_seq_lens = torch.tensor(
        [0, k_len], dtype=torch.int32, device=query_states.device
    )
    head_mask_type = torch.tensor(
        [1 for _ in range(num_heads)], device=query_states.device, dtype=torch.int32
    )
    assert head_mask_type.device == query_states.device
    assert q_cu_seq_lens.device == query_states.device
    assert k_cu_seq_lens.device == query_states.device
    assert key_states.device == query_states.device
    assert value_states.device == query_states.device
    assert approx_simple_mask.device == query_states.device
    active_simple_mask = approx_simple_mask[:, :, :q_block_num, :k_block_num].contiguous()

    attn_output = block_sparse_attn_func(
        query_states,
        key_states,
        value_states,
        q_cu_seq_lens,
        k_cu_seq_lens,
        head_mask_type,
        None,
        active_simple_mask,
        q_len,
        k_len,
        p_dropout=0.0,
        deterministic=True,
        is_causal=causal,
    )
    attn_output = attn_output.view(batch_size, q_len, num_heads, head_dim).transpose(
        1, 2
    )
    ################################

    del query_states
    num_to_compute = (k_block_num + 1) * k_block_num / 2 * num_heads
    active_attn_sums = attn_sums[:, :, :q_block_num, :k_block_num].to(torch.float32)
    retained_approx_score = (
        active_attn_sums * active_simple_mask.to(torch.float32)
    ).sum() / active_attn_sums.sum().clamp_min(1e-12)
    
    print(f"approximated prefilling Computation: {approx_simple_mask.sum() / num_to_compute}")
    print(f"retained approximated attention score: {retained_approx_score}")
    del approx_simple_mask, attn_sums
    return attn_output
