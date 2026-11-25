# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ledger and admission invariants for external-staging token capture.

These re-express the gate invariants that survive the gate's removal:
tri-state admission, commit ordering, same-call commit idempotency, and
fail-closed poisoning.
"""

from __future__ import annotations

import pytest

from nemo_gym.token_id_capture.fingerprint import FINGERPRINT_VERSION
from nemo_gym.token_id_capture.lineage import FileLineageStore, InMemoryLineageStore, _custody_columns
from nemo_gym.token_id_capture.protocols import CaptureLedger
from nemo_gym.token_id_capture.records import ParentResolutionStatus, compute_digest
from nemo_gym.token_id_capture.sink import (
    UNRESOLVED_PARENT_REASON,
    CaptureContext,
    reset_token_sink,
    resolve_parent,
    set_token_sink,
)
from nemo_gym.token_id_capture.staging.digest import (
    EMPTY_EXTRAS_DIGEST,
    compute_chain_hash,
    hash_token_ids,
)
from nemo_gym.token_id_capture.staging.records import CallRecord, CaptureLedgerCommit, RolloutManifest


USER_1 = {"role": "user", "content": "solve the task"}
ASSISTANT_1 = {"role": "assistant", "content": "first answer"}
USER_2 = {"role": "user", "content": "tool result"}
ASSISTANT_2 = {"role": "assistant", "content": "second answer"}
USER_3 = {"role": "user", "content": "follow up"}
ASSISTANT_SEEDED = {"role": "assistant", "content": "seeded turn nobody served"}

TOKENS_1 = list(range(900))
STAGING_DIGEST = "a" * 64
CHAIN_HASH_1 = compute_chain_hash(None, TOKENS_1)
CUMULATIVE_HASH_1 = hash_token_ids(TOKENS_1)


def _call_record(
    model_call_id: str,
    *,
    parent_call_id: str | None = None,
    prev_len: int = 0,
    cumulative_hash: str = CUMULATIVE_HASH_1,
    chain_hash: str = CHAIN_HASH_1,
    delta_len: int | None = None,
    admitted_at: float | None = 1_755_600_000.25,
) -> CallRecord:
    if delta_len is None:
        delta_len = len(TOKENS_1) - prev_len
    return CallRecord(
        model_call_id=model_call_id,
        parent_call_id=parent_call_id,
        staging_key=f"r1/{model_call_id}",
        weight_version=17,
        prev_len=prev_len,
        delta_len=delta_len,
        cum_len=prev_len + delta_len,
        digest=STAGING_DIGEST,
        extras_digest=EMPTY_EXTRAS_DIGEST,
        mode="text" if parent_call_id is None else "token_in",
        response_id=f"chatcmpl-{model_call_id}",
        admitted_at=admitted_at,
        chain_hash=chain_hash,
        cumulative_hash=cumulative_hash,
        fingerprint_version=FINGERPRINT_VERSION,
    )


def _commit(
    record: CallRecord,
    request_items: list[dict],
    response_items: list[dict],
    *,
    rollout_id: str = "r1",
    staging_chain: tuple[str, ...] | None = None,
) -> CaptureLedgerCommit:
    return CaptureLedgerCommit(
        rollout_id=rollout_id,
        record=record,
        staging_chain=staging_chain if staging_chain is not None else (record.staging_key,),
        request_items=request_items,
        response_items=response_items,
    )


async def _record_call_1(store, rollout_id: str = "r1") -> None:
    # Token-free custody row, exactly as the external commit hook writes it.
    await store.record(
        _commit(
            _call_record("c1"),
            [USER_1],
            [ASSISTANT_1],
            rollout_id=rollout_id,
            staging_chain=(f"{rollout_id}/c1",),
        )
    )


