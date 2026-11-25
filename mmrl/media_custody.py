"""Optional, synchronous queue-media evidence for native acceptance runs.

Read back published rows before finalization returns ownership for cleanup.
Persist hashes only; this is queue custody, not a learner-kernel observation.
"""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

FIELDS = ("pixel_values", "imgs_sizes", "num_frames")


def fingerprint(tensor):
    import torch

    value = tensor.detach().cpu().contiguous()
    return {"dtype": str(value.dtype), "shape": list(value.shape),
            "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}


def compare_parts(fields, parts):
    for name in FIELDS:
        flat, offset = fields[name].reshape(-1), 0
        for part in parts:
            expected = part["tensors"][name]
            end = offset + math.prod(expected["shape"])
            if end > flat.numel():
                raise ValueError("Canonical media is shorter than staged chain: " + name)
            if fingerprint(flat[offset:end].reshape(expected["shape"])) != expected:
                raise ValueError("Canonical media differs from staged chain: " + name)
            offset = end
        if offset != flat.numel():
            raise ValueError("Unexpected canonical media suffix: " + name)


def write_receipt(directory, value):
    """Atomic, immutable per-group records; retries cannot replace evidence."""
    directory = Path(directory)
    if not directory.is_absolute() or not directory.is_dir():
        raise ValueError("Media evidence directory must already exist and be absolute")
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    identity = json.dumps([value["job"], value["group_id"], value["rollout_ids"]])
    target = directory / (hashlib.sha256(identity.encode()).hexdigest() + ".json")
    with tempfile.NamedTemporaryFile(dir=directory, prefix=".receipt-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != payload:
                raise ValueError("Media receipt changed for the same finalized group") from None
    finally:
        temporary.unlink()
    return target


def observe_group(finalizer, group_id, rows, sample_ids, producer_tags):
    """Read staged and canonical tensors while the finalizer owns both."""
    import torch
    from nemo_gym.token_id_capture.staging.digest import hash_token_ids

    if not (len(rows) == len(sample_ids) == len(producer_tags)):
        raise ValueError("Media evidence row identities differ")
    calls, observed, excluded = {}, {}, []
    for row, sample, tags in zip(rows, sample_ids, producer_tags):
        if sample in observed or sample in excluded:
            raise ValueError("Duplicate canonical sample identity")
        if not row.valid:
            excluded.append(sample)
            continue
        items = finalizer._source.fetch_for_finalization(row.staging_keys)
        by_key = {item.staging_key: item for item in items}
        if set(by_key) != set(row.staging_keys) or len(by_key) != len(items):
            raise ValueError("Incomplete staged call readback")
        token_hash = hash_token_ids(row.token_ids)
        terminals = [item for item in items if item.snapshot.cumulative_hash == token_hash]
        if len(terminals) != 1:
            raise ValueError("Terminal call is not unique for canonical tokens")
        terminal = terminals[0]
        chain, seen, item = [], set(), terminal
        while item is not None:
            if item.staging_key in seen or item.snapshot.rollout_id != row.rollout_id:
                raise ValueError("Cyclic or cross-rollout media lineage")
            seen.add(item.staging_key)
            chain.append(item)
            parent = item.snapshot.parent_call_id
            item = by_key[row.rollout_id + "/" + parent] if parent else None
        chain.reverse()
        media_reports = []
        for item in chain:
            snapshot = item.snapshot
            report = {"staging_digest": snapshot.digest, "model_call_id": snapshot.model_call_id,
                      "parent_call_id": snapshot.parent_call_id, "weight_version": snapshot.weight_version,
                      "prev_len": snapshot.prev_len, "cum_len": snapshot.cum_len,
                      "media_present": item.media_present}
            if item.media_present:
                part = finalizer._source.fetch_media([item])[0]
                frames = (part.num_frames.reshape(-1).to(torch.int32) if part.num_frames is not None
                          else torch.ones(part.imgs_sizes.shape[0], dtype=torch.int32))
                report["tensors"] = {
                    "pixel_values": fingerprint(part.imgs.squeeze(0)),
                    "imgs_sizes": fingerprint(part.imgs_sizes.reshape(-1, 2).to(torch.int32)),
                    "num_frames": fingerprint(frames),
                }
                media_reports.append(report)
            if item.staging_key in calls and calls[item.staging_key] != report:
                raise ValueError("Staged call changed within a finalized group")
            calls[item.staging_key] = report

        def get(names):
            return finalizer._call_dp("get_samples", sample_ids=[sample],
                                      partition_id=finalizer._partition_id, select_fields=list(names))

        tokens = get(("input_ids", "input_lengths", "sample_mask"))
        length = int(tokens["input_lengths"].reshape(-1)[0].item())
        if length != len(row.token_ids) or not tokens["sample_mask"].reshape(-1)[0].item():
            raise ValueError("Published canonical row length or eligibility differs")
        actual_ids = tokens["input_ids"][0].reshape(-1).tolist()[:length]
        if hash_token_ids(actual_ids) != token_hash:
            raise ValueError("Published token IDs differ from finalized chain")
        common = {"rollout_id": row.rollout_id, "terminal_call": terminal.staging_key,
                  "token_hash": token_hash, "captured_calls": len(chain), "media_calls": len(media_reports)}
        if media_reports:
            data = get(FIELDS)
            fields = {name: data[name][0] for name in FIELDS}
            compare_parts(fields, media_reports)
            observed[sample] = common | {"pixel_and_geometry_bytes_equal": True,
                "field_fingerprints": {name: fingerprint(value) for name, value in fields.items()}}
        else:
            names = tags.get("multimodal_wire_fields")
            mask_name = "media_token_validity_mask"
            if (not isinstance(names, list) or mask_name not in names or
                    len(set(names)) != len(names) or not set(names) <= set(FIELDS) | {mask_name}):
                raise ValueError("Missing or unsupported text-only media field declaration")
            data, packed = get(names), {}
            for name in set(names) & set(FIELDS):
                geometry = tags.get(name + "__row_shapes", {}).get("shapes")
                count = data[name][0].numel()
                if count != 0 or geometry != []:
                    raise ValueError("Text-only row contains packed image data")
                packed[name] = {"numel": count, "row_shapes": geometry}
            mask = data[mask_name][0]
            if mask.dtype != torch.bool or mask.numel() < length or mask.count_nonzero().item():
                raise ValueError("Text-only row contains image positions")
            observed[sample] = common | {"text_only_queue_media_verified": True,
                "call_media_present": [False] * len(chain), "wire_fields": names,
                "packed_columns": packed, "media_token_mask": {"dtype": str(mask.dtype),
                    "numel": mask.numel(), "true_count": 0},
                "pixel_and_geometry_bytes_equal": False, "learner_dump_binding_verified": False}
    return {"schema_version": 1, "group_id": group_id, "rollout_ids": [r.rollout_id for r in rows],
            "calls": calls, "rows": observed, "invalid_samples": excluded,
            "observation_boundary": "published_queue_before_finalizer_returns",
            "learner_kernel_image_tensors_independently_verified": False}


def acceptance_job_identity():
    """Use the launcher's durable identity after Ray removes SLURM_* variables."""
    identities = [os.environ[name] for name in ("OCI_JOB_ID", "SLURM_JOB_ID")
                  if name in os.environ]
    if not identities or any(not value.isascii() or not value.isdecimal()
                             or int(value) <= 0 for value in identities):
        raise ValueError("Media custody acceptance requires an explicit Slurm job identity")
    if len(set(identities)) != 1:
        raise ValueError("Conflicting media custody job identities")
    return int(identities[0])


def maybe_observe_published_group(finalizer, group_id, rows, sample_ids, producer_tags):
    directory = os.environ.get("MMRL_MEDIA_CUSTODY_DIR")
    if not directory:
        return
    job = acceptance_job_identity()
    result = observe_group(finalizer, group_id, rows, sample_ids, producer_tags)
    result["job"] = job
    result["observer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_receipt(directory, result)
