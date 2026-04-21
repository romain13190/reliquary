"""v2.1 window state-machine smoke: open → seal → train → publish → ready.

Fully-mocked — no R2, no torch, no real drand. Exercises the
orchestration contract: _open_window advances window_n, seal_event
drives _train_and_publish, checkpoint_n increments, state ends on READY.
"""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reliquary.constants import B_BATCH, M_ROLLOUTS
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RolloutSubmission,
    WindowState,
)


@dataclass
class _FakeEnv:
    @property
    def name(self): return "fake"
    def __len__(self): return 1000
    def get_problem(self, i): return {"prompt": f"p{i}", "ground_truth": "", "id": f"p{i}"}
    def compute_reward(self, p, c): return 1.0 if "WIN" in c else 0.0


class _FakeWallet:
    class _Hk:
        ss58_address = "5FHk"
        @staticmethod
        def sign(d): return b"sig"
    hotkey = _Hk()


def _rollouts(k):
    return [
        RolloutSubmission(
            tokens=[1, 2, 3], reward=1.0 if i < k else 0.0,
            commit={
                "tokens": [1, 2, 3], "proof_version": "v5",
                "completion_text_for_test": "WIN" if i < k else "lose",
            },
        )
        for i in range(M_ROLLOUTS)
    ]


def _make_service(tmp_path, checkpoint_hash="sha256:cp"):
    """Construct a ValidationService with state paths in tmp_path and
    a mocked checkpoint store (no HF) + training stub.

    Sets publish_every=1 so every window triggers a publish, keeping
    existing test assertions simple.
    """
    from reliquary.validator.service import ValidationService
    from reliquary.validator.state_persistence import ValidatorState
    from reliquary.validator.checkpoint import ManifestEntry

    svc = ValidationService(
        wallet=_FakeWallet(), model=MagicMock(), tokenizer=MagicMock(),
        env=_FakeEnv(), netuid=99,
    )
    svc._state_path = str(tmp_path / "state.json")
    svc._state = ValidatorState(svc._state_path)
    # Publish on every window so counter tests stay simple.
    svc._publish_every = 1

    # Mock checkpoint store — start with a manifest so batcher gets a revision,
    # then return incrementing ManifestEntries on publish.
    svc._checkpoint_store = MagicMock()
    svc._checkpoint_store.current_manifest = MagicMock(return_value=ManifestEntry(
        checkpoint_n=0,
        repo_id="aivolutionedge/reliquary-sn",
        revision=checkpoint_hash,
        signature="ed25519:sig0",
    ))
    svc._checkpoint_store.publish = AsyncMock(side_effect=lambda checkpoint_n, model: ManifestEntry(
        checkpoint_n=checkpoint_n,
        repo_id="aivolutionedge/reliquary-sn",
        revision=f"rev_sha_{checkpoint_n:03d}",
        signature=f"ed25519:sig{checkpoint_n}",
    ))

    # Mock R2 archive upload.
    import reliquary.validator.service as svc_mod
    svc_mod.storage.upload_window_dataset = AsyncMock(return_value=True)

    return svc


def _patch_open_grpo_window(svc):
    """Return a context manager that replaces open_grpo_window in the service
    module with a version that injects mock verifiers (no torch, no crypto).

    The mock batcher uses the same GrpoWindowBatcher constructor but overrides
    the three verifier callables so accept_submission works without torch.
    """
    import reliquary.validator.service as svc_mod
    from reliquary.validator.batcher import GrpoWindowBatcher
    from reliquary.validator.cooldown import CooldownMap
    from reliquary.constants import BATCH_PROMPT_COOLDOWN_WINDOWS

    real_open = svc_mod.open_grpo_window

    def _mock_open(window_start, current_round, env, model, *, cooldown_map, tokenizer, bootstrap=False):
        return GrpoWindowBatcher(
            window_start=window_start,
            current_round=current_round,
            env=env,
            model=model,
            cooldown_map=cooldown_map,
            bootstrap=bootstrap,
            # Stub out torch-dependent verifiers.
            verify_commitment_proofs_fn=lambda c, m, r: (True, 1, 1),
            verify_signature_fn=lambda c, h: True,
            verify_proof_version_fn=lambda c: c.get("proof_version") == "v5",
            # Decode via the commit dict like existing smoke test does.
            completion_text_fn=lambda rollout: rollout.commit.get("completion_text_for_test", ""),
        )

    return patch.object(svc_mod, "open_grpo_window", side_effect=_mock_open)


@pytest.mark.asyncio
async def test_one_window_lap_bumps_counters(tmp_path):
    """Open → manually fire seal_event → train_and_publish →
    window_n and checkpoint_n both bumped, state = READY."""
    svc = _make_service(tmp_path)

    initial_wn = svc._state.window_n
    initial_cn = svc._state.checkpoint_n

    with _patch_open_grpo_window(svc):
        svc._open_window()
    assert svc._current_window_state == WindowState.OPEN
    assert svc._state.window_n == initial_wn + 1

    # Simulate the B-th valid submission arriving.
    svc._active_batcher.seal_event.set()

    await svc._train_and_publish()

    assert svc._state.checkpoint_n == initial_cn + 1
    assert svc._current_window_state == WindowState.READY
    assert svc._active_batcher is None
    svc._checkpoint_store.publish.assert_awaited_once()


@pytest.mark.asyncio
async def test_submission_with_matching_hash_accepted_during_open(tmp_path):
    """Inject an 8-submission batch into an OPEN batcher; seal_event fires."""
    svc = _make_service(tmp_path, checkpoint_hash="sha256:cpA")

    # Open a window (hash wired to current_manifest by _open_window)
    with _patch_open_grpo_window(svc):
        svc._open_window()
    batcher = svc._active_batcher
    assert batcher.current_checkpoint_hash == "sha256:cpA"

    # Feed B distinct submissions (k=4 → in zone)
    for i in range(B_BATCH):
        req = BatchSubmissionRequest(
            miner_hotkey=f"hk{i}",
            prompt_idx=i,
            window_start=batcher.window_start,
            signed_round=batcher.current_round,
            merkle_root="00" * 32,
            rollouts=_rollouts(k=4),
            checkpoint_hash="sha256:cpA",
        )
        resp = batcher.accept_submission(req)
        assert resp.accepted, f"unexpected reject for hk{i}: {resp.reason}"

    # seal_event should now be set.
    assert batcher.seal_event.is_set()


@pytest.mark.asyncio
async def test_submission_with_wrong_hash_rejected(tmp_path):
    svc = _make_service(tmp_path, checkpoint_hash="sha256:cpA")
    with _patch_open_grpo_window(svc):
        svc._open_window()
    batcher = svc._active_batcher

    req = BatchSubmissionRequest(
        miner_hotkey="hk", prompt_idx=0,
        window_start=batcher.window_start,
        signed_round=batcher.current_round,
        merkle_root="00" * 32,
        rollouts=_rollouts(k=4),
        checkpoint_hash="sha256:stale",  # mismatched
    )
    resp = batcher.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason.value == "wrong_checkpoint"


@pytest.mark.asyncio
async def test_timeout_partial_seal_path(tmp_path):
    """Window with only 3 valid submissions → seal_batch returns partial."""
    svc = _make_service(tmp_path, checkpoint_hash="sha256:cpA")
    with _patch_open_grpo_window(svc):
        svc._open_window()
    batcher = svc._active_batcher

    # Feed only 3 submissions
    for i in range(3):
        req = BatchSubmissionRequest(
            miner_hotkey=f"hk{i}", prompt_idx=i,
            window_start=batcher.window_start,
            signed_round=batcher.current_round,
            merkle_root="00" * 32,
            rollouts=_rollouts(k=4),
            checkpoint_hash="sha256:cpA",
        )
        batcher.accept_submission(req)

    # seal_event NOT set (fewer than B)
    assert not batcher.seal_event.is_set()

    # But _train_and_publish still advances the state machine (simulating
    # the timeout path — run loop calls seal_batch regardless).
    await svc._train_and_publish()
    assert svc._current_window_state == WindowState.READY
    assert svc._state.checkpoint_n == 1  # still bumped
