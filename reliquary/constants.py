"""GRAIL Protocol Constants — V1 Verifiable Inference.

Immutable values that all network participants must agree on.
No os.getenv() overrides. Changes require coordinated deployment.
"""

# ────────────────  GRAIL PROOF VERSION  ────────────────

GRAIL_PROOF_VERSION = "v5"

# ────────────────  CRYPTOGRAPHIC CONSTANTS  ────────────────

# Mersenne prime for modular sketch arithmetic.
PRIME_Q = 2_147_483_647

# Number of random challenge positions per completion.
CHALLENGE_K = 32

# PRF domain labels for different randomness derivations.
RNG_LABEL = {"sketch": b"sketch", "open": b"open", "sat": b"sat"}

# Transformer layer index for hidden state extraction (-1 = last layer).
LAYER_INDEX = -1

# Batch size for proof computation (log-softmax / GRAIL commitments).
# Fixed: changing causes numerical divergence between miner and validator.
PROOF_BATCH_SIZE = 16

# Top-K activation selection for sketch computation.
PROOF_TOPK = 16

# Logarithmic bucketing: buckets per sign (16 total = 8 positive + 8 negative).
PROOF_NUM_BUCKETS = 8

# Bounded coefficient range for sketch robustness: r in [-127, 127].
PROOF_COEFF_RANGE = 127

# Sketch tolerance at position 0. Covers cross-GPU drift.
# Empirical max diff across ~300M positions = 3979. Base of 6000 gives ~50% headroom.
PROOF_SKETCH_TOLERANCE_BASE = 6000

# Sketch tolerance sqrt growth factor per position.
# tolerance(P) = base + growth * sqrt(P).
PROOF_SKETCH_TOLERANCE_GROWTH = 5.0

# Attention implementation forced across all model loading paths.
# Override with GRAIL_ATTN_IMPL for test envs without flash-attn compiled
# (e.g. "eager" or "sdpa"). Production runs must stay on flash_attention_2
# because sketch commitments are bit-sensitive to attention kernel variance.
import os as _os
ATTN_IMPLEMENTATION = _os.environ.get("GRAIL_ATTN_IMPL", "flash_attention_2")

# ────────────────  TIMING (CONSENSUS)  ────────────────

# Blocks per window — 5 blocks × 12s ≈ 60s matches SLOT_DEADLINE_SECONDS.
# All roles use this to determine window boundaries. With
# WEIGHT_SUBMISSION_INTERVAL=360, that yields ROLLING_WINDOWS=72 windows of
# scoring per on-chain weight submission, providing ~72× smoothing of miner
# scores over the epoch.
WINDOW_LENGTH = 5

# Bittensor block time target average (seconds).
BLOCK_TIME_SECONDS = 12

# Typical variance in block production time (seconds).
BLOCK_TIME_VARIANCE = 3

# Network latency allowance for file uploads (seconds).
NETWORK_UPLOAD_LATENCY = 30

# Grace period = block variance + upload latency.
UPLOAD_GRACE_PERIOD = BLOCK_TIME_VARIANCE + NETWORK_UPLOAD_LATENCY

# Buffer for future drand beacon (seconds).
DRAND_FUTURE_BUFFER = 30

# Buffer subtracted from per-window deadline to leave room for final submissions.
UPLOAD_BUFFER = NETWORK_UPLOAD_LATENCY

# ────────────────  ROLLOUT GENERATION  ────────────────

# Network-wide protocol cap on completion length.
MAX_NEW_TOKENS_PROTOCOL_CAP = 8192

# Maximum token sequence length in a single rollout (prompt + completion).
MAX_TOKENS_PER_ROLLOUT = MAX_NEW_TOKENS_PROTOCOL_CAP + 4096

# ────────────────  GRPO BATCHING  ────────────────

# Number of distinct prompts to derive per window from the beacon.
PROMPTS_PER_WINDOW = 8

# Completions per prompt to collect at the slot cap. When the slot reaches
# GROUP_SIZE accepted completions, it auto-finalises; no per-class quota is
# enforced during collection. Advantage scoring (rare-class pays more)
# provides the incentive for miners to pivot toward the under-represented
# class, self-balancing the slot without a hard cap. A slot that settles
# fully one-sided ({GROUP_SIZE, 0}) has std=0 → zero payout → its emission
# share burns.
GROUP_SIZE = 32

# Maximum completions allowed in a single submission. Equals GROUP_SIZE so
# one miner can, in theory, fill an entire slot solo — but the cross-miner
# prefix-dedup and atomic batch-verification make that strategy risky and
# compute-heavy. In practice miners produce small batches (default 4).
COMPLETIONS_PER_SUBMISSION = GROUP_SIZE

# Default batch size a miner produces per submission call. Miners may submit
# any size in [1, COMPLETIONS_PER_SUBMISSION]; this is just the out-of-box
# default used by the reference MiningEngine.
MINER_BATCH_SIZE = 4

# First N generated tokens that must be distinct across the 4 completions in a batch.
DIVERSITY_PREFIX_LEN = 8

# Default HTTP port the validator listens on for miner submissions.
VALIDATOR_HTTP_PORT = 8888

# Active environment name (resolved by reliquary.environment.load_environment).
ENVIRONMENT_NAME = "gsm8k"

# Per-slot collection deadline from window start. Slots finalize as soon as
# both class quotas are full OR this timeout is reached, whichever comes first.
SLOT_DEADLINE_SECONDS = 60

# UID that receives unused slot emission budget (the burn address).
UID_BURN = 0

# ────────────────  ECONOMIC / INCENTIVE  ────────────────

# Superlinear weighting exponent for sybil resistance.
# w_i proportional to s_i^p. Splitting into k identities yields k^(1-p) * s^p < s^p.
SUPERLINEAR_EXPONENT = 4.0

# Maximum unique rollouts per miner per window that count toward weight allocation.
UNIQUE_ROLLOUTS_CAP = 5000
UNIQUE_ROLLOUTS_CAP_ENABLED = True

# ────────────────  VALIDATION RULES  ────────────────

# File size bounds for valid rollout window files.
MIN_ROLLOUT_FILE_SIZE_BYTES = 200
MAX_ROLLOUT_FILE_SIZE_BYTES = 350 * 1024 * 1024  # 350 MB

# Soft check threshold for stochastic failures.
STOCHASTIC_CHECK_FAILURE_THRESHOLD = 0.51

# ────────────────  CONTINUOUS VALIDATION  ────────────────

# How often the validator polls for new state (seconds).
POLL_INTERVAL_SECONDS = 10

# ────────────────  WEIGHT SUBMISSION  ────────────────

WEIGHT_SUBMISSION_INTERVAL = 360  # Blocks between weight submissions

# ────────────────  STORAGE  ────────────────

CHECKPOINT_PREFIX = "reliquary/checkpoints/"

# ────────────────  DEPRECATED (GRPO REFACTOR)  ────────────────
# Kept importable to avoid breaking transitive imports during the rollout.
# These knobs no longer participate in any runtime decision and will be
# removed in a follow-up cleanup once no consumer references them.

MINER_SAMPLING_ENABLED = True
MINER_SAMPLE_RATE = 0.25
MINER_SAMPLE_MIN = 2
MINER_SAMPLE_MAX = 35

ROLLOUT_SAMPLE_RATE = 0.10
ROLLOUT_SAMPLE_MIN = 16

VERIFICATION_BATCH_SIZE = 16
BATCH_FAILURE_THRESHOLD = 0.30

FAILURE_LOOKBACK_WINDOWS = 14
USED_INDICES_MAX_AGE_WINDOWS = 100

MAX_ROLLOUTS_PER_FILE = 6000

DATASET_NAME = "karpathy/climbmix-400b-shuffle"
DATASET_SPLIT = "train"
