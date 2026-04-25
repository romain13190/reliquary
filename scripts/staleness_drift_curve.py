#!/usr/bin/env python3
"""Staleness drift curve — LP deviation + sketch diff vs miner staleness.

For each k in 0..N, picks the (k+1)-th most recent published checkpoint as
the miner's model and the latest as the validator's. Generates rollouts on
MATH prompts with the miner model, then runs the validator's GRAIL sketch
+ LogprobValidator + DistributionValidator pipeline on each rollout.

Outputs a per-rollout CSV row with raw metrics so we can see where the
curve crosses LOGPROB_IS_EPS (0.01) and PROOF_SKETCH_TOLERANCE_BASE (3000).

Usage on a 2-GPU box (validator on cuda:0, miner on cuda:1):

    python scripts/staleness_drift_curve.py \\
        --repo-id aivolutionedge/reliquary-sn \\
        --last-n 8 \\
        --n-prompts 6 --m-rollouts 4 --max-tokens 192 \\
        --output /tmp/staleness_curve.csv

Single-GPU mode (slower, swaps both models on cuda:0):

    python scripts/staleness_drift_curve.py --single-gpu ...
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from statistics import median

import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from reliquary.constants import (
    ATTN_IMPLEMENTATION,
    CHALLENGE_K,
    DEFAULT_HF_REPO_ID,
    GRAIL_PROOF_VERSION,
    LAYER_INDEX,
    LOGPROB_IS_EPS,
    PROOF_SKETCH_TOLERANCE_BASE,
    PROOF_SKETCH_TOLERANCE_GROWTH,
    T_PROTO,
    TOP_K_PROTO,
    TOP_P_PROTO,
)
from reliquary.environment.math import MATHEnvironment
from reliquary.protocol.crypto import indices_from_root
from reliquary.protocol.grail_verifier import GRAILVerifier, adaptive_sketch_tolerance
from reliquary.shared.forward import forward_single_layer
from reliquary.shared.hf_compat import resolve_hidden_size
from reliquary.validator.verifier import (
    evaluate_token_distribution,
    verify_logprobs_claim,
)


def _load(repo_id: str, revision: str, device: torch.device):
    path = snapshot_download(repo_id=repo_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPLEMENTATION,
    ).to(device).eval()
    return model, path


def _generate_rollouts(model, tokenizer, prompt: str, m: int,
                      max_tokens: int, device: torch.device) -> list[dict]:
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_length = len(prompt_tokens)
    inp = torch.tensor([prompt_tokens] * m, device=device)
    with torch.no_grad():
        out = model.generate(
            inp,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=T_PROTO,
            top_p=TOP_P_PROTO,
            top_k=TOP_K_PROTO,
            pad_token_id=tokenizer.pad_token_id,
        )
    eos = tokenizer.eos_token_id
    rollouts = []
    for i in range(m):
        seq = out[i].tolist()
        gen = seq[prompt_length:]
        try:
            first_eos = gen.index(eos)
            gen = gen[: first_eos + 1]
        except ValueError:
            pass
        rollouts.append({
            "tokens": prompt_tokens + gen,
            "prompt_length": prompt_length,
            "completion_length": len(gen),
        })
    return rollouts


def _miner_commit(model, randomness: str, all_tokens: list[int],
                 prompt_length: int) -> dict:
    """Build a (sketches + claimed-logprobs) commit using the miner model."""
    hidden_dim = resolve_hidden_size(model)
    verifier = GRAILVerifier(hidden_dim=hidden_dim)
    r_vec = verifier.generate_r_vec(randomness)

    inp = torch.tensor([all_tokens], device=next(model.parameters()).device)
    with torch.no_grad():
        h, logits = forward_single_layer(model, inp, None, LAYER_INDEX)
    h = h[0]
    commitments = verifier.create_commitments_batch(h, r_vec)

    log_probs = torch.log_softmax(logits[0], dim=-1)
    token_lps = [
        log_probs[i - 1, all_tokens[i]].item()
        for i in range(prompt_length, len(all_tokens))
    ]
    return {
        "tokens": all_tokens,
        "commitments": commitments,
        "proof_version": GRAIL_PROOF_VERSION,
        "rollout": {
            "prompt_length": prompt_length,
            "completion_length": len(all_tokens) - prompt_length,
            "token_logprobs": token_lps,
        },
    }


def _validator_check(model, commit: dict, randomness: str) -> dict:
    """Run sketch + LP + distribution checks; return raw per-position diffs."""
    tokens = commit["tokens"]
    commitments = commit["commitments"]
    seq_len = len(tokens)

    hidden_dim = resolve_hidden_size(model)
    verifier = GRAILVerifier(hidden_dim=hidden_dim)
    r_vec = verifier.generate_r_vec(randomness)

    expected = min(CHALLENGE_K, seq_len)
    chal_idx = indices_from_root(tokens, randomness, seq_len, expected)

    inp = torch.tensor([tokens], device=next(model.parameters()).device)
    with torch.no_grad():
        h, logits_b = forward_single_layer(model, inp, None, LAYER_INDEX)
    h = h[0]
    logits = logits_b[0].detach().to("cpu")

    sketch_diffs = []
    sketch_passed = 0
    for idx in chal_idx:
        if idx >= seq_len:
            continue
        ok, diag = verifier.verify_commitment(
            h[idx], commitments[idx], r_vec, seq_len, idx
        )
        sketch_diffs.append(diag["sketch_diff"])
        sketch_passed += int(ok)

    rollout = commit["rollout"]
    lp_ok, lp_dev = verify_logprobs_claim(
        tokens=tokens,
        prompt_length=rollout["prompt_length"],
        completion_length=rollout["completion_length"],
        claimed_logprobs=rollout["token_logprobs"],
        logits=logits,
        challenge_randomness=randomness,
    )
    dist_ok, dist_metrics = evaluate_token_distribution(
        tokens=tokens,
        prompt_length=rollout["prompt_length"],
        completion_length=rollout["completion_length"],
        logits=logits,
        temperature=T_PROTO,
    )
    return {
        "sketch_diffs": sketch_diffs,
        "sketch_passed": sketch_passed,
        "sketch_checked": len(sketch_diffs),
        "lp_ok": lp_ok,
        "lp_dev": lp_dev,
        "dist_ok": dist_ok,
        "dist_median": dist_metrics.get("median") if dist_metrics else None,
        "dist_q10": dist_metrics.get("q10") if dist_metrics else None,
    }


def _list_recent_revisions(repo_id: str, n: int) -> list[str]:
    """Most recent commit SHAs (newest first), excluding the initial empty commit."""
    api = HfApi()
    commits = api.list_repo_commits(repo_id=repo_id, revision="main")
    return [c.commit_id for c in commits[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=DEFAULT_HF_REPO_ID)
    ap.add_argument("--last-n", type=int, default=8,
                    help="Number of recent revisions to test as miner ckpts. "
                         "Validator is the latest; miners are progressively "
                         "older (k=0 is identical, k=1 is one ckpt back, …).")
    ap.add_argument("--explicit-revisions", nargs="+", default=None,
                    help="Override --last-n: pass full list, validator first.")
    ap.add_argument("--n-prompts", type=int, default=6)
    ap.add_argument("--m-rollouts", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--randomness",
                    default=("c0ffee" * 11)[:64])
    ap.add_argument("--validator-gpu", type=int, default=0)
    ap.add_argument("--miner-gpu", type=int, default=1)
    ap.add_argument("--single-gpu", action="store_true",
                    help="Force both models onto --validator-gpu (slow).")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("staleness")

    if args.explicit_revisions:
        revs = args.explicit_revisions
    else:
        revs = _list_recent_revisions(args.repo_id, args.last_n)
    log.info("revisions (validator first): %s",
             [r[:8] for r in revs])

    val_dev = torch.device(f"cuda:{args.validator_gpu}")
    min_dev = (torch.device(f"cuda:{args.validator_gpu}") if args.single_gpu
               else torch.device(f"cuda:{args.miner_gpu}"))

    log.info("Loading validator ckpt %s", revs[0][:8])
    validator_model, val_path = _load(args.repo_id, revs[0], val_dev)
    tokenizer = AutoTokenizer.from_pretrained(val_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    env = MATHEnvironment()

    rows = []
    for k, rev in enumerate(revs):
        log.info("--- k=%d miner=%s ---", k, rev[:8])
        if args.single_gpu and k > 0:
            del miner_model  # noqa: F821
            torch.cuda.empty_cache()
        miner_model, _ = _load(args.repo_id, rev, min_dev)

        for prompt_idx in range(args.n_prompts):
            problem = env.get_problem(prompt_idx)
            rollouts = _generate_rollouts(
                miner_model, tokenizer,
                problem["prompt"], args.m_rollouts,
                args.max_tokens, min_dev,
            )
            for r_i, ro in enumerate(rollouts):
                if ro["completion_length"] == 0:
                    continue
                commit = _miner_commit(
                    miner_model, args.randomness,
                    ro["tokens"], ro["prompt_length"],
                )
                res = _validator_check(validator_model, commit, args.randomness)

                row = {
                    "k": k,
                    "rev": rev[:8],
                    "prompt_idx": prompt_idx,
                    "rollout_idx": r_i,
                    "completion_length": ro["completion_length"],
                    "sketch_passed": res["sketch_passed"],
                    "sketch_checked": res["sketch_checked"],
                    "sketch_diff_med": median(res["sketch_diffs"])
                                       if res["sketch_diffs"] else None,
                    "sketch_diff_max": max(res["sketch_diffs"])
                                       if res["sketch_diffs"] else None,
                    "lp_ok": res["lp_ok"],
                    "lp_dev": res["lp_dev"],
                    "dist_ok": res["dist_ok"],
                    "dist_median": res["dist_median"],
                    "dist_q10": res["dist_q10"],
                }
                rows.append(row)
                log.info(
                    "k=%d p=%d r=%d sk_pass=%d/%d sk_max=%s lp_dev=%.5f"
                    " dist_med=%s",
                    k, prompt_idx, r_i,
                    res["sketch_passed"], res["sketch_checked"],
                    res["sketch_diffs"] and max(res["sketch_diffs"]),
                    res["lp_dev"],
                    f"{res['dist_median']:.3f}" if res['dist_median'] else "n/a",
                )

        if not args.single_gpu:
            del miner_model
            torch.cuda.empty_cache()

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %d rows → %s", len(rows), args.output)

    # Quick stdout summary by k.
    print("\n=== summary by staleness k ===")
    print(f"thresholds: LP_EPS={LOGPROB_IS_EPS}  SKETCH_BASE={PROOF_SKETCH_TOLERANCE_BASE}\n")
    print(f"{'k':>3} {'n':>4} {'med(lp)':>10} {'p95(lp)':>10}"
          f" {'med(sk_max)':>12} {'lp_catch%':>10} {'sk_catch%':>10}")
    by_k: dict[int, list[dict]] = {}
    for r in rows:
        by_k.setdefault(r["k"], []).append(r)
    for k in sorted(by_k):
        ks = by_k[k]
        lps = sorted(r["lp_dev"] for r in ks)
        sk_max = sorted(r["sketch_diff_max"] or 0 for r in ks)
        n = len(ks)
        lp_catch = 100.0 * sum(1 for r in ks if not r["lp_ok"]) / n
        sk_catch = 100.0 * sum(
            1 for r in ks
            if r["sketch_passed"] < r["sketch_checked"]
        ) / n
        print(f"{k:>3} {n:>4} {median(lps):>10.5f}"
              f" {lps[int(0.95*(n-1))]:>10.5f}"
              f" {median(sk_max):>12.0f}"
              f" {lp_catch:>9.1f}% {sk_catch:>9.1f}%")


if __name__ == "__main__":
    main()
