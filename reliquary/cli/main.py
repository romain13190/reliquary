"""Reliquary CLI — mine and validate commands."""

import asyncio
import logging
import os

import typer

from reliquary.constants import ENVIRONMENT_NAME, VALIDATOR_HTTP_PORT

app = typer.Typer(name="reliquary", help="Reliquary — Verifiable Inference Subnet")


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@app.command()
def mine(
    use_drand: bool = typer.Option(True, help="Use drand for randomness"),
    network: str = typer.Option("finney", help="Bittensor network"),
    netuid: int = typer.Option(81, help="Subnet UID"),
    wallet_name: str = typer.Option("default", help="Wallet name"),
    hotkey: str = typer.Option("default", help="Hotkey name"),
    checkpoint: str = typer.Option(..., help="Model checkpoint path"),
    environment: str = typer.Option(ENVIRONMENT_NAME, help="Environment name"),
    validator_url: str = typer.Option(
        "",
        help=(
            "Override the validator URL (otherwise discovered from the metagraph). "
            "Useful for local testing — e.g. http://127.0.0.1:8888"
        ),
    ),
    log_level: str = typer.Option("INFO", help="Log level"),
):
    """Run Reliquary miner."""
    setup_logging(log_level)
    logger = logging.getLogger("reliquary.cli")

    os.environ["BT_NETWORK"] = network
    os.environ["NETUID"] = str(netuid)

    logger.info(
        "Starting Reliquary miner (network=%s, netuid=%d, env=%s)",
        network, netuid, environment,
    )

    async def _run():
        import bittensor as bt
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from reliquary.constants import ATTN_IMPLEMENTATION, WINDOW_LENGTH
        from reliquary.environment import load_environment
        from reliquary.infrastructure.chain import get_subtensor
        from reliquary.miner.engine import MiningEngine

        wallet = bt.Wallet(name=wallet_name, hotkey=hotkey)
        subtensor = await get_subtensor()

        logger.info("Loading models from %s...", checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        vllm_model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPLEMENTATION,
        ).to("cuda:0").eval()

        hf_model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPLEMENTATION,
        ).to("cuda:1").eval()

        env = load_environment(environment)
        engine = MiningEngine(
            vllm_model,
            hf_model,
            tokenizer,
            wallet,
            env,
            validator_url_override=validator_url or None,
        )

        logger.info("Miner ready. Entering main loop.")
        last_window = -1
        while True:
            try:
                current_block = await subtensor.get_current_block()
                window_start = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
                if window_start > last_window:
                    await engine.mine_window(
                        subtensor, window_start, use_drand=use_drand
                    )
                    last_window = window_start
                await asyncio.sleep(6)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error("Mining error: %s", e, exc_info=True)
                await asyncio.sleep(12)

    asyncio.run(_run())


@app.command()
def validate(
    use_drand: bool = typer.Option(True, help="Use drand for randomness"),
    network: str = typer.Option("finney", help="Bittensor network"),
    netuid: int = typer.Option(81, help="Subnet UID"),
    wallet_name: str = typer.Option("default", help="Wallet name"),
    hotkey: str = typer.Option("default", help="Hotkey name"),
    checkpoint: str = typer.Option(..., help="Model checkpoint path"),
    environment: str = typer.Option(ENVIRONMENT_NAME, help="Environment name"),
    http_host: str = typer.Option("0.0.0.0", help="HTTP bind address"),
    http_port: int = typer.Option(VALIDATOR_HTTP_PORT, help="HTTP listen port"),
    log_level: str = typer.Option("INFO", help="Log level"),
):
    """Run Reliquary validator."""
    setup_logging(log_level)
    logger = logging.getLogger("reliquary.cli")

    os.environ["BT_NETWORK"] = network
    os.environ["NETUID"] = str(netuid)

    logger.info(
        "Starting Reliquary validator (network=%s, netuid=%d, env=%s, http=%s:%d)",
        network, netuid, environment, http_host, http_port,
    )

    async def _run():
        import bittensor as bt
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from reliquary.constants import ATTN_IMPLEMENTATION
        from reliquary.environment import load_environment
        from reliquary.infrastructure.chain import get_subtensor
        from reliquary.validator.service import ValidationService

        wallet = bt.Wallet(name=wallet_name, hotkey=hotkey)
        subtensor = await get_subtensor()

        logger.info("Loading model from %s...", checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPLEMENTATION,
        ).to("cuda:0").eval()

        env = load_environment(environment)
        service = ValidationService(
            wallet,
            model,
            tokenizer,
            env,
            netuid,
            use_drand=use_drand,
            http_host=http_host,
            http_port=http_port,
        )
        await service.run(subtensor)

    asyncio.run(_run())


if __name__ == "__main__":
    app()
