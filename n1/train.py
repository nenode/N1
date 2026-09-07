import argparse
import random
from pathlib import Path

import torch

from .model import GPT, GPTConfig


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    inputs = torch.stack([data[start : start + block_size] for start in starts]).to(device=device, dtype=torch.long)
    targets = torch.stack([data[start + 1 : start + block_size + 1] for start in starts]).to(
        device=device, dtype=torch.long
    )
    return inputs, targets


def load_text(args: argparse.Namespace) -> bytearray:
    if args.dataset:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError(
                "Hugging Face datasets is required for --dataset; install the project dependencies first"
            ) from error

        dataset_args = {"path": args.dataset, "split": args.split, "streaming": True}
        if args.dataset_config:
            dataset_args["name"] = args.dataset_config
        stream = load_dataset(**dataset_args)
        text = bytearray()
        for row in stream:
            if args.format == "qa":
                question = row.get(args.question_column)
                answer = row.get(args.answer_column)
                context = row.get(args.context_column, "") if args.context_column else ""
                if not isinstance(question, str) or not isinstance(answer, str):
                    continue
                value = f"User: {question}\n"
                if isinstance(context, str) and context.strip():
                    value += f"Context: {context}\n"
                value += f"N1: {answer}\n\n"
            else:
                value = row.get(args.text_column)
                if not isinstance(value, str):
                    continue
            text.extend(value.encode("utf-8"))
            if args.max_chars and len(text) >= args.max_chars:
                break
        return text

    return bytearray(args.data.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train N1 on local text or a streaming Hugging Face dataset.")
    parser.add_argument("--data", type=Path, default=Path("data/train.txt"))
    parser.add_argument("--dataset", help="Hugging Face dataset ID, for example HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", help="Dataset configuration/name, when the dataset has multiple configs")
    parser.add_argument("--split", default="train", help="Dataset split to stream")
    parser.add_argument("--format", choices=("text", "qa"), default="text")
    parser.add_argument("--text-column", default="text", help="Dataset column containing training text")
    parser.add_argument("--question-column", default="instruction", help="QA dataset question/instruction column")
    parser.add_argument("--answer-column", default="response", help="QA dataset answer column")
    parser.add_argument("--context-column", default="context", help="Optional QA dataset context column")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=50_000_000,
        help="Maximum UTF-8 bytes to cache from the stream (0 means unlimited)",
    )
    parser.add_argument("--out", type=Path, default=Path("checkpoints/n1.pt"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8, help="Number of Transformer blocks")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--embedding-size", type=int, default=256, help="Transformer embedding width")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.block_size < 2 or args.batch_size < 1 or args.layers < 1 or args.heads < 1 or args.embedding_size < 1:
        parser.error("model and batch-size values must be positive; block-size must be at least 2")
    if args.embedding_size % args.heads:
        parser.error("--embedding-size must be divisible by --heads")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in the range [0, 1)")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    text = load_text(args)
    if len(text) < args.block_size + 2:
        raise ValueError("training text must contain more characters than --block-size")
    print(f"loaded {len(text):,} training bytes")
    vocabulary = list(range(256))
    data = torch.frombuffer(text, dtype=torch.uint8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = GPTConfig(
        vocab_size=len(vocabulary),
        block_size=args.block_size,
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=args.embedding_size,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    for step in range(1, args.steps + 1):
        inputs, targets = get_batch(data, args.block_size, args.batch_size, device)
        _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(f"step {step:5d}/{args.steps} | loss {loss.item():.4f} | device {device}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**model.checkpoint(), "vocabulary": vocabulary, "encoding": "utf-8-bytes"}, args.out)
    print(f"saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
