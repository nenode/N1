import argparse
import re
from pathlib import Path

import torch

from .model import GPT, GPTConfig


def generate_sentence(model: GPT, prompt_tokens: torch.Tensor, minimum_tokens: int, maximum_tokens: int, temperature: float):
    generated = model.generate(prompt_tokens, maximum_tokens, temperature)[0].tolist()
    continuation = bytes(generated[len(prompt_tokens[0]) :])
    text = continuation.decode("utf-8", errors="replace")
    if len(continuation) <= minimum_tokens:
        return generated

    sentence_end = re.search(r"[.!?](?:[\"')\]]*)?(?:\s|$)", text[minimum_tokens:])
    if sentence_end:
        end = minimum_tokens + sentence_end.end()
        return generated[: len(prompt_tokens[0]) + len(text[:end].encode("utf-8"))]
    return generated


def extract_qa_response(generated: list[int], prompt_length: int, question: str) -> str:
    """Return only the answer before the next serialized QA example."""
    continuation = bytes(generated[prompt_length:]).decode("utf-8", errors="replace")
    question = question.strip()
    if continuation.casefold().startswith(question.casefold()):
        continuation = continuation[len(question) :]
    boundary = re.search(r"\n(?:\n|User:|Human:|N1:)", continuation)
    if boundary:
        continuation = continuation[: boundary.start()]
    return continuation.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a trained N1 model.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/n1.pt"))
    parser.add_argument("--qa", action="store_true", help="Format prompts as User/N1 instruction conversations")
    parser.add_argument("--tokens", type=int, default=160, help="Minimum new tokens before completing a sentence")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=320,
        help="Hard maximum new tokens generated for one response",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    qa_mode = args.qa or args.checkpoint.stem.casefold() == "qa"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("encoding") != "utf-8-bytes":
        raise ValueError("checkpoint was created by an older incompatible trainer; train a new checkpoint")
    model = GPT(GPTConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print("N1 is ready. Type /quit to exit.")

    while True:
        try:
            prompt = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.strip() == "/quit":
            break
        prompt_text = f"User: {prompt}\nN1: " if qa_mode else prompt
        tokens = torch.tensor([[token for token in prompt_text.encode("utf-8")]], device=device)
        if args.max_tokens < args.tokens:
            parser.error("--max-tokens must be greater than or equal to --tokens")
        generated = generate_sentence(model, tokens, args.tokens, args.max_tokens, args.temperature)
        if qa_mode:
            response = extract_qa_response(generated, len(tokens[0]), prompt)
        else:
            response = bytes(generated).decode("utf-8", errors="replace")
        print("ai> " + response)


if __name__ == "__main__":
    main()
