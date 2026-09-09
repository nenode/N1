# N1

N1 is a small, trainable autoregressive language model written in Python and PyTorch. It is a real neural network that learns to predict the next character from context; it is not an intent classifier or a set of hand-written responses.

This is an educational foundation, not a ChatGPT-scale model. Its quality depends on the training data and compute available. N1 uses byte-level tokens, so it remains less efficient than a subword-tokenized model, but the default architecture is now larger (8 layers, 8 heads, 256-wide embeddings, and 512-token context).

## Run it

Create an environment and install the project:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Train N1 on the included sample data:

```bash
python -m n1.train --steps 1000
```

Talk to the trained N1 model:

```bash
python -m n1.chat
```

Chat generates at least 160 new tokens by default, then continues until it reaches a sentence boundary. Use `--tokens` to change the minimum response length and `--max-tokens` to set the hard safety limit:

```bash
python -m n1.chat --tokens 100 --max-tokens 320 --temperature 0.6
```

Generate a story automatically without entering a prompt:

```bash
python -m n1.story --checkpoint checkpoints/n1.pt
```

Use `--sentences`, `--max-tokens`, `--temperature`, and `--name` to control the story length, style, and main character:

```bash
python -m n1.story --checkpoint checkpoints/n1.pt --sentences 24 --name Alex
```

Use your own UTF-8 text by passing a file:

```bash
python -m n1.train --data path/to/your-training.txt --steps 5000
```

## Train on Hugging Face data

The trainer can stream a dataset instead of downloading the full corpus. It caches a configurable prefix locally, then trains the N1 neural language model on random context windows from that cache. This makes large datasets practical, but model quality still depends on compute and architecture size.

For a small public test:

```bash
python -m n1.train \
	--dataset roneneldan/TinyStories \
	--text-column text \
	--max-chars 5000000 \
	--steps 5000
```

For a much larger corpus, use FineWeb-Edu with a bounded local cache:

```bash
python -m n1.train \
	--dataset HuggingFaceFW/fineweb-edu \
	--dataset-config sample-10BT \
	--text-column text \
	--max-chars 500000000 \
	--steps 100000 \
	--block-size 512 \
	--batch-size 8 \
	--out checkpoints/base.pt
```

The trainer stores the corpus as compact bytes and converts only each batch to model indices, keeping peak memory much lower. Start with `--batch-size 8` on a CPU or small GPU and increase it only if resources allow. `--max-chars 0` streams until the dataset ends, which may require substantial disk space. Set a Hugging Face token in the environment only when using a private or gated dataset.

For general question answering, train a separate checkpoint on instruction data. This example formats the question and answer together, so N1 learns the conversation pattern:

```bash
python -m n1.train \
	--dataset databricks/databricks-dolly-15k \
	--format qa \
	--question-column instruction \
	--context-column context \
	--answer-column response \
	--max-chars 50000000 \
	--steps 100000 \
	--block-size 512 \
	--layers 8 \
	--heads 8 \
	--embedding-size 256 \
	--out checkpoints/qa.pt
```

Chat with that checkpoint using the matching prompt format:

```bash
python -m n1.chat --checkpoint checkpoints/qa.pt --qa --temperature 0.4
```

This teaches response patterns; it does not guarantee factual accuracy. For reliable current facts, add retrieval or external tools.

The N1 model learns statistical patterns from the text it sees. It does not automatically browse the web, verify facts, remember conversations between runs, or reason like a person. Those are separate capabilities that can be added around the language model later.

## Project layout

- `n1/model.py`: GPT-style causal Transformer architecture.
- `n1/train.py`: local/Hugging Face data loading, byte encoding, batches, optimization, and N1 checkpoints.
- `n1/chat.py`: checkpoint loading and text generation.
- `data/train.txt`: tiny example corpus for a smoke test.
