# GPT-2 Exercise Project

This is an exercise project to get familiar with GPT-2's architecture.

## Required model files

The model and tokenizer assets are not included in this repository, and Git LFS
is not required. The loaders in `main.py` fetch the following files from
`openai-community/gpt2` on Hugging Face using `hf_hub_download`:

- `model.safetensors`: pretrained GPT-2 weights, loaded by `load_weights()`.
- `tokenizer.json`: tokenizer definition, loaded by `load_tokenizer()`.

Both `main.py` and `mmlu_bench.py` use these loaders. An internet connection is
required to download assets that are not already in the Hugging Face cache;
the downloaded files are cached outside the repository for reuse. There is no
need to place them in the project directory. The previously tracked
`tokenizer_config.json` is not used by the current loaders.

The MMLU benchmark also loads the `test` and `dev` splits of the `cais/mmlu`
dataset (`all` configuration) through the `datasets` package, so that dataset
must also be downloaded or already cached.

## License

The project's code and documentation are licensed under the [MIT License](LICENSE).
The model and tokenizer files downloaded at runtime come from
[openai-community/gpt2](https://huggingface.co/openai-community/gpt2) and retain
their upstream terms. See [third-party notices](THIRD_PARTY_NOTICES.md) for the
files and license details.
