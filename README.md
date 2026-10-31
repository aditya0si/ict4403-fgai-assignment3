# English-to-French translation: four architectures

**ICT4403 Foundations of Generative AI, Assignment 3**  
**Student:** Aditya Singh (230953458)

## Objective

Compare a Transformer trained from scratch, an mT5 encoder-decoder model, and two decoder-only models (TinyLlama and Gemma) for English-to-French translation. All four models translate the same held-out test sentences. This is an educational, small-data experiment, not a production translation service.

## Data

The [OPUS-100 English-French parallel corpus](https://huggingface.co/datasets/Helsinki-NLP/opus-100) provides paired `en` and `fr` text. The experiment streams the official train, validation and test splits and takes the first 1,200 / 100 / 100 usable unique sentence pairs, respectively. It collapses whitespace and excludes pairs with either side shorter than 3 or longer than 180 characters. Case and punctuation are preserved. The script saves the selected rows in `results/` for auditability. OPUS-100 mixes domains and may contain alignment noise, duplicate content across official splits, and biased or sensitive language.

## Models and adaptation

| System | Architecture | Method |
|---|---|---|
| Basic Transformer | Encoder-decoder | Randomly initialized 2-layer encoder and decoder; train for 2 epochs |
| [mT5-small](https://huggingface.co/google/mt5-small) | Encoder-decoder | Full fine-tuning with Adafactor for 5 epochs |
| [TinyLlama 1.1B Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | LLaMA-family decoder-only | Zero-shot translation prompt |
| [Gemma 3 270M IT](https://huggingface.co/google/gemma-3-270m-it) | Decoder-only | Zero-shot translation prompt |

The scratch model uses mT5's tokenizer but **none of its pretrained model weights**. Its vocabulary choice may affect results. TinyLlama and Gemma are prompted rather than trained, so their training time is reported as `null` instead of zero. The comparison is controlled by a common test set, deterministic greedy decoding and a 64-token output limit; model size, pretraining, tokenization and adaptation method remain different, so performance differences cannot be attributed solely to architecture.

## Run in Google Colab

1. Select **Runtime → Change runtime type → GPU**.
2. Create a free Hugging Face account, accept the [Gemma model terms](https://huggingface.co/google/gemma-3-270m-it), and log in within Colab using `huggingface_hub.login()` or the `HF_TOKEN` Colab secret. Never commit a token.
3. Clone this repository, then run:

   ```bash
   pip install -r requirements.txt
   python translation_experiment.py
   ```

4. Download or commit the generated `results/` folder. Run all four systems to completion before reporting final conclusions. GPU memory and Colab session duration may vary; lower `BATCH` if memory is insufficient. The script starts each model sequentially and releases it before loading the next.

## Evaluation and results

`results/summary.json` contains corpus BLEU, chrF, parameter count, training time where applicable, total test inference time, and seconds per test sentence. `*_predictions.json` contains source, reference and prediction for qualitative review. `*_temperature.json` contains 10 sampled outputs at temperatures 0.3, 0.7 and 1.2 for both decoder-only systems. Evaluate temperature by checking fluency, fidelity, hallucination and output variability; these samples are separate from the deterministic test scores. Run `python plot_results.py` to regenerate `results/comparison.png`.

Three systems were measured locally on an NVIDIA GeForce RTX 5060 Laptop GPU. The Gemma run requires Hugging Face authentication and remains pending. Full measurements, examples, temperature observations and responsible-use discussion are in [ANALYSIS.md](ANALYSIS.md). The current highest BLEU and chrF belong to TinyLlama, though the final four-way conclusion awaits Gemma. These small-data results should not be generalized to production translation quality.

## Responsible use and qualitative review

Inspect predictions for changed names, dates, numbers, negation, gender, formality, cultural references, added information and omitted sensitive content. Translate paired examples manually or with a bilingual reviewer before claiming an error type. Possible mitigations include deduplicating and cleaning noisy pairs, balancing dialects and registers, adding domain-specific test sets, constraining output format, checking named entities and numbers, and human review for medical, legal or safety-critical content. Do not treat BLEU or chrF as a complete measure of translation quality. Specific observations from this run are documented in [ANALYSIS.md](ANALYSIS.md).

## Reproducibility notes

Seed: 4403. Sample sizes and epochs are declared at the top of the script. Hardware, package versions, elapsed times and any interrupted runs should be documented alongside final results. The official validation split is saved for tuning, although this compact baseline uses fixed hyperparameters; it is never used as test data. Do not choose settings based on test scores.

## Sources

- [OPUS-100 corpus and original paper](https://opus.nlpl.eu/opus-100.php)
- [Hugging Face OPUS-100 dataset](https://huggingface.co/datasets/Helsinki-NLP/opus-100)
- Model cards linked above.
