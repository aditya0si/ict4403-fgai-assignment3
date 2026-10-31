# Translation results and responsible use analysis

All values below are measured on the same 100 sentence English to French OPUS-100 test subset. The run used an NVIDIA GeForce RTX 5060 Laptop GPU with about 8 GB of VRAM. BLEU and chrF are corpus scores; larger is better. Inference time includes generation for all 100 sentences, with one sentence at a time for the scratch and decoder-only models and batches of two for mT5.

| Model | BLEU | chrF | Parameters | Training or fine-tuning | Test inference |
|---|---:|---:|---:|---:|---:|
| Scratch Transformer | 0.07 | 5.80 | 130,970,868 | 41.4 s | 37.1 s |
| mT5-small | 0.87 | 12.00 | 300,176,768 | 530.5 s | 34.5 s |
| TinyLlama 1.1B Chat | 13.16 | 42.29 | 1,100,048,384 | Not applicable, prompted | 115.5 s |
| Gemma 3 270M IT | 5.52 | 26.42 | 268,098,176 | Not applicable, prompted | 80.2 s |

## Interpretation

TinyLlama has the strongest measured BLEU and chrF, but it is also the largest and slowest measured model. Gemma ranks second on both translation metrics while using about one quarter of TinyLlama's parameters and less test inference time. The scratch Transformer and mT5 were exposed to only 1,200 parallel pairs. mT5 had pretrained multilingual weights but required task adaptation; the five-epoch full fine-tuning run improved its output over an earlier unsuccessful short LoRA run. Architecture alone does not explain the ranking: model size, multilingual pretraining, training data, tokenizer, optimization and prompting all differ. The 100 sentence sample is small, and the models were not tuned on the validation set, so this is an exploratory comparison.

The scratch model repeatedly emits short fragments. mT5 produces French but often omits content or repeats phrases. TinyLlama sometimes translates well; for “These laws establish a parliamentary republic and provide for the separation of powers,” it produced a close French sentence, but changed “provide for” to a stronger “guarantee.” It also sometimes prefixes “French:” or includes an explanation, contrary to the requested output format. That extra text lowers automatic scores and reduces usability.

Gemma produced nonempty output for every test sentence, but copied the English source exactly in 41 of 100 cases. For example, it returned the English sentence “Peter and I were in a car crash.” unchanged. In another case it translated “Who's going to talk to the used car salesman?” into French. Its inconsistent compliance with the translation instruction is a major limitation. Gemma was loaded in bfloat16: a float16 trial produced empty outputs and was discarded as an invalid run. The measured model is the [Unsloth-hosted Gemma 3 270M IT release](https://huggingface.co/unsloth/gemma-3-270m-it), which the model card traces to Google's instruction-tuned release.

The corpus has alignment noise. One test source says “THERE'S ONE MAN I KNOW WHO COULD NAIL HIM,” while its reference begins “Dites, je me demandais...” This is not a faithful pair; any model judged against that reference would be penalized unfairly. The selected train, validation and test subsets have no exact pair overlap after case folding, but near duplicates and broader pretraining overlap were not checked.

## Temperature experiment

The decoder-only systems also generate 10 additional translations each at temperatures 0.3, 0.7 and 1.2 with top-p 0.9. These sampled outputs are separate from the greedy scored test outputs. TinyLlama produced three distinct outputs across the three temperatures for each of the 10 sources; Gemma averaged 1.7 distinct outputs per source and often copied the English unchanged. TinyLlama’s higher temperature samples show more variation and extra commentary. For “Peter and I were in a car crash,” the 0.3 sample changes the name Peter to Pierre and includes a second translation line. The 1.2 sample adds unrelated text after a translation. This small, non-blind sample illustrates a risk; it does not establish a statistically reliable optimum temperature. Deterministic decoding is preferable for reproducible translation scoring.

## Responsible and ethical use

- **Hallucination and omission:** Added commentary, altered facts, repeated phrases and missing details can mislead readers. Constrain output to one translation, check named entities and numbers, and use bilingual review for important content.
- **Cultural representation:** The TinyLlama output for a “young Czech artist” describes him as Czechoslovak, changing the source identity. Evaluate dialects, names, register, honorifics and cultural references with reviewers familiar with both languages.
- **Bias:** OPUS-100 mixes domains and can overrepresent particular regions, institutions or speech styles. Audit performance by topic, dialect, gendered expressions and formality, and add balanced evaluation examples.
- **Sensitive content:** Do not rely on these small systems for medical, legal or safety-critical translations. Preserve negation, quantities and instructions, and require qualified human review before use.
- **Data quality:** Remove broken alignments and near duplicates, document filtering, and inspect translated examples before interpreting BLEU and chrF. Automated metrics alone cannot certify semantic fidelity.

## Conclusion and limits

TinyLlama is the best-performing model **by BLEU and chrF in this specific four-system experiment**. It is not automatically the best choice for every use: it is largest, slower than Gemma, and sometimes adds text or changes meaning. Gemma offers a smaller model but frequently copies English instead of translating. Review a stratified sample of all four prediction files with a bilingual reader before any practical use. The repository's `plot_results.py` generates the comparison chart from the measured metric files.
