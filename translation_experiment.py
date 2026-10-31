"""ICT4403 Assignment 3: reproducible English-to-French comparison.

Run in a GPU Colab session. Results are written to results/; no scores are hard-coded.
"""
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.optimization import Adafactor

SEED = 4403
TRAIN_N, VAL_N, TEST_N = 1200, 100, 100
EPOCHS = 2
BATCH = 8
MAX_SOURCE, MAX_TARGET = 64, 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("results")
OUT.mkdir(exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)


def clean_pairs(split, n):
    ds = load_dataset("Helsinki-NLP/opus-100", "en-fr", split=split, streaming=True)
    seen, rows = set(), []
    for item in ds:
        en = " ".join(item["translation"]["en"].split())
        fr = " ".join(item["translation"]["fr"].split())
        key = (en.casefold(), fr.casefold())
        if not (3 <= len(en) <= 180 and 3 <= len(fr) <= 180) or key in seen:
            continue
        seen.add(key); rows.append({"source": en, "reference": fr})
        if len(rows) == n: break
    if len(rows) < n: raise RuntimeError(f"Only {len(rows)} usable rows from {split}")
    return rows


def cached_split(name, n):
    path = OUT / f"{name}.json"
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
        if len(rows) == n: return rows
    return clean_pairs(name, n)


train = cached_split("train", TRAIN_N)
valid = cached_split("validation", VAL_N)
test = cached_split("test", TEST_N)
for name, rows in [("train", train), ("validation", valid), ("test", test)]:
    (OUT / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print("Data:", {"train": len(train), "validation": len(valid), "test": len(test)})


def score(preds, seconds, params, train_seconds, adaptation):
    refs = [r["reference"] for r in test]
    return {"bleu": BLEU().corpus_score(preds, [refs]).score,
            "chrf": CHRF().corpus_score(preds, [refs]).score,
            "inference_seconds": seconds, "seconds_per_sentence": seconds / len(test),
            "parameters": params, "train_seconds": train_seconds,
            "adaptation": adaptation}


def save_result(name, preds, metadata):
    rows = [{**row, "prediction": pred} for row, pred in zip(test, preds)]
    (OUT / f"{name}_predictions.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / f"{name}_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(name, metadata)


def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()


class PairDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows; self.tok = tokenizer
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        row = self.rows[i]
        source = self.tok.encode(row["source"], max_length=MAX_SOURCE, truncation=True)
        target = self.tok.encode(row["reference"], max_length=MAX_TARGET-1, truncation=True)
        if target and target[-1] == self.tok.eos_token_id: target = target[:-1]
        return source, target


class ScratchTransformer(nn.Module):
    def __init__(self, vocab, pad, bos, eos):
        super().__init__()
        self.pad, self.bos, self.eos = pad, bos, eos
        self.emb = nn.Embedding(vocab, 256, padding_idx=pad)
        self.pos = nn.Embedding(128, 256)
        self.net = nn.Transformer(d_model=256, nhead=4, num_encoder_layers=2,
                                  num_decoder_layers=2, dim_feedforward=512,
                                  dropout=.1, batch_first=True)
        self.head = nn.Linear(256, vocab)
    def embed(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.emb(x) * (256 ** .5) + self.pos(positions)
    def forward(self, src, tgt):
        mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
        y = self.net(self.embed(src), self.embed(tgt), tgt_mask=mask,
                     src_key_padding_mask=src.eq(self.pad), tgt_key_padding_mask=tgt.eq(self.pad),
                     memory_key_padding_mask=src.eq(self.pad))
        return self.head(y)
    @torch.no_grad()
    def translate(self, src):
        tgt = torch.tensor([[self.bos]], device=src.device)
        for _ in range(MAX_TARGET):
            nxt = self(src, tgt)[:, -1].argmax(-1, keepdim=True)
            tgt = torch.cat([tgt, nxt], dim=1)
            if nxt.item() == self.eos: break
        return tgt[0, 1:].tolist()


def run_scratch():
    # Shared subword tokenizer, but the Transformer weights start randomly.
    tok = AutoTokenizer.from_pretrained("google/mt5-small")
    pad, bos, eos = tok.pad_token_id, tok.pad_token_id, tok.eos_token_id
    model = ScratchTransformer(len(tok), pad, bos, eos).to(DEVICE)
    def collate(batch):
        sources, targets = zip(*batch)
        src = nn.utils.rnn.pad_sequence([torch.tensor(x) for x in sources], batch_first=True, padding_value=pad)
        shifted = [torch.tensor([bos] + x + [eos]) for x in targets]
        tgt = nn.utils.rnn.pad_sequence(shifted, batch_first=True, padding_value=pad)
        return src.to(DEVICE), tgt.to(DEVICE)
    loader = DataLoader(PairDataset(train, tok), batch_size=BATCH, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    sync(); start = time.perf_counter()
    model.train()
    for epoch in range(EPOCHS):
        losses = []
        for src, tgt in loader:
            opt.zero_grad()
            logits = model(src, tgt[:, :-1])
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=pad)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(loss.item())
        print("scratch epoch", epoch + 1, "loss", np.mean(losses))
    sync(); train_s = time.perf_counter() - start
    model.eval(); sync(); start = time.perf_counter()
    preds = []
    for row in test:
        ids = tok.encode(row["source"], max_length=MAX_SOURCE, truncation=True, return_tensors="pt").to(DEVICE)
        preds.append(tok.decode(model.translate(ids), skip_special_tokens=True).strip())
    sync(); infer_s = time.perf_counter() - start
    save_result("scratch_transformer", preds, score(preds, infer_s, sum(p.numel() for p in model.parameters()), train_s, "trained from scratch"))
    del model, opt; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None


def run_mt5():
    name = "google/mt5-small"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    params = sum(p.numel() for p in model.parameters())
    model.to(DEVICE)
    # Full fine-tuning is more effective here than the short LoRA run: mT5-small
    # is a denoising model, not an instruction-tuned translator.
    opt = Adafactor(model.parameters(), lr=1e-3, scale_parameter=False,
                    relative_step=False, warmup_init=False, clip_threshold=1.0)
    def batches(rows, shuffle=False):
        indices = list(range(len(rows)))
        if shuffle: random.shuffle(indices)
        for k in range(0, len(indices), 2):
            chunk = [rows[i] for i in indices[k:k+2]]
            x = tok(["translate English to French: " + r["source"] for r in chunk], padding=True, truncation=True, max_length=MAX_SOURCE, return_tensors="pt").to(DEVICE)
            y = tok(text_target=[r["reference"] for r in chunk], padding=True, truncation=True, max_length=MAX_TARGET, return_tensors="pt")["input_ids"].to(DEVICE)
            y[y == tok.pad_token_id] = -100
            yield x, y
    sync(); start = time.perf_counter(); model.train()
    for epoch in range(5):
        losses=[]; opt.zero_grad()
        for step,(x,y) in enumerate(batches(train, True), 1):
            loss=model(**x, labels=y).loss
            (loss/4).backward(); losses.append(loss.item())
            if step % 4 == 0 or step == (len(train)+1)//2:
                opt.step(); opt.zero_grad()
        print("mT5 epoch", epoch+1, "loss", np.mean(losses))
    sync(); train_s=time.perf_counter()-start
    model.eval(); preds=[]; sync(); start=time.perf_counter()
    with torch.no_grad():
        for x,_ in batches(test):
            ids=model.generate(**x, max_new_tokens=MAX_TARGET, num_beams=1, do_sample=False)
            preds += [s.strip() for s in tok.batch_decode(ids, skip_special_tokens=True)]
    sync(); infer_s=time.perf_counter()-start
    save_result("mt5_small", preds, score(preds, infer_s, params, train_s, "full fine-tuning with Adafactor"))
    del model,opt; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None


def run_causal(label, model_name):
    # Prompt-only adaptation is explicitly permitted for decoder-only models.
    tok=AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    if label == "gemma_270m":
        # Float16 produced empty Gemma outputs in this experiment. A Colab T4
        # lacks bfloat16, so use float32 there; newer GPUs can use bfloat16.
        dtype = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    else:
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    model=AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(DEVICE)
    model.eval(); params=sum(p.numel() for p in model.parameters())
    preds=[]; sync(); start=time.perf_counter()
    for row in test:
        messages=[{"role":"user","content":"Translate the following English sentence into French. Output only the translation.\nEnglish: "+row["source"]}]
        inputs=tok.apply_chat_template(messages, add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True).to(DEVICE)
        with torch.no_grad():
            ids=model.generate(**inputs, max_new_tokens=MAX_TARGET, do_sample=False, num_beams=1,
                               pad_token_id=tok.pad_token_id)
        preds.append(tok.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    sync(); infer_s=time.perf_counter()-start
    save_result(label, preds, score(preds, infer_s, params, None, "zero-shot prompting"))
    # Temperature analysis: same 10 source sentences, three temperatures, fixed seed.
    samples=[]
    for temperature in (.3,.7,1.2):
        torch.manual_seed(SEED)
        for row in test[:10]:
            messages=[{"role":"user","content":"Translate into French. Output only the translation.\nEnglish: "+row["source"]}]
            inputs=tok.apply_chat_template(messages, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True).to(DEVICE)
            with torch.no_grad():
                ids=model.generate(**inputs, max_new_tokens=MAX_TARGET, do_sample=True, temperature=temperature,
                                   top_p=.9, pad_token_id=tok.pad_token_id)
            samples.append({"temperature":temperature, **row,
                            "prediction":tok.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()})
    (OUT/f"{label}_temperature.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    del model; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    run_scratch()
    run_mt5()
    run_causal("tinyllama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    run_causal("gemma_270m", "unsloth/gemma-3-270m-it")
    results={p.stem.replace("_metrics", ""):json.loads(p.read_text()) for p in OUT.glob("*_metrics.json")}
    (OUT/"summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
