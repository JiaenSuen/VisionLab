import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import torch

from dataset import tokenize


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def count_trainable_parameters(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def save_checkpoint(state: Dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)


def load_checkpoint(path: str, map_location="cpu") -> Dict:
    """Load a full training checkpoint across recent PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_config(config: Dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def append_csv(path: str, row: Dict) -> None:
    ensure_dir(os.path.dirname(path))
    exists = os.path.isfile(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path: str, rows: Sequence[Dict]) -> None:
    ensure_dir(os.path.dirname(path))
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def seconds_to_hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(predictions: Dict[str, str], references: Dict[str, List[str]], max_n: int = 4) -> Dict[str, float]:
    pred_len = 0
    ref_len = 0
    matches = [0 for _ in range(max_n)]
    totals = [0 for _ in range(max_n)]

    for image_name, pred in predictions.items():
        cand = tokenize(pred)
        refs = [tokenize(r) for r in references.get(image_name, [])]
        if not refs:
            continue
        pred_len += len(cand)
        ref_len += min((abs(len(r) - len(cand)), len(r)) for r in refs)[1]
        for n in range(1, max_n + 1):
            cand_counts = ngram_counts(cand, n)
            totals[n - 1] += max(1, sum(cand_counts.values()))
            max_ref = Counter()
            for ref in refs:
                ref_counts = ngram_counts(ref, n)
                for gram, c in ref_counts.items():
                    if c > max_ref[gram]:
                        max_ref[gram] = c
            overlap = cand_counts & max_ref
            matches[n - 1] += sum(overlap.values())

    if pred_len == 0:
        return {f"BLEU-{i}": 0.0 for i in range(1, max_n + 1)}
    bp = 1.0 if pred_len > ref_len else math.exp(1.0 - ref_len / max(1, pred_len))
    results = {}
    for k in range(1, max_n + 1):
        precisions = [((matches[i] + 1.0) / (totals[i] + 1.0)) for i in range(k)]
        score = bp * math.exp(sum(math.log(p) for p in precisions) / k)
        results[f"BLEU-{k}"] = float(score)
    return results


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0]
        for j, y in enumerate(b, 1):
            curr.append(prev[j - 1] + 1 if x == y else max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_score(candidate: Sequence[str], reference: Sequence[str], beta: float = 1.2) -> float:
    if not candidate or not reference:
        return 0.0
    lcs = _lcs_len(candidate, reference)
    prec = lcs / len(candidate)
    rec = lcs / len(reference)
    if prec == 0 or rec == 0:
        return 0.0
    beta2 = beta * beta
    return (1 + beta2) * prec * rec / (rec + beta2 * prec)


def meteor_single(candidate: Sequence[str], reference: Sequence[str]) -> float:
    if not candidate or not reference:
        return 0.0
    ref_positions = defaultdict(list)
    for idx, tok in enumerate(reference):
        ref_positions[tok].append(idx)
    used = set()
    aligned = []
    for i, tok in enumerate(candidate):
        for pos in ref_positions.get(tok, []):
            if pos not in used:
                used.add(pos)
                aligned.append((i, pos))
                break
    matches = len(aligned)
    if matches == 0:
        return 0.0
    precision = matches / len(candidate)
    recall = matches / len(reference)
    f_mean = (10 * precision * recall) / (recall + 9 * precision + 1e-12)
    aligned.sort()
    chunks = 1
    for k in range(1, len(aligned)):
        if aligned[k][0] != aligned[k - 1][0] + 1 or aligned[k][1] != aligned[k - 1][1] + 1:
            chunks += 1
    penalty = 0.5 * (chunks / matches) ** 3
    return float((1 - penalty) * f_mean)


def cider_score(predictions: Dict[str, str], references: Dict[str, List[str]], max_n: int = 4) -> float:
    image_names = [name for name in predictions.keys() if name in references]
    n_docs = max(1, len(image_names))
    dfs = [Counter() for _ in range(max_n)]

    for name in image_names:
        for n in range(1, max_n + 1):
            grams = set()
            for ref in references[name]:
                grams.update(ngram_counts(tokenize(ref), n).keys())
            for gram in grams:
                dfs[n - 1][gram] += 1

    def tfidf(tokens: Sequence[str], n: int) -> Dict[Tuple[str, ...], float]:
        counts = ngram_counts(tokens, n)
        total = max(1, sum(counts.values()))
        vec = {}
        for gram, c in counts.items():
            idf = math.log((n_docs + 1.0) / (dfs[n - 1].get(gram, 0) + 1.0))
            vec[gram] = (c / total) * idf
        return vec

    def cosine(v1: Dict, v2: Dict) -> float:
        if not v1 or not v2:
            return 0.0
        dot = sum(v * v2.get(k, 0.0) for k, v in v1.items())
        n1 = math.sqrt(sum(v * v for v in v1.values()))
        n2 = math.sqrt(sum(v * v for v in v2.values()))
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return dot / (n1 * n2)

    scores = []
    for name in image_names:
        cand_tokens = tokenize(predictions[name])
        refs = [tokenize(r) for r in references[name]]
        img_score = 0.0
        for n in range(1, max_n + 1):
            cand_vec = tfidf(cand_tokens, n)
            sims = [cosine(cand_vec, tfidf(ref, n)) for ref in refs]
            img_score += sum(sims) / max(1, len(sims))
        scores.append(10.0 * img_score / max_n)
    return float(sum(scores) / max(1, len(scores)))


def evaluate_caption_metrics(predictions: Dict[str, str], references: Dict[str, List[str]]) -> Dict[str, float]:
    bleu = corpus_bleu(predictions, references, max_n=4)
    rouge_scores = []
    meteor_scores = []
    for image_name, pred in predictions.items():
        refs = references.get(image_name, [])
        cand_tokens = tokenize(pred)
        ref_tokens = [tokenize(r) for r in refs]
        if not ref_tokens:
            continue
        rouge_scores.append(max(rouge_l_score(cand_tokens, ref) for ref in ref_tokens))
        meteor_scores.append(max(meteor_single(cand_tokens, ref) for ref in ref_tokens))
    result = dict(bleu)
    result["METEOR"] = float(sum(meteor_scores) / max(1, len(meteor_scores)))
    result["ROUGE-L"] = float(sum(rouge_scores) / max(1, len(rouge_scores)))
    result["CIDEr"] = cider_score(predictions, references)
    return result


def format_metrics(metrics: Dict[str, float]) -> str:
    keys = ["loss", "ppl", "token_acc", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]
    parts = []
    for k in keys:
        if k in metrics:
            parts.append(f"{k}={metrics[k]:.4f}")
    return " | ".join(parts)


def write_final_report(
    path: str,
    config: Dict,
    metrics_by_phase: Dict[str, Dict[str, float]],
    model_params_m: float,
    best_epoch: int,
    best_score: float,
    elapsed_seconds: float,
) -> None:
    ensure_dir(os.path.dirname(path))
    lines = []
    lines.append("E-ConvNeXt + Transformer Image Captioning Final Report")
    lines.append("=" * 68)
    lines.append(f"Dataset root        : {config['data_dir']}")
    lines.append(f"Images path         : {config['image_dir']}")
    lines.append(f"Captions path       : {config['captions_file']}")
    lines.append(f"Encoder             : E-ConvNeXt-{config['encoder_variant']}")
    lines.append("Decoder             : Visual Resampler + Lightweight Transformer Decoder")
    lines.append(f"Trainable params    : {model_params_m:.3f} M")
    lines.append(f"Image size / MaxLen : {config['image_size']} / {config['max_len']}")
    lines.append(f"Best epoch          : {best_epoch}")
    lines.append(f"Best valid CIDEr    : {best_score:.4f}")
    lines.append(f"Total runtime       : {seconds_to_hms(elapsed_seconds)}")
    lines.append("")
    lines.append("Final split metrics")
    lines.append("-" * 68)
    for phase in ["train", "valid", "test"]:
        if phase in metrics_by_phase:
            lines.append(f"[{phase}] {format_metrics(metrics_by_phase[phase])}")
    lines.append("")
    lines.append("Analysis")
    lines.append("-" * 68)
    lines.append("1. BLEU-1/2 reflects object and phrase-level lexical grounding; BLEU-3/4 is stricter and usually lower on small caption datasets.")
    lines.append("2. METEOR and ROUGE-L provide complementary recall-oriented caption similarity checks.")
    lines.append("3. CIDEr is treated as the primary model-selection metric because it uses TF-IDF weighted n-gram consensus across references.")
    lines.append("4. The architecture reduces decoder-side visual token processing through learned resampler queries while preserving CNN multi-scale information.")
    lines.append("5. Check `Record/history.csv` for epoch-wise training/validation trends and `Record/predictions_*.csv` for qualitative caption inspection.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
