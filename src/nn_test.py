import os
import torch
from typing import Any
import sacrebleu
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

# Reuse definitions from the training module to avoid duplication.
from nn_train import Encoder, Decoder, Seq2Seq, Vocab, normalize, encode  # type: ignore


def _ensure_vocab(obj: Any) -> Vocab:
    if isinstance(obj, Vocab):
        return obj
    if hasattr(obj, "stoi") and hasattr(obj, "itos"):
        return Vocab(stoi=obj.stoi, itos=list(obj.itos))
    if isinstance(obj, dict):
        return Vocab(stoi=obj.get("stoi", {}), itos=obj.get("itos", []))
    raise ValueError("Unsupported vocab object in checkpoint")


def load_checkpoint(path: str | None = None, device: torch.device | None = None):
    if path is None:
        path = "translator_seq2seq_v2.pt"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.serialization.add_safe_globals([Vocab])

    ckpt = torch.load(path, map_location=device, weights_only=False)

    src_vocab = _ensure_vocab(ckpt.get("src_vocab"))
    tgt_vocab = _ensure_vocab(ckpt.get("tgt_vocab"))

    emb_dim = ckpt.get("emb_dim")
    hid_dim = ckpt.get("hid_dim")
    MAX_SRC_LEN = ckpt.get("MAX_SRC_LEN")
    MAX_TGT_LEN = ckpt.get("MAX_TGT_LEN")

    SRC_PAD = src_vocab.stoi.get("<pad>")
    TGT_PAD = tgt_vocab.stoi.get("<pad>")

    enc = Encoder(len(src_vocab.itos), emb_dim, hid_dim, SRC_PAD)
    dec = Decoder(len(tgt_vocab.itos), emb_dim, hid_dim, TGT_PAD)
    model = Seq2Seq(enc, dec, SRC_PAD, device).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return {
        "model": model,
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "device": device,
        "MAX_SRC_LEN": MAX_SRC_LEN,
        "MAX_TGT_LEN": MAX_TGT_LEN,
    }


@torch.no_grad()
def translate(sentence: str, model, src_vocab: Vocab, tgt_vocab: Vocab, MAX_SRC_LEN: int, MAX_TGT_LEN: int, device=None) -> str:
    device = device or next(model.parameters()).device
    sent = normalize(sentence)
    src = torch.tensor([encode(sent, src_vocab, MAX_SRC_LEN)], dtype=torch.long).to(device)

    enc_outputs, state = model.encoder(src)
    src_mask = (src != model.src_pad_idx)

    tok = torch.tensor([tgt_vocab.stoi.get("<sos>")], dtype=torch.long).to(device)
    out_ids = [tgt_vocab.stoi.get("<sos>")]

    for _ in range(MAX_TGT_LEN):
        logits, state = model.decoder(tok, state, enc_outputs, src_mask)
        tok = logits.argmax(dim=1)
        out_ids.append(tok.item())
        if tok.item() == tgt_vocab.stoi.get("<eos>"):
            break

    words = []
    for i in out_ids:
        w = tgt_vocab.itos[i]
        if w in ("<sos>", "<pad>"):
            continue
        if w == "<eos>":
            break
        words.append(w)
    return " ".join(words)


if __name__ == "__main__":
    info = load_checkpoint()
    model = info["model"]
    src_vocab = info["src_vocab"]
    tgt_vocab = info["tgt_vocab"]
    device = info["device"]
    MAX_SRC_LEN = info["MAX_SRC_LEN"]
    MAX_TGT_LEN = info["MAX_TGT_LEN"]

    # examples = [
    #     "El Parlament Europeu ha aprovat la llei.",
    #     "Necessito ajuda amb el meu equipament informàtic.",
    #     "Quin temps farà demà a Barcelona?",
    #     "M'agrada escoltar música clàssica mentre treballo.",
    #     "On és la biblioteca més propera?",
    #     "Podries recomanar-me un bon restaurant per sopar?",
    #     "Estic aprenent a programar en Python.",
    #     "La intel·ligència artificial està transformant moltes indústries.",
    #     "Quines són les últimes notícies sobre tecnologia?",
    #     "M'agradaria reservar una habitació d'hotel per a dues persones.",
    # ]
    # for s in examples:
    #     print(f"{s} -> {translate(s, model, src_vocab, tgt_vocab, MAX_SRC_LEN, MAX_TGT_LEN, device)}")
    
    #while True:
    #    print(translate(input("Escriu una frase en català: "), model, src_vocab, tgt_vocab, MAX_SRC_LEN, MAX_TGT_LEN, device))

def compute_bleu_final_20(
    ckpt_path: str | None = None,
    src_file: str = "data/europarl.en-ca.ca",
    tgt_file: str = "data/europarl.en-ca.en",
    fraction: float = 0.2,
    max_examples: int | None = None,
):
    """
    Load checkpoint (if ckpt_path is None uses default), take the final `fraction` of the
    parallel files (src_file / tgt_file) and compute corpus BLEU on model translations.
    Returns (bleu_score_float, n_examples).
    """
    info = load_checkpoint(ckpt_path)
    model = info["model"]
    src_vocab = info["src_vocab"]
    tgt_vocab = info["tgt_vocab"]
    device = info["device"]
    MAX_SRC_LEN = info["MAX_SRC_LEN"]
    MAX_TGT_LEN = info["MAX_TGT_LEN"]

    # Read files
    try:
        with open(src_file, "r", encoding="utf-8") as f:
            src_lines = [l.rstrip("\n") for l in f]
        with open(tgt_file, "r", encoding="utf-8") as f:
            tgt_lines = [l.rstrip("\n") for l in f]
    except FileNotFoundError as e:
        print("File not found:", e)
        return None, 0

    if len(src_lines) != len(tgt_lines):
        print("Warning: source and target have different lengths:", len(src_lines), len(tgt_lines))

    n = min(len(src_lines), len(tgt_lines))
    start = int(n * (1.0 - fraction))
    src_test = src_lines[start:n]
    tgt_test = tgt_lines[start:n]

    if max_examples is not None:
        src_test = src_test[:max_examples]
        tgt_test = tgt_test[:max_examples]

    hyps = []
    for s in tqdm(src_test):
        hyp = translate(s, model, src_vocab, tgt_vocab, MAX_SRC_LEN, MAX_TGT_LEN, device)
        hyps.append(hyp)

    # Try sacrebleu first, fallback to NLTK corpus_bleu
    try:

        bleu = sacrebleu.corpus_bleu(hyps, [tgt_test])
        print(f"sacreBLEU = {bleu.score:.2f}")
        return float(bleu.score), len(hyps)
    except Exception:
        try:

            # nltk expects: list_of_references: List[List[List[str]]], hypotheses: List[List[str]]
            list_of_references = [[ref.split()] for ref in tgt_test]
            hypotheses = [h.split() for h in hyps]
            score = corpus_bleu(list_of_references, hypotheses) * 100.0
            print(f"NLTK BLEU = {score:.2f}")
            return float(score), len(hyps)
        except Exception as exc:
            print("Could not compute BLEU: install sacrebleu or nltk:", exc)
            return None, len(hyps)
compute_bleu_final_20(fraction=0.1, max_examples=500)