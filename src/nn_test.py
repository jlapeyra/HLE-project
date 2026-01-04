import re
import random
import math
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# 5) Inferència (greedy)
# -------------------------
@torch.no_grad()
def translate(sentence: str, max_len=20) -> str:
    model.eval()
    sentence = normalize(sentence)
    src = torch.tensor(encode(sentence, src_vocab, MAX_SRC_LEN), dtype=torch.long).unsqueeze(0).to(device)

    enc_outputs, state = model.encoder(src)
    src_mask = (src != SRC_PAD)

    tok = torch.tensor([tgt_vocab.stoi[SOS]], dtype=torch.long).to(device)
    out_ids = [tgt_vocab.stoi[SOS]]

    for _ in range(max_len):
        logits, state = model.decoder(tok, state, enc_outputs, src_mask)
        tok = logits.argmax(dim=1)
        out_ids.append(tok.item())
        if tok.item() == tgt_vocab.stoi[EOS]:
            break

    return decode(out_ids, tgt_vocab)

print("\nProves:")
for s in ["on és això", "necessito un bany", "m'agrada una bona nit"]:
    print(f"{s} -> {translate(s)}")
