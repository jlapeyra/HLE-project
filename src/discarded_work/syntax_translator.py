# graph2graph_ud.py
# Minimal graph->graph (UD) with: GNN encoder on source UD + biaffine parser decoder for target UD
# Requirements: pip install torch
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# 1) Data structures + CoNLL-U-ish parsing
# ----------------------------

@dataclass
class Token:
    idx: int           # 1..n
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: int          # 0..n (0=root)
    deprel: str

@dataclass
class SentenceUD:
    tokens: List[Token]  # idx starts at 1

    @property
    def n(self) -> int:
        return len(self.tokens)

    def heads_tensor(self) -> torch.Tensor:
        # shape [n], each in [0..n-1] if we use 0-based nodes with root=0,
        # but we'll keep tokens as 1..n and root=0; in tensor we store head as int in [0..n]
        return torch.tensor([t.head for t in self.tokens], dtype=torch.long)

    def deprels(self) -> List[str]:
        return [t.deprel for t in self.tokens]

    def forms(self) -> List[str]:
        return [t.form for t in self.tokens]

    def uposs(self) -> List[str]:
        return [t.upos for t in self.tokens]


def parse_conllu_lines(lines: List[str]) -> SentenceUD:
    """
    Parses a single sentence in a CoNLL-U-like format:
    ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  ... (ignores remaining columns)
    Skips multiword tokens (IDs like '1-2') and empty nodes (IDs like '3.1').
    """
    tokens: List[Token] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 8:
            # allow space-separated fallback if needed
            cols = line.split()
        tok_id = cols[0]
        if "-" in tok_id or "." in tok_id:
            continue
        idx = int(tok_id)
        form = cols[1]
        lemma = cols[2]
        upos = cols[3]
        xpos = cols[4]
        feats = cols[5]
        head = int(cols[6])
        deprel = cols[7]
        tokens.append(Token(idx, form, lemma, upos, xpos, feats, head, deprel))
    # ensure sorted
    tokens.sort(key=lambda t: t.idx)
    return SentenceUD(tokens=tokens)


# ----------------------------
# 2) Vocab helpers
# ----------------------------

class Vocab:
    def __init__(self, specials: Optional[List[str]] = None):
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []
        for sp in (specials or []):
            self.add(sp)

    def add(self, s: str) -> int:
        if s not in self.stoi:
            self.stoi[s] = len(self.itos)
            self.itos.append(s)
        return self.stoi[s]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, s: str, unk: str = "<unk>") -> int:
        if s in self.stoi:
            return self.stoi[s]
        if unk in self.stoi:
            return self.stoi[unk]
        return self.add(s)


def build_vocabs(pairs: List[Tuple[SentenceUD, SentenceUD]]) -> Tuple[Vocab, Vocab, Vocab]:
    """
    Returns vocabs: form_vocab, upos_vocab, deprel_vocab
    """
    form_vocab = Vocab(specials=["<pad>", "<unk>", "<root>"])
    upos_vocab = Vocab(specials=["<pad>", "<unk>", "<root>"])
    deprel_vocab = Vocab(specials=["<pad>", "<unk>", "root"])

    for src, tgt in pairs:
        for s in (src, tgt):
            for t in s.tokens:
                form_vocab.add(t.form)
                upos_vocab.add(t.upos)
                deprel_vocab.add(t.deprel)
    return form_vocab, upos_vocab, deprel_vocab


# ----------------------------
# 3) Source GNN encoder
# ----------------------------

class MPNNLayer(nn.Module):
    """
    Simple message passing layer:
    h_i' = LayerNorm( W_self h_i + sum_{j in N(i)} W_msg [h_j, rel_emb(e_{j->i})] )
    """
    def __init__(self, d_model: int, d_rel: int):
        super().__init__()
        self.W_self = nn.Linear(d_model, d_model)
        self.W_msg = nn.Linear(d_model + d_rel, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(
        self,
        h: torch.Tensor,                 # [N, d_model]
        edge_index: torch.Tensor,        # [2, E] (src, dst) 0-based nodes
        edge_rel: torch.Tensor,          # [E, d_rel]
    ) -> torch.Tensor:
        N, d = h.shape
        src, dst = edge_index[0], edge_index[1]  # [E], [E]
        msg_in = torch.cat([h[src], edge_rel], dim=-1)           # [E, d + d_rel]
        msg = self.W_msg(msg_in)                                 # [E, d]
        agg = torch.zeros((N, d), device=h.device, dtype=h.dtype)
        agg.index_add_(0, dst, msg)                               # sum aggregation
        out = self.W_self(h) + agg
        return self.ln(torch.tanh(out))


class SourceGNNEncoder(nn.Module):
    """
    Encodes the source UD graph (tree) into node embeddings.
    Adds an explicit ROOT node at position 0; tokens are 1..n.
    """
    def __init__(
        self,
        n_forms: int,
        n_upos: int,
        n_deprel: int,
        d_model: int = 256,
        d_rel: int = 64,
        n_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model
        self.form_emb = nn.Embedding(n_forms, d_model)
        self.upos_emb = nn.Embedding(n_upos, d_model)
        self.rel_emb = nn.Embedding(n_deprel, d_rel)
        self.layers = nn.ModuleList([MPNNLayer(d_model, d_rel) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)

        # root embeddings are learned via special tokens (<root>)
        # we feed ROOT as a "token" with form="<root>", upos="<root>"
        # and use rel_emb for edge labels.

    def build_graph(
        self,
        sent: SentenceUD,
        form_ids: List[int],
        upos_ids: List[int],
        deprel_ids: List[int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          edge_index: [2, E] with edges both directions
          edge_rel:   [E] deprel-id for each directed edge (uses child's deprel for head->child and child->head)
        """
        n = sent.n
        # nodes: 0=root, 1..n=token idx
        edges_src: List[int] = []
        edges_dst: List[int] = []
        edges_rel: List[int] = []

        for tok in sent.tokens:
            child = tok.idx
            head = tok.head  # already 0..n
            rel = deprel_ids[child]  # we store deprel per token position

            # head -> child
            edges_src.append(head)
            edges_dst.append(child)
            edges_rel.append(rel)

            # child -> head (reverse edge, same rel id for simplicity)
            edges_src.append(child)
            edges_dst.append(head)
            edges_rel.append(rel)

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long, device=device)
        edge_rel = torch.tensor(edges_rel, dtype=torch.long, device=device)
        return edge_index, edge_rel

    def forward(
        self,
        sent: SentenceUD,
        form_ids: torch.Tensor,   # [n+1] including root at 0
        upos_ids: torch.Tensor,   # [n+1]
        deprel_ids: torch.Tensor, # [n+1] (per node; root deprel can be "root")
    ) -> torch.Tensor:
        """
        Returns node embeddings: [n+1, d_model]
        """
        device = form_ids.device
        h = self.form_emb(form_ids) + self.upos_emb(upos_ids)     # [N, d_model]
        h = self.drop(h)

        # Build edges (needs Python lists; OK for research/prototyping)
        edge_index, edge_rel_ids = self.build_graph(
            sent,
            form_ids.tolist(),
            upos_ids.tolist(),
            deprel_ids.tolist(),
            device=device,
        )
        edge_rel = self.rel_emb(edge_rel_ids)                     # [E, d_rel]

        for layer in self.layers:
            h = layer(h, edge_index, edge_rel)
            h = self.drop(h)
        return h


# ----------------------------
# 4) Target decoder: biaffine head + label classifier
# ----------------------------

class Biaffine(nn.Module):
    """
    Biaffine scorer: s(i,j) = x_i^T U x_j + W [x_i;x_j] + b
    We'll use the common simplified form: x_i^T U x_j
    """
    def __init__(self, d: int):
        super().__init__()
        self.U = nn.Parameter(torch.empty(d, d))
        nn.init.xavier_uniform_(self.U)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        H: [N, d] node embeddings (includes root at 0)
        returns scores: [N, N] where scores[h, j] is head h -> dep j
        """
        # scores = H U H^T
        return H @ self.U @ H.transpose(0, 1)


class Graph2GraphUD(nn.Module):
    """
    Full model:
      - encode source UD with GNN -> src_node_emb
      - encode target nodes (forms+upos) + cross-attend to source
      - predict target heads + deprel
    """
    def __init__(
        self,
        n_forms: int,
        n_upos: int,
        n_deprel: int,
        d_model: int = 256,
        d_rel: int = 64,
        gnn_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.src_enc = SourceGNNEncoder(n_forms, n_upos, n_deprel, d_model, d_rel, gnn_layers, dropout)

        # Target node initial embeddings
        self.tgt_form_emb = nn.Embedding(n_forms, d_model)
        self.tgt_upos_emb = nn.Embedding(n_upos, d_model)
        self.drop = nn.Dropout(dropout)

        # Simple cross-attention: each target node attends to all source nodes
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, dropout=dropout, batch_first=True)

        # Project after cross-attn
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        # Head scorer + label classifier
        self.head_scorer = Biaffine(d_model)
        self.deprel_clf = nn.Linear(d_model * 2, n_deprel)  # uses [head_emb; dep_emb]

    def forward(
        self,
        src_sent: SentenceUD,
        src_form_ids: torch.Tensor,   # [ns+1]
        src_upos_ids: torch.Tensor,   # [ns+1]
        src_deprel_ids: torch.Tensor, # [ns+1]
        tgt_form_ids: torch.Tensor,   # [nt+1] with root at 0
        tgt_upos_ids: torch.Tensor,   # [nt+1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          head_scores: [Nt, Nt] (including root). Use argmax over dim=0? We'll define scores[h, j].
          deprel_scores: [Nt, Nt, R] label logits for each possible head h and dep j.
        """
        # 1) encode source graph
        src_H = self.src_enc(src_sent, src_form_ids, src_upos_ids, src_deprel_ids)  # [Ns, d]

        # 2) target initial embeddings
        tgt_H0 = self.tgt_form_emb(tgt_form_ids) + self.tgt_upos_emb(tgt_upos_ids)  # [Nt, d]
        tgt_H0 = self.drop(tgt_H0)

        # 3) cross-attn: query=tgt, key/value=src
        # multiheadattention expects [B, L, D]
        tgt_ctx, _ = self.attn(
            query=tgt_H0.unsqueeze(0),
            key=src_H.unsqueeze(0),
            value=src_H.unsqueeze(0),
            need_weights=False,
        )
        tgt_H = self.proj(tgt_ctx.squeeze(0))  # [Nt, d]

        # 4) head scores (biaffine over target nodes)
        head_scores = self.head_scorer(tgt_H)  # [Nt, Nt], scores[h, j]

        # 5) label scores for each (h,j): use concat([Hh, Hj])
        Nt, d = tgt_H.shape
        Hh = tgt_H.unsqueeze(1).expand(Nt, Nt, d)  # [h, j, d] but currently [Nt, Nt, d] with first dim=h
        Hj = tgt_H.unsqueeze(0).expand(Nt, Nt, d)  # [h, j, d]
        pair = torch.cat([Hh, Hj], dim=-1)         # [Nt, Nt, 2d]
        deprel_scores = self.deprel_clf(pair)      # [Nt, Nt, R]
        return head_scores, deprel_scores


# ----------------------------
# 5) Training utilities
# ----------------------------

def make_ids_for_sentence(
    sent: SentenceUD,
    form_vocab: Vocab,
    upos_vocab: Vocab,
    deprel_vocab: Vocab,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns tensors for nodes including ROOT at index 0:
      form_ids:   [n+1]
      upos_ids:   [n+1]
      deprel_ids: [n+1]  (token's deprel; root uses 'root')
    """
    n = sent.n
    form_ids = torch.zeros(n + 1, dtype=torch.long, device=device)
    upos_ids = torch.zeros(n + 1, dtype=torch.long, device=device)
    deprel_ids = torch.zeros(n + 1, dtype=torch.long, device=device)

    form_ids[0] = form_vocab.encode("<root>")
    upos_ids[0] = upos_vocab.encode("<root>")
    deprel_ids[0] = deprel_vocab.encode("root")

    for tok in sent.tokens:
        i = tok.idx
        form_ids[i] = form_vocab.encode(tok.form)
        upos_ids[i] = upos_vocab.encode(tok.upos)
        deprel_ids[i] = deprel_vocab.encode(tok.deprel)
    return form_ids, upos_ids, deprel_ids


def loss_for_target_tree(
    head_scores: torch.Tensor,     # [Nt, Nt] scores[h, j]
    deprel_scores: torch.Tensor,   # [Nt, Nt, R]
    gold_heads: torch.Tensor,      # [Nt-1] heads for tokens 1..Nt-1 (since 0 is root)
    gold_deprel_ids: torch.Tensor, # [Nt-1] label id per token
) -> torch.Tensor:
    """
    Computes loss over dependents j=1..Nt-1 (exclude root node as dependent).
    - head loss: cross entropy over possible heads h in [0..Nt-1]
    - label loss: for gold head h*, take deprel_scores[h*, j, :]
    """
    Nt = head_scores.size(0)
    # dependents indices 1..Nt-1
    deps = torch.arange(1, Nt, device=head_scores.device)

    # head logits for each dependent j: vector over heads h
    # head_scores[h, j] -> transpose to [j, h]
    head_logits = head_scores[:, deps].transpose(0, 1)  # [Nt-1, Nt]
    head_loss = F.cross_entropy(head_logits, gold_heads)

    # label logits: pick gold head per dep
    gold_h = gold_heads  # [Nt-1]
    label_logits = deprel_scores[gold_h, deps, :]       # [Nt-1, R]
    label_loss = F.cross_entropy(label_logits, gold_deprel_ids)

    return head_loss + label_loss


# ----------------------------
# 6) Example usage / tiny training loop
# ----------------------------

def train_one_epoch(
    model: Graph2GraphUD,
    data: List[Tuple[SentenceUD, SentenceUD]],
    vocabs: Tuple[Vocab, Vocab, Vocab],
    device: torch.device,
    lr: float = 3e-4,
) -> float:
    form_vocab, upos_vocab, deprel_vocab = vocabs
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    total = 0.0
    for src_sent, tgt_sent in data:
        # ids
        src_form, src_upos, src_dep = make_ids_for_sentence(src_sent, form_vocab, upos_vocab, deprel_vocab, device)
        tgt_form, tgt_upos, tgt_dep = make_ids_for_sentence(tgt_sent, form_vocab, upos_vocab, deprel_vocab, device)

        # forward
        head_scores, deprel_scores = model(
            src_sent=src_sent,
            src_form_ids=src_form,
            src_upos_ids=src_upos,
            src_deprel_ids=src_dep,
            tgt_form_ids=tgt_form,
            tgt_upos_ids=tgt_upos,
        )

        # gold (exclude root node as dependent)
        gold_heads = tgt_sent.heads_tensor().to(device)        # [n] for tokens 1..n (values 0..n)
        gold_deprel = torch.tensor([deprel_vocab.encode(r) for r in tgt_sent.deprels()],
                                   dtype=torch.long, device=device)

        loss = loss_for_target_tree(head_scores, deprel_scores, gold_heads, gold_deprel)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total += float(loss.item())
    return total / max(1, len(data))


@torch.no_grad()
def predict_tree(
    model: Graph2GraphUD,
    src_sent: SentenceUD,
    tgt_tokens_upos: List[Tuple[str, str]],  # list of (form, upos) for L2, without root
    vocabs: Tuple[Vocab, Vocab, Vocab],
    device: torch.device,
) -> Tuple[List[int], List[str]]:
    form_vocab, upos_vocab, deprel_vocab = vocabs

    # Build a "dummy" target SentenceUD with unknown heads/labels just to size tensors
    tgt_tokens = []
    for i, (form, upos) in enumerate(tgt_tokens_upos, start=1):
        tgt_tokens.append(Token(i, form, form, upos, upos, "_", 0, "root"))
    tgt_sent = SentenceUD(tokens=tgt_tokens)

    src_form, src_upos, src_dep = make_ids_for_sentence(src_sent, form_vocab, upos_vocab, deprel_vocab, device)
    tgt_form, tgt_upos, _ = make_ids_for_sentence(tgt_sent, form_vocab, upos_vocab, deprel_vocab, device)

    model.eval()
    head_scores, deprel_scores = model(
        src_sent=src_sent,
        src_form_ids=src_form,
        src_upos_ids=src_upos,
        src_deprel_ids=src_dep,
        tgt_form_ids=tgt_form,
        tgt_upos_ids=tgt_upos,
    )

    Nt = head_scores.size(0)
    deps = torch.arange(1, Nt, device=device)
    head_logits = head_scores[:, deps].transpose(0, 1)  # [Nt-1, Nt]
    pred_heads = head_logits.argmax(dim=-1)             # [Nt-1]

    # labels conditioned on predicted head
    pred_label_ids = deprel_scores[pred_heads, deps, :].argmax(dim=-1)  # [Nt-1]
    pred_labels = [deprel_vocab.itos[i] for i in pred_label_ids.tolist()]
    return pred_heads.tolist(), pred_labels


def main_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Example: you MUST replace with real parallel annotated data ---
    # Here we just duplicate the same sentence as "target" to show it runs.
    src_lines = [
        "1\tWhile\twhile\tSCONJ\tIN\t_\t3\tmark\t_\t_",
        "2\tmuch\tmuch\tADJ\tJJ\t_\t3\tnsubj\t_\t_",
        "3\tunprecedented\tunprecedented\tADJ\tJJ\t_\t0\troot\t_\t_",
        "4\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\t_",
    ]
    tgt_lines = [
        "1\tMentre\tmentre\tSCONJ\tCS\t_\t3\tmark\t_\t_",
        "2\tmolt\tmolt\tADV\tRG\t_\t3\tadvmod\t_\t_",
        "3\tinsòlit\tinsòlit\tADJ\tAQ\t_\t0\troot\t_\t_",
        "4\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\t_",
    ]

    src = parse_conllu_lines(src_lines)
    tgt = parse_conllu_lines(tgt_lines)

    pairs = [(src, tgt)]
    vocabs = build_vocabs(pairs)

    model = Graph2GraphUD(
        n_forms=len(vocabs[0]),
        n_upos=len(vocabs[1]),
        n_deprel=len(vocabs[2]),
        d_model=192,
        d_rel=64,
        gnn_layers=3,
        dropout=0.2,
    ).to(device)

    for epoch in range(5):
        avg_loss = train_one_epoch(model, pairs, vocabs, device=device, lr=3e-4)
        print(f"epoch={epoch} loss={avg_loss:.4f}")

    # Predict a tree for a given target token list (forms+upos)
    pred_heads, pred_labels = predict_tree(
        model, src,
        tgt_tokens_upos=[("Mentre", "SCONJ"), ("molt", "ADV"), ("insòlit", "ADJ"), (".", "PUNCT")],
        vocabs=vocabs,
        device=device,
    )
    print("Pred heads (for tokens 1..n):", pred_heads)
    print("Pred labels (for tokens 1..n):", pred_labels)


if __name__ == "__main__":
    main_demo()
