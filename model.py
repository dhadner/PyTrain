import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config["n_embd"] % config["n_head"] == 0
        self.n_head = config["n_head"]
        self.n_embd = config["n_embd"]
        self.head_dim = self.n_embd // self.n_head
        self.qkv = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.proj = nn.Linear(self.n_embd, self.n_embd)

    def forward(self, x, attn_mask=None):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if attn_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config["n_embd"])
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config["n_embd"])
        self.mlp = nn.Sequential(
            nn.Linear(config["n_embd"], 4 * config["n_embd"]),
            nn.GELU(),
            nn.Linear(4 * config["n_embd"], config["n_embd"]),
        )

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config["vocab_size"], config["n_embd"])
        self.pos_emb = nn.Embedding(config["block_size"], config["n_embd"])
        self.blocks = nn.ModuleList([Block(config) for _ in range(config["n_layer"])])
        self.ln_f = nn.LayerNorm(config["n_embd"])
        self.head = nn.Linear(config["n_embd"], config["vocab_size"], bias=False)
        self.tok_emb.weight = self.head.weight
        self.apply(self._init_weights)
        print(f"Model parameters: {sum(p.numel() for p in self.parameters()):,}", flush=True)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _doc_aware_inputs(self, doc_ids):
        B, T = doc_ids.size()
        device = doc_ids.device
        new_doc = torch.zeros_like(doc_ids)
        new_doc[:, 0] = 1
        new_doc[:, 1:] = (doc_ids[:, 1:] != doc_ids[:, :-1]).long()
        arange = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        last_reset = torch.where(new_doc.bool(), arange, torch.zeros_like(arange))
        last_reset = torch.cummax(last_reset, dim=1).values
        pos = arange - last_reset
        same_doc = doc_ids.unsqueeze(-1) == doc_ids.unsqueeze(-2)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
        attn_mask = (causal.unsqueeze(0) & same_doc).unsqueeze(1)
        return pos, attn_mask

    def forward(self, idx, doc_ids=None, targets=None):
        B, T = idx.size()
        if doc_ids is not None:
            pos, attn_mask = self._doc_aware_inputs(doc_ids)
        else:
            pos = torch.arange(0, T, device=idx.device).unsqueeze(0).expand(B, T)
            attn_mask = None
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40, stop_at=None):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self.config["block_size"]:]
            logits, _ = self(idx_crop)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
            if stop_at is not None and idx_next.item() == stop_at:
                break
        return idx
