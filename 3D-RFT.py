"""Pseudocode for 3D Reweighted Finetuning (3D-RFT) for an example 3D-LLM."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreeDLLM(nn.Module):
    """Example 3D-LLM model class.
    
    Notation:
        - B: Batch size.
        - T_in: Maximum input token length.
        - T_out: Maximum output token length.
        - V: Vocabulary size.
        - D_pcd: Point cloud embedding dimension.
        - D_llm: LLM embedding dimension.
        - D: Embedding dimension.
        - ignore_idx: Ignore index for the loss calculation, defaults to -100.
    """
    def __init__(self, llm, tokenizer, pcd_projector, max_in_len=1024, max_out_len=256, device='cuda'):
        super().__init__()
        self.llm = llm                                                                            # Causal LM (e.g., LLaMA/Vicuna/Qwen)
        self.tok = tokenizer                                                                      # Tokenizer τ
        self.pcd_proj = pcd_projector                                                             # g: R^{D_pcd} → R^{D_llm}
        self.max_in_len = max_in_len
        self.max_out_len = max_out_len
        self.device = device
        self.ignore_idx = -100

    @torch.no_grad()
    def _embed_tokens(self, input_ids):
        return self.llm.get_input_embeddings()(input_ids)

    def forward(self, batch: dict, blind: bool = False):
        """Forward pass for a 3D-LLM.

        Args:
            - batch (dict): The input batch with the following keys:
                - text_prompt : list[str] length B, containing query prompts.
                - pointcloud : Tensor (B, L_pcd, D_pcd) containing 3D point clouds.
                - output_gt : list[str] length B, containing ground truth output text.
            - blind (bool): Whether to run the blind pass.
        
        Returns:
            - loss : Tensor (B,) containing the loss for each sample.
            - logits : Tensor (B, T_out, V) containing the logits for each sample.
        """
        device = self.device
        B = len(batch['text_prompt'])

        # 1) Encode text tokens
        tok = self.tok
        tok.padding_side, tok.truncation_side = 'right', 'right'
        prompt = tok(
            batch['text_prompt'],
            return_tensors='pt', 
            padding='longest', 
            truncation=True, 
            max_length=self.max_in_len
        ).to(device)                                                                                # (B, T_in) int
        prompt_emb = self._embed_tokens(prompt.input_ids)                                           # (B, T_in, D) float

        # 2) Encode 3D tokens 
        pcd = batch['pointcloud'].to(device)                                                        # (B, L_pcd, D_pcd) float
        if blind:
            pcd = torch.zeros_like(pcd)                                                             # Zero-out 3D signal in Blind model
        pcd_tokens = self.pcd_proj(pcd)                                                             # (B, L_pcd, D)
        pcd_mask = torch.ones(B, pcd_tokens.size(1), device=device, dtype=torch.long)

        # 3) Output tokens (teacher forcing); only these positions are supervised
        out = tok(
            [t + tok.eos_token for t in batch['output_gt']],
            return_tensors='pt', 
            padding='longest', 
            truncation=True, 
            max_length=self.max_out_len
        ).to(device)
        out_emb = self._embed_tokens(out.input_ids)                                                 # (B, T_out, D)

        # 4) Build sequence: [prompt || pcd || output]
        inputs_embeds  = torch.cat([prompt_emb, pcd_tokens, out_emb], dim=1)                        # (B, T, D)
        attention_mask = torch.cat([prompt.attention_mask, pcd_mask, out.attention_mask], dim=1)    # (B, T)

        # 5) Labels: supervise only the output segment
        T_out = out.attention_mask.size(1)
        labels = torch.full_like(attention_mask, self.ignore_idx)
        out_mask = out.attention_mask.bool()
        labels[:, -T_out:][out_mask] = out.input_ids[out_mask]
        labels[:, -T_out] = self.ignore_idx                                                         # Don't predict first output token

        # 6) LM forward + per-sample mean cross-entropy loss
        logits = self.llm(inputs_embeds=inputs_embeds,
                          attention_mask=attention_mask,
                          return_dict=True).logits.float()                                          # (B, T, V)

        shift_logits = logits[..., :-1, :]
        shift_labels = labels[...,  1: ]
        cross_entropy = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction='none', ignore_index=self.ignore_idx
        ).view(B, -1)
        valid = (shift_labels != self.ignore_idx).float().view(B, -1)
        num = valid.sum(1).clamp_min(1.0)
        loss = cross_entropy.sum(1) / num                                                          # (B,)
        return {'loss': loss, 'logits': logits}


def training_step(
  model: ThreeDLLM, 
  dataloader: torch.utils.data.DataLoader,
  optimizer: torch.optim.Optimizer, 
  grad_clip: float = None, 
  eps_min: float = 1e-4,
):
    """3D-RFT training step. The blind pass is run under no_grad."""
    model.train()
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # Full model
        out = model.forward(batch, blind=False)                                                     # (B,)

        # Blind reference (no gradients)
        with torch.no_grad():
            ref = model.forward(batch, blind=True)                                                  # (B,)

        loss_3drft = out['loss'] / ref['loss'].clamp_min(eps_min)                                   # (B,)
        loss_3drft = loss_3drft.mean()
        loss_3drft.backward()

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
