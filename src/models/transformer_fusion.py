import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionLayer(nn.Module):
    """
    A Cross-Attention layer where Modality A (e.g. Text) acts as the Query,
    and Modality B (e.g. Audio or Vision) acts as the Key and Value.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        # PyTorch MultiheadAttention expects sequence dimension as the first dimension 
        # when batch_first=False, but we'll use batch_first=True for simplicity.
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # query (Modality A): [batch_size, seq_len_A, embed_dim]
        # key (Modality B): [batch_size, seq_len_B, embed_dim]
        # value (Modality B): [batch_size, seq_len_B, embed_dim]
        
        # Compute cross-attention
        attn_output, attn_weights = self.mha(query, key, value, average_attn_weights=False)
        
        # Residual connection and layer normalization
        out = self.layernorm(query + self.dropout(attn_output))
        return out, attn_weights


class TextAnchoredCrossAttentionTransformer(nn.Module):
    """
    Our proposed state-of-the-art Text-Anchored Cross-Attention Transformer.
    Uses text as the stable semantic anchor (Query) to query audio (Keys/Values) 
    and vision (Keys/Values) modalities. Combines the outputs and classifies.
    """
    def __init__(self, text_dim, audio_dim, vision_dim, embed_dim=128, num_heads=4, hidden_dim=128, output_dim=1, dropout=0.2):
        super(TextAnchoredCrossAttentionTransformer, self).__init__()
        
        # 1. Modality Projection Layers (Linear projections to align dimensions)
        self.proj_text = nn.Linear(text_dim, embed_dim)
        self.proj_audio = nn.Linear(audio_dim, embed_dim)
        self.proj_vision = nn.Linear(vision_dim, embed_dim)
        
        self.pre_norm_text = nn.LayerNorm(embed_dim)
        self.pre_norm_audio = nn.LayerNorm(embed_dim)
        self.pre_norm_vision = nn.LayerNorm(embed_dim)
        
        # Positional Encoding for sequence order
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, embed_dim)) # supports up to sequence length 100
        
        # 2. Cross-Attention Layers
        # Text query attends to Audio key/value
        self.text_audio_cross = CrossAttentionLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        
        # Text query attends to Vision key/value
        self.text_vision_cross = CrossAttentionLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        
        # 3. Self-Attention Block for the fused representations
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim * 2, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(embed_dim * 2)
        
        # 4. Final Classification Head
        self.fc1 = nn.Linear(embed_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, audio, vision, return_embeddings=False):
        # text: [batch_size, seq_len, text_dim]
        # audio: [batch_size, seq_len, audio_dim]
        # vision: [batch_size, seq_len, vision_dim]
        
        batch_size, seq_len, _ = text.shape
        
        # Step 1: Project modalities to a shared embedding dimension
        t_proj = self.proj_text(text)      # [batch_size, seq_len, embed_dim]
        a_proj = self.proj_audio(audio)    # [batch_size, seq_len, embed_dim]
        v_proj = self.proj_vision(vision)  # [batch_size, seq_len, embed_dim]
        
        # Step 2: Add Positional Encodings
        # Add position embeddings up to the current sequence length
        pos = self.pos_encoder[:, :seq_len, :]
        t_proj = t_proj + pos
        a_proj = a_proj + pos
        v_proj = v_proj + pos
        
        t_proj = self.pre_norm_text(t_proj)
        a_proj = self.pre_norm_audio(a_proj)
        v_proj = self.pre_norm_vision(v_proj)
        
        # Step 3: Text-Anchored Cross-Attention
        # Text attends to Audio
        if getattr(self, 'ablate_audio', False):
            t_audio_fused = t_proj
            self.last_audio_attn_weights = None
        else:
            t_audio_fused, self.last_audio_attn_weights = self.text_audio_cross(query=t_proj, key=a_proj, value=a_proj) # [batch_size, seq_len, embed_dim]
            t_audio_fused = t_audio_fused + t_proj
        
        # Text attends to Vision
        if getattr(self, 'ablate_vision', False):
            t_vision_fused = t_proj
            self.last_vision_attn_weights = None
        else:
            t_vision_fused, self.last_vision_attn_weights = self.text_vision_cross(query=t_proj, key=v_proj, value=v_proj) # [batch_size, seq_len, embed_dim]
            t_vision_fused = t_vision_fused + t_proj
        
        # Step 4: Combine the fused features (Concatenation)
        # Result shape: [batch_size, seq_len, embed_dim * 2]
        fused = torch.cat((t_audio_fused, t_vision_fused), dim=-1)
        
        # Step 5: Self-Attention on fused representation to capture joint relations
        attn_out, _ = self.self_attn(fused, fused, fused)
        fused_attn = self.self_attn_norm(fused + self.dropout(attn_out))
        
        # Step 6: Mean Pooling across sequence dimension
        pooled = torch.mean(fused_attn, dim=1) # [batch_size, embed_dim * 2]
        
        if return_embeddings:
            return pooled
        
        # Step 7: Classification Head
        out = F.relu(self.fc1(self.dropout(pooled)))
        out = self.fc2(self.dropout(out)) # [batch_size, output_dim]
        
        return out
