import torch
import torch.nn as nn
import torch.nn.functional as F

class EarlyFusionModel(nn.Module):
    """
    Early Fusion (Feature-Level Fusion) baseline model.
    Concatenates visual, acoustic, and text features at the input level,
    projects them, runs through an LSTM/GRU layer, and classifies.
    """
    def __init__(self, text_dim, audio_dim, vision_dim, hidden_dim, output_dim=1, dropout=0.3):
        super(EarlyFusionModel, self).__init__()
        
        # Project all modalities to a unified input dimension
        self.input_dim = text_dim + audio_dim + vision_dim
        
        # LSTM/GRU sequence processor
        self.rnn = nn.GRU(
            input_size=self.input_dim, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, audio, vision):
        # text: [batch_size, seq_len, text_dim]
        # audio: [batch_size, seq_len, audio_dim]
        # vision: [batch_size, seq_len, vision_dim]
        
        # Concatenate features at the feature dimension
        # resulting shape: [batch_size, seq_len, text_dim + audio_dim + vision_dim]
        x = torch.cat((text, audio, vision), dim=-1)
        
        # Pass through RNN
        # rnn_out shape: [batch_size, seq_len, hidden_dim * 2]
        rnn_out, _ = self.rnn(x)
        
        # Pool across sequence dimension (e.g. mean pooling or taking the last step)
        pooled = torch.mean(rnn_out, dim=1) # [batch_size, hidden_dim * 2]
        
        # Classification head
        out = F.relu(self.fc1(self.dropout(pooled)))
        out = self.fc2(self.dropout(out)) # [batch_size, output_dim]
        
        return out


class UnimodalEncoder(nn.Module):
    """Simple sequence encoder for a single modality."""
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super(UnimodalEncoder, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        pooled = torch.mean(rnn_out, dim=1) # mean pool
        out = F.relu(self.fc(self.dropout(pooled)))
        return out


class LateFusionModel(nn.Module):
    """
    Late Fusion (Decision-Level Fusion) baseline model.
    Processes each modality using independent unimodal encoders,
    projects their pooled features to decision layers, and averages/combines the decisions.
    """
    def __init__(self, text_dim, audio_dim, vision_dim, hidden_dim, output_dim=1, dropout=0.3):
        super(LateFusionModel, self).__init__()
        
        # Separate unimodal encoders
        self.text_encoder = UnimodalEncoder(text_dim, hidden_dim, dropout)
        self.audio_encoder = UnimodalEncoder(audio_dim, hidden_dim, dropout)
        self.vision_encoder = UnimodalEncoder(vision_dim, hidden_dim, dropout)
        
        # Individual unimodal decision heads
        self.text_head = nn.Linear(hidden_dim, output_dim)
        self.audio_head = nn.Linear(hidden_dim, output_dim)
        self.vision_head = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text, audio, vision):
        # Extract features independently
        t_feat = self.text_encoder(text)    # [batch_size, hidden_dim]
        a_feat = self.audio_encoder(audio)  # [batch_size, hidden_dim]
        v_feat = self.vision_encoder(vision)  # [batch_size, hidden_dim]
        
        # Get individual unimodal predictions
        t_pred = self.text_head(t_feat)  # [batch_size, output_dim]
        a_pred = self.audio_head(a_feat)  # [batch_size, output_dim]
        v_pred = self.vision_head(v_feat)  # [batch_size, output_dim]
        
        # Average the final predictions (Late Fusion voting)
        final_pred = (t_pred + a_pred + v_pred) / 3.0
        
        return final_pred
