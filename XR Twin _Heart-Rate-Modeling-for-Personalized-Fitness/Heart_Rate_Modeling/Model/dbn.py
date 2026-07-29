import torch
import torch.nn as nn
import pandas as pd
from Model.modules_lstm import LSTMEncoder
from Model.modules_dense_nn import DenseNN
from dataclasses import dataclass
from Model.data import WorkoutDatasetConfig

@dataclass
class DBNConfig:
    seq_length: int
    data_config: WorkoutDatasetConfig
    learning_rate: float = 1e-3
    n_epochs: int = 10
    seed: int = 0
    lstm_hidden_dim: int = 128
    lstm_layers: int = 2
    dbn_hidden_dim: int = 64
    personalization: str = "none"
    dim_personalization: int = 8
    subject_embedding_dim: int = 8
    encoder_embedding_dim: int = 8
    dropout: float = 0.2
    clip_gradient: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Ablation switches. Both default to False, which reproduces the model that
    # was actually trained for the paper (linear emission head, no AdaFS).
    use_adafs: bool = False
    use_physiological_head: bool = False
    use_physiological_residual: bool = False
    use_contextual_residual: bool = False
    # AdaFS variant:
    #   "legacy" — original port: controller on flattened T*F window (not paper-faithful)
    #   "paper"  — §4.4 / Fig. 5: controller on latent z from history LSTM, per-step α_n^m
    adafs_variant: str = "legacy"
    # When True, enable the paper-described stack: Eq. 9 emission + paper AdaFS.
    # Residual stays off unless use_physiological_residual is also set.
    paper_faithful: bool = False
    # P2 personalization of Equation 9 (defaults preserve the pre-P2 head).
    # When True, A/B/HRmin/range are functions of subject+history embeddings only
    # (constant over the window), not of the per-step transition state.
    physio_subject_stable_params: bool = False
    # When True, intensity I(t) is f(activity_t, embeddings), so the same speed
    # can map to different effort for different subjects.
    intensity_use_embedding: bool = False
    # Bounds taken from the hybrid-ODE model of ref. [4] (Nazaret et al.,
    # ml-heart-rate-models, ode/ode.py), the codebase this model was adapted from.
    hr_min_bounds: tuple = (40.0, 90.0)
    hr_max_bounds: tuple = (140.0, 210.0)

    def __post_init__(self):
        if self.paper_faithful:
            self.use_adafs = True
            self.use_physiological_head = True
            self.adafs_variant = "paper"
            # Eq. 9 uses z = [subject/history emb, DBN state] so the transition
            # (and AdaFS reweighting of its inputs) receive gradient. Forcing
            # bounds from emb-only would detach AdaFS from the loss.


class AdaFSSoft(nn.Module):
    """
    Legacy AdaFS-soft-inspired port (not paper-faithful).

    Controller sees the entire flattened window (T * F). Prefer AdaFSPaper for §4.4.
    """

    def __init__(self, input_dim, seq_length, dropout):
        super().__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.feature_dim = input_dim // seq_length
        self.controller = ControllerMLP(
            input_dim=input_dim, embed_dims=[input_dim], dropout=dropout, feature_dim=self.feature_dim
        )

    def forward(self, field, latent=None):
        """
        Weight each feature channel at each time step by its learned relevance.

        Expects `field` of shape (batch, seq_length, feature_dim).
        `latent` is ignored (API compatibility with AdaFSPaper).
        """
        if field.dim() != 3:
            raise ValueError(f"Expected (batch, seq_length, feature_dim), got shape {tuple(field.shape)}")
        batch_size, seq_length, feature_dim = field.shape
        if seq_length != self.seq_length or feature_dim != self.feature_dim:
            raise ValueError(
                f"Expected (batch, {self.seq_length}, {self.feature_dim}), got {tuple(field.shape)}"
            )

        field = field / field.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        weights = self.controller(field.reshape(batch_size, -1))
        weights = weights.view(batch_size, seq_length, feature_dim)

        # Rescale so the average weight is 1: a plain softmax over feature_dim
        # forces the weights to sum to 1 and would shrink the signal by ~1/feature_dim
        # before it ever reaches the transition model.
        return field * weights * feature_dim


class AdaFSPaper(nn.Module):
    """
    Paper-faithful adaptive feature selection (§4.4, Figure 5).

    Controller is an MLP on the personalized latent vector z (subject + history
    encoder embedding). It produces per-feature, per-time weights α_n^m and
    reweights the DBN input features:

        features̃_{t,n} = features_{t,n} * α_n^m(z)

    Unlike the legacy port, this does not flatten the full T*F window into one
    BatchNorm MLP (which is neither in the paper nor in original AdaFS for
    time series). Softmax is over feature channels at each step.
    """

    def __init__(self, latent_dim, feature_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.controller = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, field, latent):
        """
        Parameters
        ----------
        field : Tensor (batch, T, F)
            Features to reweight (embeddings + activity + time).
        latent : Tensor (batch, T, L) or (batch, L)
            Personalized latent z from the history/subject encoder.
        """
        if field.dim() != 3:
            raise ValueError(f"Expected field (B, T, F), got {tuple(field.shape)}")
        if latent.dim() == 2:
            latent = latent.unsqueeze(1).expand(-1, field.size(1), -1)
        elif latent.dim() != 3:
            raise ValueError(f"Expected latent (B, L) or (B, T, L), got {tuple(latent.shape)}")

        # Align lengths
        t = min(field.size(1), latent.size(1))
        field = field[:, :t, :]
        latent = latent[:, :t, :]

        logits = self.controller(latent)  # (B, T, F)
        alpha = torch.softmax(logits, dim=-1)
        # Keep average scale ~1 after softmax (sum_F alpha = 1)
        return field * alpha * self.feature_dim


class ControllerMLP(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.mlp = MultiLayerPerceptron(input_dim=input_dim, embed_dims=embed_dims, dropout=dropout)

    def forward(self, emb_fields):
        output_layer = self.mlp(emb_fields)
        # Normalize across feature channels within each time step, not across the
        # whole flattened sequence.
        output_layer = output_layer.view(output_layer.size(0), -1, self.feature_dim)
        return torch.softmax(output_layer, dim=-1).reshape(output_layer.size(0), -1)

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=False):
        super().__init__()
        layers = []
        self.mlps = nn.ModuleList()
        self.out_layer = output_layer
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = embed_dim
            self.mlps.append(nn.Sequential(*layers))
            layers = []
        if self.out_layer:
            self.out = nn.Linear(input_dim, 1)

    def forward(self, x):
        for layer in self.mlps:
            x = layer(x)
        if self.out_layer:
            x = self.out(x)
        return x

class EmbeddingStore(nn.Module):
    def __init__(self, config, workouts_info):
        super().__init__()
        self.subject_id_column = config.data_config.subject_id_column
        self.workout_id_column = config.data_config.workout_id_column
        self.workouts_info = workouts_info[[self.subject_id_column, self.workout_id_column]]
        self.subject_embedding_dim = config.subject_embedding_dim
        self.initialize_subject_embeddings()
        self.encoder_input_dim = config.data_config.history_dim()
        self.encoder_embedding_dim = config.encoder_embedding_dim
        self.encoder = LSTMEncoder(self.encoder_input_dim, config.lstm_hidden_dim, config.lstm_layers, self.encoder_embedding_dim, dropout=config.dropout)
        self.dim_embedding = self.subject_embedding_dim + self.encoder_embedding_dim

    def initialize_subject_embeddings(self):
        unique_subject_ids = self.workouts_info[self.subject_id_column].unique()
        self.n_subject_embeddings = len(unique_subject_ids)
        self.subject_id_to_embedding_index = {s_id: idx for idx, s_id in enumerate(unique_subject_ids)}
        self.workout_id_to_embedding_index = {w_id: self.subject_id_to_embedding_index[s_id] for s_id, w_id in self.workouts_info[[self.subject_id_column, self.workout_id_column]].values}
        self.subject_embeddings = nn.Embedding(self.n_subject_embeddings, self.subject_embedding_dim, max_norm=5.0)

    def get_embeddings_from_workout_ids(self, workout_ids, history=None, history_lengths=None):
        embeddings = []
        device = next(self.parameters()).device
        if self.subject_embeddings is not None:
            try:
                subject_indices = [self.workout_id_to_embedding_index[wid.item()] for wid in workout_ids]
            except KeyError as e:
                print(f"Workout ID not found in the embedding index: {e}")
                return None  # Handle missing workout IDs gracefully
            subject_embeddings = self.subject_embeddings(torch.LongTensor(subject_indices).to(device))
            embeddings.append(subject_embeddings)
        if self.encoder is not None and history is not None:
            encoded_embeddings = self.encoder(history, history_lengths)
            embeddings.append(encoded_embeddings)
        embeddings = torch.cat(embeddings, dim=-1)
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1).expand(-1, history.size(1), -1)
        return embeddings
    
class TransitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TransitionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

class EmissionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EmissionModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        output = self.fc(x)
        return output

class DBNModel(nn.Module):
    def __init__(self, config, workouts_info):
        super(DBNModel, self).__init__()
        self.config = config
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
        self.embedding_store = EmbeddingStore(self.config, workouts_info)
        self.dim_embedding = self.embedding_store.dim_embedding

        input_dim = self.dim_embedding + config.data_config.n_activity_channels() + 1
        self.seq_length = config.seq_length
        self.flattened_input_dim = input_dim * self.seq_length

        self.lstm_encoder = LSTMEncoder(config.data_config.history_dim(), config.lstm_hidden_dim, config.lstm_layers, output_dim=config.encoder_embedding_dim, dropout=config.dropout, bidirectional=True)
        # Legacy AdaFS (flattened window) kept for ablation of the old port.
        self.adafs_soft = AdaFSSoft(input_dim=self.flattened_input_dim, seq_length=self.seq_length, dropout=config.dropout)
        # Paper §4.4 AdaFS: controller on latent z (subject + history embedding).
        self.adafs_paper = AdaFSPaper(
            latent_dim=self.dim_embedding,
            feature_dim=input_dim,
            hidden_dim=64,
            dropout=config.dropout,
        )

        self.transition_model = TransitionModel(input_dim, config.lstm_hidden_dim * 2, config.encoder_embedding_dim)
        self.emission_model = EmissionModel(config.encoder_embedding_dim, 1)
        residual_input_dim = config.encoder_embedding_dim
        if config.use_contextual_residual:
            residual_input_dim += self.dim_embedding + config.data_config.n_activity_channels() + 1
        self.residual_model = EmissionModel(residual_input_dim, 1)

        # Personalized scalars for Equation 9.
        # Default: z = [embeddings, state] (per-step). P2 subject-stable: embeddings only.
        self.dim_z = self.dim_embedding + config.encoder_embedding_dim
        param_input_dim = self.dim_embedding if config.physio_subject_stable_params else self.dim_z
        # Keep the Equation 9 exponent trainable. Unbounded positive A/B/I often
        # makes exp(-A-BI) collapse to zero at initialization, which turns the
        # physiological head into a saturated HRmax predictor with no useful
        # gradient for A and B.
        self.A = DenseNN(param_input_dim, 32, 8, 1, activation=nn.ReLU(), output_bounds=(1e-3, 2.0), activate_output=False)
        self.B = DenseNN(param_input_dim, 32, 8, 1, activation=nn.ReLU(), output_bounds=(1e-3, 2.0), activate_output=False)
        self.hr_min = DenseNN(param_input_dim, 32, 8, 1, activation=nn.ReLU(),
                              output_bounds=config.hr_min_bounds, activate_output=False)
        self.hr_range = DenseNN(param_input_dim, 32, 8, 1, activation=nn.ReLU(),
                                output_bounds=(1.0, config.hr_max_bounds[1] - config.hr_min_bounds[0]),
                                activate_output=False)
        # Intensity I(t): activity only (default) or activity + embeddings (P2).
        intensity_input_dim = config.data_config.n_activity_channels()
        if config.intensity_use_embedding:
            intensity_input_dim += self.dim_embedding
        self.intensity = DenseNN(intensity_input_dim, 16 if config.intensity_use_embedding else 8, 1,
                                 activation=nn.ReLU(), output_bounds=(0.0, 1.0), activate_output=False)

        self.to(self.config.device)

    def physiological_head(self, embeddings, state, activity):
        """
        Equation 9: HR(t) = HRmin + (HRmax - HRmin) * (1 - exp(-A - B * I(t)))

        HRmax is parameterized as HRmin + hr_range so that HRmax > HRmin holds by
        construction rather than by hoping the optimizer respects it.

        With physio_subject_stable_params, A/B/HRmin/range depend only on the
        subject+history embedding (constant over the window). With
        intensity_use_embedding, I(t) is personalized: f(activity_t, emb).
        """
        if self.config.physio_subject_stable_params:
            param_input = embeddings
        else:
            param_input = torch.cat([embeddings, state], dim=-1)
        a = self.A(param_input)
        b = self.B(param_input)
        hr_min = self.hr_min(param_input)
        hr_max = hr_min + self.hr_range(param_input)
        if self.config.intensity_use_embedding:
            intensity = self.intensity(torch.cat([activity, embeddings], dim=-1))
        else:
            intensity = self.intensity(activity)
        return (hr_min + (hr_max - hr_min) * (1.0 - torch.exp(-a - b * intensity))).squeeze(-1)

    def align_to_seq_length(self, values):
        if values.size(1) > self.seq_length:
            return values[:, : self.seq_length, :]
        if values.size(1) < self.seq_length:
            pad_size = self.seq_length - values.size(1)
            padding = torch.zeros(values.size(0), pad_size, values.size(2), device=values.device)
            return torch.cat([values, padding], dim=1)
        return values

    def forecast_single_workout(self, workout):
        """
        Forecast heart rate for a single workout.
        """
        activity = torch.tensor(workout['activity']).unsqueeze(0).float().to(self.config.device)
        times = torch.tensor(workout['time']).unsqueeze(0).float().to(self.config.device)
        history = torch.tensor(workout['history']).unsqueeze(0).float().to(self.config.device) if 'history' in workout else None
        history_length = torch.tensor(workout['history_length']).unsqueeze(0).float().to(self.config.device) if 'history_length' in workout else None
        workout_id = workout['workout_id']
        subject_id = workout['subject_id']
        
        # Generate predictions
        self.eval()
        with torch.no_grad():
            pred_hr = self.forecast_batch(activity, times, torch.tensor([workout_id]).to(self.config.device), torch.tensor([subject_id]).to(self.config.device), history, history_length).cpu().numpy().flatten()
        
        return {"heart_rate": pred_hr}

    def forecast_batch(self, activity, times, workout_id, subject_id, history=None, history_length=None):
        embeddings = self.embedding_store.get_embeddings_from_workout_ids(workout_id, history, history_length)

        if embeddings is None:
            raise ValueError("Embeddings could not be generated due to missing workout IDs.")

        if embeddings.size(1) != activity.size(1):
            if embeddings.size(1) > activity.size(1):
                embeddings = embeddings[:, :activity.size(1), :]
            else:
                pad_size = activity.size(1) - embeddings.size(1)
                embeddings = torch.cat([embeddings, torch.zeros(embeddings.size(0), pad_size, embeddings.size(2)).to(embeddings.device)], dim=1)

        combined_features = torch.cat([embeddings, activity, times.unsqueeze(-1)], dim=-1)

        if combined_features.size(1) != self.seq_length:
            if combined_features.size(1) > self.seq_length:
                combined_features = combined_features[:, :self.seq_length, :]
            else:
                pad_size = self.seq_length - combined_features.size(1)
                combined_features = torch.cat([combined_features, torch.zeros(combined_features.size(0), pad_size, combined_features.size(2)).to(combined_features.device)], dim=1)

        combined_features = combined_features.view(combined_features.size(0), self.seq_length, -1)
        emb = self.align_to_seq_length(embeddings)

        if self.config.use_adafs:
            if self.config.adafs_variant == "paper":
                # §4.4: controller on latent z (history/subject embedding), α_n^m on features.
                combined_features = self.adafs_paper(combined_features, emb)
            else:
                combined_features = self.adafs_soft(combined_features)

        state_predictions = self.transition_model(combined_features)

        if self.config.use_physiological_head:
            # Re-slice the inputs the head needs to the same length as the state.
            act = self.align_to_seq_length(activity)
            # Paper: A(z), B(z) from personalized latent z. Default z = [emb, state].
            physiological_predictions = self.physiological_head(emb, state_predictions, act)
            if self.config.use_physiological_residual:
                residual_input = state_predictions
                if self.config.use_contextual_residual:
                    t = self.align_to_seq_length(times.unsqueeze(-1))
                    residual_input = torch.cat([state_predictions, emb, act, t], dim=-1)
                residual = self.residual_model(residual_input).squeeze(-1)
                return physiological_predictions + residual
            return physiological_predictions

        predictions = self.emission_model(state_predictions)
        return predictions.view(predictions.size(0), -1)


def load_compatible_state_dict(model, state_dict):
    """
    Load checkpoints across ablation-only module additions.

    The model always instantiates optional heads/AdaFS variants so tests can
    verify them, but older checkpoints legitimately lack inactive modules. Treat
    inactive missing keys as compatible while still failing for live paths.
    """
    result = model.load_state_dict(state_dict, strict=False)

    live_prefixes = ["embedding_store.", "transition_model."]
    if model.config.use_physiological_head:
        live_prefixes.extend(["A.", "B.", "hr_min.", "hr_range.", "intensity."])
        if model.config.use_physiological_residual:
            live_prefixes.append("residual_model.")
    else:
        live_prefixes.append("emission_model.")

    if model.config.use_adafs:
        if model.config.adafs_variant == "paper":
            live_prefixes.append("adafs_paper.")
        else:
            live_prefixes.append("adafs_soft.")

    missing_live = [
        key for key in result.missing_keys
        if any(key.startswith(prefix) for prefix in live_prefixes)
    ]
    if missing_live:
        raise RuntimeError(f"Live checkpoint parameters missing: {missing_live}")
    return result
