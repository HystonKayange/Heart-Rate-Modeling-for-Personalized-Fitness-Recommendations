import torch
import torch.nn as nn
import pandas as pd
from Model.modules_lstm import LSTMEncoder
from Model.modules_dense_nn import DenseNN, PersonalizedScalarNN
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
    # Bounds taken from the hybrid-ODE model of ref. [4] (Nazaret et al.,
    # ml-heart-rate-models, ode/ode.py), the codebase this model was adapted from.
    hr_min_bounds: tuple = (40.0, 90.0)
    hr_max_bounds: tuple = (140.0, 210.0)

class AdaFSSoft(nn.Module):
    def __init__(self, input_dim, seq_length, dropout):
        super().__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.feature_dim = input_dim // seq_length
        self.controller = ControllerMLP(
            input_dim=input_dim, embed_dims=[input_dim], dropout=dropout, feature_dim=self.feature_dim
        )

    def forward(self, field):
        """
        Weight each feature channel at each time step by its learned relevance.

        Expects `field` of shape (batch, seq_length, feature_dim), matching the
        (z)(alpha_n^m) formulation in the paper: one weight per feature n at each
        step m.
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
            encoded_embeddings = self.encoder(history)
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
        self.adafs_soft = AdaFSSoft(input_dim=self.flattened_input_dim, seq_length=self.seq_length, dropout=config.dropout)

        self.transition_model = TransitionModel(input_dim, config.lstm_hidden_dim * 2, config.encoder_embedding_dim)
        self.emission_model = EmissionModel(config.encoder_embedding_dim, 1)

        # Personalized scalars for Equation 9. Their input is the personalized
        # latent z: the static subject/history embedding concatenated with the
        # per-step DBN state.
        self.dim_z = self.dim_embedding + config.encoder_embedding_dim
        self.A = PersonalizedScalarNN(self.dim_z, 32, 8, 1, activation=nn.ReLU(), output_activation=nn.Softplus())
        self.B = PersonalizedScalarNN(self.dim_z, 32, 8, 1, activation=nn.ReLU(), output_activation=nn.Softplus())
        self.hr_min = DenseNN(self.dim_z, 32, 8, 1, activation=nn.ReLU(), output_bounds=config.hr_min_bounds)
        self.hr_range = DenseNN(self.dim_z, 32, 8, 1, activation=nn.ReLU(),
                                output_bounds=(1.0, config.hr_max_bounds[1] - config.hr_min_bounds[0]))
        # Scalar exercise intensity I(t) derived from the activity channels.
        self.intensity = DenseNN(config.data_config.n_activity_channels(), 8, 1,
                                 activation=nn.ReLU(), output_activation=nn.Softplus())

        self.to(self.config.device)

    def physiological_head(self, embeddings, state, activity):
        """
        Equation 9: HR(t) = HRmin + (HRmax - HRmin) * (1 - exp(-A(z) - B(z) * I(t)))

        HRmax is parameterized as HRmin + hr_range so that HRmax > HRmin holds by
        construction rather than by hoping the optimizer respects it.
        """
        z = torch.cat([embeddings, state], dim=-1)
        a = self.A(z)
        b = self.B(z)
        hr_min = self.hr_min(z)
        hr_max = hr_min + self.hr_range(z)
        intensity = self.intensity(activity)
        return (hr_min + (hr_max - hr_min) * (1.0 - torch.exp(-a - b * intensity))).squeeze(-1)

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

        if self.config.use_adafs:
            combined_features = self.adafs_soft(combined_features)

        state_predictions = self.transition_model(combined_features)

        if self.config.use_physiological_head:
            # Re-slice the inputs the head needs to the same length as the state.
            emb = embeddings[:, : self.seq_length, :]
            act = activity[:, : self.seq_length, :]
            if emb.size(1) < self.seq_length:
                emb = torch.cat([emb, torch.zeros(emb.size(0), self.seq_length - emb.size(1), emb.size(2), device=emb.device)], dim=1)
            if act.size(1) < self.seq_length:
                act = torch.cat([act, torch.zeros(act.size(0), self.seq_length - act.size(1), act.size(2), device=act.device)], dim=1)
            return self.physiological_head(emb, state_predictions, act)

        predictions = self.emission_model(state_predictions)
        return predictions.view(predictions.size(0), -1)
