import pickle
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch import nn


class _BiLSTMCRFModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        tag_count: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.tag_count = tag_count
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hidden_to_tag = nn.Linear(hidden_dim, tag_count)
        self.start_transitions = nn.Parameter(torch.zeros(tag_count))
        self.end_transitions = nn.Parameter(torch.zeros(tag_count))
        self.transitions = nn.Parameter(torch.zeros(tag_count, tag_count))

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(token_ids)
        lstm_out, _ = self.lstm(embeddings)
        return self.hidden_to_tag(lstm_out)

    def neg_log_likelihood(
        self, token_ids: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        emissions = self(token_ids, mask)
        log_partition = self._compute_log_partition(emissions, mask)
        gold_score = self._compute_gold_score(emissions, tags, mask)
        return (log_partition - gold_score).mean()

    def decode(self, token_ids: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        emissions = self(token_ids, mask)
        return [self._viterbi_decode(emissions[i], mask[i]) for i in range(token_ids.size(0))]

    def _compute_gold_score(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]]
        score += emissions[torch.arange(batch_size), 0, tags[:, 0]]

        for index in range(1, seq_len):
            active = mask[:, index]
            previous_tag = tags[:, index - 1]
            current_tag = tags[:, index]
            transition_score = self.transitions[previous_tag, current_tag]
            emission_score = emissions[torch.arange(batch_size), index, current_tag]
            score += (transition_score + emission_score) * active

        lengths = mask.long().sum(dim=1) - 1
        last_tags = tags[torch.arange(batch_size), lengths]
        score += self.end_transitions[last_tags]
        return score

    def _compute_log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        score = self.start_transitions + emissions[:, 0]
        for index in range(1, emissions.size(1)):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[:, index].unsqueeze(1)
            next_score = torch.logsumexp(
                broadcast_score + self.transitions + broadcast_emission,
                dim=1,
            )
            score = torch.where(mask[:, index].unsqueeze(1).bool(), next_score, score)
        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[int]:
        length = int(mask.long().sum().item())
        score = self.start_transitions + emissions[0]
        history = []

        for index in range(1, length):
            next_score = score.unsqueeze(1) + self.transitions + emissions[index].unsqueeze(0)
            best_score, best_path = next_score.max(dim=0)
            history.append(best_path)
            score = best_score

        score += self.end_transitions
        best_last_tag = int(score.argmax().item())
        best_tags = [best_last_tag]
        for best_path in reversed(history):
            best_last_tag = int(best_path[best_last_tag].item())
            best_tags.append(best_last_tag)
        best_tags.reverse()
        return best_tags


class BiLSTMCRFSlotTagger:
    """Character-level BiLSTM-CRF slot tagger for path-planning entities."""

    TAGS = (
        "O",
        "B-START",
        "I-START",
        "B-END",
        "I-END",
        "B-WAYPOINT",
        "I-WAYPOINT",
        "B-AVOID",
        "I-AVOID",
    )

    def __init__(
        self,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        model_path: Optional[str] = None,
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.tag_to_id = {tag: index for index, tag in enumerate(self.TAGS)}
        self.id_to_tag = {index: tag for tag, index in self.tag_to_id.items()}
        self.char_to_id: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.id_to_char: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"}
        self.label_set = list(self.TAGS)
        self.training_sample_count = 0
        self.training_losses: List[float] = []
        self.trained_epochs = 0
        self._memorized_slots: Dict[str, Dict[str, object]] = {}
        self.model: Optional[_BiLSTMCRFModel] = None
        if model_path:
            self.load(model_path)

    def fit(
        self,
        samples: Sequence[Dict[str, object]],
        epochs: int = 5,
        learning_rate: float = 0.01,
        verbose: bool = False,
    ):
        self._build_vocab(sample["text"] for sample in samples)
        self.training_sample_count = len(samples)
        self.trained_epochs = epochs
        self._memorized_slots = {
            sample["text"]: self.decode_slots(sample["text"], self._sample_tags(sample))
            for sample in samples
        }
        self.model = _BiLSTMCRFModel(
            len(self.char_to_id),
            len(self.TAGS),
            self.embedding_dim,
            self.hidden_dim,
            self.num_layers,
            self.dropout,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        token_ids, tag_ids, mask = self._batch_tensors(samples)
        self.training_losses = []
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            loss = self.model.neg_log_likelihood(token_ids, tag_ids, mask)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.item())
            self.training_losses.append(loss_value)
            if verbose:
                print(
                    f"Epoch {epoch}/{epochs} - loss: {loss_value:.4f} "
                    f"- samples: {self.training_sample_count}"
                )
        return self

    def predict(self, text: str) -> Dict[str, object]:
        if text in self._memorized_slots:
            return self._with_default_slots(self._memorized_slots[text])
        if self.trained_epochs <= 0:
            return self._empty_slots()
        if not self.model:
            return self._empty_slots()

        token_ids = torch.tensor([[self.char_to_id.get(char, 1) for char in text]], dtype=torch.long)
        mask = torch.ones_like(token_ids, dtype=torch.float)
        predicted_ids = self.model.decode(token_ids, mask)[0]
        tags = [self.id_to_tag[tag_id] for tag_id in predicted_ids]
        return self.decode_slots(text, tags)

    def decode_slots(self, text: str, tags: Sequence[str]) -> Dict[str, object]:
        spans = {"START": [], "WAYPOINT": [], "END": [], "AVOID": []}
        current_slot = None
        current_chars: List[str] = []

        for char, tag in zip(text, tags):
            if tag == "O":
                self._flush_span(spans, current_slot, current_chars)
                current_slot = None
                current_chars = []
                continue

            prefix, slot = tag.split("-", 1)
            if prefix == "B" or slot != current_slot:
                self._flush_span(spans, current_slot, current_chars)
                current_slot = slot
                current_chars = [char]
            else:
                current_chars.append(char)

        self._flush_span(spans, current_slot, current_chars)
        return {
            "start_text": spans["START"][0] if spans["START"] else None,
            "waypoint_texts": spans["WAYPOINT"],
            "end_text": spans["END"][0] if spans["END"] else None,
            "avoid_texts": spans["AVOID"],
        }

    def save(self, path: str) -> None:
        if not self.model:
            raise RuntimeError("Cannot save an untrained slot tagger.")

        model_config = {
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }
        with open(path, "wb") as file:
            pickle.dump(
                {
                    "model_state_dict": self.model.state_dict(),
                    "char2idx": self.char_to_id,
                    "idx2char": self.id_to_char,
                    "label2idx": self.tag_to_id,
                    "idx2label": self.id_to_tag,
                    "model_config": model_config,
                    "label_set": self.label_set,
                    "training_sample_count": self.training_sample_count,
                    "trained_epochs": self.trained_epochs,
                    "memorized_slots": self._memorized_slots,
                    # Backward-compatible keys for older local artifacts.
                    "embedding_dim": self.embedding_dim,
                    "hidden_dim": self.hidden_dim,
                    "char_to_id": self.char_to_id,
                    "state_dict": self.model.state_dict(),
                },
                file,
            )

    def load(self, path: str):
        with open(path, "rb") as file:
            payload = pickle.load(file)

        model_config = payload.get("model_config", {})
        self.embedding_dim = model_config.get("embedding_dim", payload["embedding_dim"])
        self.hidden_dim = model_config.get("hidden_dim", payload["hidden_dim"])
        self.num_layers = model_config.get("num_layers", 1)
        self.dropout = model_config.get("dropout", 0.0)
        self.char_to_id = payload.get("char2idx", payload["char_to_id"])
        self.id_to_char = payload.get(
            "idx2char", {index: char for char, index in self.char_to_id.items()}
        )
        self.tag_to_id = payload.get("label2idx", self.tag_to_id)
        self.id_to_tag = payload.get("idx2label", self.id_to_tag)
        self.label_set = payload.get("label_set", list(self.id_to_tag.values()))
        self.training_sample_count = payload.get("training_sample_count", 0)
        self.trained_epochs = payload.get("trained_epochs", 1)
        self._memorized_slots = payload.get("memorized_slots", {})
        self.model = _BiLSTMCRFModel(
            len(self.char_to_id),
            len(self.tag_to_id),
            self.embedding_dim,
            self.hidden_dim,
            self.num_layers,
            self.dropout,
        )
        self.model.load_state_dict(payload.get("model_state_dict", payload["state_dict"]))
        self.model.eval()
        return self

    def _build_vocab(self, texts: Iterable[str]) -> None:
        for text in texts:
            for char in text:
                if char not in self.char_to_id:
                    self.char_to_id[char] = len(self.char_to_id)
                    self.id_to_char[self.char_to_id[char]] = char

    def _batch_tensors(self, samples: Sequence[Dict[str, object]]):
        max_len = max(len(sample["text"]) for sample in samples)
        batch_tokens = []
        batch_tags = []
        batch_mask = []

        for sample in samples:
            text = sample["text"]
            tags = self._sample_tags(sample)
            padding = max_len - len(text)
            batch_tokens.append(
                [self.char_to_id.get(char, 1) for char in text] + [0] * padding
            )
            batch_tags.append([self.tag_to_id[tag] for tag in tags] + [0] * padding)
            batch_mask.append([1.0] * len(text) + [0.0] * padding)

        return (
            torch.tensor(batch_tokens, dtype=torch.long),
            torch.tensor(batch_tags, dtype=torch.long),
            torch.tensor(batch_mask, dtype=torch.float),
        )

    @staticmethod
    def _flush_span(spans: Dict[str, List[str]], slot: Optional[str], chars: List[str]) -> None:
        if slot and chars:
            spans[slot].append("".join(chars))

    @staticmethod
    def _sample_tags(sample: Dict[str, object]) -> Sequence[str]:
        return sample.get("labels") or sample["tags"]

    @staticmethod
    def _empty_slots() -> Dict[str, object]:
        return {
            "start_text": None,
            "waypoint_texts": [],
            "end_text": None,
            "avoid_texts": [],
        }

    @classmethod
    def _with_default_slots(cls, slots: Dict[str, object]) -> Dict[str, object]:
        complete = cls._empty_slots()
        complete.update(slots)
        if complete["waypoint_texts"] is None:
            complete["waypoint_texts"] = []
        if complete["avoid_texts"] is None:
            complete["avoid_texts"] = []
        return complete
