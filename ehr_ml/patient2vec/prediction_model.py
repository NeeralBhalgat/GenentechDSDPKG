from typing import Any, Mapping

import torch
from torch import nn
from torch.nn.modules import transformer
import torch.nn.functional as F

# Import embedding_dot here:
#import embedding_dot

#from .labeler_task import LabelerTask
#from .rnn_model import PatientRNN
#from .sequential_task import SequentialTask
import pandas as pd
from collections import defaultdict, deque
from typing import Any, Dict, Mapping, Sequence, Tuple

# Import output.csv here.

import math
from torch import nn
from torch.autograd import Function
import torch

if torch.cuda.is_available():
    import embedding_dot_cuda

class EmbeddingDotFunction(Function):
    @staticmethod
    def forward(ctx, embedding1, embedding2, indices):
        ctx.save_for_backward(embedding1, embedding2, indices)
        if indices.shape[0] == 0:
            # There are no indices. This needs some special casing
            return torch.zeros(size=(0,), dtype=embedding1.dtype, device=embedding1.device)
        return embedding_dot_cuda.forward(embedding1, embedding2, indices)

    @staticmethod
    def backward(ctx, grad_output):
        embedding1, embedding2, indices = ctx.saved_variables
        if indices.shape[0] == 0:
            return torch.zeros_like(embedding1), torch.zeros_like(embedding2), None
        embedding1_grad, embedding2_grad = embedding_dot_cuda.backward(grad_output, embedding1, embedding2, indices)
        return embedding1_grad, embedding2_grad, None

def embedding_dot(embedding1, embedding2, indices):
    if embedding1.is_cuda:
        return EmbeddingDotFunction.apply(embedding1, embedding2, indices)
    else:
        A = nn.functional.embedding(indices[:, 0], embedding1)
        B = nn.functional.embedding(indices[:, 1], embedding2)

        return torch.sum(A * B, dim=1)

# Load the CSV file
data_path = 'ehr_ml/patient2vec/output.csv'
df = pd.read_csv(data_path)
n = 20

class SequentialTask(nn.Module):
    """
    This is paired with an encoder that outputs an encoding for each timestep.  
    This is the output (and loss) module for that encoder.  An example of an 
    encoder is PatientRNN in rnn_model.py.  
    """

    def __init__(self, config: Mapping[str, Any], info: Mapping[str, Any]):
        super().__init__()
        self.config = config
        self.info = info

        # self.scale_and_scale_function = nn.Linear(config['size'], config['num_valid_targets'] * 3)
        # self.scale_and_scale_function = nn.Linear(config['size'], config['num_valid_targets'] * 2)
        # self.delta_function = nn.Linear(config['size'], config['num_valid_targets'] * 2 * n)

        self.main_weights = torch.nn.Embedding(
            config["num_valid_targets"], config["size"] + 1
        )

        self.sub_weights = torch.nn.Embedding(
            config["num_valid_targets"] * 2 * n, 200 + 1
        )

        # self.alpha = 1
        # print('Alpha:', self.alpha)

    def forward(self, rnn_output: torch.Tensor, data: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        indices, labels, fracs, other_indices = data

        reshaped_rnn_output = rnn_output.view(-1, self.config["size"])

        bias_tensor = torch.ones(
            reshaped_rnn_output.shape[0], 1, device=reshaped_rnn_output.device
        )

        rnn_with_bias = torch.cat([bias_tensor, reshaped_rnn_output], dim=1)

        subset_rnn_with_bias = rnn_with_bias[:, :201].contiguous()

        logits = embedding_dot(
            rnn_with_bias, self.main_weights.weight, other_indices
        ) + embedding_dot(
            subset_rnn_with_bias, self.sub_weights.weight, indices
        )

        frac_logits = logits[: len(fracs)]
        frac_labels = labels[: len(fracs)]

        frac_probs = torch.sigmoid(frac_logits) * fracs
        frac_loss = torch.nn.functional.binary_cross_entropy(
            frac_probs, frac_labels, reduction="sum"
        )

        nonfrac_logits = logits[len(fracs) :]
        nonfrac_labels = labels[len(fracs) :]

        nonfrac_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            nonfrac_logits, nonfrac_labels, reduction="sum"
        )

        total_loss = (frac_loss + nonfrac_loss) / len(labels)

        return logits, total_loss

    @classmethod
    def finalize_data(
        cls,
        config: Mapping[str, Any],
        info: Mapping[str, Any],
        device: torch.device,
        initial: Sequence[torch.Tensor],
    ) -> Sequence[torch.Tensor]:
        a, b, c, d = initial
        a = torch.tensor(a, device=device, dtype=torch.int64)
        b = torch.tensor(b, device=device, dtype=torch.float)
        c = torch.tensor(c, device=device, dtype=torch.float)
        d = torch.tensor(d, device=device, dtype=torch.int64)
        return (a, b, c, d)


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 12,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        print("Got", d_model, nhead, num_encoder_layers)

        encoder_layer = transformer.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        encoder_norm = transformer.LayerNorm(d_model)
        self.encoder = transformer.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, src: torch.Tensor) -> torch.Tensor:  # type: ignore
        device = src.device
        mask = self._generate_square_subsequent_mask(len(src)).to(device)
        return self.encoder(src, mask)


class PatientRNN(nn.Module):
    def __init__(self, config: Mapping[str, Any], info: Mapping[str, Any]):
        super().__init__()
        self.config = config
        self.info = info

        self.input_code_embedding = nn.EmbeddingBag(
            config["num_first"] + 1, config["size"] + 1, mode="mean"
        )

        self.input_code_embedding1 = nn.EmbeddingBag(
            config["num_second"] + 1, (config["size"] // 4) + 1, mode="mean"
        )

        self.input_code_embedding.weight.data.normal_(mean=0.0, std=0.02)
        self.input_code_embedding1.weight.data.normal_(mean=0.0, std=0.02)

        self.norm = nn.LayerNorm(config["size"])
        self.drop = nn.Dropout(config["dropout"])

        self.model: nn.Module

        if config["use_gru"]:
            input_size = config["size"]
            self.model = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=config["size"],
                num_layers=config["gru_layers"],
                dropout=config["dropout"] if config["gru_layers"] > 1 else 0,
            )

        else:
            print("Transformer")
            self.model = Encoder(d_model=config["size"])

    def forward(self, rnn_input: Sequence[Any]) -> torch.Tensor:  # type: ignore
        (
            all_non_text_codes,
            all_non_text_offsets,
            all_non_text_codes1,
            all_non_text_offsets1,
            all_day_information,
            all_positional_encoding,
            all_lengths,
        ) = rnn_input

        size_for_embedding = (
            (self.config["size"] - 5)
            if self.config["use_gru"]
            else (self.config["size"] - 5 - 200)
        )

        embedded_non_text_codes = self.input_code_embedding(
            all_non_text_codes, all_non_text_offsets
        )[:, :size_for_embedding]

        embedded_non_text_codes1 = F.pad(
            self.input_code_embedding1(
                all_non_text_codes1, all_non_text_offsets1
            ),
            pad=[size_for_embedding - ((self.config["size"] // 4) + 1), 0],
            mode="constant",
            value=0,
        )

        all_day_information = all_day_information.unsqueeze(1)  # Add a dimension
        all_positional_encoding = all_positional_encoding.unsqueeze(1)  # Add a dimension

        items = [
            a
            for a in [
                embedded_non_text_codes + embedded_non_text_codes1,
                all_day_information,
                all_positional_encoding if not self.config["use_gru"] else None,
            ]
            if a is not None
        ]

        combined_with_day_information = torch.cat(items, dim=1,)

        # print(all_day_information.shape)
        # print(all_positional_encoding.shape)
        # print(all_non_text_codes.shape)
        # print(all_non_text_offsets.shape)

        # print(combined_with_day_information.shape)
        # print(all_lengths)

        codes_split_by_patient = [
            combined_with_day_information.narrow(0, offset, length)
            for offset, length in all_lengths
        ]

        packed_sequence = nn.utils.rnn.pack_sequence(codes_split_by_patient)

        if self.config["use_gru"]:
            output, _ = self.model(packed_sequence)

            padded_output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True
            )

            padded_output = self.drop(padded_output)

            return padded_output.contiguous()
        else:
            padded_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_sequence, batch_first=False
            )

            padded_output = padded_output.contiguous()
            
            padded_output = F.pad(padded_output, (0, 512 - padded_output.size(-1)), mode='constant', value=0)

            result = self.model(self.norm(padded_output))

            result = result.permute(1, 0, 2).contiguous()

            return result

    @classmethod
    def finalize_data(
        cls,
        config: Mapping[str, Any],
        info: Mapping[str, Any],
        device: torch.device,
        initial: Sequence[Any],
    ) -> Sequence[Any]:
        (
            all_non_text_codes,
            all_non_text_offsets,
            all_non_text_codes1,
            all_non_text_offsets1,
            all_day_information,
            all_positional_encoding,
            all_lengths,
        ) = initial

        all_non_text_codes = torch.tensor(
            all_non_text_codes, dtype=torch.long, device=device
        )
        all_non_text_offsets = torch.tensor(
            all_non_text_offsets, dtype=torch.long, device=device
        )
        all_non_text_codes1 = torch.tensor(
            all_non_text_codes1, dtype=torch.long, device=device
        )
        all_non_text_offsets1 = torch.tensor(
            all_non_text_offsets1, dtype=torch.long, device=device
        )

        all_day_information = torch.tensor(
            all_day_information, dtype=torch.float, device=device
        )

        all_positional_encoding = torch.tensor(
            all_positional_encoding, dtype=torch.float, device=device
        )

        all_lengths = [(int(a), int(b)) for (a, b) in all_lengths]

        return (
            all_non_text_codes,
            all_non_text_offsets,
            all_non_text_codes1,
            all_non_text_offsets1,
            all_day_information,
            all_positional_encoding,
            all_lengths,
        )


class LabelerTask(nn.Module):
    """
    This is paired with an encoder that outputs an encoding for each timestep.  
    This is the output (and loss) module for that encoder.  An example of an 
    encoder is PatientRNN in rnn_model.py.  
    """

    def __init__(self, config: Mapping[str, Any], info: Mapping[str, Any]):
        super().__init__()

        self.config = config
        self.info = info

        self.final_layer = nn.Linear(config["size"], 1, bias=True)

    def forward(self, rnn_output: torch.Tensor, data: Sequence[Any]) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        indices, targets = data

        flat_rnn_output = rnn_output.view(-1, self.config["size"])

        correct_output = nn.functional.embedding(indices, flat_rnn_output)

        final = self.final_layer(correct_output)

        final = final.view((final.shape[0],))

        loss = nn.functional.binary_cross_entropy_with_logits(
            final, targets, reduction="sum"
        )

        return final, loss

    @classmethod
    def finalize_data(
        cls,
        config: Mapping[str, Any],
        info: Mapping[str, Any],
        device: torch.device,
        initial: Sequence[torch.Tensor],
    ) -> Sequence[torch.Tensor]:
        indices, targets = initial
        indices = torch.tensor(indices, dtype=torch.long, device=device)
        targets = torch.tensor(targets, dtype=torch.float, device=device)
        return indices, targets




class PredictionModel(nn.Module):
    """
    Encapsulates a model that can encode a timeline, and a module that 
    defines some task.  Examples are PatientRNN and SequentialTask, 
    respectively. These two parts are kept separate b/c for the most
    part we will be using the former as an encoder, and the SequentialTask
    is simply some auxiliary task we are going to use to provide supervision. 
    For our target tasks, we are using the results of compute_embedding
    to run the former part without the auxiliary task machinery.  
    Note that this class doesn't need to know a lot of details about 
    codes vs terms, etc. 
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        info: Mapping[str, Any],
        use_cuda: bool,
        for_labeler: bool = False,
    ):
        super().__init__()
        self.config = config
        self.info = info
        self.timeline_model = PatientRNN(config, info)
        if for_labeler:
            self.labeler_module = LabelerTask(config, info)
        else:
            self.task_module = SequentialTask(config, info)
        self.use_cuda = use_cuda

    def compute_embedding_batch(self, rnn_input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.eval()
            return self.timeline_model(rnn_input)

        raise RuntimeError("Should never reach here?")

    def forward(self, batch: Mapping[str, Any]) -> Any:  # type: ignore
        rnn_input = batch["rnn"]
        if "task" in batch:
            task_input = batch["task"]
        elif "survival" in batch:
            survival_input = batch["survival"]
        elif "labeler" in batch:
            labeler_input = batch["labeler"]

        rnn_output = self.timeline_model(batch["rnn"])

        if "task" in batch:
            return self.task_module(rnn_output, batch["task"])
        elif "labeler" in batch:
            return self.labeler_module(rnn_output, batch["labeler"])
        else:
            raise ValueError("Could not find target in batch")

    @classmethod
    def finalize_data(
        cls,
        config: Mapping[str, Any],
        info: Mapping[str, Any],
        device: torch.device,
        batch: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        resulting_batch = {}

        resulting_batch["pid"] = batch["pid"].tolist()
        resulting_batch["day_index"] = batch["day_index"].tolist()
        resulting_batch["rnn"] = PatientRNN.finalize_data(
            config, info, device, batch["rnn"]
        )
        if "task" in batch:
            resulting_batch["task"] = SequentialTask.finalize_data(
                config, info, device, batch["task"]
            )
        if "labeler" in batch:
            resulting_batch["labeler"] = LabelerTask.finalize_data(
                config, info, device, batch["labeler"]
            )

        return resulting_batch
    
if __name__ == '__main__':
    dataset = df
    non_text_codes = df['code'].astype('category').cat.codes.tolist()
    non_text_offsets = [0] * len(non_text_codes) 
    non_text_codes1 = df['visit_id_x'].astype('category').cat.codes.tolist()
    non_text_offsets1 = [0] * len(non_text_codes1) 
    day_information = [0] * len(non_text_codes)  
    positional_encoding = [0] * len(non_text_codes)  
    lengths = [(0, len(non_text_codes))]    

    print('Finished loading data')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Device:', device)

    config = {
        "num_first": 1000,  
        "num_second": 1000,  
        "size": 512,          
        "dropout": 0.1,      
        "use_gru": False,    
        "gru_layers": 2,      
        "nhead": 8,         
        "dim_feedforward": 2048,  
        "num_encoder_layers": 12, 
        "num_valid_targets": 1000,
    }

    info = {}

    prepared_data = PatientRNN.finalize_data(
        config=config,
        info=info,
        device=device,
        initial=(
            non_text_codes,
            non_text_offsets,
            non_text_codes1,
            non_text_offsets1,
            day_information,
            positional_encoding,
            lengths,
        )
    )

    use_cuda = torch.cuda.is_available()
    model = PredictionModel(config=config, info=info, use_cuda=use_cuda)

    model.to(device)

    model.eval()  
    with torch.no_grad():
        embeddings = model.compute_embedding_batch(prepared_data)

    print(embeddings)


