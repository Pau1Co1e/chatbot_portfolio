from transformers import Trainer
import torch.nn.functional as func_interface

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract labels
        start_positions = inputs.pop("start_positions")
        end_positions = inputs.pop("end_positions")

        # Forward pass
        outputs = model(**inputs)
        start_logits = outputs["start_logits"]
        end_logits = outputs["end_logits"]

        # Compute the loss using Cross Entropy
        start_loss = func_interface.cross_entropy(start_logits, start_positions)
        end_loss = func_interface.cross_entropy(end_logits, end_positions)
        loss = (start_loss + end_loss) / 2

        # Optionally normalize by batch size if needed
        num_items_in_batch = inputs["input_ids"].size(0)  # Infer from input shape
        loss = loss / num_items_in_batch if num_items_in_batch > 0 else loss

        return (loss, outputs) if return_outputs else loss
