from __future__ import annotations

import torch
import torch.nn.functional as F

from datasets import build_hard_batch


def sparse_word_kd(logits: torch.Tensor, boundaries: torch.Tensor, records: list[dict],
                   tokenizer, temperature: float) -> torch.Tensor:
    """KL on candidate first pieces, normalized only over the teacher shortlist."""
    losses = []
    for row, record in enumerate(records):
        teacher = torch.tensor(record["teacher_log_probs"], device=logits.device, dtype=torch.float32)
        teacher_prob = (teacher / temperature).softmax(dim=0)
        by_piece: dict[int, torch.Tensor] = {}
        for candidate, probability in zip(record["candidates"], teacher_prob):
            pieces = tokenizer.encode(" " + candidate, out_type=int)
            piece = pieces[0] if pieces else tokenizer.unk_id()
            by_piece[piece] = by_piece.get(piece, torch.zeros((), device=logits.device)) + probability
        piece_ids = torch.tensor(list(by_piece), device=logits.device, dtype=torch.long)
        target = torch.stack([by_piece[piece] for piece in by_piece]).clamp_min(1e-9)
        target = target / target.sum()
        student = logits[row, boundaries[row], piece_ids].float() / temperature
        losses.append(torch.sum(target * (target.log() - student.log_softmax(dim=0))) * temperature**2)
    return torch.stack(losses).mean()


def sequence_word_kd(model, records: list[dict], tokenizer, max_length: int,
                     device: torch.device, temperature: float, candidate_limit: int) -> torch.Tensor:
    rows, metadata = [], []
    for record_index, record in enumerate(records):
        order = sorted(range(len(record["candidates"])),
                       key=lambda i: record["teacher_log_probs"][i], reverse=True)[:candidate_limit]
        target_index = record["candidates"].index(record["target"])
        if target_index not in order:
            order[-1] = target_index
        for candidate_index in order:
            candidate = tokenizer.encode(" " + record["candidates"][candidate_index], out_type=int)
            tag_id = tokenizer.piece_to_id({"en": "<en>", "te": "<te>", "xlit": "<xlit>"}[record.get("mode", "en")])
            context = [tag_id] + tokenizer.encode(record["context"], out_type=int)
            context = context[-max(1, max_length + 1 - len(candidate)):]
            full = (context + candidate)[:max_length + 1]
            rows.append((full[:-1], context, candidate))
            metadata.append((record_index, candidate_index))

    inputs = torch.full((len(rows), max_length), tokenizer.pad_id(), dtype=torch.long, device=device)
    for index, (x, _, _) in enumerate(rows):
        inputs[index, :len(x)] = torch.tensor(x, device=device)
    log_probs = model(inputs).float().log_softmax(dim=-1)
    student_scores = []
    for row_index, (_, context, candidate) in enumerate(rows):
        score = torch.zeros((), device=device)
        for offset, token in enumerate(candidate):
            position = len(context) - 1 + offset
            if position < max_length:
                score = score + log_probs[row_index, position, token]
        student_scores.append(score)

    losses = []
    for record_index, record in enumerate(records):
        indices = [i for i, (ri, _) in enumerate(metadata) if ri == record_index]
        candidate_indices = [metadata[i][1] for i in indices]
        teacher = torch.tensor(
            [record["teacher_log_probs"][i] for i in candidate_indices],
            device=device, dtype=torch.float32,
        )
        student = torch.stack([student_scores[i] for i in indices])
        q = (teacher / temperature).softmax(dim=0).clamp_min(1e-9)
        losses.append(torch.sum(q * (q.log() - (student / temperature).log_softmax(dim=0))) * temperature**2)
    return torch.stack(losses).mean()


def distillation_loss(model, records: list[dict], tokenizer, max_length: int,
                      device: torch.device, temperature: float,
                      hard_weight: float, soft_weight: float, sequence_weight: float,
                      include_sequence: bool, sequence_candidates: int):
    inputs, labels, boundaries, _, _ = build_hard_batch(records, tokenizer, max_length, device)
    logits = model(inputs)
    hard = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
    soft = sparse_word_kd(logits, boundaries, records, tokenizer, temperature)
    sequence = sequence_word_kd(
        model, records, tokenizer, max_length, device, temperature, sequence_candidates,
    ) if include_sequence and sequence_weight > 0 else torch.zeros((), device=device)
    total = hard_weight * hard + soft_weight * soft + sequence_weight * sequence
    return total, {"hard": hard.detach(), "soft": soft.detach(), "sequence": sequence.detach()}
