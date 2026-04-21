from __future__ import annotations

import torch

from utils.utils import inverse_sigmoid


def _get_decoder_states(model_outputs: dict) -> dict | None:
    decoder_states = model_outputs.get("decoder_states")
    if not isinstance(decoder_states, dict):
        return None
    required_keys = {"query_in", "query_out", "ref_in", "ref_out"}
    if not required_keys.issubset(decoder_states.keys()):
        return None
    return decoder_states


def get_last_layer_output_query(model_outputs: dict) -> torch.Tensor:
    decoder_states = _get_decoder_states(model_outputs)
    if decoder_states is not None:
        return decoder_states["query_out"][-1]
    return model_outputs["outputs"]


def get_last_layer_input_query(model_outputs: dict) -> torch.Tensor:
    decoder_states = _get_decoder_states(model_outputs)
    if decoder_states is not None:
        return decoder_states["query_in"][-1]
    aux_outputs = model_outputs.get("aux_outputs")
    if isinstance(aux_outputs, list) and len(aux_outputs) > 0:
        return aux_outputs[-1]["queries"]
    return get_last_layer_output_query(model_outputs)


def get_last_layer_input_ref(model_outputs: dict) -> torch.Tensor:
    decoder_states = _get_decoder_states(model_outputs)
    if decoder_states is not None:
        return inverse_sigmoid(decoder_states["ref_in"][-1])
    return model_outputs["last_ref_pts"]


def get_aux_layer_output_query(model_outputs: dict, layer_idx: int) -> torch.Tensor:
    decoder_states = _get_decoder_states(model_outputs)
    if decoder_states is not None:
        return decoder_states["query_out"][layer_idx]
    return model_outputs["aux_outputs"][layer_idx]["queries"]
