"""Reference parity checks for transformers checkpoint converters.

The machinery behind `TestCase.run_reference_parity_test`. A tiny
transformers model is built from a small config, every saved tensor is
redrawn from a keyed random generator, and the checkpoint is loaded through
keras-hub's converter. Both models then run on the
same inputs and their outputs are compared position by position.

torch, transformers and safetensors are imported inside functions, so that
importing this module needs none of them.
"""

import contextlib
import hashlib
import importlib
import os
import tempfile
import traceback

import keras
import numpy as np

from keras_hub.src.models.backbone import Backbone

# Largest per-position error of a correct port, relative to the scale of the
# output (see `per_position_error`). Over the 18 families on three backends,
# on macOS CPUs, the float32 noise of correct ports measured at most 1.1e-5
# under revealing (Mixtral on TensorFlow). Against a float64 reference, the
# rounding errors of the two sides add up to at most 1.7e-5 under revealing
# (Mixtral), so kernels that round as much keep a margin of 6 times. Qwen3
# and XLM-RoBERTa measure 1.5e-5 and 1.1e-5 under revealing, but not from
# noise: that is what their known epsilon defects leave, the same on every
# backend, and it drops to 2.3e-6 and 1.6e-6 once the defect is fixed.
PARITY_THRESHOLD = 1e-4
# The negative control must move the outputs by more than this. The smallest
# value measured over the parity tests of 18 families was 0.3.
NEGATIVE_CONTROL_MIN_ERROR = 1e-2
# The parity checks: "revealing" compares the outputs on the redrawn weights.
CHECKS = ("revealing",)
# Where to look first when the outputs differ under a check.
FAILURE_HINTS = {
    "revealing": (
        "Weights drawn at full scale: look for a swapped, transposed or "
        "mis-split tensor, layers in the wrong order, or a computation that "
        "differs from transformers (an activation, an attention scale, mask "
        "or window, a position encoding)."
    ),
}
# Length of the four input rows, long enough for every padding pattern.
SEQUENCE_LENGTH = 17
# Input ids are drawn from [MIN_TOKEN_ID, vocab_size), so that no special
# token id set in a config appears in the inputs.
MIN_TOKEN_ID = 5
# Seeds the weight draws, the inputs and the Keras initializers.
SEED = 0
# Keeps a parity test cheap enough to run on every pull request.
MAX_REFERENCE_PARAMETERS = 5_000_000
# The `max_error` of a known failure may be at most this many times what it
# measures, so that it stays close to the known error. A known failure still
# hides every defect smaller than itself: give a code path that fails its own
# test method, and keep it out of the main test.
MAX_ERROR_SLACK = 10

# Reference config fields a parity test may set, by kind. Any other field (an
# activation, a dropout, a bias switch, a rotary base) needs a maintainer to
# classify it first, because changing it can hide a bug.
CONFIG_FIELDS = {
    "vocab_size": "vocabulary",
    "hidden_size": "width",
    "n_embd": "width",
    "dim": "width",
    "embedding_size": "width",
    "num_hidden_layers": "layers",
    "n_layer": "layers",
    "n_layers": "layers",
    "num_attention_heads": "heads",
    "n_head": "heads",
    "n_heads": "heads",
    "num_key_value_heads": "heads",
    "max_position_embeddings": "positions",
    "n_positions": "positions",
    "intermediate_size": "size",
    "hidden_dim": "size",
    "head_dim": "size",
    "moe_intermediate_size": "size",
    "shared_expert_intermediate_size": "size",
    "type_vocab_size": "token_types",
    "pad_token_id": "token_id",
    "bos_token_id": "token_id",
    "eos_token_id": "token_id",
    "cls_token_id": "token_id",
    "sep_token_id": "token_id",
    "query_pre_attn_scalar": "structural",
    "sliding_window": "window",
    "local_attention": "window",
    "global_attn_every_n_layers": "structural",
    "num_local_experts": "structural",
    "num_experts": "structural",
    "num_experts_per_tok": "structural",
    "decoder_sparse_step": "structural",
    "num_hidden_groups": "structural",
    "inner_group_num": "structural",
}
# Inclusive bounds by kind of field; `None` means no bound. Six layers fit one
# period of the longest layer pattern: Gemma 3's five sliding window layers
# and one global layer. A size below 4, such as a head size of 1, hides
# transposed and mis-split tensors. An attention window longer than the
# inputs would leave the windowed attention untested.
CONFIG_BOUNDS = {
    "vocabulary": (32, 512),
    "width": (16, 96),
    "layers": (2, 6),
    "heads": (2, None),
    "positions": (SEQUENCE_LENGTH, 128),
    "size": (4, None),
    "token_id": (0, MIN_TOKEN_ID - 1),
    "window": (2, SEQUENCE_LENGTH - 1),
}
# Kinds of field that may keep the reference config default. A shape or
# structural field must differ from it, so that the test shows that the
# converter reads it.
DEFAULT_KINDS = ("token_id", "window")
# Shape and structural fields that may keep their default value, with the
# reason.
DEFAULT_FIELDS = {
    "type_vocab_size": (
        "convert_bert does not map it to `num_segments`, so BERT checkpoints "
        "load only with the default of 2 token types."
    ),
    "decoder_sparse_step": (
        "convert_qwen_moe ports the weights of a dense layer to "
        "`decoder_layer`, not to `decoder_layer.mlp`, so a checkpoint with "
        "a dense layer does not load."
    ),
}

STATUS_TEXT = {
    "shape": "shapes differ",
    "nonfinite_keras": "NaN or inf in the Keras output",
    "nonfinite_reference": "NaN or inf in the reference output",
}


def out_of_bounds(field, value):
    """Returns why `value` is out of bounds for `field`, or `None`."""
    kind = CONFIG_FIELDS[field]
    low, high = CONFIG_BOUNDS.get(kind, (None, None))
    if (low is None or value >= low) and (high is None or value <= high):
        return None
    bounds = f"at least {low}" if high is None else f"{low} to {high}"
    if kind == "token_id":
        bounds += f", as the inputs use ids from {MIN_TOKEN_ID} up"
    return (
        f"`{field}={value}` is out of bounds: a {kind.replace('_', ' ')} "
        f"field must be {bounds}."
    )


def check_config(reference_class, config):
    """Returns a list of problems with the reference config of a test.

    The rules apply to the fields given and then to the values that the
    reference uses, which its config class may derive from them (such as
    `head_dim` and `num_key_value_heads`).
    """
    import transformers

    model_class = getattr(transformers, reference_class, None)
    if model_class is None:
        return [f"`{reference_class}` is not a transformers class."]
    config_class = model_class.config_class
    defaults = config_class()
    problems = []
    for field, value in config.items():
        kind = CONFIG_FIELDS.get(field)
        if kind is None:
            problems.append(
                f"`{field}` is not a field a parity test may set; ask a "
                f"maintainer to classify `{field}`."
            )
            continue
        if not isinstance(value, int) or isinstance(value, bool):
            problems.append(f"`{field}` must be an integer, got {value!r}.")
            continue
        problem = out_of_bounds(field, value)
        if problem:
            problems.append(problem)
        if (
            kind not in DEFAULT_KINDS
            and field not in DEFAULT_FIELDS
            and getattr(defaults, field, None) == value
        ):
            problems.append(
                f"`{field}={value}` equals the {config_class.__name__} "
                "default; use another value so that the test shows the "
                f"converter reads `{field}`."
            )
    if problems:
        return problems

    reference = config_class(**config)
    for field in CONFIG_FIELDS:
        value = getattr(reference, field, None)
        if field in config or value == getattr(defaults, field, None):
            continue
        problem = isinstance(value, int) and out_of_bounds(field, value)
        if problem:
            problems.append(
                f"{problem} {config_class.__name__} derives it from the "
                "fields that the test sets."
            )
    heads = getattr(reference, "num_attention_heads", None)
    if not isinstance(heads, int):
        return problems
    # With one key/value head, or one per query head, grouping them wrongly
    # changes nothing.
    kv_heads = getattr(reference, "num_key_value_heads", None)
    if isinstance(kv_heads, int) and not 1 < kv_heads < heads:
        problems.append(
            f"`num_key_value_heads` is {kv_heads}; set it to more than 1 and "
            f"less than `num_attention_heads` ({heads}), so that the test "
            "runs grouped-query attention."
        )
    # A converter that computes the head size from the width would pass too.
    width = getattr(reference, "hidden_size", 0)
    if "head_dim" in config and config["head_dim"] == width // heads:
        problems.append(
            f"`head_dim={config['head_dim']}` equals `hidden_size // "
            "num_attention_heads`; use another value so that the test shows "
            "the converter reads `head_dim`."
        )
    return problems


def check_known_failures(known_failures):
    """Returns a list of problems with the `known_failures` of a test."""
    problems = []
    for check, value in known_failures.items():
        if check not in CHECKS:
            problems.append(
                f"`known_failures`: `{check}` is not a parity check "
                f"({', '.join(CHECKS)})."
            )
        elif not (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], str)
            and value[0].strip()
            and isinstance(value[1], (int, float))
            and value[1] > PARITY_THRESHOLD
        ):
            problems.append(
                f"`known_failures`: `{check}` needs a tuple `(reason, "
                "max_error)`: what is wrong today, and the largest "
                "per-position error it may reach, above "
                f"{PARITY_THRESHOLD:.0e}."
            )
    return problems


def apply_known_failures(comparisons, known_failures):
    """Judges the parity checks against the known failures of a test.

    `comparisons` maps each check that ran to the comparison of its
    outputs, and `known_failures` maps one to `(reason, max_error)`. A
    known failure is excused only while its largest error stays between
    `max_error / MAX_ERROR_SLACK` and `max_error`. This catches a new defect
    larger than the known one, but not a smaller one.

    Returns:
        A tuple `(failures, expected)` of lists of strings: the failures
        that are not excused, including known failures that are gone, and
        the excused known failures.
    """
    failures, expected = [], []
    for check in CHECKS:
        if check not in comparisons:
            continue
        failure = check_parity(comparisons[check])
        hint = FAILURE_HINTS[check]
        if check not in known_failures:
            if failure is not None:
                failures.append(f"{check}: {failure}\n  {hint}")
            continue
        reason, max_error = known_failures[check]
        error = largest_error(comparisons[check])
        if failure is None:
            failures.append(
                f"known failure `{check}` no longer happens; remove it from "
                f"known_failures (reason was: {reason})"
            )
        elif error > max_error:
            failures.append(
                f"{check}: {failure}\n  This is above the max_error "
                f"{max_error:g} of its known failure: a new defect, or the "
                f"known one grew (reason was: {reason})\n  {hint}"
            )
        elif error * MAX_ERROR_SLACK < max_error:
            failures.append(
                f"known failure `{check}` shrank to {error:.2e}, less than "
                f"1/{MAX_ERROR_SLACK} of its max_error {max_error:g}: if a "
                "fix landed, the remaining error is another defect; update "
                "the reason and max_error to it, or remove the entry (reason "
                f"was: {reason})"
            )
        else:
            expected.append(
                f"{check}: {reason}\n  measured: {failure} (max_error "
                f"{max_error:g})"
            )
    return failures, expected


def unit_draw(seed, key, shape):
    """Draws zero-mean, unit-variance noise keyed by `seed` and `key`.

    The generator is seeded from the SHA-256 of the key, and each value is
    the sum of four raw 53-bit integers, centred and scaled, so the draw is
    the same on every platform and numpy version.
    """
    digest = hashlib.sha256(f"{int(seed)}\x00{key}".encode("utf-8")).digest()
    generator = np.random.PCG64(int.from_bytes(digest[:16], "little"))
    size = int(np.prod(shape))
    raw = generator.random_raw(4 * size).reshape(4, size) >> np.uint64(11)
    total = raw.sum(axis=0, dtype=np.uint64)
    draw = (total.astype("float64") * 2.0**-53 - 2.0) * np.sqrt(3.0)
    return draw.reshape(shape)


def read_safetensors(path):
    """Returns `(tensors, metadata)` of a safetensors file."""
    from safetensors import safe_open

    with safe_open(path, framework="np") as file:
        tensors = {key: file.get_tensor(key) for key in file.keys()}
        return tensors, file.metadata()


def marker_roles(model, directory):
    """Finds where each saved tensor comes from, with one marker save.

    Every float parameter and persistent buffer is filled with its own
    integer code, the model is saved to `directory` and the codes are read
    back. Tied tensors share a storage and so a code. Returns a dict from
    saved key to `(kind, attribute)`, with kind `"parameter"` or `"buffer"`,
    or to `None` for an empty or integer tensor.
    """
    import torch

    sources, seen, originals = {}, set(), []
    with torch.no_grad():
        for module in model.modules():
            tensors = [
                ("parameter", name, tensor)
                for name, tensor in module.named_parameters(recurse=False)
            ] + [
                ("buffer", name, tensor)
                for name, tensor in module.named_buffers(recurse=False)
                if name not in module._non_persistent_buffers_set
            ]
            for kind, name, tensor in tensors:
                if (
                    not tensor.dtype.is_floating_point
                    or tensor.numel() == 0
                    or tensor.data_ptr() in seen
                ):
                    continue
                seen.add(tensor.data_ptr())
                code = len(sources) + 1
                sources[code] = (kind, name)
                originals.append((tensor, tensor.detach().clone()))
                tensor.fill_(code)
        model.save_pretrained(directory)
        for tensor, original in originals:
            tensor.copy_(original)
    marked, _ = read_safetensors(os.path.join(directory, "model.safetensors"))
    roles = {}
    for key, value in marked.items():
        if value.size == 0 or not np.issubdtype(value.dtype, np.floating):
            roles[key] = None
            continue
        codes = np.unique(value)
        if len(codes) != 1 or float(codes[0]) not in sources:
            raise ValueError(
                f"Cannot tell which module the saved tensor `{key}` comes from."
            )
        roles[key] = sources[int(codes[0])]
    return roles


def redraw(path, roles):
    """Rewrites a safetensors file with keyed random values.

    Matrices get `min(0.6, 1.2 / sqrt(min(shape[0], shape[-1])))` times a
    unit draw. Vectors get their constructed value (if constant, else 0) plus
    0.25 times a unit draw. Running statistics are redrawn; other buffers
    are kept. The metadata, which holds the aliases of tied weights, is
    kept.
    """
    from safetensors.numpy import save_file

    tensors, metadata = read_safetensors(path)
    for key in sorted(tensors):
        value, role = tensors[key], roles[key]
        if role is None:
            continue
        kind, name = role
        if kind == "buffer":
            if name == "running_var":
                noise = unit_draw(SEED, key, value.shape) / np.sqrt(12.0)
                new_value = 0.5 + np.clip(noise + 0.5, 0.0, 1.0)
            elif name == "running_mean":
                new_value = 0.25 * unit_draw(SEED, key, value.shape)
            else:
                continue
        else:
            noise = unit_draw(SEED, key, value.shape)
            if value.ndim >= 2:
                std = 1.2 / np.sqrt(min(value.shape[0], value.shape[-1]))
                new_value = min(0.6, std) * noise
            else:
                flat = value.ravel()
                constant = bool(np.all(flat == flat[0]))
                new_value = (float(flat[0]) if constant else 0.0) + 0.25 * noise
        tensors[key] = np.asarray(new_value).astype(value.dtype)
    save_file(tensors, path, metadata=metadata)


@contextlib.contextmanager
def hidden_progress_bars():
    """Hides the progress bars of transformers, then restores the setting.

    Hiding or showing the bars of transformers resets the settings of
    huggingface_hub, its global one and those of its groups, so these are
    restored too.
    """
    from huggingface_hub import utils as hub_utils
    from transformers.utils import logging

    # `huggingface_hub.utils.tqdm` is also the name of a class there.
    hub_tqdm = importlib.import_module("huggingface_hub.utils.tqdm")
    # The settings of huggingface_hub, its groups included. Not public API:
    # without it, only the global setting is restored.
    hub_states = getattr(hub_tqdm, "progress_bar_states", None)
    saved_states = dict(hub_states) if isinstance(hub_states, dict) else None
    hub_disabled = hub_utils.are_progress_bars_disabled()
    enabled = logging.is_progress_bar_enabled()
    logging.disable_progress_bar()
    try:
        yield
    finally:
        if enabled:
            logging.enable_progress_bar()
        if saved_states is not None:
            hub_states.clear()
            hub_states.update(saved_states)
        elif hub_disabled:
            hub_utils.disable_progress_bars()
        else:
            hub_utils.enable_progress_bars()


def build_reference(reference_class, config, directory):
    """Saves a tiny reference model to `directory`; returns it reloaded.

    The model is built from `config` and saved with `save_pretrained`. Its
    weights are redrawn, and it is loaded again in float32 with the library
    default attention implementation.
    """
    import torch
    import transformers

    model_class = getattr(transformers, reference_class)
    reference_config = model_class.config_class(**config)
    torch.manual_seed(SEED)
    model = model_class(reference_config).eval()
    num_parameters = sum(p.numel() for p in model.parameters())
    if num_parameters > MAX_REFERENCE_PARAMETERS:
        raise ValueError(
            f"The reference model has {num_parameters:,} parameters, more "
            f"than {MAX_REFERENCE_PARAMETERS:,}. Use a smaller config."
        )
    with hidden_progress_bars():
        model.save_pretrained(directory)
        with tempfile.TemporaryDirectory() as scratch:
            roles = marker_roles(model, scratch)
        path = os.path.join(directory, "model.safetensors")
        redraw(path, roles)
        model, info = model_class.from_pretrained(
            directory,
            dtype=torch.float32,
            local_files_only=True,
            output_loading_info=True,
        )
    for name in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        if info.get(name):
            raise ValueError(
                f"Reloading the reference gave {name}: {sorted(info[name])}"
            )
    return model.eval()


def text_inputs(vocabulary_size):
    """Returns `(token_ids, padding_mask)`, four rows of `SEQUENCE_LENGTH`.

    Row 0 is all valid, row 1 is right-padded, row 2 is left-padded and row
    3 has a hole and right padding.
    """
    rng = np.random.default_rng(SEED)
    shape = (4, SEQUENCE_LENGTH)
    token_ids = rng.integers(MIN_TOKEN_ID, vocabulary_size, size=shape)
    padding_mask = np.ones(shape, dtype="int32")
    padding_mask[1, 11:] = 0
    padding_mask[2, :6] = 0
    padding_mask[3, 3:6] = 0
    padding_mask[3, 14:] = 0
    return token_ids.astype("int32"), padding_mask


def keras_inputs(backbone, token_ids, padding_mask):
    """Returns the inputs of `backbone`, with segment ids if it takes them.

    Segment ids are 0 in the first half of each row and 1 in the second.
    """
    segment_ids = np.zeros_like(token_ids)
    segment_ids[:, SEQUENCE_LENGTH // 2 :] = 1
    inputs = {
        "token_ids": token_ids,
        "padding_mask": padding_mask,
        "segment_ids": segment_ids,
    }
    return {name: inputs[name] for name in backbone.input if name in inputs}


def reference_outputs(model, inputs):
    """Runs the reference on Keras-style inputs; returns numpy outputs.

    `hidden` is the last hidden state of the base model, `pooled` its pooled
    output if it has one, and `logits` the output of the head if the model
    has one.
    """
    import torch

    kwargs = {
        "input_ids": torch.tensor(inputs["token_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(
            inputs["padding_mask"], dtype=torch.long
        ),
    }
    if "segment_ids" in inputs:
        kwargs["token_type_ids"] = torch.tensor(
            inputs["segment_ids"], dtype=torch.long
        )
    outputs = {}
    with torch.no_grad():
        base_outputs = model.base_model(**kwargs)
        outputs["hidden"] = base_outputs.last_hidden_state.numpy()
        if getattr(base_outputs, "pooler_output", None) is not None:
            outputs["pooled"] = base_outputs.pooler_output.numpy()
        if model.base_model is not model:
            logits = getattr(model(**kwargs), "logits", None)
            if logits is not None:
                outputs["logits"] = logits.numpy()
    return outputs


def keras_outputs(backbone, task, inputs):
    """Runs the Keras models; returns numpy outputs named as the reference."""
    outputs = {}
    backbone_outputs = backbone(inputs)
    if isinstance(backbone_outputs, dict):
        outputs["hidden"] = backbone_outputs["sequence_output"]
        if "pooled_output" in backbone_outputs:
            outputs["pooled"] = backbone_outputs["pooled_output"]
    else:
        outputs["hidden"] = backbone_outputs
    if task is not None:
        outputs["logits"] = task(inputs)
    return {k: keras.ops.convert_to_numpy(v) for k, v in outputs.items()}


def per_position_error(keras_output, reference_output, padding_mask=None):
    """Compares two outputs position by position, in float64.

    At each position where `padding_mask` (shape `(batch, positions)`, if
    given) is not 0, the largest absolute difference is divided by the root
    mean square of the reference at that position, so that the error does
    not depend on the scale of the output. Where the reference is zero, the
    output must be zero too. The error is the largest of these values.

    Returns:
        A tuple `(status, error)`: `status` is `"ok"`, `"shape"`,
        `"nonfinite_keras"` or `"nonfinite_reference"`, and `error` is a
        float if `status` is `"ok"`, else `None`.
    """
    keras_output = np.asarray(keras_output, dtype="float64")
    reference_output = np.asarray(reference_output, dtype="float64")
    if keras_output.shape != reference_output.shape:
        return "shape", None
    if padding_mask is not None:
        valid = np.asarray(padding_mask).astype(bool)
        keras_output = keras_output[valid]
        reference_output = reference_output[valid]
    if not np.isfinite(reference_output).all():
        return "nonfinite_reference", None
    if not np.isfinite(keras_output).all():
        return "nonfinite_keras", None
    width = reference_output.shape[-1]
    keras_output = keras_output.reshape(-1, width)
    reference_output = reference_output.reshape(-1, width)
    rms = np.sqrt(np.mean(reference_output**2, axis=-1))
    difference = np.abs(keras_output - reference_output).max(axis=-1)
    error = difference / np.maximum(rms, np.finfo("float64").tiny)
    return "ok", float(error.max())


def compare_outputs(keras_outputs, reference_outputs, padding_mask):
    """Returns `{name: (status, error)}` for outputs both sides have."""
    return {
        name: per_position_error(
            keras_outputs[name],
            reference_outputs[name],
            None if name == "pooled" else padding_mask,
        )
        for name in reference_outputs
        if name in keras_outputs
    }


def largest_error(comparison):
    """Returns the largest error of a comparison; inf if one is not ok."""
    return max(
        error if status == "ok" else float("inf")
        for status, error in comparison.values()
    )


def check_parity(comparison):
    """Returns `None` if all outputs are within the threshold."""
    if largest_error(comparison) <= PARITY_THRESHOLD:
        return None
    return (
        f"per-position error above {PARITY_THRESHOLD:.0e}: "
        f"{format_comparison(comparison)}"
    )


def format_comparison(comparison):
    """Formats a comparison as, e.g., `"hidden 1.23e-06, logits 3.40e-06"`."""
    return ", ".join(
        f"{name} {error:.2e}"
        if status == "ok"
        else f"{name}: {STATUS_TEXT[status]}"
        for name, (status, error) in comparison.items()
    )


def load_model(directory, task_cls=None):
    """Loads a checkpoint as a user would; returns `(backbone, task)`."""
    keras.utils.set_random_seed(SEED)
    if task_cls is None:
        return Backbone.from_preset(directory), None
    task = task_cls.from_preset(directory, preprocessor=None)
    return task.backbone, task


def describe_load_failure(load_call, error):
    """Describes an exception raised by `from_preset`.

    The last three frames outside this file usually hold the converter line.
    """
    import transformers

    frames = [
        frame
        for frame in traceback.extract_tb(error.__traceback__)
        if frame.filename != __file__
    ]
    lines = [f"load: `{load_call}` raised {type(error).__name__}: {error}"]
    lines += [line.rstrip("\n") for line in traceback.format_list(frames[-3:])]
    lines.append(
        "  The checkpoint was written by `save_pretrained` of transformers "
        f"{transformers.__version__}. No check after the load ran."
    )
    return "\n".join(lines)


def checked_load(directory, backbone_cls, task_cls):
    """Loads a checkpoint.

    Returns a tuple `(backbone, task, problem)`. `problem` describes
    an exception raised by `from_preset`, or a load that returned other
    classes than `backbone_cls` and `task_cls`; it is `None` if the load
    worked.
    """
    load_call = f"{(task_cls or Backbone).__name__}.from_preset"
    try:
        backbone, task = load_model(directory, task_cls)
    except Exception as e:
        return None, None, describe_load_failure(load_call, e)
    loaded = (type(backbone), type(task) if task_cls else None)
    wanted = (backbone_cls, task_cls)
    problem = None
    if loaded != wanted:
        problem = (
            f"load: `{load_call}` returned "
            f"{' and '.join(c.__name__ for c in loaded if c)}, expected "
            f"{' and '.join(c.__name__ for c in wanted if c)}."
        )
    return backbone, task, problem


def check_negative_control(backbone, task, inputs, reference, padding_mask):
    """Rolls the rows of the token embedding by one and compares again.

    Returns `(error, problems)`: the largest error with the rows rolled, and
    a problem if it is not above `NEGATIVE_CONTROL_MIN_ERROR`.
    """
    embeddings = backbone.token_embedding.embeddings
    original = np.array(keras.ops.convert_to_numpy(embeddings.value))
    embeddings.assign(np.roll(original, 1, axis=0))
    try:
        rolled = keras_outputs(backbone, task, inputs)
    finally:
        embeddings.assign(original)
    error = largest_error(compare_outputs(rolled, reference, padding_mask))
    if error > NEGATIVE_CONTROL_MIN_ERROR:
        return error, []
    return error, [
        f"negative control: rolling the rows of `{embeddings.path}` moved "
        f"the outputs by only {error:.2e} (needs more than "
        f"{NEGATIVE_CONTROL_MIN_ERROR:.0e}). The comparison is blind: a bug "
        "in the parity test machinery, not in the model."
    ]


def run_checks(
    model_type,
    reference_class,
    config,
    backbone_cls,
    task_cls,
    directory,
):
    """Runs the parity checks of one `model_type`.

    The arguments are those of `TestCase.run_reference_parity_test`, and
    `directory` is an empty directory for the checkpoints. Returns a tuple
    `(problems, comparisons, notes)`: the failures that no known
    failure can excuse (a failed load,
    a wrong test argument, test machinery that does not work), the
    comparison of the outputs under each check that ran, and what was
    measured.
    """
    problems, comparisons, notes = [], {}, []
    # The checkpoints, each in a folder of the name of its check.
    references = {
        check: build_reference(
            reference_class, config, os.path.join(directory, check)
        )
        for check in CHECKS
    }
    saved_model_type = references["revealing"].config.model_type
    if saved_model_type != model_type:
        problems.append(
            f"`{reference_class}` saves model_type `{saved_model_type}`, not "
            f"`{model_type}`."
        )
        return problems, comparisons, notes
    token_ids, padding_mask = text_inputs(
        references["revealing"].config.vocab_size
    )
    controlled = False

    # Every Keras load and call runs on the CPU: on a GPU, JAX and
    # TensorFlow compute float32 matmuls in reduced precision by default.
    with keras.device("cpu"):
        for check in CHECKS:
            path = os.path.join(directory, check)
            backbone, task, problem = checked_load(path, backbone_cls, task_cls)
            if problem is not None:
                problems.append(problem)
                break
            inputs = keras_inputs(backbone, token_ids, padding_mask)
            reference = reference_outputs(references[check], inputs)
            outputs = keras_outputs(backbone, task, inputs)
            comparison = compare_outputs(outputs, reference, padding_mask)
            comparisons[check] = comparison
            if check == "revealing":
                if task_cls is not None and "logits" not in reference:
                    problems.append(
                        f"`{reference_class}` has no logits to compare with "
                        f"`{task_cls.__name__}`; save a class with a head."
                    )
            notes.append(f"{check}: {format_comparison(comparison)}")
            if not controlled and check_parity(comparison) is None:
                # Once, under the first run whose parity passed.
                controlled = True
                error, control_problems = check_negative_control(
                    backbone, task, inputs, reference, padding_mask
                )
                notes.append(f"negative control under {check}: {error:.2g}")
                problems += control_problems
    if comparisons and not controlled:
        notes.append("negative control: not run, as no parity check passed")
    return problems, comparisons, notes
