"""Reference parity tests for transformers checkpoint converters.

`run_reference_parity_test` checks a converter against transformers itself. It
builds a tiny transformers model from a test's config, draws random weights,
saves it with `save_pretrained`, loads that checkpoint with keras-hub's
`from_preset` on the CPU, and compares the outputs of the two models at every
valid position. It does this for each row of `RUNS`, each with its own
checkpoint: full-size weights, small weights, and small weights with every
normalization epsilon set by the helper. Each check fails when its largest
error is above `THRESHOLD`. One more load of the first checkpoint is audited:
the converter must read every stored tensor and assign every Keras weight.

A test supplies a small config, and names, each with a reason, the stored
tensors the converter need not read and the weights the model computes instead
of loading.

The file reads top-down: the constants of the verdict, the entry point and its
run loop, then one section per step: the argument rules, the reference model,
the Keras model, the load audit and the comparison. torch, transformers and
safetensors are imported inside functions, so that importing this module needs
none of them.
"""

import contextlib
import fnmatch
import hashlib
import math
import os
import tempfile
import types
from unittest import mock

import keras
import numpy as np
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.utils import preset_utils
from keras_hub.src.utils.transformers import safetensor_utils

# A check fails when an output differs from transformers by more than this,
# relative to the root mean square of the transformers output at the same
# position (see `per_position_error`). Correct ports measure at most 1.1e-5 on
# macOS CPUs (Mixtral on TensorFlow, full_weights). Provisional: to be
# calibrated on the Linux CI runners from the "Measured" rows that each test
# prints (`pytest -rA`). The seeded defects of reference_parity_test.py need
# it between 5.2e-5 and 2.5e-4.
THRESHOLD = 1e-4

# Outside normalization layers, the small_weights and epsilon checks multiply
# the noise of every weight by this (see `redraw`), so that activations are
# small and an epsilon matters.
SMALL_WEIGHTS = 0.01

# The runs of every parity test: (check, scale of the noise outside
# normalization layers, value given to every epsilon field of the config).
# - full_weights shows swapped, transposed or misrouted tensors, and
#   computations that differ from transformers.
# - small_weights shows a wrong normalization epsilon.
# - epsilon shows an epsilon that the converter does not map, or that the
#   backbone hard-codes, even where the transformers and keras-hub defaults
#   agree. No single value shows every such defect (query/key norms show best
#   at 3e-4, ModernBERT's MLP norms only at 3e-3), so the check runs at both
#   and is judged on the larger error. It runs only if the config has an
#   epsilon field.
RUNS = (
    ("full_weights", 1.0, None),
    ("small_weights", SMALL_WEIGHTS, None),
    ("epsilon", SMALL_WEIGHTS, 3e-4),
    ("epsilon", SMALL_WEIGHTS, 3e-3),
)
# The checks, in order.
CHECKS = tuple(dict.fromkeys(check for check, _, _ in RUNS))

# Where to look first when a check fails.
HINTS = {
    "full_weights": (
        "Look for a swapped, transposed or mis-split tensor, layers in the "
        "wrong order, or a computation that differs from transformers: an "
        "activation, an attention scale, mask or window, a position encoding."
    ),
    "small_weights": (
        "Look for a normalization epsilon. If full_weights fails too, fix "
        "that first."
    ),
    "epsilon": (
        "Look at how the converter maps {fields}, and for an epsilon "
        "hard-coded in the backbone or its layers. If full_weights fails too, "
        "fix that first."
    ),
}


def run_reference_parity_test(
    testcase,
    *,
    model_type,
    reference_class,
    cls,
    config,
    ignored_reference_tensors=None,
    derived_weights=None,
):
    """Runs one parity test; see `TestCase.run_reference_parity_test`."""
    if keras.config.backend() == "openvino":
        testcase.skipTest("Reference parity tests do not run on OpenVINO.")
    for module in ("torch", "transformers", "safetensors"):
        pytest.importorskip(module)
    ignored = ignored_reference_tensors or {}
    derived = derived_weights or {}
    header = (
        f"Reference parity of model_type `{model_type}` ({reference_class} "
        f"against {getattr(cls, '__name__', cls)}):"
    )
    problems = check_arguments(
        model_type, reference_class, cls, config, ignored, derived
    )
    if problems:
        testcase.fail("\n".join([header] + problems))

    fields = epsilon_fields(reference_config(reference_class, config))
    with tempfile.TemporaryDirectory() as directory:
        errors, problems, notes = run_checks(
            model_type=model_type,
            reference_class=reference_class,
            cls=cls,
            config=config,
            ignored=ignored,
            derived=derived,
            fields=fields,
            directory=directory,
        )
    failures = judge(errors, fields)
    lines = [header] + problems
    if problems and failures:
        lines.append("Fix the problems above first; they often cause these:")
    lines += failures
    lines += ["Measured:"] + [f"  {note}" for note in notes]
    message = "\n".join(lines)
    if problems or failures:
        testcase.fail(message)
    # A passing test raises nothing to carry its rows, so print them: the
    # threshold is calibrated from them.
    print(message)


def run_checks(
    *,
    model_type,
    reference_class,
    cls,
    config,
    ignored,
    derived,
    fields,
    directory,
):
    """Runs every row of `RUNS`, and audits the load of the first.

    `fields` are the epsilon fields of the reference config; without them, the
    epsilon rows do not run.

    Returns:
        A tuple `(errors, problems, notes)`. `errors` maps each check that
        ran to its largest error over its rows. `problems` fail the test: a
        load that returned other classes, a task with no logits to compare,
        and what the audit found. `notes` name the converter and give every
        row, for the message.
    """
    errors, problems, notes = {}, [], []
    # Every Keras load and call runs on the CPU: on a GPU, JAX and TensorFlow
    # compute float32 matmuls in reduced precision by default.
    with keras.device("cpu"):
        for index, (check, scale, epsilon) in enumerate(RUNS):
            if epsilon is not None and not fields:
                continue
            label = check
            if epsilon is not None:
                label += f" ({', '.join(fields)} = {epsilon:.0e})"
            path = os.path.join(directory, str(index))
            reference = build_reference(
                reference_class, config, path, scale, epsilon
            )
            checkpoint = f"{label} checkpoint of model_type `{model_type}`"
            backbone, task = load(path, cls, checkpoint)
            inputs = keras_inputs(backbone, reference.config.vocab_size)
            expected = reference_outputs(reference, inputs)
            if index == 0:
                notes.append(f"converter: {converter_name(path)}")
                problems += check_classes(cls, backbone, task)
                if task is not None and "logits" not in expected:
                    problems.append(
                        f"load: `{reference_class}` has no output named "
                        f"`logits` to compare with `{cls.__name__}`. Use a "
                        "transformers class with a language model head, or "
                        "test the backbone."
                    )
                if problems:
                    break
                problems += audit(path, cls, checkpoint, ignored, derived)
            error, text = compare(
                keras_outputs(backbone, task, inputs),
                expected,
                inputs["padding_mask"],
            )
            errors[check] = max(error, errors.get(check, 0.0))
            notes.append(f"{label}: {text}")
    return errors, problems, notes


def judge(errors, fields):
    """Judges each check on its largest error.

    Returns:
        The message lines of the checks that fail.
    """
    failures = []
    for check in CHECKS:
        if check not in errors:
            continue
        error = errors[check]
        hint = HINTS[check].format(fields=", ".join(f"`{f}`" for f in fields))
        if error > THRESHOLD:
            failures.append(
                f"{check}: {error:.3g}, above {THRESHOLD:.0e}. {hint}"
            )
    return failures


# ---------------------------------------------------------------------------
# Argument rules: checked before anything is built.

# Reference config fields a test may set, each with the name of its bound in
# `BOUNDS` (None: no bound). Any other field (an activation, a dropout, a
# rotary base) keeps its transformers default, because changing it can hide a
# bug: a field is added here, with a reviewer, before a test may set it. The
# epsilon check sets the epsilon fields.
FIELDS = {
    "vocab_size": "vocabulary",
    "hidden_size": "size",
    "n_embd": "size",
    "dim": "size",
    "intermediate_size": "size",
    "hidden_dim": "size",
    "head_dim": "size",
    "moe_intermediate_size": "size",
    "shared_expert_intermediate_size": "size",
    "num_hidden_layers": "layers",
    "n_layer": "layers",
    "n_layers": "layers",
    "num_attention_heads": "heads",
    "n_head": "heads",
    "n_heads": "heads",
    "num_key_value_heads": "heads",
    "max_position_embeddings": "positions",
    "n_positions": "positions",
    "sliding_window": "window",
    "local_attention": "window",
    "pad_token_id": "token_id",
    "bos_token_id": "token_id",
    "eos_token_id": "token_id",
    "cls_token_id": "token_id",
    "sep_token_id": "token_id",
    "global_attn_every_n_layers": None,
    "num_local_experts": None,
    "num_experts": None,
    "num_experts_per_tok": None,
    "decoder_sparse_step": None,
}
# Four rows of 17 tokens: long enough for right, left and interior padding.
SEQUENCE_LENGTH = 17
# Input ids are drawn from MIN_TOKEN_ID up, so that no special token id that a
# config sets appears in the inputs.
MIN_TOKEN_ID = 5
# Inclusive bounds (high None: no upper bound), with their reasons. A bound
# applies to the value of its field in the reference config, whether the test
# set it or not, and to each item of a list; a default of 0 or None, which
# means off or unset, is not checked.
BOUNDS = {
    "vocabulary": (32, None, "a smaller vocabulary barely varies the inputs"),
    "size": (4, None, "a size below 4 hides transposed and mis-split tensors"),
    "layers": (2, None, "one layer cannot show layers in the wrong order"),
    "heads": (2, None, "one head hides head split and merge bugs"),
    "positions": (SEQUENCE_LENGTH, None, "the inputs are that long"),
    "window": (
        2,
        SEQUENCE_LENGTH - 1,
        "a window as long as the inputs leaves windowed attention untested",
    ),
    "token_id": (0, MIN_TOKEN_ID - 1, "the inputs use only larger ids"),
}
# Keeps a test to a few seconds per backend. `build_reference` checks it.
MAX_PARAMETERS = 5_000_000


def check_arguments(model_type, reference_class, cls, config, ignored, derived):
    """Returns the problems with the arguments of a test."""
    import transformers

    problems = []
    if not (isinstance(cls, type) and issubclass(cls, (Backbone, CausalLM))):
        problems.append(
            f"cls: {cls!r} is not a keras-hub backbone or causal LM class; "
            "other tasks are not supported."
        )
    model_class = getattr(transformers, str(reference_class), None)
    if not (
        isinstance(model_class, type)
        and issubclass(model_class, transformers.PreTrainedModel)
    ):
        return problems + [
            f"reference_class: `{reference_class}` is not a transformers "
            "model class."
        ]
    config_class = model_class.config_class
    if config_class.model_type != model_type:
        problems.append(
            f"model_type: `{reference_class}` saves model_type "
            f"`{config_class.model_type}`, not `{model_type}`."
        )
    problems += check_config(config_class, config)
    for argument, entries in (
        ("ignored_reference_tensors", ignored),
        ("derived_weights", derived),
    ):
        problems += [
            f"{argument}: `{pattern}` needs a reason."
            for pattern, reason in entries.items()
            if not (isinstance(reason, str) and reason.strip())
        ]
    return problems


def check_config(config_class, config):
    """Returns the problems with the reference config of a test."""
    defaults = config_class()
    problems = []
    for field, value in config.items():
        if field not in FIELDS:
            problems.append(
                f"config: `{field}` is not a field a parity test may set. The "
                "epsilon check sets the epsilon fields. Test how the converter "
                "maps any other field in a unit test of "
                "`convert_backbone_config`, or add it to `FIELDS`, with the "
                "reason, in your pull request."
            )
        elif type(value) is not int:
            problems.append(f"config: `{field}` must be an int, not {value!r}.")
        # A token id may keep its default: its bound keeps it out of the
        # inputs, and the test need not show that the converter reads it.
        elif not field.endswith("_token_id") and value == getattr(
            defaults, field, None
        ):
            problems.append(
                f"config: `{field}={value}` is the {config_class.__name__} "
                "default; use another value, so that the test shows that the "
                f"converter reads `{field}`."
            )
    if problems:
        return problems

    reference = config_class(**config)
    # A config may store a field under another name (GPT-2 keeps
    # `hidden_size` as `n_embd`). Check each stored field once, under the
    # bound of the first of its names in FIELDS.
    aliases = getattr(config_class, "attribute_map", {})
    written = {aliases.get(field, field) for field in config}
    bounds = {}
    for field, bound in FIELDS.items():
        if bound is not None:
            bounds.setdefault(aliases.get(field, field), bound)
    for field, bound in bounds.items():
        low, high, reason = BOUNDS[bound]
        value = getattr(reference, field, None)
        for item in value if isinstance(value, list) else [value]:
            if type(item) is not int or (item == 0 and field not in written):
                continue
            if item < low or (high is not None and item > high):
                where = "is" if field in written else "defaults to"
                limits = (
                    f"{low} or more" if high is None else f"{low} to {high}"
                )
                problems.append(
                    f"config: `{field}` {where} {item}; it must be {limits}, "
                    f"as {reason}."
                )
    heads = getattr(reference, "num_attention_heads", None)
    kv_heads = getattr(reference, "num_key_value_heads", None)
    if (
        type(heads) is int
        and type(kv_heads) is int
        and not 1 < kv_heads < heads
    ):
        problems.append(
            f"config: `num_key_value_heads` is {kv_heads}; it must be more "
            f"than 1 and less than `num_attention_heads` ({heads}), so that "
            "grouped-query attention runs."
        )
    width = getattr(reference, "hidden_size", None)
    if (
        "head_dim" in config
        and type(heads) is int
        and type(width) is int
        and heads > 0
        and config["head_dim"] == width // heads
    ):
        problems.append(
            f"config: `head_dim={config['head_dim']}` is `hidden_size // "
            "num_attention_heads`; use another value, so that the test shows "
            "that the converter reads `head_dim`."
        )
    return problems


# ---------------------------------------------------------------------------
# The reference model.

# Seeds torch and Keras, the weight draws and the inputs.
SEED = 0
# A matrix gets noise of standard deviation 1.2 / sqrt(its smaller
# dimension), at most 0.6: large enough that a swap or transpose changes the
# outputs, small enough that activations stay in range.
MATRIX_GAIN = 1.2
MATRIX_MAX_STD = 0.6
# A vector gets its constructed value (1 for a norm gain, 0 for a bias) plus
# noise of this standard deviation.
VECTOR_STD = 0.25


def reference_config(reference_class, config):
    """Returns the transformers config that a test describes."""
    import transformers

    return getattr(transformers, reference_class).config_class(**config)


def epsilon_fields(config):
    """Returns the names of the normalization epsilon fields of a config.

    These are the float fields whose name contains "eps". Sub-configs are not
    searched: no covered model has one.
    """
    return sorted(
        name
        for name, value in vars(config).items()
        if isinstance(value, float) and "eps" in name.lower()
    )


def build_reference(reference_class, config, directory, scale, epsilon):
    """Saves a tiny transformers model to `directory`; returns it reloaded.

    The model is built from `config`, with every epsilon field set to
    `epsilon` unless it is None. Its weights are redrawn (see `redraw`), and
    it is saved with `save_pretrained`, then loaded back in float32, so that
    the reference is what transformers reads from the files that keras-hub
    reads.
    """
    import torch
    import transformers

    model_class = getattr(transformers, reference_class)
    model_config = reference_config(reference_class, config)
    if epsilon is not None:
        for name in epsilon_fields(model_config):
            setattr(model_config, name, epsilon)
    try:
        # Counted on the meta device, which allocates no memory.
        with torch.device("meta"):
            count = sum(
                p.numel() for p in model_class(model_config).parameters()
            )
        if count > MAX_PARAMETERS:
            raise ValueError(
                f"The reference model has {count:,} parameters, more than "
                f"{MAX_PARAMETERS:,}. Use a smaller config."
            )
        torch.manual_seed(SEED)
        model = model_class(model_config)
        redraw(model, scale)
        model.save_pretrained(directory)
        model, info = model_class.from_pretrained(
            directory,
            dtype=torch.float32,
            local_files_only=True,
            output_loading_info=True,
        )
    except Exception as error:
        error.add_note(
            f"Raised while building `{reference_class}` from the test config."
        )
        raise
    for name in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        if info.get(name):
            raise ValueError(
                f"Reloading the reference gave {name}: {sorted(info[name])}"
            )
    return model.eval()


def redraw(model, scale):
    """Replaces every parameter of a torch model with keyed random values.

    A tensor of two or more dimensions gets zero-mean noise, scaled to its
    smaller matrix dimension. A vector gets its constructed value if that is
    constant (1 for a norm gain, 0 for a bias), else 0, plus small noise:
    centring on the constructed value keeps Gemma-style `1 + weight` norms
    right. Outside normalization layers, the noise is multiplied by `scale`.
    Buffers keep their constructed values. Draws are keyed by parameter name,
    so a transformers release that renames a model's parameters redraws them.
    """
    import torch

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.numel() == 0:
                continue
            module = model.get_submodule(name.rpartition(".")[0])
            noise = unit_draw(SEED, name, tuple(parameter.shape))
            if "norm" not in type(module).__name__.lower():
                noise = noise * scale
            if parameter.dim() >= 2:
                std = MATRIX_GAIN / math.sqrt(min(parameter.shape[-2:]))
                value = min(MATRIX_MAX_STD, std) * noise
            else:
                flat = parameter.detach().reshape(-1)
                constant = bool((flat == flat[0]).all())
                center = float(flat[0]) if constant else 0.0
                value = center + VECTOR_STD * noise
            parameter.copy_(torch.as_tensor(value, dtype=parameter.dtype))


def unit_draw(seed, key, shape):
    """Draws zero-mean, unit-variance noise keyed by `seed` and `key`.

    The generator is seeded from the SHA-256 of the key, and each value is
    the sum of four uniform draws, centred and scaled. Raw bit-generator
    output, unlike numpy's samplers, is the same on every platform and numpy
    version, so the draws are too.
    """
    digest = hashlib.sha256(f"{int(seed)}\x00{key}".encode("utf-8")).digest()
    generator = np.random.PCG64(int.from_bytes(digest[:16], "little"))
    size = int(np.prod(shape))
    raw = generator.random_raw(4 * size).reshape(4, size) >> np.uint64(11)
    total = raw.sum(axis=0, dtype=np.uint64)
    # Four uniforms on [0, 1) sum to a mean of 2 and a variance of 1/3.
    draw = (total.astype("float64") * 2.0**-53 - 2.0) * np.sqrt(3.0)
    return draw.reshape(shape)


def reference_outputs(model, inputs):
    """Runs the reference on Keras inputs; returns numpy outputs.

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


# ---------------------------------------------------------------------------
# The Keras model.


def load(directory, cls, checkpoint):
    """Loads a checkpoint as a user would; returns `(backbone, task)`.

    `task` is None when `cls` is a backbone. `checkpoint` describes the
    checkpoint in the note added to an exception.
    """
    import transformers

    keras.utils.set_random_seed(SEED)
    try:
        if issubclass(cls, CausalLM):
            task = cls.from_preset(directory, preprocessor=None)
            return task.backbone, task
        return cls.from_preset(directory), None
    except Exception as error:
        error.add_note(
            f"Raised by `{cls.__name__}.from_preset` while loading the "
            f"{checkpoint}, written by transformers {transformers.__version__}."
        )
        raise


def converter_name(directory):
    """Returns the name of the converter module that loads `directory`."""
    converter = preset_utils.get_preset_loader(directory).converter
    return converter.__name__.rpartition(".")[2]


def check_classes(cls, backbone, task):
    """Returns a problem if the load returned other classes than `cls`."""
    loaded = backbone if task is None else task
    if type(loaded) is not cls:
        return [f"load: returned {type(loaded).__name__}, not {cls.__name__}."]
    if task is not None and type(backbone) is not cls.backbone_cls:
        return [
            f"load: the backbone is {type(backbone).__name__}, not "
            f"{cls.backbone_cls.__name__}."
        ]
    return []


def keras_inputs(backbone, vocabulary_size):
    """Returns four rows of `SEQUENCE_LENGTH` tokens, as `backbone` takes them.

    Row 0 is all valid, row 1 is right-padded, row 2 is left-padded and row
    3 has a hole and right padding. Segment ids, if the backbone takes them,
    are 0 in the first half of each row and 1 in the second.
    """
    rng = np.random.default_rng(SEED)
    shape = (4, SEQUENCE_LENGTH)
    token_ids = rng.integers(MIN_TOKEN_ID, vocabulary_size, size=shape)
    padding_mask = np.ones(shape, dtype="int32")
    padding_mask[1, 11:] = 0
    padding_mask[2, :6] = 0
    padding_mask[3, 3:6] = 0
    padding_mask[3, 14:] = 0
    segment_ids = np.zeros(shape, dtype="int32")
    segment_ids[:, SEQUENCE_LENGTH // 2 :] = 1
    inputs = {
        "token_ids": token_ids.astype("int32"),
        "padding_mask": padding_mask,
        "segment_ids": segment_ids,
    }
    return {name: inputs[name] for name in backbone.input if name in inputs}


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


# ---------------------------------------------------------------------------
# The load audit: one more load of the first checkpoint, never compared.


def audit(directory, cls, checkpoint, ignored, derived):
    """Returns problem lines for what the converter does not read or assign.

    The checkpoint is loaded once more under `recording`, which sets every
    float Keras weight to NaN before the converter runs. A stored tensor is
    unread if no conversion read it; a Keras weight is unassigned if it still
    holds NaN, or was built after the conversion. The test's
    `ignored_reference_tensors` and `derived_weights` excuse them; each of
    their patterns must match something to excuse, and nothing else.
    """
    with recording(directory) as (reads, seen):
        try:
            backbone, task = load(directory, cls, f"{checkpoint}, audited")
        except Exception as error:
            error.add_note(
                "Raised only under the load audit. Either the converter reads "
                "a weight before it assigns it (the audit fills every weight "
                "with NaN until then), or the audit failed: see the last "
                "frame."
            )
            raise
    weights = (backbone if task is None else task).weights
    paths = [weight.path for weight in weights]
    stored = stored_keys(directory)
    unread = [key for key in stored if key not in reads]
    unassigned = [w for w in weights if id(w) not in seen or holds_nan(w)]
    pattern_problems = check_patterns(
        "ignored_reference_tensors", ignored, stored, unread, "reads"
    ) + check_patterns(
        "derived_weights",
        derived,
        paths,
        [w.path for w in unassigned],
        "assigns",
    )
    unread = [key for key in unread if not matches(key, ignored)]
    unassigned = [w for w in unassigned if not matches(w.path, derived)]
    problems = []
    if unread:
        problems.append(
            f"unread: {len(unread)} of {len(stored)} stored tensors were never "
            f"read by {converter_name(directory)}:\n"
            + listing(unread)
            + "\n  Port them. A part the Keras class does not have, such as "
            "a head, goes in `ignored_reference_tensors`, with the reason."
        )
    if unassigned:
        # Paths can repeat; then the index in `weights` tells them apart.
        index = {id(w): i for i, w in enumerate(weights)}
        names = [
            f"{w.path} {tuple(w.shape)}"
            + (f" (weights[{index[id(w)]}])" if paths.count(w.path) > 1 else "")
            for w in unassigned
        ]
        problems.append(
            f"unassigned: {len(unassigned)} of {len(weights)} Keras weights "
            f"were never assigned:\n{listing(names)}\n  Port them. A weight "
            "the model computes from its config goes in `derived_weights`, "
            "with the reason."
        )
    return problems + pattern_problems


@contextlib.contextmanager
def recording(directory):
    """Records what a load of `directory` reads, and fills weights with NaN.

    It patches two things, and undoes both on exit:
    - the converter that `from_preset` picks for `directory`: each
      `convert_weights` or `convert_head` call first sets every float weight
      it has not seen yet to NaN;
    - the `safetensors` module that `SafetensorLoader` uses: every tensor
      read is recorded by its stored key.

    Yields:
        A tuple `(reads, seen)`: the set of stored keys read, and a dict from
        `id()` to every weight that a conversion call saw. Keras variables
        are not hashable, and holding them keeps their ids unique.
    """
    converter = preset_utils.get_preset_loader(directory).converter
    reads, seen = set(), {}

    def fill_then_convert(convert):
        def wrapper(model, *args, **kwargs):
            for weight in model.weights:
                if id(weight) not in seen:
                    seen[id(weight)] = weight
                    if "float" in str(weight.dtype):
                        weight.assign(np.full(weight.shape, np.nan, "float32"))
            return convert(model, *args, **kwargs)

        return wrapper

    safetensors = safetensor_utils.safetensors

    def safe_open(*args, **kwargs):
        return RecordingFile(safetensors.safe_open(*args, **kwargs), reads)

    with contextlib.ExitStack() as stack:
        module = types.SimpleNamespace(**vars(safetensors))
        module.safe_open = safe_open
        stack.enter_context(
            mock.patch.object(safetensor_utils, "safetensors", module)
        )
        for name in ("convert_weights", "convert_head"):
            if hasattr(converter, name):
                wrapper = fill_then_convert(getattr(converter, name))
                stack.enter_context(mock.patch.object(converter, name, wrapper))
        yield reads, seen


class RecordingFile:
    """An open safetensors file that records the key of every tensor read."""

    def __init__(self, file, reads):
        self.file = file
        self.reads = reads

    def __enter__(self):
        # SafetensorLoader keeps what `__enter__` returns.
        self.file.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self.file.__exit__(*exc_info)

    def __getattr__(self, name):
        return getattr(self.file, name)

    def get_tensor(self, key):
        tensor = self.file.get_tensor(key)
        self.reads.add(key)
        return tensor


def stored_keys(directory):
    """Returns the keys of the tensors saved in `directory`."""
    from safetensors import safe_open

    keys = []
    for name in sorted(os.listdir(directory)):
        if name.endswith(".safetensors"):
            path = os.path.join(directory, name)
            with safe_open(path, framework="np") as file:
                keys += list(file.keys())
    return keys


def holds_nan(weight):
    """Whether a float Keras weight holds NaN."""
    if "float" not in str(weight.dtype):
        return False
    return bool(np.isnan(keras.ops.convert_to_numpy(weight.value)).any())


def matches(name, patterns):
    """Whether `name` matches any of the fnmatch `patterns`."""
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def check_patterns(argument, patterns, names, excusable, verb):
    """Returns the problems with the patterns of an exemption argument.

    Each pattern must match at least one name of `excusable`, and no other
    name of `names`; `verb` says what the converter does with the others.
    """
    problems = []
    for pattern in patterns:
        matched = [name for name in names if fnmatch.fnmatchcase(name, pattern)]
        used = [name for name in matched if name not in excusable]
        if len(used) == len(matched):
            problems.append(
                f"{argument}: `{pattern}` matches nothing to excuse; remove it."
            )
        elif used:
            problems.append(
                f"{argument}: `{pattern}` also matches `{used[0]}`, which the "
                f"converter {verb}; make it specific."
            )
    return problems


def listing(items, limit=8):
    """Lists items one per line, indented, at most `limit` of them."""
    lines = [f"    {item}" for item in items[:limit]]
    if len(items) > limit:
        lines.append(f"    ... and {len(items) - limit} more")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The comparison.


def compare(actual, expected, padding_mask):
    """Compares the Keras outputs with the reference outputs both have.

    The pooled output has no positions, so no mask applies to it.

    Returns:
        A tuple `(error, text)`: the largest error over the outputs, and a
        text such as `"hidden 1.20e-06, logits 3.40e-06"`.
    """
    errors, parts = [], []
    for name, reference in expected.items():
        if name in actual:
            mask = None if name == "pooled" else padding_mask
            error, note = per_position_error(actual[name], reference, mask)
            errors.append(error)
            parts.append(f"{name} {note or f'{error:.2e}'}")
    return max(errors), ", ".join(parts)


def per_position_error(keras_output, reference_output, padding_mask=None):
    """Compares two outputs position by position, in float64.

    At each position where `padding_mask` (shape `(batch, positions)`, if
    given) is not 0, the largest absolute difference is divided by the root
    mean square of the reference at that position, so that the small outputs
    of the small_weights and epsilon checks are judged against their own
    scale. The error is the largest of these values.

    Returns:
        A tuple `(error, note)`: the error and an empty note, or `inf` and
        why the outputs cannot be compared.
    """
    keras_output = np.asarray(keras_output, dtype="float64")
    reference_output = np.asarray(reference_output, dtype="float64")
    if keras_output.shape != reference_output.shape:
        return math.inf, (
            f"shapes differ: {keras_output.shape} in Keras, "
            f"{reference_output.shape} in transformers"
        )
    if padding_mask is not None:
        valid = np.asarray(padding_mask).astype(bool)
        keras_output = keras_output[valid]
        reference_output = reference_output[valid]
    if not np.isfinite(reference_output).all():
        return math.inf, "NaN or inf in the transformers output"
    if not np.isfinite(keras_output).all():
        return math.inf, "NaN or inf in the Keras output"
    width = reference_output.shape[-1]
    keras_output = keras_output.reshape(-1, width)
    reference_output = reference_output.reshape(-1, width)
    rms = np.sqrt(np.mean(reference_output**2, axis=-1))
    difference = np.abs(keras_output - reference_output).max(axis=-1)
    error = difference / np.maximum(rms, np.finfo("float64").tiny)
    return float(error.max()), ""
