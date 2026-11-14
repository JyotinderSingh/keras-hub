import ast
import pathlib
import textwrap

from keras_hub.src.tests.test_case import TestCase

TRANSFORMERS_DIR = pathlib.Path(__file__).parent

# Dispatch keys of `preset_loader.py` that have no reference parity test yet,
# as `{model_type: reason}`. Blockers were found with transformers 5.22.2.
NOT_YET_COVERED = {
    "bart": (
        "Reads `num_hidden_layers`, which transformers does not save for "
        "BART. Encoder-decoder inputs are a later step."
    ),
    "blip-2": "Vision-language model: image inputs are a later step.",
    "deit": "Vision model: image inputs are a later step.",
    "diffusion_gemma": (
        "Reads `final_logit_softcapping`, which transformers does not save. "
        "Image inputs are a later step."
    ),
    "dinov2": "Vision model: image inputs are a later step.",
    "dinov2_with_registers": "Vision model: image inputs are a later step.",
    "dinov3_vit": "Vision model: image inputs are a later step.",
    "embedding_gemma2": (
        "A text-only config does not load: the converter needs a vision "
        "config. Image and audio inputs are a later step."
    ),
    "esm": (
        "Cannot read a checkpoint written by `save_pretrained`: it asks for "
        "`encoder.layer.0.attention.LayerNorm.weight`, which is not there."
    ),
    "gemma3": (
        "Expects `vision_tower.vision_model.*` keys, which transformers "
        "saves as `vision_tower.*`. Image inputs are a later step."
    ),
    "gemma3n": (
        "The vision tower has no size fields and keeps 294M parameters. "
        "Image and audio inputs are a later step."
    ),
    "gemma4": "Multimodal model: image and audio inputs are a later step.",
    "gemma4_assistant": "Not onboarded yet (a drafter model).",
    "gemma4_text": (
        "Reads `global_head_dim`, which transformers saves in "
        "`per_layer_config`. The backbone also needs `position_ids`."
    ),
    "gpt_oss": (
        "Reads only MXFP4 expert tensors (`*_blocks`, `*_scales`), which a "
        "float32 reference does not have."
    ),
    "metaclip_2": "Vision-language model: image inputs are a later step.",
    "mistral3": "Vision-language model: image inputs are a later step.",
    "muse_glimmer": "Vision-language model: image inputs are a later step.",
    "muse_glimmer_assistant": "Not onboarded yet (a drafter model).",
    "paligemma": (
        "Expects `vision_tower.vision_model.*` keys, which transformers "
        "saves as `vision_tower.*`. Image inputs are a later step."
    ),
    "qwen2_vl": "Vision-language model: image inputs are a later step.",
    "qwen3_5": "Vision-language model: image inputs are a later step.",
    "qwen3_5_moe": "Vision-language model: image inputs are a later step.",
    "qwen3_moe": "Reads `num_experts`, which transformers does not save.",
    "sam3_video": "Segmentation model: image and prompts are a later step.",
    "smolvlm": (
        "Reads `rope_theta`, which transformers saves in `rope_parameters`. "
        "Image inputs are a later step."
    ),
    "swin": "Vision model: image inputs are a later step.",
    "t5gemma": "Not onboarded yet: encoder-decoder inputs are a later step.",
    "t5gemma2": "Encoder-decoder model with images: a later step.",
    "vit": "Vision model: image inputs are a later step.",
}


def dispatched_model_types(test):
    """Returns the model types that an `if` test of the dispatch chain picks.

    Returns `None` unless the test is `model_type == "x"` or
    `model_type in ("x", "y")`.
    """
    if not (
        isinstance(test, ast.Compare)
        and ast.unparse(test.left) == "model_type"
        and len(test.ops) == 1
    ):
        return None
    try:
        value = ast.literal_eval(test.comparators[0])
    except ValueError:
        return None
    if isinstance(test.ops[0], ast.Eq) and isinstance(value, str):
        return [value]
    if (
        isinstance(test.ops[0], ast.In)
        and isinstance(value, (tuple, list, set))
        and all(isinstance(item, str) for item in value)
    ):
        return list(value)
    return None


def dispatch_keys(path):
    """Reads the dispatch chain of `preset_loader.py`.

    Returns:
        A tuple `(keys, problems)`. `keys` maps each model type to its
        converter module, from every `if model_type == "x":` or
        `if model_type in ("x", "y"):` branch whose body has the statement
        `self.converter = convert_x`. `problems` lists every other
        assignment to `self.converter`, which this test cannot read.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keys, read = {}, []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        model_types = dispatched_model_types(node.test)
        for statement in node.body:
            if (
                model_types
                and isinstance(statement, ast.Assign)
                and [ast.unparse(t) for t in statement.targets]
                == ["self.converter"]
                and isinstance(statement.value, ast.Name)
            ):
                keys.update((key, statement.value.id) for key in model_types)
                read.append(statement.targets[0])
    problems = [
        f"{path.name}:{node.lineno}: cannot tell which `model_type` selects "
        'this converter. Write its branch as `if model_type == "x":` or '
        '`if model_type in ("x", "y"):`, or update `dispatch_keys()` in '
        "this test."
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, ast.Store)
        and ast.unparse(node) == "self.converter"
        and node not in read
    ]
    return keys, problems


def is_test_class(node):
    """Whether pytest collects a class: a `TestCase` or a `Test...` class."""
    bases = [ast.unparse(base).split(".")[-1] for base in node.bases]
    return "TestCase" in bases or node.name.startswith("Test")


def sets_pytestmark(body):
    """Whether a module or class body sets `pytestmark`.

    A `pytestmark` marks every test of the module or class, as a decorator
    would.
    """
    return any(
        isinstance(node, ast.Name)
        and node.id == "pytestmark"
        and isinstance(node.ctx, ast.Store)
        for statement in body
        if not isinstance(statement, (ast.FunctionDef, ast.ClassDef))
        for node in ast.walk(statement)
    )


def skip_lines(method):
    """Returns the lines of a method body that can skip or xfail the test.

    These are uses of `skipTest`, `pytest.skip`, `pytest.importorskip`,
    `pytest.xfail`, `unittest.SkipTest` and the `skip` and `skipif` markers.
    """
    names = ("skipTest", "skip", "skipif", "importorskip", "xfail", "SkipTest")
    return [
        node.lineno
        for statement in method.body
        for node in ast.walk(statement)
        if getattr(node, "attr", getattr(node, "id", None)) in names
    ]


def statement_index(method, call):
    """Returns the index of `call` in `method.body`, or `None`.

    It is `None` unless the call is a statement of its own in the method
    body: not inside an `if`, `for`, `while`, `try` or `with` block, and not
    part of a larger expression.
    """
    for index, statement in enumerate(method.body):
        if isinstance(statement, ast.Expr) and statement.value is call:
            return index
    return None


def can_return(statements):
    """Whether `statements` hold a `return` of the function they are in.

    A `return` inside a nested function or class does not count.
    """
    nested = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for node in statements:
        if isinstance(node, ast.Return):
            return True
        if not isinstance(node, nested) and can_return(
            ast.iter_child_nodes(node)
        ):
            return True
    return False


def parity_calls(tree):
    """Yields `(class, method, call)` for each parity test call in a module.

    Only calls inside the methods of a module-level class count.
    """
    for test_class in tree.body:
        if not isinstance(test_class, ast.ClassDef):
            continue
        for method in test_class.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            for node in ast.walk(method):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "run_reference_parity_test"
                ):
                    yield test_class, method, node


def parity_tests(directory):
    """Finds the `run_reference_parity_test` calls in `directory`.

    Returns:
        A tuple `(tests, problems)`: `{model_type: [test file names]}`, and
        the calls whose `model_type` is not a string literal keyword, or
        that pytest would not run on every pull request: calls outside a
        test method of a test class, calls that a decorator or a
        `pytestmark` marks, for example with `pytest.mark.large`, calls in a
        method that can skip itself, calls that are not a statement of
        their own in the method body, and calls after a `return`.
    """
    tests = {}
    problems = []
    for path in sorted(directory.glob("*_test.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for test_class, method, call in parity_calls(tree):
            where = f"{path.name}:{call.lineno}"
            name = f"`{test_class.name}.{method.name}`"
            index = statement_index(method, call)
            if not (
                is_test_class(test_class) and method.name.startswith("test")
            ):
                problems.append(
                    f"{where}: pytest does not run {name}. Call "
                    "`run_reference_parity_test` in a method named "
                    "`test...` of a `TestCase` class."
                )
            elif (
                test_class.decorator_list
                or method.decorator_list
                or sets_pytestmark(tree.body)
                or sets_pytestmark(test_class.body)
            ):
                problems.append(
                    f"{where}: remove the decorators and `pytestmark` that "
                    f"mark {name}; a reference parity test runs on every "
                    "pull request and every backend."
                )
            elif skip_lines(method):
                problems.append(
                    f"{path.name}:{skip_lines(method)[0]}: remove the skip "
                    f"from {name}; a reference parity test runs on every pull "
                    "request and every backend. The helper skips it where it "
                    "cannot run."
                )
            elif index is None:
                problems.append(
                    f"{where}: call `run_reference_parity_test` as a "
                    f"statement of its own in the body of {name}, not inside "
                    "an `if`, `for`, `while`, `try` or `with` block; a "
                    "reference parity test runs on every pull request and "
                    "every backend."
                )
            elif can_return(method.body[:index]):
                problems.append(
                    f"{where}: {name} can return before it calls "
                    "`run_reference_parity_test`; remove the `return`, as a "
                    "reference parity test runs on every pull request and "
                    "every backend."
                )
            model_type = None
            for keyword in call.keywords:
                if keyword.arg == "model_type":
                    model_type = getattr(keyword.value, "value", None)
            if isinstance(model_type, str):
                tests.setdefault(model_type, []).append(path.name)
            else:
                problems.append(
                    f"{where}: pass `model_type` to "
                    "`run_reference_parity_test` as a string literal "
                    "keyword, so that this test can find it."
                )
    return tests, problems


def coverage_problems(directory, not_yet_covered):
    """Checks that every dispatch key has a reference parity test.

    Args:
        directory: The folder of `preset_loader.py` and the converter tests.
        not_yet_covered: `{model_type: reason}` for keys without a test.

    Returns:
        A list of problems, empty if there are none.
    """
    dispatch, problems = dispatch_keys(directory / "preset_loader.py")
    tests, test_problems = parity_tests(directory)
    problems += test_problems
    if not dispatch:
        return problems + [
            "Found no dispatch keys in preset_loader.py; update "
            "`dispatch_keys()` in this test."
        ]
    for model_type, converter in sorted(dispatch.items()):
        if model_type not in tests and model_type not in not_yet_covered:
            problems.append(
                f"`{model_type}` has no reference parity test. Add one to "
                f"{converter}_test.py (see CONTRIBUTING_MODELS.md), or list "
                "it in NOT_YET_COVERED with the reason."
            )
    for model_type, reason in sorted(not_yet_covered.items()):
        if model_type not in dispatch:
            problems.append(
                f"NOT_YET_COVERED lists `{model_type}`, which is not a "
                "dispatch key of preset_loader.py; remove it."
            )
        elif model_type in tests:
            problems.append(
                f"`{model_type}` has a reference parity test now; remove it "
                f"from NOT_YET_COVERED (reason was: {reason})"
            )
    for model_type, files in sorted(tests.items()):
        for file in files:
            if model_type not in dispatch:
                problems.append(
                    f"{file} tests `{model_type}`, which is not a dispatch "
                    "key of preset_loader.py."
                )
            elif file != f"{dispatch[model_type]}_test.py":
                problems.append(
                    f"The reference parity test of `{model_type}` is in "
                    f"{file}; move it to {dispatch[model_type]}_test.py, the "
                    f"test file of the converter that `{model_type}` is "
                    "routed to."
                )
    return problems


class ReferenceParityCoverageTest(TestCase):
    def test_every_dispatch_key_has_a_reference_parity_test(self):
        problems = coverage_problems(TRANSFORMERS_DIR, NOT_YET_COVERED)
        if problems:
            self.fail("\n".join(problems))

    def test_coverage_problems(self):
        sources = {
            "preset_loader.py": """
                class TransformersPresetLoader(PresetLoader):
                    def __init__(self, preset, config):
                        model_type = self.config["model_type"]
                        if model_type == "alpha":
                            self.converter = convert_alpha
                        elif model_type in ("beta", "beta_v2"):
                            self.converter = convert_beta
                        elif model_type == "gamma":
                            self.converter = convert_gamma
                        elif model_type == "epsilon":
                            self.converter = convert_epsilon
                        else:
                            raise ValueError(model_type)
            """,
            "convert_alpha_test.py": """
                class TestTask(TestCase):
                    def test_reference_parity(self):
                        self.run_reference_parity_test(model_type="alpha")

                    def test_reference_parity_beta(self):
                        self.run_reference_parity_test(model_type="beta")

                    @pytest.mark.large
                    def test_reference_parity_large(self):
                        self.run_reference_parity_test(model_type="alpha")

                    def reference_parity(self):
                        self.run_reference_parity_test(model_type="alpha")


                @pytest.mark.skip
                class TestSkipped(TestCase):
                    def test_reference_parity(self):
                        self.run_reference_parity_test(model_type="alpha")


                class Helpers:
                    def test_reference_parity(self):
                        self.run_reference_parity_test(model_type="alpha")


                class TestMarked(TestCase):
                    pytestmark = pytest.mark.large

                    def test_reference_parity(self):
                        self.run_reference_parity_test(model_type="alpha")


                class TestSkipsItself(TestCase):
                    def test_reference_parity(self):
                        if keras.config.backend() != "torch":
                            self.skipTest("Slow outside torch.")
                        self.run_reference_parity_test(model_type="alpha")

                    def test_reference_parity_importorskip(self):
                        pytest.importorskip("tensorflow_text")
                        self.run_reference_parity_test(model_type="alpha")


                class TestPlacement(TestCase):
                    def test_reference_parity_in_if(self):
                        if keras.config.backend() == "torch":
                            self.run_reference_parity_test(model_type="alpha")

                    def test_reference_parity_in_with(self):
                        with self.subTest("alpha"):
                            self.run_reference_parity_test(model_type="alpha")

                    def test_reference_parity_after_return(self):
                        if keras.config.backend() != "torch":
                            return
                        self.run_reference_parity_test(model_type="alpha")

                    def test_reference_parity_with_helper(self):
                        def config():
                            return {"vocab_size": 211}

                        self.run_reference_parity_test(model_type="alpha")
            """,
            "convert_beta_test.py": """
                class TestTask(TestCase):
                    def test_reference_parity(self):
                        self.run_reference_parity_test(model_type="beta_v2")
                        self.run_reference_parity_test(model_type=MODEL_TYPE)
                        self.run_reference_parity_test(model_type="omega")
            """,
            "convert_epsilon_test.py": """
                pytestmark = pytest.mark.large


                class TestTask(TestCase):
                    def test_reference_parity(self):
                        self.run_reference_parity_test(model_type="epsilon")
            """,
        }
        directory = pathlib.Path(self.get_temp_dir())
        for name, source in sources.items():
            (directory / name).write_text(textwrap.dedent(source))
        not_yet_covered = {
            "beta_v2": "Image inputs are a later step.",
            "delta": "Not onboarded yet.",
        }
        marked = (
            "{}: remove the decorators and `pytestmark` that mark {}; a "
            "reference parity test runs on every pull request and every "
            "backend."
        )
        not_run = (
            "{}: pytest does not run {}. Call `run_reference_parity_test` in "
            "a method named `test...` of a `TestCase` class."
        )
        skips = (
            "{}: remove the skip from {}; a reference parity test runs on "
            "every pull request and every backend. The helper skips it where "
            "it cannot run."
        )
        nested = (
            "{}: call `run_reference_parity_test` as a statement of its own "
            "in the body of {}, not inside an `if`, `for`, `while`, `try` or "
            "`with` block; a reference parity test runs on every pull "
            "request and every backend."
        )
        after_return = (
            "{}: {} can return before it calls `run_reference_parity_test`; "
            "remove the `return`, as a reference parity test runs on every "
            "pull request and every backend."
        )
        self.assertEqual(
            coverage_problems(directory, not_yet_covered),
            [
                marked.format(
                    "convert_alpha_test.py:11",
                    "`TestTask.test_reference_parity_large`",
                ),
                not_run.format(
                    "convert_alpha_test.py:14", "`TestTask.reference_parity`"
                ),
                marked.format(
                    "convert_alpha_test.py:20",
                    "`TestSkipped.test_reference_parity`",
                ),
                not_run.format(
                    "convert_alpha_test.py:25",
                    "`Helpers.test_reference_parity`",
                ),
                marked.format(
                    "convert_alpha_test.py:32",
                    "`TestMarked.test_reference_parity`",
                ),
                skips.format(
                    "convert_alpha_test.py:38",
                    "`TestSkipsItself.test_reference_parity`",
                ),
                skips.format(
                    "convert_alpha_test.py:42",
                    "`TestSkipsItself.test_reference_parity_importorskip`",
                ),
                nested.format(
                    "convert_alpha_test.py:49",
                    "`TestPlacement.test_reference_parity_in_if`",
                ),
                nested.format(
                    "convert_alpha_test.py:53",
                    "`TestPlacement.test_reference_parity_in_with`",
                ),
                after_return.format(
                    "convert_alpha_test.py:58",
                    "`TestPlacement.test_reference_parity_after_return`",
                ),
                "convert_beta_test.py:5: pass `model_type` to "
                "`run_reference_parity_test` as a string literal keyword, so "
                "that this test can find it.",
                marked.format(
                    "convert_epsilon_test.py:7",
                    "`TestTask.test_reference_parity`",
                ),
                "`gamma` has no reference parity test. Add one to "
                "convert_gamma_test.py (see CONTRIBUTING_MODELS.md), or list "
                "it in NOT_YET_COVERED with the reason.",
                "`beta_v2` has a reference parity test now; remove it from "
                "NOT_YET_COVERED (reason was: Image inputs are a later step.)",
                "NOT_YET_COVERED lists `delta`, which is not a dispatch key "
                "of preset_loader.py; remove it.",
                "The reference parity test of `beta` is in "
                "convert_alpha_test.py; move it to convert_beta_test.py, the "
                "test file of the converter that `beta` is routed to.",
                "convert_beta_test.py tests `omega`, which is not a dispatch "
                "key of preset_loader.py.",
            ],
        )
        (directory / "preset_loader.py").write_text("")
        self.assertEqual(
            coverage_problems(directory, {})[-1],
            "Found no dispatch keys in preset_loader.py; update "
            "`dispatch_keys()` in this test.",
        )

    def test_dispatch_keys(self):
        path = pathlib.Path(self.get_temp_dir()) / "preset_loader.py"
        source = """
            class TransformersPresetLoader(PresetLoader):
                def __init__(self, preset, config):
                    model_type = self.config["model_type"]
                    if model_type == "alpha":
                        self.converter = convert_alpha
                    elif model_type in ("beta", "beta_v2"):
                        self.converter = convert_beta
                    elif model_type == "gamma":
                        from . import convert_gamma

                        self.converter = convert_gamma
                    elif model_type == "delta" or model_type == "eta":
                        self.converter = convert_delta
                    elif self.config["model_type"] == "zeta":
                        self.converter = convert_zeta
                    elif model_type == "theta":
                        self.converter = converter = convert_theta
                    else:
                        raise ValueError(model_type)
        """
        path.write_text(textwrap.dedent(source))
        keys, problems = dispatch_keys(path)
        self.assertEqual(
            keys,
            {
                "alpha": "convert_alpha",
                "beta": "convert_beta",
                "beta_v2": "convert_beta",
                "gamma": "convert_gamma",
            },
        )
        # A branch this test cannot read fails, rather than go unchecked.
        message = (
            "preset_loader.py:{}: cannot tell which `model_type` selects this "
            'converter. Write its branch as `if model_type == "x":` or `if '
            'model_type in ("x", "y"):`, or update `dispatch_keys()` in this '
            "test."
        )
        self.assertEqual(
            problems, [message.format(line) for line in (14, 16, 18)]
        )
