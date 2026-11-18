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
    "gemma4_assistant": (
        "Drafter model: it runs with a target model, which the helper does "
        "not build."
    ),
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
    "muse_glimmer_assistant": (
        "Drafter model: it runs with a target model, which the helper does "
        "not build."
    ),
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
    "t5gemma": (
        "Encoder-decoder model: encoder-decoder inputs are a later step."
    ),
    "t5gemma2": "Encoder-decoder model with images: a later step.",
    "vit": "Vision model: image inputs are a later step.",
}


UNREAD_BRANCH = (
    "{}: cannot tell which `model_type` selects this converter. Write the "
    'branch as `if model_type == "x":` or `if model_type in ("x", "y"):`, '
    "with `self.converter = convert_x` in its body."
)
METHOD_SHAPE = (
    "{}: a reference parity test runs on every pull request. Make the call "
    "the last statement of a `test...` method of a module-level `TestCase` "
    "class, with `model_type` a string literal. Before it, the method may "
    "hold only a docstring and assignments of string constants. Add no "
    "decorators to the method or class, and set no `pytestmark` in the class "
    "or module."
)


def parse(path):
    """Returns the tree of a file, and a map from each node to its parent."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return tree, {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}


def is_string(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def selected_model_types(test):
    """Returns the model types an `if` test selects, or `[]`.

    The test, written back by `ast.unparse`, must be `model_type == "x"` or
    `model_type in ("x", "y")`, with the string literals it holds.
    """
    values = tuple(node.value for node in ast.walk(test) if is_string(node))
    forms = [f"model_type == {value!r}" for value in values]
    forms.append(f"model_type in {values!r}")
    return list(values) if values and ast.unparse(test) in forms else []


def dispatch_keys(path):
    """Returns `({model_type: converter module}, problems)`.

    Each assignment to `self.converter` must be `self.converter = convert_x`
    in the body of an `if` that `selected_model_types` reads.
    """
    tree, parent = parse(path)
    converters, problems = {}, []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and ast.unparse(node) == "self.converter"
        ):
            continue
        statement, branch = parent[node], parent[parent[node]]
        model_types = []
        if isinstance(branch, ast.If) and statement in branch.body:
            model_types = selected_model_types(branch.test)
        if (
            model_types
            and getattr(statement, "targets", None) == [node]
            and isinstance(statement.value, ast.Name)
        ):
            converters.update(dict.fromkeys(model_types, statement.value.id))
        else:
            problems.append(UNREAD_BRANCH.format(f"{path.name}:{node.lineno}"))
    return converters, problems


def sets_pytestmark(node):
    """Whether a module or class sets `pytestmark` outside its functions.

    A nested class is skipped too: its `pytestmark` marks only its own tests.
    """
    return any(
        isinstance(name, ast.Name)
        and name.id == "pytestmark"
        and isinstance(name.ctx, ast.Store)
        for statement in node.body
        if not isinstance(statement, (ast.FunctionDef, ast.ClassDef))
        for name in ast.walk(statement)
    )


def parity_model_type(call, parent):
    """Returns the `model_type` of a `run_reference_parity_test` call.

    Returns `None` unless the call has the shape `METHOD_SHAPE` asks for.
    """
    statement = parent[call]
    method = parent[statement]
    cls = parent.get(method)
    module = parent.get(cls)
    bases = [ast.unparse(b).split(".")[-1] for b in getattr(cls, "bases", [])]
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    if (
        isinstance(method, ast.FunctionDef)
        and method.body[-1] is statement
        and method.name.startswith("test")
        and not method.decorator_list
        and all(
            isinstance(other, (ast.Expr, ast.Assign)) and is_string(other.value)
            for other in method.body[:-1]
        )
        and isinstance(cls, ast.ClassDef)
        and isinstance(module, ast.Module)
        and "TestCase" in bases
        and not cls.decorator_list
        and not sets_pytestmark(cls)
        and not sets_pytestmark(module)
        and is_string(keywords.get("model_type"))
    ):
        return keywords["model_type"].value
    return None


def coverage_problems(directory, not_yet_covered):
    """Returns the problems of the reference parity tests in `directory`."""
    converters, problems = dispatch_keys(directory / "preset_loader.py")
    tests = set()  # `(model_type, file name)` of each well-formed test.
    for path in sorted(directory.glob("convert_*_test.py")):
        tree, parent = parse(path)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func).split(".")[-1]
            == "run_reference_parity_test"
        ]
        for call in sorted(calls, key=lambda node: node.lineno):
            model_type = parity_model_type(call, parent)
            if model_type is None:
                where = f"{path.name}:{call.lineno}"
                problems.append(METHOD_SHAPE.format(where))
            else:
                tests.add((model_type, path.name))
    tested, listed = {t for t, _ in tests}, set(not_yet_covered)
    problems += [
        f"`{t}` has no reference parity test. Add one to "
        f"{converters[t]}_test.py (see CONTRIBUTING_MODELS.md), or list it in "
        "NOT_YET_COVERED with the reason."
        for t in sorted(converters.keys() - tested - listed)
    ]
    problems += [
        f"`{t}` has a reference parity test; remove it from NOT_YET_COVERED."
        for t in sorted(listed & tested)
    ]
    problems += [
        f"NOT_YET_COVERED lists `{t}`, which preset_loader.py does not "
        "dispatch; remove it."
        for t in sorted(listed - converters.keys())
    ]
    for t, file in sorted(tests):
        if t not in converters:
            problems.append(
                f"{file} tests `{t}`, which preset_loader.py does not dispatch."
            )
        elif file != f"{converters[t]}_test.py":
            problems.append(
                f"{file} tests `{t}`; move the test to {converters[t]}_test.py."
            )
    return problems


class ReferenceParityCoverageTest(TestCase):
    def test_every_dispatch_key_has_a_reference_parity_test(self):
        problems = coverage_problems(TRANSFORMERS_DIR, NOT_YET_COVERED)
        if problems:
            self.fail("\n".join(problems))

    def test_coverage_problems(self):
        # Every line that ends with `# Bad.` gives one problem.
        files = {
            "preset_loader.py": """
            if model_type == "misplaced":
                self.converter = convert_b
            elif model_type in ("ok", "listed", "covered", "bad"):
                self.converter = convert_a
            elif model_type == "x" or model_type == "y":
                self.converter = convert_c  # Bad.
            elif model_type == "z":
                self.converter = converter = convert_c  # Bad.
            else:
                self.converter = convert_c  # Bad.""",
            "convert_a_test.py": """
            class TestA(TestCase):
                def test_ok(self):
                    "A docstring."
                    reason = "A string constant."
                    self.run_reference_parity_test(model_type="ok")
                def test_covered(self):
                    self.run_reference_parity_test(model_type="covered")
                def test_misplaced(self):
                    self.run_reference_parity_test(model_type="misplaced")
                def test_unknown(self):
                    self.run_reference_parity_test(model_type="unknown")
                def parity(self):
                    self.run_reference_parity_test(model_type="bad")  # Bad.
                @pytest.mark.large
                def test_decorated(self):
                    self.run_reference_parity_test(model_type="bad")  # Bad.
                def test_skip(self):
                    self.skipTest("Slow.")
                    self.run_reference_parity_test(model_type="bad")  # Bad.
                def test_in_if(self):
                    if keras.config.backend() == "torch":
                        self.run_reference_parity_test(model_type="bad")  # Bad.
                def test_after_return(self):
                    if keras.config.backend() != "torch":
                        return
                    self.run_reference_parity_test(model_type="bad")  # Bad.
                def test_two_calls(self):
                    self.run_reference_parity_test(model_type="bad")  # Bad.
                    self.run_reference_parity_test(model_type="bad")  # Bad.
                def test_not_literal(self):
                    self.run_reference_parity_test(model_type=NAME)  # Bad.
                class TestNested(TestCase):
                    def test_nested(self):
                        self.run_reference_parity_test(model_type="bad")  # Bad.
            @pytest.mark.large
            class TestDecorated(TestCase):
                def test_decorated(self):
                    self.run_reference_parity_test(model_type="bad")  # Bad.
            class TestMarked(TestCase):
                pytestmark = pytest.mark.large
                def test_marked(self):
                    self.run_reference_parity_test(model_type="bad")  # Bad.
            class Base(TestCase):
                pytestmark = pytest.mark.large
            class TestInherited(Base):
                def test_inherited(self):
                    self.run_reference_parity_test(model_type="bad")  # Bad.
            class Helpers:
                def test_helper(self):
                    self.run_reference_parity_test(model_type="bad")  # Bad.""",
            "convert_b_test.py": """
            pytestmark = pytest.mark.large
            class TestB(TestCase):
                def test_b(self):
                    self.run_reference_parity_test(model_type="bad")  # Bad.""",
        }
        directory = pathlib.Path(self.get_temp_dir())
        expected = []
        for name, source in files.items():
            (directory / name).write_text(textwrap.dedent(source))
            message = METHOD_SHAPE if "test" in name else UNREAD_BRANCH
            expected += [
                message.format(f"{name}:{number}")
                for number, line in enumerate(source.splitlines(), 1)
                if line.endswith("# Bad.")
            ]
        # A call that breaks the method-shape rule is not a test of `bad`.
        expected += [
            "`bad` has no reference parity test. Add one to convert_a_test.py "
            "(see CONTRIBUTING_MODELS.md), or list it in NOT_YET_COVERED with "
            "the reason.",
            "`covered` has a reference parity test; remove it from "
            "NOT_YET_COVERED.",
            "NOT_YET_COVERED lists `stale`, which preset_loader.py does not "
            "dispatch; remove it.",
            "convert_a_test.py tests `misplaced`; move the test to "
            "convert_b_test.py.",
            "convert_a_test.py tests `unknown`, which preset_loader.py does "
            "not dispatch.",
        ]
        not_yet_covered = dict.fromkeys(["listed", "covered", "stale"], "A.")
        self.assertEqual(
            coverage_problems(directory, not_yet_covered), expected
        )
