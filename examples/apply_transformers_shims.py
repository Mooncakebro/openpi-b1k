"""Apply the openpi transformers compatibility shims to the ACTIVE venv.

The fork's ``transformers_replace/`` bundle mixes 4.49-style modeling files
with the 5.5.4 pin. This script patches the installed ``transformers`` so the
JAX->PyTorch converter and serving both work. **Idempotent** — safe to re-run
after every ``uv sync`` / venv rebuild.

Usage (openpi venv activated):

    python examples/apply_transformers_shims.py

Shims:
  1. ``LossKwargs`` in ``transformers/utils/__init__.py`` — dropped from 5.x
     utils; the replace files still import it. Must be ``typing.TypedDict``
     (same metaclass as 5.5.4's ``FlashAttentionKwargs``).
  2. ``HybridCache`` placeholder in ``cache_utils.py`` — rearchitected away in
     5.5.4; the replace paligemma imports it only for isinstance dispatch that
     5.5.4 can never trigger.
  3. ``GemmaConfig.use_bidirectional_attention = None`` — 5.5.4's
     ``PaliGemmaConfig.__post_init__`` reads it off the replaced 4.49-style
     config, which never defined it.
  4. RoPE: the replaced 4.49 modeling files select ``rope_type == "default"``
     (plain rotary), which 5.5.4's ``ROPE_INIT_FUNCTIONS`` no longer registers,
     and its linear function requires a ``factor`` that plain-RoPE configs
     never set. Alias ``"default"`` to the linear function and default
     ``factor`` to 1.0 (identity scaling — numerically the same as plain RoPE).
"""

from __future__ import annotations

import site
import sys
from pathlib import Path

MARKER = "# ---- openpi bridge shim"


def append_once(path: Path, text: str) -> None:
    current = path.read_text()
    if MARKER in current:
        print(f"skip (already patched): {path.name}")
        return
    path.open("a").write("\n" + text + "\n")
    print(f"patched: {path.name}")


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    current = path.read_text()
    if old not in current:
        print(f"skip (already patched): {label}")
        return
    count = current.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: pattern found {count} times in {path}")
    path.write_text(current.replace(old, new))
    print(f"patched: {label}")


def main() -> int:
    sp = Path(site.getsitepackages()[0])
    tf = sp / "transformers"

    # 1. LossKwargs (typing.TypedDict — NOT typing_extensions: metaclass clash)
    append_once(
        tf / "utils" / "__init__.py",
        MARKER
        + "\nfrom typing import Optional, TypedDict\n"
        + "class LossKwargs(TypedDict, total=False):\n"
        + '    """Keyword arguments to be passed to the loss function."""\n'
        + "    num_items_in_batch: Optional[int]\n",
    )

    # 2. HybridCache placeholder (dead isinstance dispatch in replace paligemma)
    append_once(
        tf / "cache_utils.py",
        MARKER
        + "\nclass HybridCache(Cache):\n"
        + '    """Compatibility placeholder for the openpi transformers_replace."""\n'
        + "    pass\n",
    )

    # 3. use_bidirectional_attention class default on the replaced GemmaConfig
    cfg_path = tf / "models" / "gemma" / "configuration_gemma.py"
    if MARKER not in cfg_path.read_text():
        cfg_path.open("a").write(
            "\n"
            + MARKER
            + "\n# 5.5.4's PaliGemmaConfig.__post_init__ reads this attr off the\n"
            + "# replaced 4.49-style GemmaConfig; None lets it apply its default.\n"
            + "GemmaConfig.use_bidirectional_attention = None\n"
        )
        print("patched: configuration_gemma.py")
    else:
        print("skip (already patched): configuration_gemma.py")

    # 4a. factor default (plain-RoPE configs never set it). Target the unique
    # 3-line block inside _compute_linear_scaling_rope_parameters; the same
    # .get line exists in the dynamic/yarn functions, so anchor on the block.
    rope = tf / "modeling_rope_utils.py"
    replace_once(
        rope,
        """    config.standardize_rope_params()
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters
    factor = rope_parameters_dict["factor"]""",
        """    config.standardize_rope_params()
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters
    factor = rope_parameters_dict.get("factor", 1.0)  # openpi shim: plain rotary""",
        "rope factor default",
    )
    # 4b. register "default" (4.49 plain rotary) as linear with factor 1.0
    rope_text = rope.read_text()
    if '"default": _compute_linear_scaling_rope_parameters' not in rope_text:
        old = '    "proportional": _compute_proportional_rope_parameters,\n}'
        if old not in rope_text:
            raise RuntimeError("ROPE_INIT_FUNCTIONS closing pattern not found")
        rope.write_text(
            rope_text.replace(
                old,
                '    "proportional": _compute_proportional_rope_parameters,\n'
                '    "default": _compute_linear_scaling_rope_parameters,  # openpi shim: plain rotary\n'
                "}",
            )
        )
        print("patched: ROPE_INIT_FUNCTIONS default alias")
    else:
        print("skip (already patched): ROPE_INIT_FUNCTIONS default alias")

    # Verify
    from transformers.utils import LossKwargs  # noqa: F401
    from transformers.cache_utils import Cache, HybridCache  # noqa: F401
    from transformers import GemmaConfig
    from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding

    emb = GemmaRotaryEmbedding(config=GemmaConfig())
    print(f"ALL SHIMS OK — inv_freq shape {tuple(emb.inv_freq.shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
