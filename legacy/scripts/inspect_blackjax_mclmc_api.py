#!/usr/bin/env python
"""Inspect the installed BlackJAX MCLMC API."""

from __future__ import annotations

import inspect


def main() -> int:
    try:
        import blackjax
    except ImportError as exc:
        raise SystemExit(
            "BlackJAX is not installed. Install with `python -m pip install blackjax` "
            "or the project `samplers` extra."
        ) from exc

    print(f"blackjax_version: {getattr(blackjax, '__version__', 'unknown')}")
    for name in ("mclmc", "adjusted_mclmc", "adjusted_mclmc_dynamic"):
        api = getattr(blackjax, name, None)
        print(f"{name}: {api is not None}")
        if api is None:
            continue
        for attr in ("init", "build_kernel"):
            fn = getattr(api, attr, None)
            print(f"{name}.{attr}: {fn is not None}")
            if fn is not None:
                try:
                    print(f"{name}.{attr}.signature: {inspect.signature(fn)}")
                except (TypeError, ValueError) as exc:
                    print(f"{name}.{attr}.signature_error: {exc}")
    for name in ("mclmc_find_L_and_step_size", "adjusted_mclmc_find_L_and_step_size"):
        fn = getattr(blackjax, name, None)
        print(f"{name}: {fn is not None}")
        if fn is not None:
            try:
                print(f"{name}.signature: {inspect.signature(fn)}")
            except (TypeError, ValueError) as exc:
                print(f"{name}.signature_error: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
