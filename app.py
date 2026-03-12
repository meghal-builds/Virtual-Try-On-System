"""Backward-compatible entrypoint for Streamlit frontend."""

import runpy

if __name__ == "__main__":
    runpy.run_module("frontend.app", run_name="__main__")
