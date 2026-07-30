"""DEPRECATED: Legacy Flask application entry point.

Please use the new FastAPI entry point instead:
    python main.py
    
This file is kept temporarily for backward compatibility and fallback purposes.
"""
import warnings

warnings.warn(
    "app.py is deprecated and will be removed in a future release. "
    "Please run 'python main.py' to start the FastAPI server.",
    DeprecationWarning,
    stacklevel=2,
)

from src import config, app

if __name__ == "__main__":
    app.run(host= config.HOST,
            port= config.PORT,
            debug= config.DEBUG)