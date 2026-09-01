"""Local runtime launcher.

The standard way every member runs the app::

    python -m launcher

It reads the active runtime-config revision, starts the FastAPI backend (and,
optionally, the local frontend dev server), and watches
``<app-data>/control/restart.request``. On a restart request it applies the new
revision, polls ``/health`` and -- if the app does not come up healthy within
``LAUNCHER_HEALTH_TIMEOUT_SECONDS`` -- restores the previous revision and
starts again. Progress is reported through ``restart.status`` for the
Settings UI.
"""

from launcher.core import Launcher, RuntimeSpec, main  # noqa: F401
