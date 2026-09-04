"""
Legacy compatibility shim.

The old entry-point was `uvicorn main:app`.
This shim re-exports the new application so existing scripts / deployment
configs continue to work without modification.

Preferred entry-point going forward:
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""
from app.main import app  # noqa: F401 — re-export for backwards compatibility