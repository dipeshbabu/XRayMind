"""Regression tests for static API routes that sit near dynamic path params."""

from xraymind.api import app


def _route_paths() -> list[str]:
    return [getattr(route, "path", "") for route in app.routes]


def test_process_next_job_route_is_not_shadowed_by_job_id_route() -> None:
    """The static worker route must be registered before /jobs/{job_id}.

    FastAPI/Starlette route matching is order sensitive. If /jobs/{job_id}
    appears first, POST /jobs/process-next can be captured by the dynamic route
    and fail path validation instead of reaching the worker endpoint.
    """

    paths = _route_paths()
    assert "/jobs/process-next" in paths
    assert "/jobs/{job_id}" in paths or "/jobs/{job_id:int}" in paths

    static_index = paths.index("/jobs/process-next")
    dynamic_candidates = [
        idx
        for idx, path in enumerate(paths)
        if path in {"/jobs/{job_id}", "/jobs/{job_id:int}"}
    ]
    assert dynamic_candidates
    assert static_index < min(dynamic_candidates)
