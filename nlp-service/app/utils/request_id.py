"""Request ID context management and middleware for request tracing."""

import contextvars
from typing import Callable
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Context variable to hold the request ID for the current async task
request_id_ctx_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


def get_current_request_id() -> str:
    """Retrieve the current request ID from the context variable or return empty string."""
    return request_id_ctx_var.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that ensures every incoming HTTP request has a correlation request ID.
    If 'X-Request-ID' header is supplied, it is preserved; otherwise a new UUID4 is generated.
    The ID is placed into contextvars and injected into the response header.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        incoming_req_id = request.headers.get("X-Request-ID")
        req_id = incoming_req_id if incoming_req_id and incoming_req_id.strip() else str(uuid.uuid4())

        token = request_id_ctx_var.set(req_id)
        # Also store on request state for easy access in handlers
        request.state.request_id = req_id

        try:
            response: Response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            return response
        finally:
            request_id_ctx_var.reset(token)
