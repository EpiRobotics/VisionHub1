"""TCP JSONL server for VisionHub per-project inference.

Each project gets its own TCP server on a dedicated port.
Protocol: JSON Lines (one JSON object per line, terminated by \\n).

Supported commands:
- INFER: Run inference on an image
- PING: Health check
- STATUS: Get project status
- SET_ACTIVE_MODEL: Switch model version
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from app.result_schema import (
    ErrorCode,
    InferResult,
    TcpResponse,
    make_error_result,
    make_tcp_error,
)

logger = logging.getLogger(__name__)


class TcpProjectServer:
    """TCP JSONL server for a single project."""

    def __init__(
        self,
        project_id: str,
        host: str,
        port: int,
        on_infer: Any,  # async callable(project_id, job_id, image_path, options) -> InferResult
        on_status: Any,  # async callable(project_id) -> dict
        on_set_model: Any,  # async callable(project_id, version) -> bool
        log_buffer: Any = None,  # ProjectLogBuffer instance
    ):
        self.project_id = project_id
        self.host = host
        self.port = port
        self._on_infer = on_infer
        self._on_status = on_status
        self._on_set_model = on_set_model
        self._log_buffer = log_buffer
        self._server: asyncio.AbstractServer | None = None
        self._running = False

    def _log(self, level: str, msg: str) -> None:
        """Write to both Python logger and the project log buffer."""
        if level == "INFO":
            logger.info("[%s] %s", self.project_id, msg)
        elif level == "ERROR":
            logger.error("[%s] %s", self.project_id, msg)
        else:
            logger.debug("[%s] %s", self.project_id, msg)
        if self._log_buffer is not None:
            self._log_buffer.append(level, "TCP", msg)

    async def start(self) -> None:
        """Start the TCP server."""
        self._server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port,
        )
        self._running = True
        logger.info(
            "TCP server started for project '%s' on %s:%d",
            self.project_id,
            self.host,
            self.port,
        )

    async def stop(self) -> None:
        """Stop the TCP server."""
        self._running = False
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        logger.info("TCP server stopped for project '%s'", self.project_id)

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single client connection (supports multiple requests)."""
        peer = writer.get_extra_info("peername")
        self._log("INFO", f"Client connected: {peer}")

        try:
            while self._running:
                try:
                    line = await asyncio.wait_for(reader.readline(), timeout=300)
                except asyncio.TimeoutError:
                    break

                if not line:
                    break  # Client disconnected

                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue

                self._log("DEBUG", f"RECV: {line_str[:500]}")
                response = await self._process_request(line_str)
                self._log("DEBUG", f"SEND: {response[:500]}")
                response_bytes = (response + "\n").encode("utf-8")
                writer.write(response_bytes)
                await writer.drain()
        except ConnectionResetError:
            self._log("INFO", f"Client disconnected: {peer}")
        except Exception as exc:
            self._log("ERROR", f"Error handling client {peer}: {exc}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _process_request(self, line: str) -> str:
        """Parse a JSON line request and dispatch to handler."""
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            self._log("ERROR", f"Invalid JSON: {e}")
            resp = make_tcp_error("UNKNOWN", ErrorCode.INVALID_CMD, f"Invalid JSON: {e}")
            return resp.model_dump_json()

        cmd = request.get("cmd", "").upper()
        self._log("INFO", f"CMD={cmd} job_id={request.get('job_id', '-')}")

        if cmd == "PING":
            return self._handle_ping()
        elif cmd == "INFER":
            return await self._handle_infer(request)
        elif cmd == "STATUS":
            return await self._handle_status()
        elif cmd == "SET_ACTIVE_MODEL":
            return await self._handle_set_model(request)
        else:
            resp = make_tcp_error(cmd, ErrorCode.INVALID_CMD, f"Unknown command: {cmd}")
            return resp.model_dump_json()

    def _handle_ping(self) -> str:
        """Handle PING command."""
        resp = TcpResponse(
            ok=True,
            cmd="PING",
            data={
                "project_id": self.project_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )
        return resp.model_dump_json()

    async def _handle_infer(self, request: dict[str, Any]) -> str:
        """Handle INFER command."""
        job_id = request.get("job_id", "")
        image_path = request.get("image_path", "")
        options = request.get("options", {})

        if not job_id:
            resp = make_tcp_error("INFER", ErrorCode.INVALID_CMD, "Missing job_id")
            return resp.model_dump_json()
        if not image_path:
            resp = make_tcp_error("INFER", ErrorCode.INVALID_CMD, "Missing image_path")
            return resp.model_dump_json()

        try:
            result: InferResult = await self._on_infer(
                self.project_id, job_id, image_path, options
            )
            self._log("INFO", f"INFER result: pred={result.pred} score={result.score:.4f} job={job_id}")
            return result.to_json_line()
        except Exception as e:
            self._log("ERROR", f"INFER failed: {e}")
            result = make_error_result(
                job_id=job_id,
                project_id=self.project_id,
                code=ErrorCode.INFER_FAILED,
                message=str(e),
            )
            return result.to_json_line()

    async def _handle_status(self) -> str:
        """Handle STATUS command."""
        try:
            status_data = await self._on_status(self.project_id)
            resp = TcpResponse(ok=True, cmd="STATUS", data=status_data)
            return resp.model_dump_json()
        except Exception as e:
            resp = make_tcp_error("STATUS", ErrorCode.INTERNAL_ERROR, str(e))
            return resp.model_dump_json()

    async def _handle_set_model(self, request: dict[str, Any]) -> str:
        """Handle SET_ACTIVE_MODEL command."""
        version = request.get("version", "")
        if not version:
            resp = make_tcp_error("SET_ACTIVE_MODEL", ErrorCode.INVALID_CMD, "Missing version")
            return resp.model_dump_json()

        try:
            success = await self._on_set_model(self.project_id, version)
            if success:
                resp = TcpResponse(
                    ok=True,
                    cmd="SET_ACTIVE_MODEL",
                    data={"version": version, "message": "Model switched successfully"},
                )
            else:
                resp = make_tcp_error(
                    "SET_ACTIVE_MODEL",
                    ErrorCode.MODEL_NOT_LOADED,
                    f"Failed to switch to version: {version}",
                )
            return resp.model_dump_json()
        except Exception as e:
            resp = make_tcp_error("SET_ACTIVE_MODEL", ErrorCode.INTERNAL_ERROR, str(e))
            return resp.model_dump_json()


class TcpServerManager:
    """Manages TCP servers for all projects."""

    def __init__(self) -> None:
        self._servers: dict[str, TcpProjectServer] = {}

    async def start_server(
        self,
        project_id: str,
        host: str,
        port: int,
        on_infer: Any,
        on_status: Any,
        on_set_model: Any,
        log_buffer: Any = None,
    ) -> TcpProjectServer:
        """Start a TCP server for a project."""
        if project_id in self._servers:
            await self.stop_server(project_id)

        server = TcpProjectServer(
            project_id=project_id,
            host=host,
            port=port,
            on_infer=on_infer,
            on_status=on_status,
            on_set_model=on_set_model,
            log_buffer=log_buffer,
        )
        await server.start()
        self._servers[project_id] = server
        return server

    async def stop_server(self, project_id: str) -> None:
        """Stop a project's TCP server."""
        server = self._servers.pop(project_id, None)
        if server is not None:
            await server.stop()

    async def stop_all(self) -> None:
        """Stop all TCP servers."""
        for pid in list(self._servers.keys()):
            await self.stop_server(pid)

    def get_server(self, project_id: str) -> TcpProjectServer | None:
        return self._servers.get(project_id)

    def list_servers(self) -> dict[str, dict[str, Any]]:
        """Return info about all running TCP servers."""
        result: dict[str, dict[str, Any]] = {}
        for pid, server in self._servers.items():
            result[pid] = {
                "host": server.host,
                "port": server.port,
                "running": server._running,
            }
        return result
