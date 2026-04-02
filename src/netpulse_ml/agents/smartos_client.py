"""SmartOS ubus WebSocket client for real device management.

Communicates with SmartOS routers via the JUCI WebSocket API.
Endpoint: ws://<router_ip>/websocket/

Authentication uses JUCI's challenge-response flow:
1. Send "challenge" with username -> get {token, salt}
2. Compute md5crypt(password, salt) -> password_hash
3. Generate random client_challenge
4. Compute md5(client_challenge + server_token + password_hash) -> response
5. Send "login" with [username, response, client_challenge] -> get {success: session_id}
"""

import asyncio
import hashlib
import json
import random
import string

import structlog
import websockets

from netpulse_ml.config import settings

log = structlog.get_logger()


def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _md5crypt(password: str, salt: str) -> str:
    """MD5-based password hash ($1$ format) compatible with JUCI's md5crypt.js."""
    from passlib.hash import md5_crypt
    return md5_crypt.using(salt=salt).hash(password)


def _random_string(length: int = 31) -> str:
    chars = string.ascii_letters + string.digits
    return "".join(random.choice(chars) for _ in range(length))


class SmartOSClient:
    """Async WebSocket client for SmartOS router JUCI API."""

    def __init__(
        self,
        host: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self._host = host or settings.smartos_host
        self._username = username or settings.smartos_username
        self._password = password or settings.smartos_password
        self._ws_url = f"ws://{self._host}/websocket/"
        self._ws = None
        self._call_id = 0
        self._sid = "00000000000000000000000000000000"
        self._pending: dict[int, asyncio.Future] = {}

    async def connect(self) -> None:
        """Open WebSocket connection."""
        if not self._host:
            raise RuntimeError("SmartOS host not configured (set SMARTOS_HOST)")
        self._ws = await websockets.connect(self._ws_url, ping_interval=30)
        # Start background listener
        asyncio.create_task(self._listener())
        log.info("SmartOS WebSocket connected", host=self._host)

    async def _listener(self) -> None:
        """Background task to receive WebSocket messages and resolve pending requests."""
        try:
            async for raw in self._ws:
                data = json.loads(raw)
                msg_id = data.get("id")
                if msg_id and msg_id in self._pending:
                    self._pending[msg_id].set_result(data)
        except websockets.exceptions.ConnectionClosed:
            log.warning("SmartOS WebSocket closed")

    async def _request(self, method: str, params: list) -> dict:
        """Send a JSON-RPC request and wait for the response."""
        if self._ws is None:
            await self.connect()

        self._call_id += 1
        msg_id = self._call_id
        msg = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}

        future = asyncio.get_running_loop().create_future()
        self._pending[msg_id] = future

        await self._ws.send(json.dumps(msg))
        response = await asyncio.wait_for(future, timeout=10)
        del self._pending[msg_id]

        if "error" in response:
            raise RuntimeError(f"RPC error: {response['error']}")

        return response.get("result", {})

    async def login(self) -> str:
        """Authenticate using JUCI challenge-response."""
        # Step 1: Get challenge
        challenge_resp = await self._request("challenge", [self._username])
        server_token = challenge_resp.get("token", "")
        salt = challenge_resp.get("salt", "")

        if not server_token:
            raise RuntimeError(f"Challenge failed: {challenge_resp}")

        # Step 2: Compute response
        password_hash = _md5crypt(self._password, salt)
        client_challenge = _random_string(31)
        response = _md5(client_challenge + server_token + password_hash)

        # Step 3: Login
        login_resp = await self._request("login", [self._username, response, client_challenge])
        session_id = login_resp.get("success", "")

        if not session_id:
            raise RuntimeError(f"Login failed: {login_resp}")

        self._sid = session_id
        log.info("SmartOS authenticated", host=self._host)
        return session_id

    async def call(self, obj: str, method: str, params: dict | None = None) -> dict:
        """Make an authenticated ubus call.

        Object names use JUCI's /-prefixed format: /system, /uci, /network.wireless, etc.
        If the caller passes a bare name (no /), we auto-prefix it.
        """
        if not obj.startswith("/"):
            obj = f"/{obj}"
        result = await self._request("call", [self._sid, obj, method, params or {}])
        # ubus returns [status_code, data]
        if isinstance(result, list):
            if result[0] != 0:
                raise RuntimeError(f"ubus {obj}.{method} returned status {result[0]}")
            return result[1] if len(result) > 1 else {}
        return result

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None

    # -----------------------------------------------------------------------
    # High-level methods (used by agent tools)
    # -----------------------------------------------------------------------

    async def get_system_info(self) -> dict:
        board = await self.call("system", "board")
        info = await self.call("system", "info")
        return {**board, **info}

    async def reboot(self) -> dict:
        log.warning("Rebooting SmartOS device", host=self._host)
        return await self.call("system", "reboot")

    async def get_sqm_config(self) -> dict:
        try:
            return await self.call("uci", "get", {"config": "sqm"})
        except RuntimeError:
            return {"error": "sqm config not found"}

    async def enable_sqm(self, interface: str = "wan") -> dict:
        log.info("Enabling SQM", host=self._host, interface=interface)
        await self.call("uci", "set", {"config": "sqm", "section": interface, "values": {"enabled": "1"}})
        await self.call("uci", "commit", {"config": "sqm"})
        return {"success": True, "action": "sqm_enabled", "interface": interface}

    async def get_usteer_info(self) -> dict:
        return await self.call("usteer", "local_info")

    async def get_usteer_clients(self) -> dict:
        return await self.call("usteer", "get_clients")

    async def steer_client(self, mac_address: str, reason: int = 1, ban_time: int = 30000) -> dict:
        log.info("Steering client", host=self._host, mac=mac_address)
        return await self.call("usteer", "del_client", {"address": mac_address, "reason": reason, "ban_time": ban_time})

    async def get_wireless_status(self) -> dict:
        return await self.call("network.wireless", "status")

    async def validate_firmware(self, path: str) -> dict:
        return await self.call("system", "validate_firmware_image", {"path": path})

    async def upgrade_firmware(self, path: str, keep_config: bool = True) -> dict:
        log.warning("Starting firmware upgrade", host=self._host, path=path)
        return await self.call("system", "sysupgrade", {"path": path, "force": False, "backup": "/tmp/backup.tar.gz" if keep_config else ""})

    async def get_rate_limits(self) -> dict:
        return await self.call("ratelimit", "dump")

    async def set_rate_limit(self, device: str, download: str = "", upload: str = "") -> dict:
        return await self.call("ratelimit", "device_set", {"device": device, "download": download, "upload": upload})
