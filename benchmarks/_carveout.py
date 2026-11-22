"""GB10 display-carveout weight backing for expert microbenchmarks (cuteafd #2546).

spark-attic's ``atticd`` lends the GB10 display carveout as 2 MiB slices: one
flat JSON request per AF_UNIX SOCK_SEQPACKET message, the lent slice's POSIX
fd beside the alloc reply (SCM_RIGHTS). CUDA imports each fd and maps it with
the ordinary VMM calls. :class:`CarveoutBacking` maps ``[carveout | cuMemCreate
tail]`` into one virtual range, so a weight image larger than the lendable
carveout fills it first and continues in ordinary device memory with no seam
visible to kernels.

Standard library and ctypes only (no build step, no torch at import), so any
bench can reuse it. Clients send no request the daemon does not document:
``info`` and ``alloc``; a lend never exceeds ``info.largest`` (atticd checks
``free`` but not ``largest`` before asking RM, and three RM failures reset its
client).
"""

from __future__ import annotations

import array
import ctypes
import json
import os
import socket
from dataclasses import dataclass, field

DEFAULT_SOCKET = "/run/spark-attic/attic.sock"
DEFAULT_KEY_FILE = "/etc/spark-attic/key"
SLICE = 2 << 20
KEY_MAX = 127
REPLY_MAX = 4096
MAX_SLICES = 64  # atticd's default per-client lend limit
# Released protocol range: v0.1.0..v0.1.7 share one wire format (status/info/alloc,
# 2 MiB slices, flat JSON, SCM_RIGHTS); info gained "pages" in 0.1.3.
SUPPORTED_MAJOR_MINOR = ((0, 1),)


class CarveoutError(RuntimeError):
    """atticd or the CUDA mapping refused; the caller falls back or fails."""


def key_ok(key: str) -> bool:
    return 0 < len(key) <= KEY_MAX and all(
        0x21 <= ord(c) <= 0x7E and c not in '"\\' for c in key)


def read_key(path: str = DEFAULT_KEY_FILE) -> str:
    try:
        with open(path, encoding="ascii") as handle:
            key = handle.readline().strip()
    except OSError as error:
        raise CarveoutError(f"attic key {path}: {error.strerror}") from error
    if not key_ok(key):
        raise CarveoutError("attic key is not 1..127 printable bytes without space, quote or backslash")
    return key


def encode_request(key: str, op: str, size: int | None = None) -> bytes:
    if not key_ok(key):
        raise CarveoutError("attic key is not sendable")
    request = {"key": key, "op": op}
    if size is not None:
        request["size"] = int(size)
    return json.dumps(request, separators=(",", ":")).encode("ascii")


def decode_reply(message: bytes, fds: list[int], flags: int, want_fd: bool) -> tuple[dict, int | None]:
    """Validate one reply; closes every received fd on refusal."""
    try:
        if not message:
            raise CarveoutError("atticd closed the connection")
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
            raise CarveoutError("atticd reply was truncated")
        try:
            reply = json.loads(message)
        except ValueError as error:
            raise CarveoutError("atticd reply is not JSON") from error
        if not isinstance(reply, dict):
            raise CarveoutError("atticd reply is not a JSON object")
        if reply.get("ok") is not True:
            raise CarveoutError(f"atticd refused: {reply.get('error', 'no reason given')}")
        if len(fds) != (1 if want_fd else 0):
            raise CarveoutError(f"atticd sent {len(fds)} descriptors with its reply")
    except CarveoutError:
        for fd in fds:
            os.close(fd)
        raise
    return reply, (fds[0] if want_fd else None)


def check_version(status: dict) -> str:
    if status.get("name") != "spark-attic":
        raise CarveoutError(f"daemon is not spark-attic: name {status.get('name')!r}")
    version = status.get("version")
    parts = str(version).split(".") if isinstance(version, str) else []
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:3]):
        raise CarveoutError(f"spark-attic version {version!r} is not x.y.z")
    if (int(parts[0]), int(parts[1])) not in SUPPORTED_MAJOR_MINOR:
        raise CarveoutError(f"spark-attic {version} is outside the supported protocol range 0.1.x")
    if status.get("state") != "ready":
        raise CarveoutError(f"spark-attic {version} is {status.get('state')!r}: {status.get('why', '')}")
    return version


class AtticClient:
    """One atticd connection. Lent slices stay lent until :meth:`close`."""

    def __init__(self, key: str, sock_path: str = DEFAULT_SOCKET, timeout: float = 5.0,
                 sock: socket.socket | None = None):
        self.key = key
        if sock is None:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET | socket.SOCK_CLOEXEC)
            sock.settimeout(timeout)
            try:
                sock.connect(sock_path)
            except OSError as error:
                sock.close()
                raise CarveoutError(f"no atticd at {sock_path}: {error.strerror or error}") from error
        self._sock = sock

    def _ask(self, payload: bytes, want_fd: bool) -> tuple[dict, int | None]:
        if self._sock is None:
            raise CarveoutError("attic connection is closed")
        try:
            self._sock.send(payload)
            fds = array.array("i")
            message, ancillary, flags, _ = self._sock.recvmsg(
                REPLY_MAX, socket.CMSG_SPACE(fds.itemsize))
        except OSError as error:
            raise CarveoutError(f"atticd: {error.strerror or error}") from error
        received: list[int] = []
        for level, kind, data in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                usable = len(data) - len(data) % fds.itemsize
                fds.frombytes(data[:usable])
        received.extend(fds)
        return decode_reply(message, received, flags, want_fd)

    def status(self) -> dict:
        reply, _ = self._ask(encode_request(self.key, "status"), False)
        return reply

    def check_version(self) -> str:
        """The daemon's version from status (info carries none); refuse unknown ranges."""
        return check_version(self.status())

    def info(self) -> dict:
        reply, _ = self._ask(encode_request(self.key, "info"), False)
        for name in ("size", "free", "largest"):
            if not isinstance(reply.get(name), int) or reply[name] < 0:
                raise CarveoutError(f"atticd info reply lacks {name}")
        return {name: reply[name] for name in ("size", "free", "largest")}

    def alloc(self, nbytes: int) -> int:
        if nbytes <= 0 or nbytes % SLICE:
            raise CarveoutError(f"{nbytes} bytes is not a whole number of 2 MiB slices")
        reply, fd = self._ask(encode_request(self.key, "alloc", nbytes), True)
        if reply.get("size") != nbytes:
            os.close(fd)
            raise CarveoutError(f"atticd lent {reply.get('size')!r} bytes, asked for {nbytes}")
        return fd

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None


# --------------------------------------------------------------------------- CUDA driver

class _Location(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _Prop(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("handle_types", ctypes.c_int), ("location", _Location),
                ("win32_metadata", ctypes.c_void_p), ("alloc_flags", ctypes.c_ubyte * 8)]


class _Access(ctypes.Structure):
    _fields_ = [("location", _Location), ("flags", ctypes.c_int)]


_U64 = ctypes.c_ulonglong
_SIGNATURES = {
    "cuGetErrorName": [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
    "cuCtxGetCurrent": [ctypes.POINTER(ctypes.c_void_p)],
    "cuCtxGetDevice": [ctypes.POINTER(ctypes.c_int)],
    "cuMemGetAllocationGranularity": [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(_Prop), ctypes.c_int],
    "cuMemImportFromShareableHandle": [ctypes.POINTER(_U64), ctypes.c_void_p, ctypes.c_int],
    "cuMemCreate": [ctypes.POINTER(_U64), ctypes.c_size_t, ctypes.POINTER(_Prop), _U64],
    "cuMemAddressReserve": [ctypes.POINTER(_U64), ctypes.c_size_t, ctypes.c_size_t, _U64, _U64],
    "cuMemMap": [_U64, ctypes.c_size_t, ctypes.c_size_t, _U64, _U64],
    "cuMemSetAccess": [_U64, ctypes.c_size_t, ctypes.POINTER(_Access), ctypes.c_size_t],
    "cuMemUnmap": [_U64, ctypes.c_size_t],
    "cuMemAddressFree": [_U64, ctypes.c_size_t],
    "cuMemRelease": [_U64],
}
_PINNED, _DEVICE, _POSIX_FD, _READWRITE = 1, 1, 1, 3


class Driver:
    """libcuda entry points; the caller has a current context (torch.cuda.set_device)."""

    def __init__(self, path: str = "libcuda.so.1"):
        lib = ctypes.CDLL(path)
        for name, args in _SIGNATURES.items():
            function = getattr(lib, name)
            function.argtypes = args
            function.restype = ctypes.c_int
        self.lib = lib

    def check(self, status: int, what: str) -> None:
        if status:
            name = ctypes.c_char_p()
            self.lib.cuGetErrorName(status, ctypes.byref(name))
            label = name.value.decode() if name.value else f"error {status}"
            raise CarveoutError(f"{what}: {label}")

    def device(self) -> int:
        context = ctypes.c_void_p()
        self.check(self.lib.cuCtxGetCurrent(ctypes.byref(context)), "cuCtxGetCurrent")
        if not context.value:
            raise CarveoutError("no current CUDA context: select the device first")
        ordinal = ctypes.c_int()
        self.check(self.lib.cuCtxGetDevice(ctypes.byref(ordinal)), "cuCtxGetDevice")
        return ordinal.value

    @staticmethod
    def prop(device: int, handle_types: int = 0) -> _Prop:
        prop = _Prop()
        prop.type, prop.handle_types = _PINNED, handle_types
        prop.location.type, prop.location.id = _DEVICE, device
        return prop

    def granularity(self, device: int) -> int:
        value = ctypes.c_size_t()
        prop = self.prop(device, _POSIX_FD)
        self.check(self.lib.cuMemGetAllocationGranularity(ctypes.byref(value), ctypes.byref(prop), 0),
                   "cuMemGetAllocationGranularity")
        return value.value


# --------------------------------------------------------------------------- backing

@dataclass
class CarveoutBacking:
    """One virtual range: carveout slices first, then an ordinary cuMemCreate tail.

    ``requested`` is the image size rounded up to 2 MiB. Close it (or use it as
    a context manager) only after every tensor over it is dead and the device
    has drained: unmapping memory a kernel still reads faults the context.
    """

    requested: int
    device: int
    ptr: int = 0
    carveout_bytes: int = 0
    tail_bytes: int = 0
    lends: list[int] = field(default_factory=list)
    _parts: list[tuple[int, int]] = field(default_factory=list)  # (handle, bytes) in map order
    _mapped: list[tuple[int, int]] = field(default_factory=list)  # (address, bytes)
    _driver: Driver | None = None
    _client: AtticClient | None = None
    note: str = ""
    version: str = ""

    def record(self) -> dict:
        return {"backing": "carveout", "requested_bytes": self.requested,
                "carveout_bytes": self.carveout_bytes, "tail_bytes": self.tail_bytes,
                "lends": list(self.lends), "note": self.note, "attic_version": self.version}

    def close(self) -> str | None:
        """Undo in reverse; returns the first error (never raises on cleanup)."""
        error = None
        lib = self._driver.lib if self._driver else None
        if lib is not None:
            for address, nbytes in reversed(self._mapped):
                status = lib.cuMemUnmap(address, nbytes)
                error = error or (status and f"cuMemUnmap: {status}")
            self._mapped.clear()
            if self.ptr:
                status = lib.cuMemAddressFree(self.ptr, self.requested)
                error = error or (status and f"cuMemAddressFree: {status}")
                self.ptr = 0
            for handle, _ in reversed(self._parts):
                status = lib.cuMemRelease(handle)
                error = error or (status and f"cuMemRelease: {status}")
            self._parts.clear()
        if self._client is not None:
            self._client.close()
            self._client = None
        return error or None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def plan_lends(requested: int, largest: int, max_slices: int = MAX_SLICES) -> list[int]:
    """Carveout lends for an image: at most ``largest`` (2 MiB rounded down), one lend."""
    if requested <= 0 or requested % SLICE:
        raise CarveoutError("requested bytes must be a positive multiple of 2 MiB")
    usable = min(requested, largest - largest % SLICE)
    return [usable] if usable > 0 and max_slices > 0 else []


def map_backing(requested: int, *, key: str, sock_path: str = DEFAULT_SOCKET,
                driver: Driver | None = None, client: AtticClient | None = None) -> CarveoutBacking:
    """Borrow up to ``requested`` bytes of carveout and map ``[carveout | tail]``."""
    requested = -(-int(requested) // SLICE) * SLICE
    driver = driver or Driver()
    device = driver.device()
    if SLICE % driver.granularity(device):
        raise CarveoutError("device granularity does not divide the 2 MiB attic slice")
    backing = CarveoutBacking(requested=requested, device=device, _driver=driver)
    try:
        backing._client = client or AtticClient(key, sock_path)
        backing.version = backing._client.check_version()
        info = backing._client.info()
        for size in plan_lends(requested, int(info["largest"])):
            fd = backing._client.alloc(size)
            try:
                handle = _U64()
                driver.check(driver.lib.cuMemImportFromShareableHandle(
                    ctypes.byref(handle), ctypes.c_void_p(fd), _POSIX_FD), "cuMemImportFromShareableHandle")
            finally:
                os.close(fd)
            backing._parts.append((handle.value, size))
            backing.lends.append(size)
            backing.carveout_bytes += size
        tail = requested - backing.carveout_bytes
        if tail:
            handle = _U64()
            prop = driver.prop(device)
            driver.check(driver.lib.cuMemCreate(ctypes.byref(handle), tail, ctypes.byref(prop), 0),
                         f"cuMemCreate tail {tail}")
            backing._parts.append((handle.value, tail))
            backing.tail_bytes = tail
        base = _U64()
        driver.check(driver.lib.cuMemAddressReserve(ctypes.byref(base), requested, SLICE, 0, 0),
                     "cuMemAddressReserve")
        backing.ptr = base.value
        at = base.value
        for handle, nbytes in backing._parts:
            driver.check(driver.lib.cuMemMap(at, nbytes, 0, handle, 0), "cuMemMap")
            backing._mapped.append((at, nbytes))
            at += nbytes
        access = _Access()
        access.location.type, access.location.id, access.flags = _DEVICE, device, _READWRITE
        driver.check(driver.lib.cuMemSetAccess(base.value, requested, ctypes.byref(access), 1),
                     "cuMemSetAccess")
    except BaseException:
        backing.close()
        raise
    if not backing.carveout_bytes:
        backing.note = f"atticd had nothing to lend (largest {info['largest']})"
    return backing


def tensor_view(torch, backing: CarveoutBacking, offset: int, nbytes: int):
    """A flat uint8 CUDA tensor over [offset, offset + nbytes) of the backing (zero copy)."""
    if offset < 0 or nbytes <= 0 or offset + nbytes > backing.requested or not backing.ptr:
        raise CarveoutError("tensor view outside the mapped backing")

    class _View:
        __cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1",
                                    "data": (backing.ptr + offset, False), "version": 3,
                                    "strides": None, "stream": None}

    return torch.as_tensor(_View(), device=torch.device("cuda", backing.device))
