"""CPU tests for the spark-attic client framing used by --weight-backing carveout."""

from __future__ import annotations

import array
import json
import os
import socket
import threading

import pytest

from benchmarks import _carveout as carveout

KEY = "attic_0123456789abcdef01234567"


def _serve(server, replies):
    """Answer each request with the next (body, send_fd) and record what arrived."""
    seen = []
    for body, send_fd in replies:
        request = server.recv(1024)
        seen.append(json.loads(request))
        if send_fd:
            read_end, write_end = os.pipe()
            os.close(write_end)
            server.sendmsg([json.dumps(body).encode()],
                           [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [read_end]))])
            os.close(read_end)
        else:
            server.send(json.dumps(body).encode())
    return seen


def _client(replies):
    client_sock, server = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    result = {}
    thread = threading.Thread(target=lambda: result.setdefault("seen", _serve(server, replies)))
    thread.start()
    return carveout.AtticClient(KEY, sock=client_sock), thread, result, server


STATUS_016 = {"ok": True, "name": "spark-attic", "version": "0.1.6", "state": "ready", "why": "",
              "size": 2143289344, "free": 2143289344, "largest": 2143289344, "lent": 0,
              "draining": 0, "clients": 1, "regions": []}
INFO_016 = {"ok": True, "size": 2143289344, "free": 2143289344, "largest": 2143289344,
            "pages": "2MB pages"}


def test_requests_are_flat_json_and_alloc_returns_one_descriptor():
    client, thread, result, server = _client(
        [(STATUS_016, False), (INFO_016, False), ({"ok": True, "size": 1 << 30}, True)])
    try:
        assert client.check_version() == "0.1.6"
        assert client.info() == {"size": 2143289344, "free": 2143289344, "largest": 2143289344}
        fd = client.alloc(1 << 30)
        assert fd >= 0
        os.close(fd)
    finally:
        thread.join(5)
        client.close()
        server.close()
    assert result["seen"] == [{"key": KEY, "op": "status"}, {"key": KEY, "op": "info"},
                              {"key": KEY, "op": "alloc", "size": 1 << 30}]


def test_info_before_pages_field_is_accepted():
    old = {key: value for key, value in INFO_016.items() if key != "pages"}  # v0.1.0..0.1.2
    client, thread, _, server = _client([(old, False)])
    try:
        assert client.info()["largest"] == old["largest"]
    finally:
        thread.join(5)
        client.close()
        server.close()


@pytest.mark.parametrize("version,state,ok", [
    ("0.1.0", "ready", True), ("0.1.7", "ready", True), ("0.1.6", "refusing", False),
    ("0.2.0", "ready", False), ("1.0.0", "ready", False), ("dev", "ready", False)])
def test_version_range_and_state(version, state, ok):
    status = {**STATUS_016, "version": version, "state": state}
    if ok:
        assert carveout.check_version(status) == version
    else:
        with pytest.raises(carveout.CarveoutError):
            carveout.check_version(status)
    with pytest.raises(carveout.CarveoutError):
        carveout.check_version({**status, "name": "other-lender"})


def test_refusal_closes_any_descriptor_and_raises():
    client, thread, _, server = _client([({"ok": False, "error": "only 0 MiB free"}, True)])
    before = set(os.listdir("/proc/self/fd"))
    try:
        with pytest.raises(carveout.CarveoutError, match="only 0 MiB free"):
            client.alloc(2 << 20)
    finally:
        thread.join(5)
        client.close()
        server.close()
    assert set(os.listdir("/proc/self/fd")) <= before


def test_alloc_without_descriptor_and_wrong_size_are_refused():
    client, thread, _, server = _client([({"ok": True, "size": 2 << 20}, False),
                                         ({"ok": True, "size": 4 << 20}, True)])
    try:
        with pytest.raises(carveout.CarveoutError, match="0 descriptors"):
            client.alloc(2 << 20)
        with pytest.raises(carveout.CarveoutError, match="lent 4194304"):
            client.alloc(2 << 20)
    finally:
        thread.join(5)
        client.close()
        server.close()


def test_closed_connection_and_bad_requests():
    client_sock, server = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    server.close()
    client = carveout.AtticClient(KEY, sock=client_sock)
    with pytest.raises(carveout.CarveoutError):
        client.info()
    client.close()
    with pytest.raises(carveout.CarveoutError):
        carveout.encode_request('bad"key', "info")
    with pytest.raises(carveout.CarveoutError):
        carveout.AtticClient(KEY, sock=socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)[0]).alloc(3)
    with pytest.raises(carveout.CarveoutError, match="no atticd"):
        carveout.AtticClient(KEY, sock_path="/nonexistent/attic.sock")


def test_lend_plan_never_exceeds_largest_and_stays_slice_aligned():
    assert carveout.plan_lends(4 << 30, 2143289344) == [2143289344]
    assert carveout.plan_lends(8 << 20, 2143289344) == [8 << 20]
    assert carveout.plan_lends(8 << 20, (3 << 20) + 5) == [2 << 20]
    assert carveout.plan_lends(8 << 20, 0) == []
    with pytest.raises(carveout.CarveoutError):
        carveout.plan_lends(3, 1 << 30)


def test_key_file_rules(tmp_path):
    path = tmp_path / "key"
    path.write_text(KEY + "\n")
    assert carveout.read_key(str(path)) == KEY
    path.write_text("has space\n")
    with pytest.raises(carveout.CarveoutError):
        carveout.read_key(str(path))
    with pytest.raises(carveout.CarveoutError):
        carveout.read_key(str(tmp_path / "absent"))
