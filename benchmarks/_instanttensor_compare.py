"""Distributed InstantTensor transport for the checkpoint comparison corpus.

The installed native range interface receives original filenames and offsets.
No checkpoint files or installed package files are modified. Separate ranks
consume borrowed tensors immediately into the benchmark's CUDA IPC targets.
"""

from __future__ import annotations

import ctypes
from datetime import timedelta
import hashlib
import importlib.util
from importlib.metadata import version
import multiprocessing as mp
from pathlib import Path
import time
import traceback


def source_ranges(experts):
    tensors = sorted((t for e in experts for t in e["tensors"]), key=lambda t: (t.path, t.offset))
    groups = []
    for tensor in tensors:
        if groups and groups[-1][-1].path == tensor.path and groups[-1][-1].offset + groups[-1][-1].nbytes == tensor.offset:
            groups[-1].append(tensor)
        else:
            groups.append([tensor])
    paths, offsets = [], []
    for group in groups:
        index = len(paths)
        paths.append(group[0].path)
        offsets.extend((index, tensor.offset) for tensor in group)
        offsets.append((index, group[-1].offset + group[-1].nbytes))
    return paths, offsets, tensors


def _load_native(path):
    spec = importlib.util.spec_from_file_location("_gds_shard_reads", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _worker(connection, rank, devices, ipc, native_path, rendezvous, experts):
    try:
        import torch
        import torch.distributed as dist
        from b12x.loader._gds_kernels import compile_copies
        from b12x.loader._gds_native import load

        torch.set_num_threads(1)
        device = devices[rank]
        torch.cuda.set_device(device)
        gds = load()
        import instanttensor
        from instanttensor import Backend, safe_open

        class CufileStatus(ctypes.Structure):
            _fields_ = [("err", ctypes.c_int), ("cu_err", ctypes.c_int)]

        cufile_path = next(line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                           if "/libcufile.so" in line)
        cufile = ctypes.CDLL(cufile_path)
        cufile.cuFileDriverOpen.restype = CufileStatus
        status = cufile.cuFileDriverOpen()
        if status.err:
            raise RuntimeError(f"cuFileDriverOpen: {status.err}, CUDA {status.cu_err}")

        native = _load_native(native_path)
        base = native.import_ipc(device, ipc[0])
        destination = base + ipc[1]
        program = compile_copies(device)[0]
        dist.init_process_group("nccl", init_method="file://" + rendezvous,
                                rank=rank, world_size=len(devices), timeout=timedelta(seconds=300))
        warmup = torch.zeros(1, device=device)
        dist.all_reduce(warmup)
        torch.cuda.synchronize(device)
        group = dist.group.WORLD
        communicator = group._get_backend(torch.device("cuda", device))._comm_ptr()
        paths, offsets, tensors = source_ranges(experts)
        # Match safe_open's parameter selection using only the selected corpus.
        configs = {}
        for name, backend in (("instanttensor_default", None), ("instanttensor_cufile", Backend.CUFILE)):
            selector = object.__new__(safe_open)
            selector.filename = paths
            selector.device = torch.device("cuda", device)
            selector.world_size = len(devices)
            selector.process_group = group
            selector.tensor_sizes = [tensor.nbytes for tensor in tensors]
            selector.total_tensor_size = sum(selector.tensor_sizes)
            selector._determine_io_params(None, None, None, None, backend)
            selector._determine_buffer_size(None)
            configs[name] = dict(backend=selector.backend.name, backend_id=selector.backend.value,
                                 buffer_size=selector.buffer_size, chunk_size=selector.chunk_size,
                                 concurrency=selector.concurrency, io_depth=selector.io_depth)
        library_paths = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                                if any(key in line for key in ("libnccl.so", "libcufile.so", "libcudart.so"))})
        package = Path(instanttensor.__file__).parent
        identity = dict(version=version("instanttensor"), device=device, rank=rank, configs=configs,
                        logical_ranges=len(paths), tensors=len(tensors), source_bytes=sum(t.nbytes for t in tensors),
                        hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [package / "_impl.py", Path(instanttensor._C.__file__), *map(Path, library_paths)]})
        connection.send(("ready", identity))
        stream = torch.cuda.current_stream(device)
        while True:
            method = connection.recv()
            if method == "close":
                break
            config = configs[method]
            before_io = int(next(line.split()[1] for line in Path("/proc/self/io").read_text().splitlines()
                                 if line.startswith("read_bytes:")))
            if config["backend"] == "CUFILE":
                gds.start_stats(3)
                before_transport = gds.synchronous_transport_stats()
            else:
                before_transport = None
            dist.barrier()
            before_free = torch.cuda.mem_get_info(device)[0]
            started = time.perf_counter()
            handle = instanttensor._C.open(paths, device, communicator, config["buffer_size"],
                                          config["chunk_size"], config["concurrency"], config["io_depth"],
                                          config["backend_id"], offsets)
            opened = time.perf_counter()
            ring_device_bytes = before_free - torch.cuda.mem_get_info(device)[0]
            try:
                for index, tensor in enumerate(tensors):
                    stream.synchronize()
                    value = torch.from_dlpack(instanttensor._C.get_dl_tensor(handle, index, tensor.nbytes))
                    if value.device.index != device:
                        raise RuntimeError(f"InstantTensor returned device {value.device}, expected {device}")
                    rows, width, offset = tensor.shard(rank, len(devices))
                    native.peer_gather(device, program.function, value.data_ptr() + offset,
                                       destination + tensor.destination_offset, width, rows,
                                       tensor.width, width, stream.cuda_stream)
                stream.synchronize()
            finally:
                stream.synchronize()
                destinations_ready = time.perf_counter()
                instanttensor._C.close(handle)
            finished = time.perf_counter()
            after_io = int(next(line.split()[1] for line in Path("/proc/self/io").read_text().splitlines()
                                if line.startswith("read_bytes:")))
            if before_transport is not None:
                transport = {key: value - before_transport[key]
                             for key, value in gds.synchronous_transport_stats().items()}
                if transport["posix_reads"] or transport["read_errors"] or not transport["nvfs_reads"] + transport["p2p_reads"]:
                    raise RuntimeError(f"InstantTensor CUFILE transport qualification failed: {transport}")
            else:
                transport = None
            connection.send(("result", dict(started=started, finished=finished, seconds=finished - started,
                                            open_seconds=opened - started, ring_device_bytes=ring_device_bytes,
                                            until_final_destination_seconds=destinations_ready - started,
                                            close_seconds=finished - destinations_ready,
                                            process_read_bytes=after_io - before_io, transport=transport,
                                            config=config)))
        stream.synchronize()
        native.close_ipc(device, base)
        dist.destroy_process_group()
        connection.send(("closed", None))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
        raise


class InstantTensorRanks:
    def __init__(self, devices, targets, experts, native, output):
        context = mp.get_context("spawn")
        self.connections, self.processes = [], []
        try:
            for rank, device in enumerate(devices):
                parent, child = context.Pipe()
                ipc = native.export_ipc(device, targets[rank].data_ptr())
                process = context.Process(target=_worker, args=(child, rank, devices, ipc, native.__file__,
                                                                str(output / "instanttensor-nccl"), experts))
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            self.identity = self._receive("ready")
        except BaseException:
            self.abort()
            raise

    def _receive(self, expected):
        deadline = time.monotonic() + 300
        pending = set(range(len(self.connections)))
        results = [None] * len(pending)
        while pending:
            for index in tuple(pending):
                connection = self.connections[index]
                if connection.poll(0.01):
                    kind, value = connection.recv()
                    if kind != expected:
                        raise RuntimeError(f"InstantTensor rank {index}: expected {expected}, received {kind}: {value}")
                    results[index] = value
                    pending.remove(index)
                elif not self.processes[index].is_alive():
                    raise RuntimeError(f"InstantTensor rank {index} exited with {self.processes[index].exitcode}")
            if time.monotonic() > deadline:
                raise RuntimeError(f"InstantTensor ranks timed out waiting for {expected}: {sorted(pending)}")
        return results

    def run(self, method):
        try:
            for connection in self.connections:
                connection.send(method)
            results = self._receive("result")
            return dict(seconds=max(r["finished"] for r in results) - min(r["started"] for r in results), ranks=results)
        except BaseException:
            self.abort()
            raise

    def abort(self):
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(10)
            if process.is_alive():
                process.kill()
                process.join()

    def close(self):
        if not all(p.is_alive() for p in self.processes):
            self.abort()
            return
        try:
            for connection in self.connections:
                connection.send("close")
            self._receive("closed")
            for process in self.processes:
                process.join(10)
        finally:
            self.abort()
