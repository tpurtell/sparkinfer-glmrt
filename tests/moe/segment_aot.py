"""Exercise the same generated header and object ABI that the native engine uses."""
from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import torch

from b12x.integration.cuteafd import validate_exported_header


class ExportedProgram:
    def __init__(self, program, directory: Path, stem: str):
        directory.mkdir(parents=True, exist_ok=True)
        symbol = "cuteafd_" + stem
        program.export_to_c(str(directory), stem, symbol)
        validate_exported_header(program, directory / f"{stem}.h", symbol)
        scalar_types = {"int32": "int32_t", "int64": "int64_t", "float32": "float"}
        declarations = [f"void *{operand.name}" for operand in program.operands]
        declarations += [f"{scalar_types[scalar.ctype]} {scalar.name}" for scalar in program.scalars]
        declarations += ["cudaStream_t stream"]
        arguments = [operand.name for operand in program.operands] + [scalar.name for scalar in program.scalars] + ["stream"]
        # Include and invoke the validated generated wrapper, rather than calling
        # the JIT object or rebuilding a parallel ABI in Python.
        source = directory / f"{stem}_bridge.cu"
        source.write_text(f'''#include "{stem}.h"
static {symbol}_Kernel_Module_t module;
extern "C" void load() {{ {symbol}_Kernel_Module_Load(&module); }}
extern "C" int launch({", ".join(declarations)}) {{
    return cute_dsl_{symbol}_wrapper(&module, {", ".join(arguments)});
}}
''')
        library = directory / f"{stem}.so"
        subprocess.run(["nvcc", "--shared", "-Xcompiler", "-fPIC", str(source),
                        str(directory / f"{stem}.o"), "-lcuda", "-o", str(library)],
                       check=True, capture_output=True, text=True, timeout=120)
        self.library = ctypes.CDLL(str(library))
        self.library.load.argtypes = []
        self.library.load.restype = None
        self.library.load()
        self.operands = program.operands
        self.scalars = program.scalars
        ctypes_types = {"int32": ctypes.c_int32, "int64": ctypes.c_int64, "float32": ctypes.c_float}
        self.library.launch.argtypes = ([ctypes.c_void_p] * len(self.operands)
            + [ctypes_types[scalar.ctype] for scalar in self.scalars] + [ctypes.c_void_p])
        self.library.launch.restype = ctypes.c_int

    def launch(self, *tensors, scalars, stream=None):
        assert len(tensors) == len(self.operands) and len(scalars) == len(self.scalars)
        address = torch.cuda.current_stream().cuda_stream if stream is None else int(stream)
        status = self.library.launch(*(tensor.data_ptr() for tensor in tensors), *scalars, address)
        assert status == 0, f"exported segment launch failed with CUDA status {status}"
