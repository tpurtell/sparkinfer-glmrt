#define _GNU_SOURCE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufile.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* Research-only batched reads; no checkpoint loader dispatch changes. */
#define CAPSULE "b12x.benchmark.gds_shard_reads"
typedef struct { int fd; int64_t size; CUfileHandle_t handle; } input_t;
typedef struct {
    int device;
    unsigned depth, files;
    CUfileBatchHandle_t batch;
    CUfileIOParams_t *params;
    CUfileIOEvents_t *events;
    unsigned char *completed;
    input_t *inputs;
    PyObject *input_owner;
    int files_frozen;
    void *registered;
    uint64_t registered_bytes;
} reader_t;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void destroy(PyObject *capsule) {
    reader_t *r = PyCapsule_GetPointer(capsule, CAPSULE);
    if (!r) return;
    cudaSetDevice(r->device);
    if (r->batch) cuFileBatchIODestroy(r->batch);
    if (r->registered) cuFileBufDeregister(r->registered);
    if (!r->input_owner) {
        for (unsigned i = 0; i < r->files; i++) {
            cuFileHandleDeregister(r->inputs[i].handle);
            close(r->inputs[i].fd);
        }
        free(r->inputs);
    }
    Py_XDECREF(r->input_owner);
    free(r->params); free(r->events); free(r->completed); free(r);
}

static PyObject *create(PyObject *self, PyObject *args) {
    (void)self;
    int device;
    unsigned depth;
    if (!PyArg_ParseTuple(args, "iI", &device, &depth)) return NULL;
    if (!depth || depth > 128) return PyErr_Format(PyExc_ValueError, "batch size must be 1..128");
    reader_t *r = calloc(1, sizeof(*r));
    if (!r) return PyErr_NoMemory();
    r->device = device; r->depth = depth;
    PyObject *capsule = PyCapsule_New(r, CAPSULE, destroy);
    if (!capsule) { free(r); return NULL; }
    r->params = calloc(depth, sizeof(*r->params));
    r->events = calloc(depth, sizeof(*r->events));
    r->completed = calloc(depth, 1);
    if (!r->params || !r->events || !r->completed) { Py_DECREF(capsule); return PyErr_NoMemory(); }
    cudaError_t gpu = cudaSetDevice(device);
    if (gpu != cudaSuccess) {
        Py_DECREF(capsule);
        return PyErr_Format(PyExc_RuntimeError, "cudaSetDevice: %s", cudaGetErrorString(gpu));
    }
    CUfileError_t status = cuFileBatchIOSetUp(&r->batch, depth);
    if (status.err != CU_FILE_SUCCESS) {
        Py_DECREF(capsule);
        return PyErr_Format(PyExc_RuntimeError, "cuFileBatchIOSetUp: %d CUDA %d", status.err, status.cu_err);
    }
    return capsule;
}

static PyObject *add_file(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule;
    const char *path;
    if (!PyArg_ParseTuple(args, "Os", &capsule, &path)) return NULL;
    reader_t *r = PyCapsule_GetPointer(capsule, CAPSULE);
    if (!r) return NULL;
    if (r->files_frozen) return PyErr_Format(PyExc_RuntimeError, "file inventory is frozen");
    int fd = open(path, O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd < 0) return PyErr_SetFromErrnoWithFilename(PyExc_OSError, path);
    struct stat s;
    if (fstat(fd, &s)) { close(fd); return PyErr_SetFromErrno(PyExc_OSError); }
    input_t *inputs = realloc(r->inputs, (r->files + 1) * sizeof(*inputs));
    if (!inputs) { close(fd); return PyErr_NoMemory(); }
    r->inputs = inputs;
    CUfileDescr_t desc = {.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD};
    desc.handle.fd = fd;
    CUfileHandle_t handle;
    CUfileError_t status = cuFileHandleRegister(&handle, &desc);
    if (status.err != CU_FILE_SUCCESS) {
        close(fd);
        return PyErr_Format(PyExc_RuntimeError, "cuFileHandleRegister: %d CUDA %d", status.err, status.cu_err);
    }
    r->inputs[r->files] = (input_t){fd, s.st_size, handle};
    return PyLong_FromUnsignedLong(r->files++);
}

static PyObject *share_files(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *target, *source;
    if (!PyArg_ParseTuple(args, "OO", &target, &source)) return NULL;
    reader_t *r = PyCapsule_GetPointer(target, CAPSULE), *owner = PyCapsule_GetPointer(source, CAPSULE);
    if (!r || !owner) return NULL;
    if (r == owner || r->files || r->device != owner->device || r->input_owner)
        return PyErr_Format(PyExc_ValueError, "incompatible cuFile handle sharing");
    r->inputs = owner->inputs; r->files = owner->files;
    r->files_frozen = owner->files_frozen = 1;
    r->input_owner = source; Py_INCREF(source);
    Py_RETURN_NONE;
}

static PyObject *execute(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule;
    Py_buffer records;
    if (!PyArg_ParseTuple(args, "Oy*", &capsule, &records)) return NULL;
    reader_t *r = PyCapsule_GetPointer(capsule, CAPSULE);
    if (!r) { PyBuffer_Release(&records); return NULL; }
    char error[256] = {0};
    uint64_t reads = 0, bytes = 0, submissions = 0;
    double submit_seconds = 0, wait_seconds = 0;
    if (records.len % 64) snprintf(error, sizeof(error), "invalid descriptor size");
    size_t jobs = records.len / 64;
    for (size_t i = 0; i < jobs && !error[0]; i++) {
        uint64_t a[8]; memcpy(a, (char *)records.buf + i * 64, 64);
        if (a[0] >= r->files || !a[2] || !a[3] || a[4] || !a[5] ||
            a[1] > INT64_MAX || a[2] > INT64_MAX - a[1] || a[6] > INT64_MAX ||
            (a[6] && a[5] - 1 > (INT64_MAX - a[1] - a[2]) / a[6]) ||
            a[1] + (a[5] - 1) * a[6] + a[2] > (uint64_t)r->inputs[a[0]].size ||
            (a[5] > 1 && a[7] < a[2]) ||
            (a[7] && a[5] - 1 > (UINT64_MAX - a[3] - a[2]) / a[7]))
            snprintf(error, sizeof(error), "invalid byte-copy descriptor");
    }
    Py_BEGIN_ALLOW_THREADS
    cudaError_t gpu = cudaSetDevice(r->device);
    if (gpu != cudaSuccess) snprintf(error, sizeof(error), "cudaSetDevice: %s", cudaGetErrorString(gpu));
    size_t job = 0;
    uint64_t row = 0;
    while (job < jobs && !error[0]) {
        unsigned count = 0;
        while (count < r->depth && job < jobs) {
            uint64_t a[8]; memcpy(a, (char *)records.buf + job * 64, 64);
            CUfileIOParams_t *p = &r->params[count];
            *p = (CUfileIOParams_t){.mode = CUFILE_BATCH, .fh = r->inputs[a[0]].handle,
                .opcode = CUFILE_READ, .cookie = (void *)(uintptr_t)count};
            p->u.batch.devPtr_base = (void *)(uintptr_t)a[3];
            p->u.batch.devPtr_offset = row * a[7];
            p->u.batch.file_offset = a[1] + row * a[6];
            p->u.batch.size = a[2];
            r->completed[count++] = 0;
            if (++row == a[5]) { row = 0; job++; }
        }
        double begin = now();
        CUfileError_t status = cuFileBatchIOSubmit(r->batch, count, r->params, 0);
        submit_seconds += now() - begin;
        submissions++;
        if (status.err != CU_FILE_SUCCESS) {
            snprintf(error, sizeof(error), "cuFileBatchIOSubmit: %d CUDA %d", status.err, status.cu_err);
            cuFileBatchIOCancel(r->batch);
        }
        unsigned left = count;
        double deadline = now() + 30;
        begin = now();
        while (left) {
            unsigned nr = r->depth;
            struct timespec timeout = {.tv_sec = 0, .tv_nsec = 100000000};
            status = cuFileBatchIOGetStatus(r->batch, left, &nr, r->events, &timeout);
            if (status.err != CU_FILE_SUCCESS || now() > deadline) {
                /* This app must not free final tensors with unproven DMA completion. */
                fprintf(stderr, "Unproven batch completion: cuFile %d CUDA %d, %u pending; exiting prototype.\n",
                        status.err, status.cu_err, left);
                fflush(stderr); _exit(70);
            }
            for (unsigned i = 0; i < nr; i++) {
                CUfileIOEvents_t *event = &r->events[i];
                uintptr_t slot = (uintptr_t)event->cookie;
                if (slot >= count || r->completed[slot]) {
                    fprintf(stderr, "Invalid completion cookie; exiting prototype.\n");
                    fflush(stderr); _exit(70);
                }
                if (event->status == CUFILE_PENDING || event->status == CUFILE_WAITING) continue;
                r->completed[slot] = 1; left--; reads++;
                if (event->status == CUFILE_COMPLETE && (ssize_t)event->ret >= 0) bytes += event->ret;
                if (!error[0] && (event->status != CUFILE_COMPLETE || event->ret != r->params[slot].u.batch.size))
                    snprintf(error, sizeof(error), "batch completion at offset %lld: status %d, returned %lld, expected %zu",
                        (long long)r->params[slot].u.batch.file_offset, event->status,
                        (long long)event->ret, r->params[slot].u.batch.size);
            }
        }
        wait_seconds += now() - begin;
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&records);
    if (error[0]) return PyErr_Format(PyExc_RuntimeError, "%s", error);
    return Py_BuildValue("{s:K,s:K,s:K,s:d,s:d}", "reads", reads, "api_read_bytes", bytes,
                        "submissions", submissions, "submit_seconds", submit_seconds, "wait_seconds", wait_seconds);
}

static PyObject *register_buffer(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule;
    unsigned long long pointer, bytes;
    if (!PyArg_ParseTuple(args, "OKK", &capsule, &pointer, &bytes)) return NULL;
    reader_t *r = PyCapsule_GetPointer(capsule, CAPSULE);
    if (!r) return NULL;
    if (r->registered) return PyErr_Format(PyExc_RuntimeError, "buffer already registered");
    cudaSetDevice(r->device);
    CUfileError_t status = cuFileBufRegister((void *)(uintptr_t)pointer, bytes, 0);
    if (status.err != CU_FILE_SUCCESS)
        return PyErr_Format(PyExc_RuntimeError, "cuFileBufRegister: %d CUDA %d", status.err, status.cu_err);
    r->registered = (void *)(uintptr_t)pointer;
    r->registered_bytes = bytes;
    Py_RETURN_NONE;
}

static PyObject *read_whole(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule;
    unsigned file;
    unsigned long long offset, bytes, buffer_offset;
    if (!PyArg_ParseTuple(args, "OIKKK", &capsule, &file, &offset, &bytes, &buffer_offset)) return NULL;
    reader_t *r = PyCapsule_GetPointer(capsule, CAPSULE);
    if (!r) return NULL;
    if (file >= r->files || !r->registered || offset >= (uint64_t)r->inputs[file].size ||
        buffer_offset > r->registered_bytes || bytes > r->registered_bytes - buffer_offset)
        return PyErr_Format(PyExc_ValueError, "invalid aligned whole-tensor read");
    uint64_t expected = (uint64_t)r->inputs[file].size - offset;
    if (expected > bytes) expected = bytes;
    ssize_t result;
    Py_BEGIN_ALLOW_THREADS
    cudaSetDevice(r->device);
    result = cuFileRead(r->inputs[file].handle, r->registered, bytes, offset, buffer_offset);
    Py_END_ALLOW_THREADS
    if (result != (ssize_t)expected)
        return PyErr_Format(PyExc_RuntimeError, "cuFileRead whole tensor: expected %llu, received %lld", (unsigned long long)expected, (long long)result);
    return PyLong_FromSsize_t(result);
}

static PyObject *peer_gather(PyObject *self, PyObject *args) {
    (void)self;
    int device;
    unsigned long long kernel, source, destination, bytes, rows, source_stride, destination_stride, stream;
    if (!PyArg_ParseTuple(args, "iKKKKKKKK", &device, &kernel, &source, &destination,
                         &bytes, &rows, &source_stride, &destination_stride, &stream)) return NULL;
    void *scratch = NULL;
    void *params[] = {&source, &destination, &bytes, &rows, &source_stride,
                     &destination_stride, &scratch, &scratch};
    CUresult result;
    Py_BEGIN_ALLOW_THREADS
    cudaSetDevice(device);
    result = cuLaunchKernel((CUfunction)(uintptr_t)kernel, (unsigned)((rows * bytes + 1023) / 1024),
                            1, 1, 128, 1, 1, 0, (CUstream)(uintptr_t)stream, params, NULL);
    Py_END_ALLOW_THREADS
    if (result != CUDA_SUCCESS) return PyErr_Format(PyExc_RuntimeError, "peer byte-gather launch: CUDA %d", result);
    Py_RETURN_NONE;
}

static PyObject *close_reader(PyObject *self, PyObject *capsule) {
    (void)self;
    reader_t *r = PyCapsule_GetPointer(capsule, CAPSULE);
    if (!r) return NULL;
    cudaSetDevice(r->device);
    if (r->batch) { cuFileBatchIODestroy(r->batch); r->batch = NULL; }
    if (r->registered) { cuFileBufDeregister(r->registered); r->registered = NULL; }
    if (!r->input_owner) {
        for (unsigned i = 0; i < r->files; i++) {
            cuFileHandleDeregister(r->inputs[i].handle);
            close(r->inputs[i].fd);
        }
        free(r->inputs);
    }
    r->inputs = NULL; r->files = 0;
    Py_RETURN_NONE;
}

static PyObject *export_ipc(PyObject *self, PyObject *args) {
    (void)self;
    int device; unsigned long long pointer;
    if (!PyArg_ParseTuple(args, "iK", &device, &pointer)) return NULL;
    cudaSetDevice(device);
    CUdeviceptr base; size_t bytes;
    CUresult status = cuMemGetAddressRange(&base, &bytes, (CUdeviceptr)pointer);
    if (status != CUDA_SUCCESS) return PyErr_Format(PyExc_RuntimeError, "IPC allocation range: CUDA %d", status);
    cudaIpcMemHandle_t handle;
    cudaError_t error = cudaIpcGetMemHandle(&handle, (void *)(uintptr_t)base);
    if (error != cudaSuccess) return PyErr_Format(PyExc_RuntimeError, "IPC export: %s", cudaGetErrorString(error));
    return Py_BuildValue("(y#K)", &handle, (Py_ssize_t)sizeof(handle), pointer - (unsigned long long)base);
}

static PyObject *import_ipc(PyObject *self, PyObject *args) {
    (void)self;
    int device; Py_buffer data;
    if (!PyArg_ParseTuple(args, "iy*", &device, &data)) return NULL;
    if (data.len != (Py_ssize_t)sizeof(cudaIpcMemHandle_t)) {
        PyBuffer_Release(&data);
        return PyErr_Format(PyExc_ValueError, "invalid CUDA IPC handle");
    }
    cudaIpcMemHandle_t handle; memcpy(&handle, data.buf, sizeof(handle));
    PyBuffer_Release(&data);
    cudaSetDevice(device);
    void *pointer;
    cudaError_t error = cudaIpcOpenMemHandle(&pointer, handle, cudaIpcMemLazyEnablePeerAccess);
    if (error != cudaSuccess) return PyErr_Format(PyExc_RuntimeError, "IPC import: %s", cudaGetErrorString(error));
    return PyLong_FromUnsignedLongLong((uintptr_t)pointer);
}

static PyObject *close_ipc(PyObject *self, PyObject *args) {
    (void)self;
    int device; unsigned long long pointer;
    if (!PyArg_ParseTuple(args, "iK", &device, &pointer)) return NULL;
    cudaSetDevice(device);
    cudaError_t error = cudaIpcCloseMemHandle((void *)(uintptr_t)pointer);
    if (error != cudaSuccess) return PyErr_Format(PyExc_RuntimeError, "IPC close: %s", cudaGetErrorString(error));
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"create", create, METH_VARARGS, NULL}, {"add_file", add_file, METH_VARARGS, NULL},
    {"execute", execute, METH_VARARGS, NULL},
    {"register_buffer", register_buffer, METH_VARARGS, NULL},
    {"share_files", share_files, METH_VARARGS, NULL},
    {"read_whole", read_whole, METH_VARARGS, NULL},
    {"peer_gather", peer_gather, METH_VARARGS, NULL},
    {"close", close_reader, METH_O, NULL},
    {"export_ipc", export_ipc, METH_VARARGS, NULL},
    {"import_ipc", import_ipc, METH_VARARGS, NULL},
    {"close_ipc", close_ipc, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_gds_shard_reads", NULL, -1, methods,
                                  NULL, NULL, NULL, NULL};
PyMODINIT_FUNC PyInit__gds_shard_reads(void) { return PyModule_Create(&module); }
