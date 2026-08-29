from b12x._lib import compiler


def test_only_dynamic_moe_compile_is_host_capacity_limited():
    dynamic = compiler.KernelCompileSpec.from_facts(
        "integration.tp_moe.dynamic",
        1,
        ("quant_mode", "nvfp4"),
    )
    other = compiler.KernelCompileSpec.from_facts(
        "integration.tp_moe.micro_direct",
        1,
        ("quant_mode", "nvfp4"),
    )

    assert compiler._host_compile_capacity_limited(dynamic)
    assert not compiler._host_compile_capacity_limited(other)
    assert not compiler._host_compile_capacity_limited(None)


def test_call_cute_compile_holds_host_capacity_lock(monkeypatch):
    events = []
    compile_spec = compiler.KernelCompileSpec.from_facts(
        "integration.tp_moe.dynamic",
        1,
        ("quant_mode", "nvfp4"),
    )

    class Guard:
        def __enter__(self):
            events.append(("enter", compile_spec))

        def __exit__(self, exc_type, exc, traceback):
            events.append(("exit", compile_spec))

    def compile_callable(func, *args, **kwargs):
        events.append(("compile", func, args, kwargs))
        return "compiled"

    monkeypatch.setattr(compiler, "_cute_compile_progress_enabled", lambda: False)
    monkeypatch.setattr(
        compiler,
        "_host_compile_capacity_lock",
        lambda spec: Guard(),
    )

    result = compiler._call_cute_compile(
        compile_callable,
        test_call_cute_compile_holds_host_capacity_lock,
        ("arg",),
        {"option": "value"},
        compile_spec=compile_spec,
        cache_key="0" * 64,
    )

    assert result == "compiled"
    assert events == [
        ("enter", compile_spec),
        (
            "compile",
            test_call_cute_compile_holds_host_capacity_lock,
            ("arg",),
            {"option": "value"},
        ),
        ("exit", compile_spec),
    ]
