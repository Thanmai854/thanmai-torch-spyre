# Test Discovery Report

**Query:** tests for bmm

**Matches found:** 77 tests

**PyTorch root:** /Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch

**Torch-Spyre root:** /Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/torch-spyre

**Generated:** 2026-05-08T12:24:12.824480Z

---

## Matched Tests

### 1. `test_baddbmm` — `export/test_export.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/export/test_export.py:15767–15768` |
| **Classification** | `AS_IS` |
| **Module** | `export` |
| **Description** | The test defines a module that computes a bias‑added batch matrix multiplication using torch.ops.aten.baddbmm.default, exports it with dynamic batch dimension support, and verifies that the exported m... |
| **Dtypes** | torch.float16 |
| **LLM Dtype** | torch.float16|torch.float16|torch.float16|torch.float16 |
| **Shapes** | (64, 1, 192), (64, 1, 64), (64, 2048, 64), (64, 64, 192) |
| **LLM Shape** | (64, 64, 192)|(64, 1, 192)|(64, 2048, 64)|(64, 1, 64)|(4, 4)|(4, 4)|(4, 4) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_baddbmm(self):
        class M(torch.nn.Module):
```

---

### 2. `test_bmm_out_dtype` — `inductor/test_max_autotune.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_max_autotune.py:2233–2249` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that torch.bmm with an explicit out_dtype works correctly under different autotuning settings. It first ensures that enabling max_autotune with the TRITON backend raises an InductorE... |
| **Dtypes** | torch.float16, torch.float32 |
| **LLM Dtype** | torch.float32|torch.float16|torch.float16 |
| **Shapes** | (2, 3, 4), (2, 4, 5) |
| **LLM Shape** | (2, 3, 4)|(2, 4, 5) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_bmm_out_dtype(self):
        def f(a, b):
            return torch.bmm(a, b, out_dtype=torch.float32)

        a = torch.randn(2, 3, 4, device=GPU_TYPE, dtype=torch.float16)
        b = torch.randn(2, 4, 5, device=GPU_TYPE, dtype=torch.float16)
        expected = torch.bmm(a.float(), b.float())
        with config.patch(
            max_autotune=False,
            max_autotune_gemm_backends="ATEN",
        ):
            compiled_f = torch.compile(f)
            out, code = run_and_get_code(compiled_f, a, b)
            FileCheck().check("extern_kernels.bmm_dtype").run(code[0])
            self.assertEqual(out, expected, atol=1e-3, rtol=1e-3)

    @unittest.skipIf(config.cpp_wrapper, "out_dtype override not supported for AOTI")
```

---

### 3. `test_bmm_outer_product_k_is_one` — `inductor/test_mmdecomp.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_mmdecomp.py:177–187` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test creates two batched tensors with a singleton inner dimension (k=1), computes their batch matrix product using torch.bmm, and verifies that the custom decomp_bmm function returns a valid resul... |
| **Shapes** | (32, 1, 256), (32, 8, 1) |
| **LLM Shape** | (32, 8, 1)|(32, 1, 256)|(b, m, 1)|(b, 1, n)|(b, m, lhs_k_unbacked)|(b, rhs_k_unbacked, n) |
| **Device Classification** | device_generic |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_bmm_outer_product_k_is_one(self, device):
        t1 = torch.randn(32, 8, 1, device=device)
        t2 = torch.randn(32, 1, 256, device=device)
        expected = torch.bmm(t1, t2)

        out = decomp_bmm(t1, t2)

        self.assertIsNot(out, NotImplemented)
        self.assertEqual(expected, out)

    @unittest.skipIf(not HAS_GPU, "GPU tests require triton")
```

---

### 4. `test_bmm_outer_product_k_is_one_with_unbacked_k` — `inductor/test_mmdecomp.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_mmdecomp.py:188–221` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test checks the behavior of the batched matrix multiplication decomposition (decomp_bmm) when the inner dimension k equals one for an outer‑product pattern, using tensors that contain unbacked sym... |
| **Shapes** | (b, 1, n), (b, m, 1), (b, m, lhs_k_unbacked), (b, rhs_k_unbacked, n) |
| **LLM Shape** | (b, m, 1)|(b, 1, n)|(b, m, lhs_k_unbacked)|(b, rhs_k_unbacked, n) |
| **Device Classification** | device_generic |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_bmm_outer_product_k_is_one_with_unbacked_k(self, device):
        if device == "cpu":
            self.skipTest("unbacked symints require GPU fake tensors")

        shape_env = ShapeEnv()
        with FakeTensorMode(shape_env=shape_env):
            b, m, n = [shape_env.create_unbacked_symint() for _ in range(3)]
            lhs_k_unbacked, rhs_k_unbacked = [
                shape_env.create_unbacked_symint() for _ in range(2)
            ]

            lhs_static_k = torch.empty((b, m, 1), device=device)
            rhs_static_k = torch.empty((b, 1, n), device=device)
            lhs_unbacked_k = torch.empty((b, m, lhs_k_unbacked), device=device)
            rhs_unbacked_k = torch.empty((b, rhs_k_unbacked, n), device=device)

            self.assertIsNot(
                decomp_bmm(lhs_static_k, rhs_static_k),
                NotImplemented,
            )
            self.assertIs(
                decomp_bmm(lhs_static_k, rhs_unbacked_k),
                NotImplemented,
            )
            self.assertIs(
                decomp_bmm(lhs_unbacked_k, rhs_static_k),
                NotImplemented,
            )
            self.assertIs(
                decomp_bmm(lhs_unbacked_k, rhs_unbacked_k),
                NotImplemented,
            )

    @config.patch(coordinate_descent_tuning=False)
```

---

### 5. `test_bmm` — `functorch/test_vmap.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/functorch/test_vmap.py:1964–2006` |
| **Classification** | `AS_IS` |
| **Module** | `functorch` |
| **Description** | The test verifies that functorch's vmap correctly vectorizes the torch.bmm operation across various batch dimensions and raises appropriate errors on shape mismatches. It exercises left‑only, right‑on... |
| **Shapes** | (2, 2), (2, 5, 3), (B0, 2), (B0, 2, 2, 2), (B0, 2, 3, 5), (B0, 2, 5, 3), (B0, 3, 3, 2), (B0, B1, 2, 5, 3), (B1, 2, 3, 5), (B1, B0, 2, 3, 5) |
| **LLM Shape** | B0, B1|2, 2, 2|B0, 2|2, 2|3, 3, 2|2, 2, 2|B0, 3, 3, 2|2, 2|B0, 2, 2, 2|2, 5, 3|B0, 2, 3, 5|2, 5, 3|B1, B0, 2, 3, 5|2, 5, 3|2, 5, 3|B0, 2, 3, 5|2, 5, 3|B0, 2, 3, 5|B0, 2, 5, 3|B1, B0, 2, 3, 5|B0, B1, 2, 5, 3|B1, 2, 3, 5|B0, 2, 5, 3|B0, 2|B0, 3|B0, 0|B0, 0 |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_bmm(self):
        op = torch.bmm
        test = self._vmap_test
        B0, B1 = 7, 11

        # shape mismatch
        msg = ""
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 3, 3, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2, 2, 2))

        # left arg is vmapped
        test(op, (torch.rand(B0, 2, 3, 5), torch.rand(2, 5, 3)), in_dims=(0, None))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 2, 3, 5), torch.rand(2, 5, 3)),
            in_dims=(1, None),
        )

        # right arg is vmapped
        test(op, (torch.rand(2, 5, 3), torch.rand(B0, 2, 3, 5)), in_dims=(None, 0))
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(2, 5, 3), torch.rand(B1, B0, 2, 3, 5)),
            in_dims=(None, 1),
        )

        # both args are vmapped
        test(op, (torch.rand(B0, 2, 3, 5), torch.rand(B0, 2, 5, 3)))
        test(
            vmap(op),
            (torch.rand(B1, B0, 2, 3, 5), torch.rand(B0, B1, 2, 5, 3)),
            in_dims=(1, 0),
        )
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 2, 3, 5), torch.rand(B0, 2, 5, 3)),
            in_dims=(None, 0),
        )

```

---

### 6. `test_bmm_horizontal_fusion` — `inductor/test_native_matmul.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_native_matmul.py:162–176` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test defines a function that performs two batched matrix multiplications and combines their results with elementwise arithmetic, then verifies that the compiled code correctly fuses the two bmm op... |
| **Dtypes** | torch.int64 |
| **LLM Dtype** | torch.int64 |
| **Shapes** | (128, 16, 128)|(128, 128, 16)|(128, 16, 128)|(128, 128, 16)|(32, 16, 128)|(64, 128, 16)|(32, 16)|(32, 16)|(32, 16)|(128, 16) |
| **LLM Shape** | (B, M, K)|(M * K, K, 1)|(B, K, N)|(K * N, N, 1)|(128, 16, 128, 16)|(N_in, R)|(R, 1)|(N_w, R, J)|(R * J, J, 1)|(B, I)|(I, 1)|(B,)|(N_out, J)|(N_out, J) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_bmm_horizontal_fusion(self):
        def f(x, y, z, w):
            bmm1 = torch.bmm(x, y)
            bmm2 = torch.bmm(z, w)
            return bmm1 - bmm2 + bmm1 * bmm2

        B, M, K, N = 128, 16, 128, 16
        x = rand_strided((B, M, K), (M * K, K, 1), device=GPU_TYPE)
        y = rand_strided((B, K, N), (K * N, N, 1), device=GPU_TYPE)
        z = rand_strided((B, M, K), (M * K, K, 1), device=GPU_TYPE)
        w = rand_strided((B, K, N), (K * N, N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y, z, w))
        self._check_code(f, (x, y, z, w), 1, 2)

```

---

### 7. `test_bmm_no_z_broadcast` — `inductor/test_native_matmul.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_native_matmul.py:243–268` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test compiles a function performing a batched matrix multiplication (torch.bmm) on non‑contiguous GPU tensors and inspects the generated Triton kernel to ensure correct indexing without broadcasti... |
| **LLM Dtype** | torch.int64|torch.int64 |
| **Shapes** | (128, 16, 128)|(128, 128, 16) |
| **LLM Shape** | (G, M, K, P)|(M * K * P, K * P, P, 1)|(NB, K, N)|(K * N, N, 1)|(G,)|(G, P)|(NA, M, N)|(B, M, K)|(M * K, K, 1)|(B, K, N)|(K * N, N, 1) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_bmm_no_z_broadcast(self):
        def f(x, y):
            z = torch.bmm(x, y)
            return z

        B, M, K, N = 128, 16, 128, 16
        x = rand_strided((B, M, K), (M * K, K, 1), device=GPU_TYPE)
        y = rand_strided((B, K, N), (K * N, N, 1), device=GPU_TYPE)

        f = torch.compile(f)
        code = run_and_get_triton_code(f, x, y)

        FileCheck().check_not("tl.arange(0, ZBLOCK)[:, None, None, None]").check(
            "tl.arange(0, ZBLOCK)"
        ).check("tl.arange(0, YBLOCK)[None, :, None, None]").check(
            "tl.arange(0, XBLOCK)[None, None, :, None]"
        ).check("tl.arange(0, R0_BLOCK)[None, None, None, :]").run(code)


if HAS_GPU:
    torch.set_default_device(GPU_TYPE)

if __name__ == "__main__":
    if HAS_GPU:
        run_tests()

```

---

### 8. `test_bmm_non_standard_batch_stride` — `inductor/test_nv_universal_gemm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_nv_universal_gemm.py:230–234` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test creates batched matrices with non‑standard (non‑largest) batch strides by transposing tensors, runs torch.bmm both eagerly and through torch.compile with the NVGEMM backend, and checks that t... |
| **Dtypes** | torch.float16, torch.bfloat16 |
| **LLM Dtype** | torch.float16|torch.bfloat16 |
| **Shapes** | (k, batch, n), (m, batch, k) |
| **LLM Shape** | 8|64|256|128|(m, batch, k)|(batch, m, k)|(k, batch, n)|(batch, k, n) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_bmm_non_standard_batch_stride(self, dtype):
        """Test BMM path with non-standard batch strides."""
        batch, m, n, k = 8, 64, 256, 128
        device = "cuda"

```

---

### 9. `test_pad_bmm_dyn_b` — `inductor/test_pad_mm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_pad_mm.py:198–203` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that a batched matrix multiplication (torch.bmm) works correctly when the batch dimension is marked dynamic and shape padding is forced. It compiles the model with torch.compile, che... |
| **Dtypes** | torch.float32 |
| **LLM Dtype** | torch.float32 |
| **Shapes** | (B, K, N), (B, M, K) |
| **LLM Shape** | B, M, K, N | (B, M, K) | (B, K, N) | (10, 128, 33, 40) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_pad_bmm_dyn_b(self):
        B = 10
        M = 128
        K = 33
        N = 40

```

---

### 10. `test_pad_bmm_dyn_k` — `inductor/test_pad_mm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_pad_mm.py:229–234` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test checks that a batched matrix multiplication (bmm) with a dynamic K dimension is correctly padded and compiled by torch.compile, and that the generated kernel uses the expected padded N size.... |
| **Dtypes** | torch.float32 |
| **LLM Dtype** | torch.float32|torch.float32|torch.float32|torch.float32 |
| **Shapes** | (B, K, N), (B, M, K) |
| **LLM Shape** | B, M, K|B, K, N|B, M, K|B, K, N|B, M, K|B, K, N|B, M, K|B, K, N |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_pad_bmm_dyn_k(self):
        B = 10
        M = 128
        K = 40
        N = 41

```

---

### 11. `test_pad_bmm_dyn_bm` — `inductor/test_pad_mm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_pad_mm.py:260–265` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that a batched matrix multiplication (torch.bmm) with dynamic dimensions is correctly padded and compiled by torch.compile, and that the compiled result matches the eager execution r... |
| **Dtypes** | torch.float32 |
| **LLM Dtype** | torch.float32|torch.float32|torch.float32|torch.float32|torch.float32|torch.float32 |
| **Shapes** | (B, K, N), (B, M, K) |
| **LLM Shape** | B, M, K|B, K, N|B, M, K|B, K, N|M, N|M, K|K, N |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_pad_bmm_dyn_bm(self):
        B = 10
        M = 128
        K = 40
        N = 41

```

---

### 12. `test_bmm_to_mm` — `inductor/test_pattern_matcher.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_pattern_matcher.py:658–690` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test compiles a function performing torch.bmm and verifies that for a batch size of 1 the generated kernel is rewritten to a regular matrix multiplication (mm), while for larger batch sizes it rem... |
| **Shapes** | (1, 16, 8), (1, 8, 32), (3, 16, 8), (3, 8, 32) |
| **LLM Shape** | 1, 16, 8|1, 8, 32|3, 16, 8|3, 8, 32 |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_bmm_to_mm(self):
        def fn(a, b):
            return torch.bmm(a, b)

        a = torch.randn(1, 16, 8, device=GPU_TYPE)
        b = torch.randn(1, 8, 32, device=GPU_TYPE)

        result, (code,) = run_and_get_code(torch.compile(fn), a, b)

        expected = fn(a, b)
        torch.testing.assert_close(result, expected)

        # The mm kernel should use ATen (because we set max_autotune_gemm_backends = ATEN).
        # Its name should contain `aten.bmm` since this is the original aten op where the bmm came from.
        if HAS_GPU:
            FileCheck().check("extern_kernels.mm(").check_not(
                "extern_kernels.bmm("
            ).run(code)
        else:
            FileCheck().check("extern_kernels.bmm(")

        a_multi = torch.randn(3, 16, 8, device=GPU_TYPE)
        b_multi = torch.randn(3, 8, 32, device=GPU_TYPE)

        result_multi, (code_multi,) = run_and_get_code(
            torch.compile(fn), a_multi, b_multi
        )

        expected_multi = fn(a_multi, b_multi)
        torch.testing.assert_close(result_multi, expected_multi)

        FileCheck().check("extern_kernels.bmm(").run(code_multi)

```

---

### 13. `test_bmm` — `inductor/test_select_algorithm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_select_algorithm.py:220–221` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test compiles a simple function performing a batched matrix multiplication (torch.bmm) and executes it with random GPU tensors of shapes (2, 8, 32) and (2, 32, 8). It then verifies that the induct... |
| **Shapes** | (2, 32, 8), (2, 8, 32) |
| **LLM Shape** | (2, 8, 32)|(2, 32, 8) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_bmm(self):
        @torch.compile
```

---

### 14. `test_baddbmm` — `inductor/test_select_algorithm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_select_algorithm.py:264–265` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test compiles a function that performs a batched matrix multiplication with addition using torch.baddbmm and runs it on random GPU tensors. It then verifies that the inductor autotuning mechanism ... |
| **Shapes** | (2, 1, 8), (2, 32, 8), (2, 8, 32) |
| **LLM Shape** | (2, 8, 32)|(2, 32, 8)|(2, 1, 8) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_baddbmm(self):
        @torch.compile
```

---

### 15. `test_bmm` — `inductor/test_snode_runtime.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_snode_runtime.py:172–182` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test defines a function that performs a batched matrix multiplication (torch.bmm) on two 3‑D tensors and verifies that the measured runtime is non‑zero.... |
| **Shapes** | (10, 10, 10)|(10, 10)|(10) |
| **LLM Shape** | (10, 10, 10)|(10, 10, 10) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_bmm(self):
        def f(a, b):
            return torch.bmm(a, b)

        inp = (
            T(10, 10, 10),
            T(10, 10, 10),
        )
        self.assertNotZero(calculate_runtime(f, *inp))


```

---

### 16. `test_baddbmm` — `inductor/test_torchinductor.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_torchinductor.py:9165–9186` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies the correctness of the aten.baddbmm operation, which performs a batch matrix multiplication of tensors b and c, scales the result by beta, and adds tensor a. It runs the operation wi... |
| **LLM Dtype** | torch.bool |
| **Shapes** | (6, 1, 100), (6, 128, 64), (6, 64, 100) |
| **LLM Shape** | (6, 128, 64)|(6, 64, 100)|(6, 1, 100)|(128, 1)|(1, 128)|(128, 128)|(6, 1, 100)|(6, 128, 100)|(800, 256, 7, 7)|(601, 256, 7, 7)|(1024, 4, 2)|(4, 1, 1)|(100, 256, 7, 7) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_baddbmm(self):
        def fn(a, b, c, beta):
            return aten.baddbmm(a, b, c, beta=beta)

        b = torch.randn(6, 128, 64)
        c = torch.randn(6, 64, 100)
        options = itertools.product(
            [torch.randn(6, 1, 100), torch.randn(6, 1, 100).fill_(torch.nan)],
            [0.0, 1.0],
        )
        for a, beta in options:
            self.common(
                fn,
                [a, b, c, beta],
                # Mismatched elements: 1212 / 76800 (1.6%)
                # Greatest absolute difference: 0.001953125 at index (0, 0, 93) (up to 1e-05 allowed)
                # Greatest relative difference: 1.0 at index (3, 19, 4) (up to 0.001 allowed)
                atol=0.002,
                rtol=0.001,
            )

    @config.patch({"triton.max_tiles": 2})
```

---

### 17. `test_baddbmm` — `onnx/test_pytorch_onnx_onnxruntime.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/onnx/test_pytorch_onnx_onnxruntime.py:7830–7831` |
| **Classification** | `AS_IS` |
| **Module** | `onnx` |
| **Description** | The test defines a simple module that applies torch.baddbmm with custom alpha and beta scalars to three random tensors and verifies that the model can be exported to ONNX and run with ONNX Runtime.... |
| **Shapes** | (10, 3, 4), (10, 3, 5), (10, 4, 5), (5) |
| **LLM Shape** | (10, 3, 5)|(10, 3, 4)|(10, 4, 5)|(2, 3, 5)|(4, 5, 6)|(0)|(4) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_baddbmm(self):
        class MyModule(torch.nn.Module):
```

---

### 18. `test_baddbmm_dynamic` — `onnx/test_pytorch_onnx_onnxruntime.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/onnx/test_pytorch_onnx_onnxruntime.py:7843–7844` |
| **Classification** | `AS_IS` |
| **Module** | `onnx` |
| **Description** | The test defines a simple module that computes torch.baddbmm on three batched tensors with scalar coefficients and verifies the exported ONNX model against PyTorch using ONNX Runtime.... |
| **Shapes** | (10, 3, 4), (10, 3, 5), (10, 4, 5), (3.5), (5) |
| **LLM Shape** | (10, 3, 5)|(10, 3, 4)|(10, 4, 5) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_baddbmm_dynamic(self):
        class MyModule(torch.nn.Module):
```

---

### 19. `test_baddbmm_symint` — `test_dynamic_shapes.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_dynamic_shapes.py:1595–1613` |
| **Classification** | `AS_IS` |
| **Module** | `dynamic_shapes` |
| **Description** | The test creates unbacked symbolic dimensions and meta tensors for a batch matrix multiplication with bias, then invokes torch.baddbmm under FakeTensorMode to ensure the operation works with dynamic s... |
| **Shapes** | (B, K, N), (B, M, K), (B, M, N) |
| **LLM Shape** | (B, M, K)|(B, K, N)|(B, M, N) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_baddbmm_symint(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)

        B, M, K, N = [shape_env.create_unbacked_symint() for _ in range(4)]

        with fake_mode:
            A = torch.empty((B, M, K), device="meta")
            Bmat = torch.empty((B, K, N), device="meta")
            bias3 = torch.empty((B, M, N), device="meta")

            _ = torch.baddbmm(bias3, A, Bmat)


@skipIfTorchDynamo(
    "Creating ShapeEnv fails for confusing reasons (also we never expect dynamo to see code like this)"
)
```

---

### 20. `test_bmm` — `test_legacy_vmap.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_legacy_vmap.py:1307–1349` |
| **Classification** | `AS_IS` |
| **Module** | `legacy_vmap` |
| **Description** | The test verifies that the vectorized map (vmap) correctly handles torch.bmm under various batching configurations and raises appropriate errors on shape mismatches. It exercises left‑only, right‑only... |
| **Shapes** | (2, 2), (2, 5, 3), (B0, 2), (B0, 2, 2, 2), (B0, 2, 3, 5), (B0, 2, 5, 3), (B0, 3, 3, 2), (B0, B1, 2, 5, 3), (B1, 2, 3, 5), (B1, B0, 2, 3, 5) |
| **LLM Shape** | (B0, 3)|(B0, 3, 5)|(B0, B1, 3, 5)|(4, 4)|(2, B0, 3)|(B0, 0, 3)|(B0, 2, 2, 2)|(B0, 2)|(2, 2)|(B0, 3, 3, 2)|(2, 2)|(B0, 2, 2, 2)|(B0, 2, 3, 5)|(2, 5, 3)|(B1, B0, 2, 3, 5)|(2, 5, 3)|(B0, 2, 3, 5)|(B0, 2, 5, 3)|(B1, B0, 2, 3, 5)|(B0, B1, 2, 5, 3)|(B1, 2, 3, 5)|(B0, 2, 5, 3)|[3]|[3]|[4, 4]|[3]|[3] |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_bmm(self):
        op = torch.bmm
        test = self._vmap_test
        B0, B1 = 7, 11

        # shape mismatch
        msg = "Shape mismatch"
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 3, 3, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2, 2, 2))

        # left arg is vmapped
        test(op, (torch.rand(B0, 2, 3, 5), torch.rand(2, 5, 3)), in_dims=(0, None))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 2, 3, 5), torch.rand(2, 5, 3)),
            in_dims=(1, None),
        )

        # right arg is vmapped
        test(op, (torch.rand(2, 5, 3), torch.rand(B0, 2, 3, 5)), in_dims=(None, 0))
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(2, 5, 3), torch.rand(B1, B0, 2, 3, 5)),
            in_dims=(None, 1),
        )

        # both args are vmapped
        test(op, (torch.rand(B0, 2, 3, 5), torch.rand(B0, 2, 5, 3)))
        test(
            vmap(op),
            (torch.rand(B1, B0, 2, 3, 5), torch.rand(B0, B1, 2, 5, 3)),
            in_dims=(1, 0),
        )
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 2, 3, 5), torch.rand(B0, 2, 5, 3)),
            in_dims=(None, 0),
        )

```

---

### 21. `test_baddbmm_nan_input_with_zero_beta` — `test_linalg.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_linalg.py:7850–7863` |
| **Classification** | `NEEDS_REFACTOR` |
| **Module** | `linalg` |
| **Description** | The test iterates over two tensor shapes, creates random matrices mat1 and mat2, and verifies that torch.baddbmm with beta=0.0 returns the same result as torch.bmm regardless of NaN values present in ... |
| **Dtypes** | torch.float |
| **LLM Dtype** | torch.half|torch.half|torch.half|torch.half|torch.half|torch.half|torch.float|dtype|dtype|dtype|dtype|dtype|torch.int16|torch.int32|torch.int64|torch.float16|torch.float32|torch.float64|torch.float32|torch.float32|dtype|torch.float32 |
| **Shapes** | (shape) |
| **LLM Shape** | [3, 2, 2]|[2, 20, 20]|(1, 2, 2)|(1, 2, 2)|(1, 2, 2)|128x128|128x1000|1000x128|3x128x128|3x128x1000|3x1000x128 |
| **Device Classification** | device_generic |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_baddbmm_nan_input_with_zero_beta(self, device, dtype):
        for shape in [[3, 2, 2], [2, 20, 20]]:
            mat1, mat2 = (torch.randn(shape, dtype=dtype, device=device) for _ in range(2))
            inputs = [torch.randn(shape, dtype=dtype, device=device),
                      torch.randn(shape, dtype=dtype, device=device).fill_(torch.nan)]
            outs = [None, torch.randn(shape, dtype=dtype, device=device),
                    torch.randn(shape, dtype=dtype, device=device).fill_(torch.nan)]
            options = itertools.product(inputs, outs)
            for input, out in options:
                y_ref = torch.bmm(mat1, mat2)
                y = torch.baddbmm(input, mat1, mat2, beta=0.0, out=out)
                self.assertEqual(y_ref, y)

    @dtypes(torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64)
```

---

### 22. `test_baddbmm_input_dtypes_compatibility` — `test_linalg.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_linalg.py:7864–7878` |
| **Classification** | `NEEDS_REFACTOR` |
| **Module** | `linalg` |
| **Description** | The test verifies that torch.baddbmm enforces matching input dtypes and correctly writes results to an out tensor when dtypes are compatible. It checks error raising for mismatched dtypes and validate... |
| **Dtypes** | torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64 |
| **LLM Dtype** | torch.half|torch.half|torch.half|torch.float|torch.int16|torch.int32|torch.int64|torch.float16|torch.float32|torch.float64|torch.float32|torch.float32|torch.float32|torch.half|torch.half|torch.half|torch.half |
| **Shapes** | (1, 2, 2) |
| **LLM Shape** | [3, 128, 128]|[3, 128, 1000]|[3, 1000, 128]|[3, 2, 2]|[2, 20, 20]|[1, 2, 2]|[1, 2, 2]|[1, 2, 2]|[1, 2, 2]|[65537, 22, 64]|[65537, 64, 22]|[65537, 22, 22] |
| **Device Classification** | device_generic |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_baddbmm_input_dtypes_compatibility(self, device, dtype):
        batch1 = torch.rand((1, 2, 2), dtype=torch.float32, device=device)
        batch2 = torch.rand((1, 2, 2), dtype=torch.float32, device=device)
        input_tensor = torch.rand((1, 2, 2), device=device).to(dtype)
        if dtype != torch.float32:
            with self.assertRaisesRegex(RuntimeError, "Input dtypes must be the same"):
                y = torch.baddbmm(input_tensor, batch1, batch2, beta=0.0)
        else:
            out = torch.randn((1, 2, 2), dtype=dtype, device=device).fill_(torch.nan)
            y_ref = torch.bmm(batch1, batch2)
            y = torch.baddbmm(input_tensor, batch1, batch2, beta=0.0, out=out)
            self.assertEqual(out, y_ref)

    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "cublas runtime error")
    @onlyCUDA
```

---

### 23. `test_bmm` — `test_linalg.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_linalg.py:8639–8643` |
| **Classification** | `NEEDS_REFACTOR` |
| **Module** | `linalg` |
| **Description** | The test verifies the correctness of torch.bmm across a variety of scenarios, including different data types, device types, non‑contiguous inputs, broadcasting, and zero‑sized dimensions. It also chec... |
| **Dtypes** | torch.bfloat16, torch.half, torch.float32 |
| **LLM Dtype** | torch.float32|torch.float64|torch.half|torch.bfloat16|dtype|torch.bfloat16|torch.float32 |
| **Shapes** | (num_batches, M, N), (num_batches, M, O), (num_batches, N, O), (shape1), (shape2) |
| **LLM Shape** | [1, 10]|[2, 2, 2]|[3, 1, 1]|(num_batches, M, N)|(num_batches, N, O)|(num_batches if b1 else 1, M if b2 else 1, N if b3 else 1)|(num_batches if b4 else 1, N if b5 else 1, O if b6 else 1)|(num_batches, M, N)|(num_batches, N, O)|(num_batches if z1 else 0, M if z2 else 0, N if z3 else 0)|(num_batches if z1 else 0, N if z3 else 0)|23|15|12 |
| **Device Classification** | device_generic |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_bmm(self, device, dtype):
        batch_sizes = [1, 10]
        M, N, O = 23, 15, 12
        numpy_dtype = dtype if dtype != torch.bfloat16 else torch.float32

```

---

### 24. `test_addbmm` — `test_linalg.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_linalg.py:8734–8741` |
| **Classification** | `NEEDS_REFACTOR` |
| **Module** | `linalg` |
| **Description** | The test verifies the correctness of torch.addbmm across a variety of tensor layouts, permutations, and broadcasting scenarios for different floating point dtypes. It computes a reference result using... |
| **Dtypes** | torch.bfloat16, torch.half, torch.float32 |
| **LLM Dtype** | dtype|torch.half|torch.bfloat16|torch.float32|floating_and_complex_types_and(torch.bfloat16, torch.half) |
| **Shapes** | (num_batches, M, N), (num_batches, N, O), (ref) |
| **LLM Shape** | (num_batches, M, N)|(num_batches, N, O)|(M, N)|(N, O)|(0, 1, 2)|(0, 1)|(num_batches if s1 else 1, M if s2 else 1, N if s3 else 1)|(num_batches if s4 else 1, N if s5 else 1, O if s6 else 1)|16|17|18|2 |
| **Device Classification** | device_generic |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_addbmm(self, device, dtype):
        num_batches = 2
        M, N, O = 16, 17, 18

        if dtype == torch.bfloat16:
            if self.device_type == 'cpu':
                self.precision = 1  # 43 vs 43.75

```

---

### 25. `test_baddbmm` — `test_linalg.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_linalg.py:8791–8795` |
| **Classification** | `NEEDS_REFACTOR` |
| **Module** | `linalg` |
| **Description** | The test verifies the correctness of torch.baddbmm across a variety of tensor layouts, dtypes, and broadcasting scenarios. It constructs batched matrices, optionally permutes them to create non‑contig... |
| **Dtypes** | torch.bfloat16, torch.half, torch.float32 |
| **LLM Dtype** | dtype|torch.bfloat16|torch.half|torch.float32|torch.bfloat16|torch.half|dtype|dtype|dtype|dtype|dtype|dtype|dtype|numpy_dtype|dtype|dtype|dtype|dtype|dtype|dtype|dtype|torch.half|torch.bfloat16 |
| **Shapes** | (num_batches, M, N), (num_batches, N, O), (ref) |
| **LLM Shape** | (num_batches, M, N)|(num_batches, N, O)|(num_batches if s1 else 1, M if s2 else 1, N if s3 else 1)|(num_batches if s4 else 1, N if s5 else 1, O if s6 else 1)|(num_batches, M, N)|(num_batches, N, O)|(num_batches if z1 else 0, M if z2 else 0, N if z3 else 0)|(num_batches if z1 else 0, N if z3 else 0, O if z4 else 0)|(0, 1, 2)|(12, 8, 50)|10|8|50 |
| **Device Classification** | device_generic |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_baddbmm(self, device, dtype):
        num_batches = 10
        M, N, O = 12, 8, 50


```

---

### 26. `test_bmm` — `test_namedtensor.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_namedtensor.py:1819–1862` |
| **Classification** | `AS_IS` |
| **Module** | `namedtensor` |
| **Description** | The test iterates over all device types and verifies that torch.bmm correctly infers and propagates named dimensions, handles missing names, respects the out= argument, and raises appropriate errors f... |
| **Shapes** | (7,3,2)|(7,2,5)|(7,3,2)|(7,2,5)|(3,3,3)|(3,3,3)|(3,3,3) |
| **LLM Shape** | N:7,A:3,B:2|N:7,A:2,B:5|7,3,2|N:7,A:2,B:5|N:7,A:3,B:2|7,2,5|0|N:7,A:3,B:2|N:7,B:2,A:5|N:3,A:3,B:3|M:3,A:3,B:3|N:3,A:3,B:3|None:3,N:3,B:3 |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_bmm(self):
        for device in get_all_device_types():
            # full names
            self._test_name_inference(
                torch.bmm, device=device,
                args=(create('N:7,A:3,B:2'), create('N:7,A:2,B:5')),
                expected_names=('N', 'A', 'B'))

            # no name on left tensor
            self._test_name_inference(
                torch.bmm, device=device,
                args=(create('7,3,2'), create('N:7,A:2,B:5')),
                expected_names=('N', None, 'B'))

            # no name on right tensor
            self._test_name_inference(
                torch.bmm, device=device,
                args=(create('N:7,A:3,B:2'), create('7,2,5')),
                expected_names=('N', 'A', None))

            # out=
            self._test_name_inference(
                out_fn(torch.bmm), device=device,
                args=(create('0'), create('N:7,A:3,B:2'), create('N:7,A:2,B:5')),
                expected_names=('N', 'A', 'B'))

            # duplicate names after mm
            self._test_name_inference(
                torch.bmm, device=device,
                args=(create('N:7,A:3,B:2'), create('N:7,B:2,A:5')),
                maybe_raises_regex='with duplicate names')

            # matching error (batch dimensions must be alignable)
            self._test_name_inference(
                torch.bmm, device=device,
                args=(create('N:3,A:3,B:3'), create('M:3,A:3,B:3')),
                maybe_raises_regex='do not match')

            # misalignment (batch dimension is getting contracted)
            self._test_name_inference(
                torch.bmm, device=device,
                args=(create('N:3,A:3,B:3'), create('None:3,N:3,B:3')),
                maybe_raises_regex='misaligned')

```

---

### 27. `test_bmm_cpu` — `test_nestedtensor.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_nestedtensor.py:2328–2332` |
| **Classification** | `NEEDS_REFACTOR` |
| **Module** | `nestedtensor` |
| **Description** | The test verifies that batched matrix multiplication (torch.bmm) works correctly on CPU for NestedTensor objects with float and double dtypes.... |
| **Dtypes** | torch.float, torch.double |
| **LLM Dtype** | dtype|torch.float16|dtype|torch.float16|dtype|torch.float|torch.double|torch.float16|torch.bfloat16|torch.float|torch.double|torch.float|torch.double|torch.float|torch.double |
| **LLM Shape** | (2, 8)|(3, 16)|(8, 8)|(16, 8)|(2, 3)|(6, 7)|(2, 3)|(6, 7) |
| **Device Classification** | cpu |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_bmm_cpu(self, device, dtype):
        self._test_bmm(device, dtype)

    # cannot test torch.float16 because: RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'
    @dtypes(torch.float, torch.double)
```

---

### 28. `test_bmm_batch2_last_dim_size_is_one` — `inductor/test_mmdecomp.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_mmdecomp.py:166–176` |
| **Classification** | `NEEDS_REFACTOR` |
| **Module** | `inductor` |
| **Description** | The test creates two random float32 tensors of shapes (1, 32, 2) and (1, 2, 1) on the specified device and runs a batched matrix multiplication (bmm) through the compilation comparison helper.... |
| **Dtypes** | torch.float32 |
| **LLM Dtype** | torch.float32|torch.float|torch.bfloat16 |
| **Shapes** | (1, 2, 1), (1, 32, 2) |
| **LLM Shape** | (1, 32, 2)|(1, 2, 1)|(32, 8, 1)|(32, 1, 256)|(b, m, 1)|(b, 1, n)|(b, m, lhs_k_unbacked)|(b, rhs_k_unbacked, n) |
| **Device Classification** | device_generic |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_bmm_batch2_last_dim_size_is_one(self, device):
        fudge = 3
        rtol = default_rtol[torch.float32] * fudge
        atol = default_atol[torch.float32] * fudge

        t1 = torch.randn(1, 32, 2, device=device)
        t2 = torch.randn(1, 2, 1, device=device)

        run_comp_nocomp(torch_bmm, t1, t2, rtol=rtol, atol=atol)

    @config.patch(coordinate_descent_tuning=False)
```

---

### 29. `test_bmm_outer_product_permuted_inputs` — `inductor/test_mmdecomp.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_mmdecomp.py:222–255` |
| **Classification** | `NEEDS_REFACTOR` |
| **Module** | `inductor` |
| **Description** | The test verifies that the custom decomp_bmm function correctly handles batch matrix multiplication when the input tensors have permuted (non‑contiguous) memory layouts. It compares the result against... |
| **LLM Dtype** | torch.float|torch.bfloat16|torch.int|dtype |
| **Shapes** | (1, M, B), (B, 1, N), (B, M, 1), (M, B, 1), (N, B, 1) |
| **LLM Shape** | (b, m, 1)|(b, 1, n)|(b, m, lhs_k_unbacked)|(b, rhs_k_unbacked, n)|(M, B, 1)|(B, 1, N)|(N, B, 1)|(1, M, B)|(4, 8, 16)|B=4|M=8|N=16|[[1], [2], [3], [4]]|[[1, 2, 3, 4]] |
| **Device Classification** | device_generic |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_bmm_outer_product_permuted_inputs(self, device):
        B, M, N = 4, 8, 16

        cases = [
            # LHS: batch dim permuted
            (
                torch.randn(M, B, 1, device=device).permute(1, 0, 2),
                torch.randn(B, 1, N, device=device),
            ),
            # RHS: batch dim permuted
            (
                torch.randn(B, M, 1, device=device),
                torch.randn(N, B, 1, device=device).permute(1, 2, 0),
            ),
            # Both permuted
            (
                torch.randn(M, B, 1, device=device).permute(1, 0, 2),
                torch.randn(N, B, 1, device=device).permute(1, 2, 0),
            ),
            # LHS: fully transposed [1, M, B] -> [B, M, 1]
            (
                torch.randn(1, M, B, device=device).permute(2, 1, 0),
                torch.randn(B, 1, N, device=device),
            ),
        ]

        for t1, t2 in cases:
            expected = torch.bmm(t1, t2)
            out = decomp_bmm(t1, t2)
            self.assertIsNot(out, NotImplemented)
            self.assertEqual(expected, out, exact_stride=True)

    @unittest.skipIf(not HAS_GPU, "GPU tests require triton")
    @parametrize("dtype", [torch.float, torch.bfloat16, torch.int])
```

---

### 30. `test_bmm_multithreaded` — `test_torch.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_torch.py:10673–10682` |
| **Classification** | `AS_IS` |
| **Module** | `torch` |
| **Description** | The test checks that batched matrix multiplication (torch.bmm) produces correct and deterministic results when executed with multiple CPU threads. It exercises a variety of input configurations, inclu... |
| **Dtypes** | torch.float32 |
| **LLM Dtype** | torch.float32|dtype |
| **Shapes** | (num_batches, M, N), (num_batches, M, O), (num_batches, N, O), (shape1), (shape2) |
| **LLM Shape** | (num_batches, M, N)|(num_batches, N, O)|(M, N)|(N, O)|(num_batches if b1 else 1, M if b2 else 1, N if b3 else 1)|(num_batches if b4 else 1, N if b5 else 1, O if b6 else 1)|(num_batches if z1 else 0, M if z2 else 0, N if z3 else 0)|(num_batches if z1 else 0, N if z3 else 0, O if z4 else 0)|23|8|12|[1, 10] |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_bmm_multithreaded(self):
        device = 'cpu'
        num_threads = torch.get_num_threads()

        torch.set_num_threads(4)
        batch_sizes = [1, 10]
        M, N, O = 23, 8, 12
        dtype = torch.float32
        numpy_dtype = dtype

```

---

### 31. `test_max_autotune_cutlass_backend_bmm` — `inductor/test_cutlass_backend.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_cutlass_backend.py:879–890` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that batched matrix multiplication (torch.bmm) works correctly when the CUTLASS backend is selected and max‑autotune is enabled, covering static and dynamic shape scenarios, AOTI com... |
| **Dtypes** | torch.float16, torch.bfloat16 |
| **LLM Dtype** | torch.bfloat16|torch.float16|torch.bfloat16 |
| **Shapes** | (B, M, K), (B, N, K), (M, K) |
| **LLM Shape** | (10, 4096, 2048, 25728)|(20, 2048, 1024, 12864)|(B, M, N, K)|(M, K)|(B, -1, -1)|(B, N, K)|(0, 2, 1) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_max_autotune_cutlass_backend_bmm(
        self,
        dynamic: bool,
        use_aoti: bool = False,
        max_autotune_gemm_backends: str = "CUTLASS",
        dtype: torch.dtype = torch.float16,
        use_expand: bool = False,
    ):
        """
        Main test for bmm.
        """

```

---

### 32. `test_bmm_flexible_layout` — `inductor/test_cpu_select_algorithm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_cpu_select_algorithm.py:3062–3063` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test defines a module that reshapes, clones, permutes, and scales tensors before performing a batch matrix multiplication (bmm). It runs the module with random inputs and verifies the output again... |
| **LLM Dtype** | u.dtype |
| **Shapes** | (2, 24, 512, 64), (48, 512, 64) |
| **LLM Shape** | [-1, 512, 64]|[0, 2, 1]|2, 24, 512, 64|48, 512, 64 |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_bmm_flexible_layout(self):
        class M(torch.nn.Module):
```

---

### 33. `test_bmm_freezing` — `inductor/test_cpu_select_algorithm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_cpu_select_algorithm.py:2732–2733` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that a batched matrix multiplication (bmm) involving a frozen (requires_grad=False) weight parameter is correctly compiled by TorchInductor, producing exactly one templated C++ kerne... |
| **Dtypes** | torch.float, torch.bfloat16, torch.half |
| **LLM Dtype** | torch.float|torch.bfloat16|torch.half|dtype |
| **Shapes** | (bs, Kdim, Ndim), (bs, Mdim, Kdim) |
| **LLM Shape** | (bs, Mdim, Kdim)|(bs, Kdim, Ndim)|(1,)|(192,)|(196,)|(64, 65)|(64, 61)|(bs, Mdim, Kdim)|(bs, Kdim, Ndim)|(12, 10, 62)|(12, 62, 61)|((0, 1, 2), (0, 2, 1))|((0, 1, 2), (1, 2, 0))|((0, 1, 2), (1, 0, 2))|((1, 0, 2), (0, 1, 2))|((1, 0, 2), (1, 2, 0)) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_bmm_freezing(self, dtype, bs, Mdim, Kdim, Ndim):
        class M(torch.nn.Module):
```

---

### 34. `test_bmm_amx` — `inductor/test_cpu_select_algorithm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_cpu_select_algorithm.py:2676–2677` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that a batched matrix multiplication (bmm) operation selects the correct AMX or BRGEMM kernel in TorchInductor and produces numerically accurate results across a range of output dime... |
| **Dtypes** | torch.bfloat16, torch.half |
| **LLM Dtype** | torch.bfloat16|torch.half|torch.float|torch.bfloat16|torch.half |
| **Shapes** | (bs, Kdim, Ndim), (bs, Mdim, Kdim) |
| **LLM Shape** | (bs, Mdim, Kdim)|(bs, Kdim, Ndim)|(2,)|(16, 32)|(32,)|(3, 16, 32, 48, 128, 1024, 1025) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_bmm_amx(self, dtype, bs, Mdim, Kdim, Ndim):
        class M(torch.nn.Module):
```

---

### 35. `test_bmm_multiple_dynamic` — `inductor/test_aot_inductor.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_aot_inductor.py:1796–1799` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test compiles and runs a simple model that performs torch.bmm on inputs with a dynamic batch dimension, using multiple example inputs of varying batch sizes on a GPU with the Triton backend.... |
| **Shapes** | (batch, K, N), (batch, M, K) |
| **LLM Shape** | batch, M, K|batch, K, N|1, 2048|2048, M, K|128, M, K|1024, M, K|128, 2048|128, 2048 |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_bmm_multiple_dynamic(self):
        if self.device == "cpu":
            raise unittest.SkipTest("using triton backend only is not supported on CPU")

```

---

### 36. `test_cutlass_backend_subproc_bmm` — `inductor/test_cutlass_backend.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_cutlass_backend.py:388–414` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that the CUTLASS backend can be autotuned in a subprocess for batched matrix multiplication (bmm) when using torch.compile. It compiles a bmm operation with specific autotuning setti... |
| **LLM Dtype** | half |
| **Shapes** | (B, M, K), (B, N, K) |
| **LLM Shape** | (B, M, K)|(B, N, K)|(10, 4096, 2048, 25728)|10|4096|2048|25728|(128, 16)|(128, 16)|(512, 16) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_cutlass_backend_subproc_bmm(self):
        """
        Test autotune_in_subproc works for bmm.
        """

        B, M, N, K = 10, 4096, 2048, 25728

        a = torch.randn(B, M, K).to(GPU_TYPE).half()
        b = torch.randn(B, N, K).to(GPU_TYPE).half().permute(0, 2, 1)

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "compile_threads": 4,
                "cutlass.cutlass_max_profiling_configs": 4,
            }
        ):
            Y_compiled = torch.compile(torch.bmm)(a, b)
            Y = torch.bmm(a, b)
            torch.testing.assert_close(Y_compiled, Y)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @parametrize("dynamic", (False, True))
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
```

---

### 37. `test_bmm` — `inductor/test_cpu_select_algorithm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_cpu_select_algorithm.py:2652–2653` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that the inductor backend correctly compiles and executes a batched matrix multiplication (bmm) for various batch sizes, dimensions, and data types, and that exactly one templated ke... |
| **Dtypes** | torch.float, torch.bfloat16, torch.half |
| **LLM Dtype** | torch.bfloat16|torch.float|torch.half |
| **Shapes** | (bs, Kdim, Ndim), (bs, Mdim, Kdim) |
| **LLM Shape** | (2, 32)|(bs, Mdim, Kdim)|(bs, Kdim, Ndim)|1|50|192|196|84|385|2|16|32|3|48|128|1024|1025 |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_bmm(self, dtype, bs, Mdim, Kdim, Ndim):
        class M(torch.nn.Module):
```

---

### 38. `test_decompose_bmm_cpu` — `inductor/test_decompose_mem_bound_mm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_decompose_mem_bound_mm.py:164–187` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test runs a compiled module that performs a batch matrix multiplication (bmm) on randomly generated tensors and checks whether the Inductor optimizer decomposes the bmm operation based on the batc... |
| **Shapes** | (b, k, n), (b, m, k) |
| **LLM Shape** | (1, 2, 2, 2)|(2, 2, 2, 2)|(b, m, k)|(b, k, n)|(20480, 5, 2)|(20480, 32, 2)|(2048, 2, 2)|(m, k) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_decompose_bmm_cpu(self, b, m, n, k, should_decompose):
        torch._logging.set_logs(inductor=logging.DEBUG)
        mat1 = torch.randn(b, m, k)
        mat2 = torch.randn(b, k, n)

        counters.clear()

        module = MyModule2()
        traced = torch.compile(module)
        input = [mat1, mat2]
        self.compare_pred(module, traced, input)

        expected_val = 1 if should_decompose else 0
        self.assertEqual(
            counters["inductor"]["decompose_bmm"],
            expected_val,
        )
        counters.clear()

    @parametrize(
        "m,k,n, should_decompose",
        [(20480, 5, 2, True), (20480, 32, 2, False), (2048, 2, 2, False)],
    )
    @parametrize("has_bias", [True, False])
```

---

### 39. `test_permute_bmm_fusion` — `inductor/test_fx_fusion.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_fx_fusion.py:133–134` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test defines a module that permutes a 3‑D tensor and then performs a batched matrix multiplication with a stored tensor. It applies a FX fusion pass that should replace the permute + torch.bmm pat... |
| **Shapes** | (batch, k, m), (batch, k, n) |
| **LLM Shape** | batch, k, n | 6, 16, 8, 4 | (batch, k, n) | (batch, k, m) | 0, 2, 1 |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_permute_bmm_fusion(self):
        class TestModule(torch.nn.Module):
```

---

### 40. `test_bmm3` — `fx/test_z3_gradual_types.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/fx/test_z3_gradual_types.py:187–188` |
| **Classification** | `AS_IS` |
| **Module** | `fx` |
| **Description** | The test defines a module that performs a batch matrix multiplication (torch.bmm) on two tensors with mismatched batch dimensions, symbolically traces the module, transforms shape constraints, and che... |
| **Shapes** | (2, 3, 3)|(1, 3, 2) |
| **LLM Shape** | [2, 3, 3]|[1, 3, 2]|[1, 2, 3, 4]|[2050, 1024] |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_bmm3(self):
        class BasicBlock(torch.nn.Module):
```

---

### 41. `test_bmm2` — `fx/test_z3_gradual_types.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/fx/test_z3_gradual_types.py:168–169` |
| **Classification** | `AS_IS` |
| **Module** | `fx` |
| **Description** | The test defines a module that performs a batch matrix multiplication (bmm) on a dynamic input tensor and a statically‑typed tensor, traces it with torch.fx, converts the traced graph into Z3 constrai... |
| **Shapes** | (1, 2, 3), (1, 3, 2) |
| **LLM Shape** | [Dyn, 2, 3]|[1, 3, 2]|[2, 3, 3]|[1, 2, 3, 4] |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_bmm2(self):
        class BasicBlock(torch.nn.Module):
```

---

### 42. `test_bmm` — `fx/test_z3_gradual_types.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/fx/test_z3_gradual_types.py:149–150` |
| **Classification** | `AS_IS` |
| **Module** | `fx` |
| **Description** | The test defines a module that performs a batch matrix multiplication (torch.bmm) on two tensors, traces it with torch.fx, converts the traced graph into Z3 constraints, and checks that the inferred o... |
| **Shapes** | (1, 2, 3), (1, 3, 2) |
| **LLM Shape** | [Dyn, 2, 3]|[1, 3, 2]|[2, 3, 3] |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_bmm(self):
        class BasicBlock(torch.nn.Module):
```

---

### 43. `test_bmm_with_y_storage_offset` — `inductor/test_cpu_select_algorithm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_cpu_select_algorithm.py:2949–2950` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test builds a module that performs a batch matrix multiplication where the right-hand operand is a contiguous slice with a non‑zero storage offset. It runs the module through the Inductor compiler... |
| **Dtypes** | torch.float, torch.bfloat16, torch.half |
| **LLM Dtype** | torch.float|torch.bfloat16|torch.half|dtype|y.dtype |
| **Shapes** | (3, ?), (bs, Kdim, Ndim), (bs, Mdim, Kdim) |
| **LLM Shape** | (3, *y.shape)|(bs, Mdim, Kdim)|(bs, Kdim, Ndim) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_bmm_with_y_storage_offset(self, dtype, bs, Mdim, Kdim, Ndim):
        class M(torch.nn.Module):
```

---

### 44. `test_bmm_amp` — `inductor/test_cpu_select_algorithm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_cpu_select_algorithm.py:2706–2707` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that a simple batched matrix multiplication module runs correctly under CPU automatic mixed precision (AMP) for several dtypes and that the inductor compiler generates exactly one te... |
| **Dtypes** | torch.float, torch.bfloat16, torch.half |
| **LLM Dtype** | dtype|torch.float|torch.bfloat16|torch.half|torch.bfloat16 |
| **Shapes** | (bs, Kdim, Ndim), (bs, Mdim, Kdim) |
| **LLM Shape** | (bs, Mdim, Kdim)|(bs, Kdim, Ndim)|(1,)|(192,)|(196,)|(84,)|(64, 65) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_bmm_amp(self, dtype, bs, Mdim, Kdim, Ndim):
        class M(torch.nn.Module):
```

---

### 45. `test_bmm_2d_permute` — `inductor/test_cpu_select_algorithm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_cpu_select_algorithm.py:2767–2775` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that batched matrix multiplication works correctly when the input tensors are reshaped and permuted according to various dimension orders. It runs the forward pass of a small module ... |
| **Dtypes** | torch.float, torch.bfloat16, torch.half |
| **LLM Dtype** | torch.float|torch.bfloat16|torch.half|dtype |
| **Shapes** | (bs, Kdim, Ndim), (bs, Mdim, Kdim), (o) |
| **LLM Shape** | (bs, Mdim, Kdim)|(bs, Kdim, Ndim)|(0, 1, 2)|(0, 2, 1)|(1, 2, 0)|(1, 0, 2)|12|10|62|64|61|x_order[0]|x_order[1] * x_order[2]|w_order[0]|w_order[1] * w_order[2] |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_bmm_2d_permute(self, Ndim, order, dtype):
        # TODO: Support bmm with transposed X
        bs = 12
        Mdim = 10
        Kdim = 62
        x_args = (bs, Mdim, Kdim)
        w_args = (bs, Kdim, Ndim)
        inverse_order = [torch.argsort(torch.tensor(o)).tolist() for o in order]

```

---

### 46. `test_bmm_self_permute` — `inductor/test_cpu_select_algorithm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_cpu_select_algorithm.py:2810–2811` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test defines a simple module that computes a batch matrix multiplication of a tensor with its own permuted version, runs it through the Inductor compiler, and checks that a single templated C++ ke... |
| **Dtypes** | torch.float, torch.float16, torch.bfloat16 |
| **LLM Dtype** | torch.float|torch.float16|torch.bfloat16|torch.float |
| **Shapes** | (bs, Mdim, Kdim) |
| **LLM Shape** | (bs, Mdim, Kdim)|(0, 1, 2)|(0, 2, 1)|(bs, Mdim, Mdim)|(5,)|(64,)|(96,)|(3, 64)|(16,)|(32,)|(64,) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_bmm_self_permute(self, bs, Mdim, Kdim, dtype):
        class M(torch.nn.Module):
```

---

### 47. `test_bmm_self_square` — `inductor/test_cpu_select_algorithm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_cpu_select_algorithm.py:2831–2832` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test defines a simple module that computes a batch matrix multiplication of a tensor with itself, runs it with random input, and verifies the output against the reference while checking that exact... |
| **Dtypes** | torch.float |
| **LLM Dtype** | torch.float|torch.float16|torch.bfloat16|torch.float|torch.float|torch.float32|torch.bfloat16|torch.half |
| **Shapes** | (bs, Mdim, Mdim) |
| **LLM Shape** | (5,)|(3, 64)|(bs, Mdim, Mdim)|(5,)|(16,)|(32,)|(64,)|(bs, Kdim, Ndim)|(Mdim, Kdim)|(5,)|(384,)|(96,)|(64, 65) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_bmm_self_square(self, bs, Mdim, dtype):
        class M(torch.nn.Module):
```

---

### 48. `test_bmm2` — `inductor/test_torchinductor.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_torchinductor.py:4322–4336` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test defines a function that permutes the first input tensor and then performs a batch matrix multiplication with a second tensor. It runs this function through the TorchInductor testing harness w... |
| **LLM Dtype** | torch.int8|torch.uint8|torch.float|torch.int64 |
| **Shapes** | (1, 8, 8) |
| **LLM Shape** | (1, 8, 8)|(8, 8)|(8, 8)|(256, 256)|(256, 256)|(4, 8)|(2, 3)|(9)|b.shape[1] |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_bmm2(self):
        def fn(a, b):
            return torch.bmm(a.permute(0, 2, 1), b)

        self.common(
            fn,
            (
                torch.randn(1, 8, 8),
                torch.randn(1, 8, 8),
            ),
            check_lowp=False,
        )

    @skipIfPy312  # segfaults
    @skipCUDAIf(not SM80OrLater, "Requires sm80")
```

---

### 49. `test_bmm1` — `inductor/test_torchinductor.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_torchinductor.py:4298–4321` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test defines a function performing two batched matrix multiplications, one with plain inputs and one with elementwise offsets, and verifies it under TorchInductor for two different input shape con... |
| **LLM Dtype** | torch.int8 |
| **Shapes** | (1, 16, 8), (1, 8, 10), (2, 8, 8) |
| **LLM Shape** | (2, 8, 8)|(2, 8, 8)|(1, 16, 8)|(1, 8, 10)|(1, 8, 8)|(1, 8, 8)|(8, 8)|(8, 8)|(8, 8)|(8, 8)|(8,)|(8,)|(256, 256)|(256, 256) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_bmm1(self):
        def fn(a, b):
            return (
                torch.bmm(a, b),
                torch.bmm(a + 1, b + 2) + 3,
            )

        self.common(
            fn,
            (
                torch.randn(2, 8, 8),
                torch.randn(2, 8, 8),
            ),
            check_lowp=False,
        )
        self.common(
            fn,
            (
                torch.randn(1, 16, 8),
                torch.randn(1, 8, 10),
            ),
            check_lowp=False,
        )

```

---

### 50. `test_sp24_matmuls_bmm` — `test_sparse_semi_structured.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_sparse_semi_structured.py:1034–1043` |
| **Classification** | `AS_IS` |
| **Module** | `sparse_semi_structured` |
| **Description** | The test creates a dense tensor `a` and a batched tensor `b`, converts `a` to a sparse semi‑structured format, and attempts to perform a batched matrix multiplication using the `@` operator. It verifi... |
| **Dtypes** | torch.float16 |
| **LLM Dtype** | torch.float16 |
| **Shapes** | (5, 6, 128), (64, 128) |
| **LLM Shape** | [64, 128]|[5, 6, 128] |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_sp24_matmuls_bmm(self) -> None:
        a = torch.randn([64, 128], device="cuda", dtype=torch.float16)
        b = torch.randn([5, 6, 128], device="cuda", dtype=torch.float16)
        a_m = sparse24_largest_mask_2d(a)
        a_s = to_sparse_semi_structured(a)

        with pytest.raises(NotImplementedError):
            torch.testing.assert_close(a_s @ b, (a * a_m) @ b, **atol_rtol_kw[a.dtype])


```

---

### 51. `test_bmm_with_broadcasted_mat1` — `inductor/test_cpu_select_algorithm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_cpu_select_algorithm.py:2854–2855` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test defines a module that expands a 2‑D tensor to a batch dimension and performs a batched matrix multiplication with a 3‑D tensor. It runs the compiled model, checks numerical correctness, and v... |
| **Dtypes** | torch.float |
| **LLM Dtype** | torch.float|dtype|torch.float32|torch.bfloat16|torch.half |
| **Shapes** | (Mdim, Kdim), (bs, Kdim, Ndim) |
| **LLM Shape** | (5,)|(16,)|(32,)|(64,)|(Mdim, Kdim)|(bs, Kdim, Ndim)|(bs, Mdim, Kdim)|(bs, Kdim, Ndim)|(bs, Mdim, Ndim)|(64, 65)|2|-1 |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_bmm_with_broadcasted_mat1(self, bs, Mdim, Kdim, Ndim, dtype):
        class M(torch.nn.Module):
```

---

### 52. `test_bmm_with_pointwise` — `inductor/test_cpu_select_algorithm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_cpu_select_algorithm.py:2890–2891` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test builds a module that performs a batched matrix multiplication (x @ w) followed by a pointwise epilogue operation (relu, add, sub, mul, or div) using a third tensor. It runs the module with ra... |
| **Dtypes** | torch.float32, torch.bfloat16, torch.half |
| **LLM Dtype** | torch.float32|torch.bfloat16|torch.half |
| **Shapes** | (bs, Kdim, Ndim), (bs, Mdim, Kdim), (bs, Mdim, Ndim) |
| **LLM Shape** | (5,)|(384,)|(96,)|(64, 65)|(bs, Mdim, Kdim)|(bs, Kdim, Ndim)|(bs, Mdim, Ndim)|(8, 8, 3136, 8)|(200704, 8, 64, 1)|(64, 3137, 8)|(64, 8, 8)|(8, 8, 3137, 8)|[0, 0, 1, 0, 0, 0]|(8, 8, 3137, 8) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_bmm_with_pointwise(self, bs, Mdim, Kdim, Ndim, epilogue, dtype):
        class M(torch.nn.Module):
```

---

### 53. `test_bmm_with_fused_epilogues` — `inductor/test_cpu_select_algorithm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_cpu_select_algorithm.py:2913–2914` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test defines a module that performs a reshape, batch matrix multiplication, scalar multiplication, constant padding, and addition, then verifies that the Inductor compiler fuses the multiplication... |
| **Dtypes** | torch.float32, torch.bfloat16, torch.half |
| **LLM Dtype** | torch.float32|torch.bfloat16|torch.half|torch.float |
| **Shapes** | (8, 8, 3136, 8), (8, 8, 3137, 8), (8, 8, 8, 8) |
| **LLM Shape** | (8, 8, 3136, 8)|(200704, 8, 64, 1)|[64, 3137, 8]|[64, 8, 8]|[8, 8, 3137, 8]|[0, 0, 1, 0, 0, 0]|(8, 8, 3137, 8)|(8, 8, 8, 8)|[1, 50]|[192]|[196]|[84, 385] |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_bmm_with_fused_epilogues(self, dtype):
        class M(torch.nn.Module):
```

---

### 54. `test_aoti_bmm_unique_identifiers` — `inductor/test_cpu_select_algorithm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_cpu_select_algorithm.py:2972–2981` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test defines a simple module that performs two consecutive batched matrix multiplications and runs it through the AOTInductor pipeline. It verifies numerical correctness against eager execution an... |
| **Dtypes** | torch.float |
| **LLM Dtype** | dtype|torch.float|torch.bfloat16|torch.half|y.dtype |
| **Shapes** | (3, 64, 64) |
| **LLM Shape** | (3, 64, 64)|(3, *y.shape) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_aoti_bmm_unique_identifiers(self, dtype):
        try:
            try:
                from . import test_aot_inductor_utils
            except ImportError:
                import test_aot_inductor_utils
        except Exception:
            # skip this UT if import failed
            return

```

---

### 55. `test_bmm_with_pointwise_dynamic_shapes` — `inductor/test_cpu_select_algorithm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_cpu_select_algorithm.py:3179–3180` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that TorchInductor correctly generates a templated batched matrix multiplication kernel with a fused ReLU epilogue when input dimensions are partially dynamic. It runs the module on ... |
| **Dtypes** | torch.float, torch.bfloat16, torch.half |
| **LLM Dtype** | torch.float|torch.bfloat16|torch.half |
| **Shapes** | (bs, Kdim, Ndim), (bs, Mdim, Kdim) |
| **LLM Shape** | (5,)|(384,)|(96,)|(64, 65)|(bs, Mdim, Kdim)|(bs, Kdim, Ndim)|(-1, Mdim, Kdim)|(-1, Kdim, Ndim)|(bs, 8, Mdim, Kdim)|(bs, 8, Kdim, Ndim)|(bs * 8, Mdim, Ndim) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_bmm_with_pointwise_dynamic_shapes(self, bs, Mdim, Kdim, Ndim, dtype):
        class M(torch.nn.Module):
```

---

### 56. `test_bmm_with_pointwise_with_reshape_dynamic_shapes` — `inductor/test_cpu_select_algorithm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_cpu_select_algorithm.py:3209–3211` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that a module performing a reshaped batched matrix multiplication followed by a ReLU epilogue and addition works correctly with dynamic batch dimensions and that the Inductor compile... |
| **Dtypes** | torch.float, torch.bfloat16, torch.half |
| **LLM Dtype** | torch.float|torch.bfloat16|torch.half|dtype |
| **Shapes** | (bs, 8, Kdim, Ndim), (bs, 8, Mdim, Kdim) |
| **LLM Shape** | bs, 8, Mdim, Kdim|bs, 8, Kdim, Ndim|bs * 8, Mdim, Ndim|-1, Mdim, Kdim|-1, Kdim, Ndim|5|8|512|64 |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_bmm_with_pointwise_with_reshape_dynamic_shapes(
        self, bs, Mdim, Kdim, Ndim, dtype
    ):
```

---

### 57. `test_bmm_epilogue_dynamic_reshape` — `inductor/test_cpu_select_algorithm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_cpu_select_algorithm.py:3242–3244` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test builds a module that reshapes two 4‑D tensors into 3‑D batches, performs a batched matrix multiplication, reshapes the result back, runs a series of integer tensor manipulations and an embedd... |
| **Dtypes** | torch.float, torch.bfloat16, torch.float32, torch.int64 |
| **LLM Dtype** | dtype|torch.float|torch.bfloat16|torch.int64|torch.float32 |
| **Shapes** | (32, 8), (512, 512), (bs, 8, 512, 64), (bs, 8, 64, 512) |
| **LLM Shape** | [bs, 8, Mdim, Kdim]|[bs, 8, Kdim, Ndim]|[bs * 8, Mdim, Ndim]|[mul_91, 512, 64]|[mul_91, 64, 512]|[arg131_1, 8, 512, 512]|[512, 512]|[512, 512]|[512, 512]|[512, 512]|[2, 0, 1]|[arg131_1, 1, 1, 512]|-1|Mdim|Kdim|Ndim |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_bmm_epilogue_dynamic_reshape(self, dtype):
        bs = 5

```

---

### 58. `test_triton_template_generated_code_caching_bmm` — `inductor/test_max_autotune.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_max_autotune.py:2508–2516` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test compiles a function containing two batched matrix multiplications using Triton template code, runs it, and verifies that the compiled results match eager execution while checking the generate... |
| **Shapes** | (10, 10, 22), (10, 22, 30) |
| **LLM Shape** | 10, 22|22, 30|10, 11|8, 30|3, 30|10, 10, 22|10, 22, 30|10, 40|40, 30 |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_triton_template_generated_code_caching_bmm(self):
        def func_test1(x, y, z, m):
            a = torch.bmm(x, y)
            b = torch.bmm(z, m)
            return a, b

        a = torch.rand(10, 10, 22, device=GPU_TYPE)
        b = torch.rand(10, 22, 30, device=GPU_TYPE)

```

---

### 59. `test_bmm_large_batch_reversed_pid` — `inductor/test_native_matmul.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_native_matmul.py:227–242` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test compiles a function performing a large‑batch batch matrix multiplication (B=65537) and inspects the generated Triton kernel to ensure the program ID offsets are ordered as expected (z, y, x).... |
| **Shapes** | (65537, 16, 16)|(65537, 16, 16) |
| **LLM Shape** | (B, M, K)|(M * K, K, 1)|(B, K, N)|(K * N, N, 1)|65537, 16, 16, 16|128, 16, 128, 16 |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_bmm_large_batch_reversed_pid(self):
        def f(x, y):
            z = torch.bmm(x, y)
            return z

        B, M, K, N = 65537, 16, 16, 16
        x = rand_strided((B, M, K), (M * K, K, 1), device=GPU_TYPE)
        y = rand_strided((B, K, N), (K * N, N, 1), device=GPU_TYPE)

        f = torch.compile(f)
        code = run_and_get_triton_code(f, x, y)

        FileCheck().check("zoffset = tl.program_id(0)").check(
            "yoffset = tl.program_id(1)"
        ).check("xoffset = tl.program_id(2)").run(code)

```

---

### 60. `test_bmm_fusion_complex2` — `inductor/test_native_matmul.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_native_matmul.py:204–205` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test defines a function that gathers a sub‑tensor from B using index tensor Ak, performs a double‑contracted einsum between Av and the gathered tensor, and accumulates the result into out using in... |
| **Dtypes** | torch.int64 |
| **LLM Dtype** | torch.int64|torch.int64 |
| **Shapes** | (0, NA), (0, NB), (G), (G, P), (NA, M, N) |
| **LLM Shape** | (G, M, K, P)|(M * K * P, K * P, P, 1)|(NB, K, N)|(K * N, N, 1)|(G,)|(G, P)|(NA, M, N) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_bmm_fusion_complex2(self):
        # out[Ai[g],m,n] += Av[g,m,k,p] * B[Ak[g,p],k,n]
```

---

### 61. `test_bmm_fusion_complex1` — `inductor/test_native_matmul.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_native_matmul.py:177–178` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test defines a function that gathers slices from an input tensor and a weight tensor using integer index tensors, computes a three‑operand einsum (batched matrix multiplication with a vector), and... |
| **Dtypes** | torch.int64 |
| **LLM Dtype** | torch.int64 |
| **Shapes** | (0, N_in), (0, N_out), (0, N_w), (B), (B, I), (N_out, J) |
| **LLM Shape** | (B, M, K)|(M * K, K, 1)|(B, K, N)|(K * N, N, 1)|(B, M, K)|(M * K, K, 1)|(B, K, N)|(K * N, N, 1)|(N_in, R)|(R, 1)|(N_w, R, J)|(R * J, J, 1)|(B, I)|(I, 1)|(B, I)|(B,)|(B, I)|(N_out, J)|(G, M, K, P)|(M * K * P, K * P, P, 1)|(NB, K, N)|(K * N, N, 1)|(G,)|(G, P)|(NA, M, N)|(B, I, R)|(B, R, J)|(B, I)|(B, I, J)|(-1, J)|(-1, N_out) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_bmm_fusion_complex1(self):
        # Out[O[b,i],j] += Input[I[b,i],r] * Weight[W[b],r,j] * Val[b,i]
```

---

### 62. `test_bmm_dynamic_bm_stride` — `inductor/test_cpu_select_algorithm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_cpu_select_algorithm.py:3301–3306` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test defines a simple module that performs a batched matrix multiplication with a permuted weight tensor, marks certain dimensions of the inputs as dynamic, runs the module under TorchInductor, an... |
| **Dtypes** | torch.float |
| **LLM Dtype** | torch.float32|torch.float|dtype |
| **Shapes** | (Kdim, Mdim, bs), (bs, Mdim, Kdim) |
| **LLM Shape** | [arg131_1, 1, 1, 512]|[bs, 8, 512, 64]|[bs, 8, 64, 512]|[32, 8]|[bs, Mdim, Kdim]|[Kdim, Mdim, bs] |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_bmm_dynamic_bm_stride(self):
        bs = 8
        Mdim = 256
        Kdim = 64
        dtype = torch.float

```

---

### 63. `test_bmm_vertical_fusion` — `inductor/test_native_matmul.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_native_matmul.py:148–161` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test defines a function that performs a batched matrix multiplication followed by ReLU, then combines the result with a quadratic term. It runs the function on random strided GPU tensors and check... |
| **Shapes** | (128, 16, 128)|(128, 128, 16) |
| **LLM Shape** | (B, M, K)|(M * K, K, 1)|(B, K, N)|(K * N, N, 1)|128|16|128|16 |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_bmm_vertical_fusion(self):
        def f(x, y):
            z = torch.bmm(x, y)
            w = torch.nn.functional.relu(z)
            v = w + z * z
            return v

        B, M, K, N = 128, 16, 128, 16
        x = rand_strided((B, M, K), (M * K, K, 1), device=GPU_TYPE)
        y = rand_strided((B, K, N), (K * N, N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y))
        self._check_code(f, (x, y), 1, 1)

```

---

### 64. `test_fallback` — `inductor/test_fxir_backend.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_fxir_backend.py:209–213` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test compiles a function that uses torch.addbmm with random batch tensors and verifies that the compiled graph contains two ATen fallback operations (randint.low_out and addbmm.default). It checks... |
| **Shapes** | (2, 3, 5), (2, 5, 4), (3, 4) |
| **LLM Shape** | (2, 3, 5)|(2, 5, 4)|(3, 4)|(8, 8)|(8, 8) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_fallback(self):
        """
        Test a program that calls aten fallbacks.
        """

```

---

### 65. `test_batched_mm` — `inductor/test_mmdecomp.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_mmdecomp.py:144–165` |
| **Classification** | `NEEDS_REFACTOR` |
| **Module** | `inductor` |
| **Description** | The test verifies that batched matrix multiplication (torch.bmm) and batched add-matrix-multiplication (torch.baddbmm) produce numerically correct results across multiple batch sizes, matrix dimension... |
| **Dtypes** | torch.bfloat16, torch.float |
| **LLM Dtype** | torch.float|torch.bfloat16|torch.float32 |
| **LLM Shape** | (9, 1, 1, 4)|(bs, a1_0, a1_1)|(bs, a2_0, a2_1)|(bs, a1_0, a2_1)|(1, 32, 2)|(1, 2, 1)|(32, 8, 1)|(32, 1, 256)|(a1_0, a1_1)|(a2_0, a2_1)|(a1_0, a2_1) |
| **Device Classification** | device_generic |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_batched_mm(self, device, dtype, bs):
        fudge = 3
        rtol = default_rtol[dtype] * fudge
        atol = default_atol[dtype] * fudge

        for t_size in ts_list:
            ((a1_0, a1_1, a2_0, a2_1)) = t_size

            t1 = rand_math_tensor((bs, a1_0, a1_1), dtype=dtype, device=device)
            t2 = rand_math_tensor((bs, a2_0, a2_1), dtype=dtype, device=device)
            tadd = rand_math_tensor((bs, a1_0, a2_1), dtype=dtype, device=device)

            run_comp_nocomp(torch_bmm, t1, t2, rtol=rtol, atol=atol)

            for alpha in (0, 1, -1, 0.5, -0.5):
                for beta in (0, 1, -1, 0.5, -0.5):
                    run_comp_nocomp(
                        torch_baddbmm, tadd, t1, t2, alpha, beta, rtol=rtol, atol=atol
                    )

    @unittest.skipIf(not HAS_GPU, "GPU tests require triton")
    @config.patch(coordinate_descent_tuning=True)
```

---

### 66. `test_autotune_gemm_choice_validation` — `inductor/test_max_autotune.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_max_autotune.py:2699–2731` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test iterates over several matrix‑multiplication‑related ops (mm, addmm, bmm, baddbmm, mm_plus_mm) with and without max_autotune enabled, generates appropriate input tensors, runs the operation un... |
| **LLM Dtype** | torch.float32|torch.float32 |
| **Shapes** | (128), (128, 256), (256, 128), (4, 128, 128), (4, 128, 256), (4, 256, 128) |
| **LLM Shape** | (128, 256)|(256, 128)|(128)|(4, 128, 256)|(4, 256, 128)|(4, 128, 128)|(128, 256)|(256, 128) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_autotune_gemm_choice_validation(self, op, max_autotune):
        def generate_inputs_and_func(op_name):
            # Base config with just x and w
            base_inputs = [
                torch.randn(128, 256, device=GPU_TYPE),
                torch.randn(256, 128, device=GPU_TYPE),
            ]
            func = torch.mm
            if op_name == "mm":
                # default
                pass
            elif op_name == "addmm":
                # Add bias for addmm
                base_inputs = [torch.randn(128, device=GPU_TYPE)] + base_inputs
                func = torch.addmm
            elif op_name in ["bmm", "baddbmm"]:
                # Override for batch dimensions
                base_inputs[0] = torch.randn(4, 128, 256, device=GPU_TYPE)
                base_inputs[1] = torch.randn(4, 256, 128, device=GPU_TYPE)
                func = torch.bmm
                if op_name == "baddbmm":
                    # Add batch bias
                    base_inputs = [
                        torch.torch.randn(4, 128, 128, device=GPU_TYPE)
                    ] + base_inputs
                    func = torch.baddbmm
            elif op_name == "mm_plus_mm":
                # Add second matrix pair
                base_inputs += [
                    torch.randn(128, 256, device=GPU_TYPE),
                    torch.randn(256, 128, device=GPU_TYPE),
                ]

```

---

### 67. `test_fixed_layout_at_lowering` — `inductor/test_max_autotune.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_max_autotune.py:2803–2809` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that the max‑autotune path for various matrix multiplication kernels (mm, addmm, bmm, and a combination of two mm calls) correctly pads input tensors and preserves the expected outpu... |
| **Dtypes** | torch.bfloat16, torch.float32 |
| **LLM Dtype** | torch.bfloat16|torch.float32 |
| **Shapes** | (1490), (4608, 1490), (4608, 512), (512, 4608, 8), (8, 4608, 1490) |
| **LLM Shape** | (4608, 512)|(4608, 1490)|1490|(512, 4608, 8)|(8, 4608, 1490)|64|32, 128|128, 64|[1, 0]|[2, 0, 1] |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_fixed_layout_at_lowering(self):
        """
        Test that max-autotune with addmm/bmm/mm_plus_mm correctly handles
        padding and maintains correct output strides. Specifically, when matrix
        b with shape (4608, 1490) is padded, its stride should become 1536.
        """

```

---

### 68. `test_1_sized_with_0_strided` — `test_linalg.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_linalg.py:10506–10516` |
| **Classification** | `NEEDS_REFACTOR` |
| **Module** | `linalg` |
| **Description** | The test creates two batched tensors with custom strides (including a zero stride), performs a batch matrix multiplication using torch.bmm, and checks the result against a NumPy reference implementati... |
| **Dtypes** | torch.float, torch.double, torch.float32 |
| **LLM Dtype** | torch.float|torch.double|dtype=dtype|torch.float32|torch.float64 |
| **Shapes** | (8, 1, 64), (8, 64, 512) |
| **LLM Shape** | (8, 1, 64)|[8, 1, 64]|[64, 0, 1]|(8, 64, 512)|[8, 64, 512]|[64, 1, 512]|[0.052, -0.2115, 0.6913]|[-0.3229, -0.8374, 0.8391]|[0.2550, 0.8769, -0.4884]|[0.6063, 0.4343, -1.4166] |
| **Device Classification** | device_generic |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_1_sized_with_0_strided(self, device, dtype):
        a = make_tensor((8, 1, 64), dtype=dtype, device=device)
        a_strided = torch.as_strided(a, size=[8, 1, 64], stride=[64, 0, 1])
        b = make_tensor((8, 64, 512), dtype=dtype, device=device)
        b_strided = torch.as_strided(b, size=[8, 64, 512], stride=[64, 1, 512])
        res = torch.bmm(a_strided, b_strided)
        expect = torch.from_numpy(
            a_strided.cpu().numpy() @ b_strided.cpu().numpy()).to(device=device, dtype=dtype)
        self.assertEqual(expect, res)

    @onlyCUDA
```

---

### 69. `test_broadcast_fused_matmul` — `test_linalg.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_linalg.py:9710–9718` |
| **Classification** | `AS_IS` |
| **Module** | `linalg` |
| **Description** | The test iterates over several linear‑algebra functions (baddbmm, addbmm, addmm, addmv, addr) and verifies that broadcasting the first argument works correctly by comparing results with a manually exp... |
| **LLM Dtype** | float() |
| **Shapes** | (1, 8) |
| **LLM Shape** | [batch_dim, n_dim, p_dim]|[batch_dim, n_dim, m_dim]|[batch_dim, m_dim, p_dim]|[n_dim, p_dim]|[batch_dim, n_dim, m_dim]|[batch_dim, m_dim, p_dim]|[n_dim, p_dim]|[n_dim, m_dim]|[m_dim, p_dim]|[n_dim]|[n_dim, m_dim]|[m_dim]|[n_dim, m_dim]|[n_dim]|[m_dim]|t0_dims_full|t0_dims_small|t1_dims|t2_dims |
| **Device Classification** | device_generic |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_broadcast_fused_matmul(self, device):
        fns = ["baddbmm", "addbmm", "addmm", "addmv", "addr"]

        for fn in fns:
            batch_dim = random.randint(1, 8)
            n_dim = random.randint(1, 8)
            m_dim = random.randint(1, 8)
            p_dim = random.randint(1, 8)

```

---

### 70. `test_blas_nan_out` — `test_linalg.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_linalg.py:6898–6924` |
| **Classification** | `NEEDS_REFACTOR` |
| **Module** | `linalg` |
| **Description** | The test verifies that BLAS operations (torch.mv, torch.mm, torch.bmm) correctly overwrite NaN‑filled output tensors and produce results without NaNs, matching the default return behavior.... |
| **Dtypes** | torch.half, torch.bfloat16 |
| **LLM Dtype** | torch.half|torch.bfloat16|torch.bfloat16|torch.half|dtype|dtype|dtype|dtype|dtype|dtype|dtype|dtype|dtype|dtype|dtype|dtype|dtype|dtype|dtype|dtype|dtype|dtype|dtype|dtype |
| **Shapes** | (b, m, n), (b, n, p), (b, p, m), (m), (m, n), (n, p), (p, m) |
| **LLM Shape** | (m, n)|(m,)|(m,)|(n, p)|(b, m, n)|(b, p, m)|(b, n, p) |
| **Device Classification** | device_generic |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_blas_nan_out(self, device, dtype):
        # These functions should work correctly with NaN filled outputs,
        # but need special handling, see [NOTE: cpu_zero]
        b = 3
        n = 5
        m = 7
        p = 11

        # torch.mv
        nm = torch.randn((m, n), device=device).t()
        _m = torch.randn((), device=device).expand(m)
        _m_out = torch.full((m,), float('nan'), device=device)
        self.assertEqual(torch.mv(nm, _m), torch.mv(nm, _m, out=_m_out))
        self.assertEqual(0, torch.isnan(torch.mv(nm, _m)).sum())

        # torch.mm
        mp = torch.randn((p, m), device=device).t()
        np_out = torch.full((n, p), float('nan'), device=device)
        self.assertEqual(torch.mm(nm, mp), torch.mm(nm, mp, out=np_out))

        # torch.bmm
        bnm = torch.randn((b, m, n), device=device).transpose(1, 2)
        bmp = torch.randn((b, p, m), device=device).transpose(1, 2)
        bnp_out = torch.full((b, n, p), float('nan'), device=device)
        self.assertEqual(torch.bmm(bnm, bmp), torch.bmm(bnm, bmp, out=bnp_out))

    @onlyCPU  # not supported by CUBLAS
```

---

### 71. `test_combine_profiles` — `inductor/test_analysis.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_analysis.py:573–623` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test generates three separate GPU execution traces from distinct models (addmm, bmm, and pointwise), combines them using the inductor CLI '--combine' option, and verifies that the resulting combin... |
| **Dtypes** | torch.float, torch.float16 |
| **LLM Dtype** | dtype|torch.float|torch.float16|dtype |
| **Device Classification** | device_generic |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_combine_profiles(self, device, dtype):
        """
        Test combining multiple profiles into a single profile.
        """
        if device == "cpu" or torch.version.hip is not None:
            return

        # Create three different models to generate different traces
        om1 = _test_model(device, dtype, addmm=True, bmm=False)
        om2 = _test_model(device, dtype, addmm=False, bmm=True)
        om3 = _pointwise_test_model(device, dtype)

        # Generate three separate traces
        trace1, trace2 = trace_files()
        trace3 = f"{TMP_DIR}/trace3-{uuid.uuid4()}.json"
        combined_trace = f"{TMP_DIR}/combined-{uuid.uuid4()}.json"

        # Generate first trace
        torch._dynamo.reset()
        with fresh_inductor_cache():
            with torch.profiler.profile(record_shapes=True) as p1:
                om1()
        p1.export_chrome_trace(trace1)

        # Generate second trace
        torch._dynamo.reset()
        with fresh_inductor_cache():
            with torch.profiler.profile(record_shapes=True) as p2:
                om2()
        p2.export_chrome_trace(trace2)

        # Generate third trace
        torch._dynamo.reset()
        with fresh_inductor_cache():
            with torch.profiler.profile(record_shapes=True) as p3:
                om3()
        p3.export_chrome_trace(trace3)

        # Combine the three traces
        with patch(
            "sys.argv",
            [
                *prefix,
                "--combine",
                trace1,
                trace2,
                trace3,
                combined_trace,
            ],
        ):
            main()
```

---

### 72. `test_some_batched` — `inductor/test_mmdecomp.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_mmdecomp.py:276–293` |
| **Classification** | `NEEDS_REFACTOR` |
| **Module** | `inductor` |
| **Description** | The test runs batched matrix multiplication (torch.bmm) on small constant tensors repeated across a batch dimension, varying batch size, data type, and device. It compares the compiled and uncompiled ... |
| **Dtypes** | torch.float, torch.bfloat16, torch.int |
| **LLM Dtype** | torch.float|torch.bfloat16|torch.int|dtype |
| **LLM Shape** | [[[1], [2], [3], [4]]]|[[[1, 2, 3, 4]]]|bs|(a1_0, a1_1)|(a2_0, a2_1) |
| **Device Classification** | device_generic |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_some_batched(self, device, dtype, bs):
        # this Pytorch data type is not fully supported on cuda today
        # - unfortunately we can't skipIf because we don't see the actual params in skipIf
        if device.startswith(GPU_TYPE) and dtype == torch.int:
            return

        run_comp_nocomp(
            torch_bmm,
            init_tensor([[[1], [2], [3], [4]]] * bs, dtype=dtype, device=device),
            init_tensor([[[1, 2, 3, 4]]] * bs, dtype=dtype, device=device),
        )
        run_comp_nocomp(
            torch_bmm,
            init_tensor([[[1, 2, 3, 4]]] * bs, dtype=dtype, device=device),
            init_tensor([[[1], [2], [3], [4]]] * bs, dtype=dtype, device=device),
        )

    @parametrize("dtype", [torch.float, torch.bfloat16])
```

---

### 73. `test_batchmatmul` — `inductor/test_native_matmul.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_native_matmul.py:136–147` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test defines a function that performs a batched matrix multiplication (torch.bmm) on two 3‑D tensors and checks that the inductor compilation produces correct results and expected generated code o... |
| **LLM Dtype** | torch.long |
| **Shapes** | (256, 128, 128)|(256, 128, 128)|(128, 16, 128)|(128, 16, 128)|(128, 16, 128)|(128, 16, 128)|(128, 16, 128)|(128, 16, 128)|(32, 16, 128)|(128, 128) |
| **LLM Shape** | (M, K)|(K, N)|(M, K)|(K, N)|(M, K)|(K, N)|(M, K)|(K, N)|(M, N)|(B, M, K)|(M * K, K, 1)|(B, K, N)|(K * N, N, 1)|(B, M, K)|(M * K, K, 1)|(B, K, N)|(K * N, N, 1)|(B, M, K)|(M * K, K, 1)|(B, K, N)|(K * N, N, 1)|(B, M, K)|(M * K, K, 1)|(B, K, N)|(K * N, N, 1)|(B, M, K)|(M * K, K, 1)|(B, K, N)|(K * N, N, 1)|(N_in, R) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_batchmatmul(self):
        def f(x, y):
            z = torch.bmm(x, y)
            return z

        B, M, K, N = 256, 128, 128, 128
        x = rand_strided((B, M, K), (M * K, K, 1), device=GPU_TYPE)
        y = rand_strided((B, K, N), (K * N, N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y))
        self._check_code(f, (x, y), 1, 1)

```

---

### 74. `test_pad_batch` — `inductor/test_pad_mm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_pad_mm.py:392–406` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that torch.compile correctly pads batched matrix multiplication inputs and outputs for float16 tensors, ensuring alignment and that the compiled result matches the eager bmm output.... |
| **Dtypes** | torch.float16 |
| **LLM Dtype** | torch.float16 |
| **Shapes** | (batch_size, k, n), (batch_size, m, k) |
| **LLM Shape** | [a, b]|[4, 5]|[5, 6]|[a]|[batch_size, m, k]|[batch_size, k, n]|(3, 8, 16)|[25, 25] |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_pad_batch(self):
        m = 6
        n = 9
        k = 11
        batch_size = 3
        mat1 = torch.ones((batch_size, m, k), device=GPU_TYPE, dtype=torch.float16)
        mat2 = torch.ones((batch_size, k, n), device=GPU_TYPE, dtype=torch.float16)
        expected_alignment = get_alignment_size(mat1)

        if expected_alignment != 8:
            raise AssertionError("Alignment for float16 should be 8")
        if not can_pad(mat1, mat2, torch.ops.aten.bmm):
            raise AssertionError("This should pass the common padding criteria")

        @torch.compile()
```

---

### 75. `test_op` — `test_flop_counter.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_flop_counter.py:86–134` |
| **Classification** | `AS_IS` |
| **Module** | `flop_counter` |
| **Description** | The test runs a series of tensor operations (mm, bmm, addmm, baddbmm, conv2d, conv1d) inside a FlopCounterMode context and checks that the reported total FLOPs match manually computed expectations.... |
| **Shapes** | (4, 5)|(5, 6)|(3, 4, 5)|(3, 5, 6)|(4, 6)|(4, 1)|(6)|(3, 4, 6)|(2, 3, 6, 6)|(6, 3, 4, 4)|(2, 3, 6)|(6, 3, 4)|(4, 5)|(5, 6)|(7, 4, 6)|(7, 6, 7) |
| **LLM Shape** | (2, 32, 128, 64)|(2, 8, 128, 64)|(2, 5, 128, 64)|(4, 5)|(5, 6)|(4, 6)|(5, 6)|(4, 6)|(4, 5)|(5, 6)|(4, 1)|(4, 5)|(5, 6)|(6)|(4, 5)|(5, 6)|(3, 4, 5)|(3, 5, 6)|(3, 4, 6)|(3, 4, 5)|(3, 5, 6)|(2, 3, 6, 6)|(6, 3, 4, 4)|(2, 3, 6)|(6, 3, 4)|(4, 5)|(5, 6)|(7, 4, 6)|(7, 6, 7) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_op(self):
        with FlopCounterMode() as mode:
            torch.mm(T(4, 5), T(5, 6))
        # 4 * 6 * 2 * 5 = 240
        self.assertExpectedInline(get_total_flops(mode), """240""")

        with mode:
            torch.bmm(T(3, 4, 5), T(3, 5, 6))
        # 3 * 4 * 6 * 2 * 5 = 720
        self.assertExpectedInline(get_total_flops(mode), """720""")

        with mode:
            torch.addmm(T(4, 6), T(4, 5), T(5, 6))
            torch.addmm(T(4, 1), T(4, 5), T(5, 6))
            torch.addmm(T(6), T(4, 5), T(5, 6))

        # 4 * 6 * 2 * 5 = 240
        self.assertExpectedInline(get_total_flops(mode), """720""")

        with mode:
            torch.baddbmm(T(3, 4, 6), T(3, 4, 5), T(3, 5, 6))

        # 3 * 4 * 6 * 2 * 5 = 720
        self.assertExpectedInline(get_total_flops(mode), """720""")

        with mode:
            torch.conv2d(T(2, 3, 6, 6), T(6, 3, 4, 4), padding=1)

        # out_image_size = 2 * 5 * 5
        # kernel_size = 4 * 4
        # c_out = 6
        # c_in = 3
        # out_image_size * kernel_size * c_out * 2 * c_in

        # NB: I don't think this properly accounts for padding?
        self.assertExpectedInline(get_total_flops(mode), """28800""")

        with mode:
            torch.conv1d(T(2, 3, 6), T(6, 3, 4), padding=1)

        # out_image_size = 2 * 5
        # kernel_size = 4
        # c_out = 6
        # c_in = 3
        # out_image_size * kernel_size * c_out * 2 * c_in

        # NB: I don't think this properly accounts for padding?
        self.assertExpectedInline(get_total_flops(mode), """1440""")

```

---

### 76. `test_mm_and_friends` — `inductor/test_unbacked_symints.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_unbacked_symints.py:232–235` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test compiles a function that expands input tensors using a runtime scalar and then applies a matrix multiplication variant (mm, bmm, or addmm). It runs the compiled function on GPU and checks tha... |
| **LLM Dtype** | torch.int |
| **Shapes** | (1), (1, 32), (100), (32, 1) |
| **LLM Shape** | [u0, 1, 16]|[32, 16]|[1, 16]|[u0, 32]|[32, u0]|[10, u0, 32]|[10, 32, u0]|[1, 32]|[32, 1]|[u0]|[16, 16] |
| **Device Classification** | device_generic |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_mm_and_friends(self, device, torch_fn, coordinate_descent_tuning):
        if torch_fn == torch.addmm:
            torch_fn = functools.partial(torch_fn, torch.ones(1, device=device))

```

---

### 77. `test_baddmm` — `inductor/test_max_autotune.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_max_autotune.py:1318–1319` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test builds a module that computes bias + batch matrix multiplication (baddbmm) on FP16 tensors, compiles it with torch.compile using the max‑autotune mode, runs it, and verifies the compiled resu... |
| **Dtypes** | torch.float16 |
| **LLM Dtype** | torch.float16|torch.float16|torch.float16 |
| **Shapes** | (64, 1, 192), (64, 2048, 64), (64, 64, 192) |
| **LLM Shape** | 0, 256, 14, 14|256, 256|1|3|64, 64, 192|64, 1, 192|64, 2048, 64|1, 1|0, 0|1, 1|0, 0|1, 1|0, 0|1|3|64|1, 1|1, 1|0, 0|4, 3, 224, 224|224|224 |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_baddmm(self, search_space):
        class M(torch.nn.Module):
```

---

