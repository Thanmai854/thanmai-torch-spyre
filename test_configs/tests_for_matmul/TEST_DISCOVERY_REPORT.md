# Test Discovery Report

**Query:** tests for matmul

**Matches found:** 155 tests

**PyTorch root:** /Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch

**Torch-Spyre root:** /Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/torch-spyre

**Generated:** 2026-05-08T12:24:07.320794Z

---

## Matched Tests

### 1. `test_matmul` — `torch_np/test_binary_ufuncs.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/torch_np/test_binary_ufuncs.py:152–156` |
| **Classification** | `AS_IS` |
| **Module** | `torch_np` |
| **Description** | The test verifies that the custom `matmul` implementation produces the same result as NumPy's `matmul` for simple 1‑D inputs.... |
| **Shapes** | (1,)|(1,) |
| **LLM Shape** | [0.5]|[0.6] |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_matmul(self):
        assert_allclose(
            np.matmul([0.5], [0.6]), matmul([0.5], [0.6]), atol=1e-7, check_dtype=False
        )

```

---

### 2. `test_prioritize_cheaper_matmul` — `functorch/test_ac.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/functorch/test_ac.py:269–276` |
| **Classification** | `AS_IS` |
| **Module** | `functorch` |
| **Description** | The test defines a function that performs two matrix multiplications followed by a cosine and a sum reduction, then measures memory usage and FLOPs under eager execution and under a constrained memory... |
| **Shapes** | (1, 4)|(2, 2)|(4, 4)|(6, 2)|(2, 6)|(512, 512) |
| **LLM Shape** | (1, 4)|(2, 2)|(4, 4)|(6, 2)|(2, 6) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_prioritize_cheaper_matmul(self):
        def f(xs, ws):
            xs = [torch.mm(x, w).cos() for x, w in zip(xs, ws)]
            return sum(x.sum() for x in xs)

        x1, w1 = create_pair(1, 4)
        x2, w2 = create_pair(2, 2)

```

---

### 3. `test_weird_matmul_case` — `functorch/test_vmap.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/functorch/test_vmap.py:1605–1620` |
| **Classification** | `AS_IS` |
| **Module** | `functorch` |
| **Description** | The test creates two random tensors of shapes (5,2,2,2) and (5,7,2) and applies a nested vmap over torch.matmul to ensure the operation executes without error.... |
| **Shapes** | (5, 2, 2, 2), (5, 7, 2) |
| **LLM Shape** | (5, 2, 2, 2)|(5, 7, 2) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_weird_matmul_case(self):
        # Check that this doesn't crash.
        # https://github.com/pytorch/functorch/issues/417
        x = torch.randn(5, 2, 2, 2)
        y = torch.randn(5, 7, 2)

        vmap(vmap(torch.matmul, in_dims=(None, 0)))(x, y)

    @parametrize(
        "case",
        (
            (torch.clamp_min_, TensorFactory.randn),
            (torch.clamp_max_, TensorFactory.randn),
        ),
        name_fn=lambda x: x[0].__name__,
    )
```

---

### 4. `test_matmul_trivial` — `inductor/test_auto_chunker.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_auto_chunker.py:104–106` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test invokes a shared matrix‑multiplication verification routine with softmax disabled, exercising the auto‑chunker on a simple matmul scenario.... |
| **Dtypes** | torch.bfloat16 |
| **LLM Dtype** | torch.bfloat16|dtype |
| **Shapes** | (32, 1024, 768)|(32, 1024) |
| **LLM Shape** | B|T|C|V|32|1024|768|50257 |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_matmul_trivial(self):
        self.common_matmul_test(has_softmax=False)

```

---

### 5. `test_matmul_softmax_dynamic_shape` — `inductor/test_auto_chunker.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_auto_chunker.py:117–120` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test runs a matrix multiplication followed by a softmax operation with dynamic input shapes, using the common_matmul_test helper.... |
| **Dtypes** | torch.bfloat16 |
| **LLM Dtype** | torch.bfloat16 |
| **Shapes** | (32, 1024, 768)|(32, 1024) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_matmul_softmax_dynamic_shape(self):
        self.common_matmul_test(has_softmax=True, dynamic_shape=True)

    @config.patch("auto_chunker.num_chunk", 4)
```

---

### 6. `test_matmul_out` — `torch_np/numpy_tests/core/test_multiarray.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/torch_np/numpy_tests/core/test_multiarray.py:2674–2685` |
| **Classification** | `AS_IS` |
| **Module** | `torch_np` |
| **Description** | The test verifies that NumPy's matmul correctly handles the 'out' argument, including cases where the output array overlaps with the input memory.... |
| **LLM Dtype** | dtype |
| **Shapes** | (18) |
| **LLM Shape** | (2, 3, 3)|a[::-1, ...] |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_matmul_out(self):
        # overlapping memory
        a = np.arange(18).reshape(2, 3, 3)
        b = np.matmul(a, a)
        c = np.matmul(a, a, out=a)
        assert_(c is a)
        assert_equal(c, b)
        a = np.arange(18).reshape(2, 3, 3)
        c = np.matmul(a, a, out=a[::-1, ...])
        assert_(c.base is a.base)
        assert_equal(c, b)

```

---

### 7. `test_matmul_cpu` — `inductor/test_external_callables.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_external_callables.py:46–59` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test creates a 128x128 scaled identity matrix, compiles a MatMulModule with an external CPU matmul implementation, and compares its output to the same module compiled without the external call usi... |
| **Shapes** | (128, 128) |
| **LLM Shape** | 128, 128|128|128 |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_matmul_cpu(self):
        # 2I + 2I == (2I)(2I)
        x = torch.eye(128, 128) * 2
        opt_fn = torch.compile(
            MatMulModule(),
            options={"max_autotune": True, "external_matmul": [matmul_cpu]},
        )
        opt_fn_golden = torch.compile(MatMulModule(), options={"max_autotune": True})
        torch.testing.assert_close(
            opt_fn(x),
            opt_fn_golden(x),
            msg=f"torch.compile(..., external_matmul = {matmul_cpu}) failed",
        )

```

---

### 8. `test_matmul_cuda` — `inductor/test_external_callables.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_external_callables.py:80–99` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test compiles a simple matrix multiplication module for a CUDA device, once using a custom external matmul implementation (matmul_cuda) and once with the default implementation, then checks that b... |
| **Shapes** | (128, 128) |
| **LLM Shape** | 128|128 |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_matmul_cuda(self):
        device = torch.device(device_type)
        x = (torch.eye(128, 128) * 2).to(device=device)
        opt_fn = torch.compile(
            MatMulModule().to(device),
            options={"max_autotune": True, "external_matmul": [matmul_cuda]},
        )
        opt_fn_golden = torch.compile(
            MatMulModule().to(device), options={"max_autotune": True}
        )
        torch.testing.assert_close(
            opt_fn(x),
            opt_fn_golden(x),
            msg=f"torch.compile(..., external_matmul = {matmul_cuda}) failed",
        )


if __name__ == "__main__":
    run_tests()

```

---

### 9. `test_float32_matmul_precision` — `inductor/test_flex_attention.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_flex_attention.py:3755–3771` |
| **Classification** | `NEEDS_REFACTOR` |
| **Module** | `inductor` |
| **Description** | The test creates three float32 tensors, runs the FlexAttention operation with a custom score modifier under the highest float32 matmul precision, and compares gradients between eager and torch.compile... |
| **Dtypes** | torch.float32 |
| **LLM Dtype** | torch.float32|torch.float64 |
| **LLM Shape** | (2, 2, 128, 32)|(2, 2, 11, 4) |
| **Device Classification** | device_generic |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_float32_matmul_precision(self, device):
        make_tensor = functools.partial(
            torch.zeros,
            (2, 2, 128, 32),
            device=device,
            dtype=torch.float32,
            requires_grad=False,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()
        query.fill_(0.2)
        key.fill_(0.3)
        value.fill_(0.4)

        query.requires_grad = True
        key.requires_grad = True
        value.requires_grad = True

```

---

### 10. `test_matmul` — `test_namedtensor.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_namedtensor.py:1863–1942` |
| **Classification** | `AS_IS` |
| **Module** | `namedtensor` |
| **Description** | The test iterates over all supported devices and checks torch.matmul's named‑tensor name inference for a variety of input dimensionalities, confirming correct output names and proper error handling.... |
| **LLM Shape** | 7|3|2|2|5|0|N:7,A:3,B:2|7,2,5|N:7,A:2,B:5|N:3,A:3,B:3|M:3,A:3,B:3|None:3,N:3,B:3|A:2|B:2|A:3,C:2|B:2|A:5,C:3,D:2|C:2|A:2,B:3|A:3,B:2,D:5|A:3,B:2|A:2,B:3 |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_matmul(self):
        for device in get_all_device_types():
            # input tensors are less than 1D
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create(''), create('A:2')),
                maybe_raises_regex='at least 1D')
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create('A:2'), create('')),
                maybe_raises_regex='at least 1D')

            # 1D @ 1D
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create('A:2'), create('B:2')),
                expected_names=[])

            # ND @ 1D
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create('A:3,C:2'), create('B:2')),
                expected_names=['A'])
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create('A:5,C:3,D:2'), create('B:2')),
                expected_names=['A', 'C'])

            # 1D @ ND
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create('C:2'), create('A:2,B:3')),
                expected_names=['B'])
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create('C:2'), create('A:3,B:2,D:5')),
                expected_names=['A', 'D'])

            # 2D @ 2D
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create('A:3,B:2'), create('A:2,B:3')),
                expected_names=['A', 'B'])
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create('A:3,B:2'), create('B:2,A:5')),
                maybe_raises_regex='with duplicate names')

            # ND @ ND where N >= 2
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create('C:5,A:3,B:2'), create('A:2,B:3')),
                expected_names=['C', 'A', 'B'])
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create('C:5,A:3,B:2'), create('None:1,A:2,B:3')),
                expected_names=['C', 'A', 'B'])
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create('C:5,A:3,B:2'), create('None:2,None:1,A:2,B:3')),
                expected_names=[None, 'C', 'A', 'B'])

            # out=
            self._test_name_inference(
                out_fn(torch.matmul), device=device,
                args=(create('0'), create('N:7,A:3,B:2'), create('N:7,A:2,B:5')),
                expected_names=('N', 'A', 'B'))

            # duplicate names after mm
            self._test_name_inference(
                torch.bmm, device=device,
                args=(create('N:7,A:3,B:2'), create('N:7,B:2,A:5')),
                maybe_raises_regex='with duplicate names')

            # misalignment (batch dimension is getting contracted)
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create('N:3,A:3,B:3'), create('A:3,N:3,B:3')),
                maybe_raises_regex='do not match')

```

---

### 11. `test_matmul_mv` — `test_linalg.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_linalg.py:10429–10450` |
| **Classification** | `NEEDS_REFACTOR` |
| **Module** | `linalg` |
| **Description** | The test creates a 50,000 × 50,000 matrix of ones and a length‑50,000 random vector, performs a matrix‑vector multiplication with torch.matmul, and verifies the result.... |
| **Dtypes** | torch.float, torch.half, torch.bfloat16, torch.float16, torch.float32 |
| **LLM Dtype** | torch.float|torch.half|torch.bfloat16|dtype|dtype|dtype|torch.float32|torch.float16|torch.bfloat16 |
| **Shapes** | (n), (n, n) |
| **LLM Shape** | (n, n)|(n,)|(q_len, q_len)|(100000, 100000)|50_000|n |
| **Device Classification** | device_generic |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_matmul_mv(self, device, dtype):
        # Regression test for https://github.com/pytorch/pytorch/issues/150637
        # Such matrix will take more than 4Gb in memory

        # It is expected that we have very large errors when we are summing
        # 50,000 random numbers in low precision dtypes using 2 different
        # reduction paths so atol,rtol values above reflect this.
        n = 50_000
        A = torch.ones(n, n, dtype=dtype, device=device)
        B = torch.randn(n, dtype=dtype, device=device)
        C = torch.matmul(A, B)

        # Sanity Checks
        self.assertEqual(C.shape, (n,))
        self.assertEqual(C.dtype, dtype)
        self.assertFalse(torch.isnan(C).any())
        self.assertFalse(torch.isinf(C).any())

        self.assertEqual(C, B.sum().expand(B.shape))

    @onlyCUDA
    @largeTensorTest("40GB")
```

---

### 12. `test_broadcast_batched_matmul` — `test_linalg.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_linalg.py:9750–9756` |
| **Classification** | `AS_IS` |
| **Module** | `linalg` |
| **Description** | The test generates random matrix dimensions and batch shapes, creates tensors with matching and broadcastable batch dimensions, and verifies that torch.matmul correctly performs batched matrix multipl... |
| **LLM Dtype** | float|float|float|float|float |
| **Shapes** | (1, 3), (1, 8), (torch_result) |
| **LLM Shape** | [batch_dim, n_dim, p_dim]|[batch_dim, n_dim, m_dim]|[batch_dim, m_dim, p_dim]|[n_dim, p_dim]|[batch_dim, n_dim, m_dim]|[batch_dim, m_dim, p_dim]|[n_dim, p_dim]|[n_dim, m_dim]|[m_dim, p_dim]|[n_dim]|[n_dim, m_dim]|[m_dim]|[n_dim, m_dim]|[m_dim, p_dim]|[n_dim, p_dim]|[n_dim, m_dim]|[m_dim]|[n_dim]|[m_dim, p_dim]|[m_dim]|[p_dim]|[1, m_dim]|[m_dim, 1]|lhs_mat_dims|rhs_mat_dims|rhs_dims|lhs_dims|small_dims|dim0_dims|full_batch_dims + full_mat_dims |
| **Device Classification** | device_generic |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_broadcast_batched_matmul(self, device):
        n_dim = random.randint(1, 8)
        m_dim = random.randint(1, 8)
        p_dim = random.randint(1, 8)
        full_batch_dims = [random.randint(1, 3) for i in range(random.randint(1, 3))]
        (batch_dims_small, _, _) = self._select_broadcastable_dims(full_batch_dims)

```

---

### 13. `test_my_matmul` — `cpp_extensions/test_libtorch_agnostic.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/cpp_extensions/test_libtorch_agnostic.py:1709–1742` |
| **Classification** | `AS_IS` |
| **Module** | `cpp_extensions` |
| **Description** | The test verifies that the custom operation libtorch_agnostic.ops.my_matmul produces identical results to torch.matmul for a variety of input dimensionalities, including standard 2‑D matrix multiplica... |
| **Shapes** | (2, 3, 4), (2, 4, 5), (3, 4), (3, 5), (4), (4, 5) |
| **LLM Shape** | [3, 4]|[4, 5]|[3, 5]|[3, 4, 5]|[4, 5]|[3, 5]|[3, 4]|[4]|[4, 5]|[3, 4]|[4]|[2, 3, 4]|[2, 4, 5] |
| **Device Classification** | device_generic |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_my_matmul(self, device):
        """Test matmul op."""
        import libtorch_agn_2_9 as libtorch_agnostic

        # Test 2D x 2D matrix multiplication
        a = torch.randn(3, 4, device=device)
        b = torch.randn(4, 5, device=device)
        result = libtorch_agnostic.ops.my_matmul(a, b)
        expected = torch.matmul(a, b)
        self.assertEqual(result, expected)
        self.assertEqual(result.shape, torch.Size([3, 5]))

        # Test 1D x 2D (vector-matrix)
        v = torch.randn(4, device=device)
        m = torch.randn(4, 5, device=device)
        result_vm = libtorch_agnostic.ops.my_matmul(v, m)
        expected_vm = torch.matmul(v, m)
        self.assertEqual(result_vm, expected_vm)

        # Test 2D x 1D (matrix-vector)
        m2 = torch.randn(3, 4, device=device)
        v2 = torch.randn(4, device=device)
        result_mv = libtorch_agnostic.ops.my_matmul(m2, v2)
        expected_mv = torch.matmul(m2, v2)
        self.assertEqual(result_mv, expected_mv)

        # Test batched matmul
        batch_a = torch.randn(2, 3, 4, device=device)
        batch_b = torch.randn(2, 4, 5, device=device)
        result_batch = libtorch_agnostic.ops.my_matmul(batch_a, batch_b)
        expected_batch = torch.matmul(batch_a, batch_b)
        self.assertEqual(result_batch, expected_batch)

    @skipIfTorchVersionLessThan(2, 10)
```

---

### 14. `test_matmul` — `test_jit_fuser_te.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_jit_fuser_te.py:1714–1717` |
| **Classification** | `AS_IS` |
| **Module** | `jit_fuser_te` |
| **Description** | The test iterates over a set of data types, devices, and input shape pairs, creates tensors, computes a reference result with eager torch.matmul, JIT‑traces the same operation, and checks that the tra... |
| **Dtypes** | torch.bfloat16, torch.float16 |
| **LLM Dtype** | torch.float16|torch.bfloat16|dtype|self.dtypes |
| **LLM Shape** | [128, 128]|[10, 10]|[1, 16]|[16, 128]|[128]|[128, 128]|[3]|[3, 4]|[4]|[10, 3, 4]|[4]|[10, 4, 5]|[4, 5]|size_x|size_y|size |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_matmul(self):
        if self.dynamic_shapes:
            self.skipTest("don't run conv with dynamic shapes")

```

---

### 15. `test_merge_matmuls` — `test_fx_experimental.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_fx_experimental.py:1867–1872` |
| **Classification** | `AS_IS` |
| **Module** | `fx_experimental` |
| **Description** | The test verifies the torch.fx.experimental.merge_matmul graph transformation by constructing modules with multiple torch.matmul calls, applying the transformation, and checking that the optimized gra... |
| **Shapes** | (3, 3), (3, 4), (5, 4) |
| **LLM Shape** | (3, 3)|(3, 3)|(3, 4) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_merge_matmuls(self):
        """
        A collection of test cases for torch.fx.experimental.merge_matmul,
        a graph transformation that merges matrix multiplication operations.
        """
        # Utility function for counting matmuls for test assertions.
```

---

### 16. `test_profiler_matmul_dim_fp16_pattern` — `profiler/test_profiler.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/profiler/test_profiler.py:3439–3454` |
| **Classification** | `AS_IS` |
| **Module** | `profiler` |
| **Description** | The test profiles a matrix multiplication (x @ x) on several float16 CUDA tensors of varying shapes, applies the MatMulDimInFP16Pattern to the profiling data, and checks that the number of matched eve... |
| **Dtypes** | torch.float16 |
| **LLM Dtype** | torch.float16 |
| **Shapes** | (200, 200), (201, 201), (3, 200, 200), (3, 97, 97) |
| **LLM Shape** | (201, 201)|(3, 97, 97)|(200, 200)|(3, 200, 200) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_profiler_matmul_dim_fp16_pattern(self):
        cases = (
            (1, torch.randn((201, 201), device="cuda", dtype=torch.float16)),
            (1, torch.randn((3, 97, 97), device="cuda", dtype=torch.float16)),
            (0, torch.randn((200, 200), device="cuda", dtype=torch.float16)),
            (0, torch.randn((3, 200, 200), device="cuda", dtype=torch.float16)),
        )
        num_matched = []
        for _, x in cases:
            with profile(with_stack=True, record_shapes=True) as prof:
                x @ x
            pattern = MatMulDimInFP16Pattern(prof)
            num_matched.append(len(pattern.matched_events()))
        self.assertEqual(num_matched, [i for i, _ in cases])

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
```

---

### 17. `test_matmul` — `onnx/test_pytorch_onnx_onnxruntime.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/onnx/test_pytorch_onnx_onnxruntime.py:5352–5353` |
| **Classification** | `AS_IS` |
| **Module** | `onnx` |
| **Description** | The test defines a simple module that performs torch.matmul on two inputs and verifies its correctness when exported to ONNX and executed with ONNX Runtime. It runs the model with both floating‑point ... |
| **Shapes** | (10), (3, 4), (4, 5) |
| **LLM Shape** | (3, 4)|(4, 5)|(2, 3, 4)|(2, 4, 5)|(7, 3, 5)|[[1.0, 2.0, 3.0], [1.0, 1.0, 2.0]] |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_matmul(self):
        class MatmulModel(torch.nn.Module):
```

---

### 18. `test_matmul_batch` — `onnx/test_pytorch_onnx_onnxruntime.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/onnx/test_pytorch_onnx_onnxruntime.py:5365–5366` |
| **Classification** | `AS_IS` |
| **Module** | `onnx` |
| **Description** | The test defines a simple module that performs torch.matmul on two inputs and verifies its ONNX export and execution for batched matrix multiplication. It runs the model with both floating‑point tenso... |
| **Shapes** | (10), (2, 3, 4), (2, 4, 5) |
| **LLM Shape** | (2, 3, 4)|(2, 4, 5)|(2, 3, 4)|(2, 4, 5) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_matmul_batch(self):
        class MatmulModel(torch.nn.Module):
```

---

### 19. `test_profiler_fp32_matmul_pattern` — `profiler/test_profiler.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/profiler/test_profiler.py:3331–3340` |
| **Classification** | `AS_IS` |
| **Module** | `profiler` |
| **Description** | The test profiles a single FP32 matrix multiplication on CUDA and checks whether the FP32MatMulPattern correctly identifies the operation based on TF32 usage.... |
| **Shapes** | (100, 100) |
| **LLM Shape** | (100, 100)|(50, 50)|(100, 100)|(100,)|(10,) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_profiler_fp32_matmul_pattern(self):
        x = torch.ones((100, 100), device="cuda")
        with profile(with_stack=True) as prof:
            x = x @ x
        pattern = FP32MatMulPattern(prof)
        has_tf32 = 0 if pattern.skip else 1
        num_matched = len(pattern.matched_events())
        self.assertEqual(num_matched, has_tf32)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
```

---

### 20. `test_prioritize_cheaper_matmul2` — `functorch/test_ac.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/functorch/test_ac.py:308–315` |
| **Classification** | `AS_IS` |
| **Module** | `functorch` |
| **Description** | The test defines a function that performs three matrix multiplications followed by element‑wise cosine and sums the results, then measures memory usage and FLOPs under different memory budgets. It ver... |
| **Shapes** | (4, 4)|(6, 2)|(2, 6) |
| **LLM Shape** | (4, 4)|(6, 2)|(2, 6)|(1, 1) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_prioritize_cheaper_matmul2(self):
        def f(xs, ws):
            xs = [torch.mm(x, w).cos() for x, w in zip(xs, ws)]
            return sum(x.sum() for x in xs)

        data = [(4, 4), (6, 2), (2, 6)]
        xs, ws = zip(*[create_pair(a, b) for a, b in data])

```

---

### 21. `test_matmul_dup` — `inductor/test_external_callables.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_external_callables.py:60–79` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test creates a 128x128 scaled identity matrix, compiles a module with a duplicated external matmul callable, and checks that its output matches the same module compiled without external callables.... |
| **Shapes** | (128, 128) |
| **LLM Shape** | 128, 128|128|128 |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_matmul_dup(self):
        # 2I + 2I == (2I)(2I)
        x = torch.eye(128, 128) * 2
        # This should only register the first external call
        opt_fn = torch.compile(
            MatMulModule(),
            options={"max_autotune": True, "external_matmul": [matmul_dup, matmul_dup]},
        )
        opt_fn_golden = torch.compile(MatMulModule(), options={"max_autotune": True})
        torch.testing.assert_close(
            opt_fn(x),
            opt_fn_golden(x),
            msg=f"torch.compile(..., external_matmul = {matmul_dup}) failed",
        )

    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "CUDA and XPU not found")
    @unittest.skipIf(
        torch.cuda.is_available() and torch.cuda.get_device_capability() < (7, 0),
        "Triton does not support device capability < 7.0",
    )
```

---

### 22. `test_broadcast_matmul` — `onnx/test_pytorch_onnx_shape_inference.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/onnx/test_pytorch_onnx_shape_inference.py:199–226` |
| **Classification** | `AS_IS` |
| **Module** | `onnx` |
| **Description** | The test builds several ONNX graphs containing a MatMul node with constant inputs of varying ranks and shapes, runs ONNX shape inference, and checks that the inferred output shape matches the expected... |
| **LLM Dtype** | Float |
| **Shapes** | (2), (3, 1, 2, 1), (5, 1, 2) |
| **LLM Shape** | (5, 1, 2)|(3, 1, 2, 1)|(3, 5, 1, 1)|(2)|(3, 1, 1)|(5, 1)|() |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_broadcast_matmul(self):
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(5, 1, 2))
        constant_2 = self.insert_tensor_constant(g, torch.ones(3, 1, 2, 1))
        shape = g_op(g, "MatMul", constant, constant_2)
        self.run_test(g, shape.node(), expect_tensor("Float", shape=(3, 5, 1, 1)))

        # test when first input is of rank 1
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(2))
        constant_2 = self.insert_tensor_constant(g, torch.ones(3, 1, 2, 1))
        shape = g_op(g, "MatMul", constant, constant_2)
        self.run_test(g, shape.node(), expect_tensor("Float", shape=(3, 1, 1)))

        # test when second input is of rank 1
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(5, 1, 2))
        constant_2 = self.insert_tensor_constant(g, torch.ones(2))
        shape = g_op(g, "MatMul", constant, constant_2)
        self.run_test(g, shape.node(), expect_tensor("Float", shape=(5, 1)))

        # test when both inputs are of rank 1
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(2))
        constant_2 = self.insert_tensor_constant(g, torch.ones(2))
        shape = g_op(g, "MatMul", constant, constant_2)
        self.run_test(g, shape.node(), expect_tensor("Float", shape=()))

```

---

### 23. `test_matmul_raises` — `torch_np/numpy_tests/core/test_multiarray.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/torch_np/numpy_tests/core/test_multiarray.py:5694–5700` |
| **Classification** | `AS_IS` |
| **Module** | `torch_np` |
| **Description** | The test verifies that invoking the matmul operator with unsupported int8 scalar inputs raises an exception.... |
| **LLM Dtype** | bool|int8 |
| **LLM Shape** | (2, 0)|0|(3, 4, 5)|(-2, -1)|(-1, -2)|(1, 2)|(0, 1)|(4, 4, 3)|(3, 4, 4)|(4, 5)|(1, 80) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_matmul_raises(self):
        assert_raises(
            (RuntimeError, TypeError, ValueError), self.matmul, np.int8(5), np.int8(5)
        )

    @xpassIfTorchDynamo_np  # (reason="torch supports inplace matmul, and so do we")
    @skipif(numpy.__version__ >= "1.26", reason="This is fixed in numpy 1.26")
```

---

### 24. `test_matmul` — `dynamo/cpython/3_13/test_operator.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/dynamo/cpython/3_13/test_operator.py:284–288` |
| **Classification** | `AS_IS` |
| **Module** | `dynamo` |
| **Description** | The test verifies error handling for the module's matmul function and ensures that the Python @ operator correctly invokes a custom __matmul__ implementation when Dynamo's graph‑break checking is disa... |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_matmul(self):
        operator = self.module
        self.assertRaises(TypeError, operator.matmul)
        self.assertRaises(TypeError, operator.matmul, 42, 42)
        with torch._dynamo.error_on_graph_break(False):
```

---

### 25. `test_matmul_axes` — `torch_np/numpy_tests/core/test_multiarray.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/torch_np/numpy_tests/core/test_multiarray.py:5718–5732` |
| **Classification** | `AS_IS` |
| **Module** | `torch_np` |
| **Description** | The test creates a 3‑dimensional tensor and exercises np.matmul with explicit axes specifications, verifies the resulting shapes, checks that swapping axes yields an equivalent tensor, and also tests ... |
| **LLM Dtype** | np.int8|np.float64 |
| **Shapes** | (3) |
| **LLM Shape** | (3, 4, 5)|(3, 4, 4)|(4, 4, 3)|(4, 5)|(1, 80)|(2, 2) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_matmul_axes(self):
        a = np.arange(3 * 4 * 5).reshape(3, 4, 5)
        c = np.matmul(a, a, axes=[(-2, -1), (-1, -2), (1, 2)])
        if c.shape != (3, 4, 4):
            raise AssertionError(f"shape mismatch: {c.shape} != (3, 4, 4)")
        d = np.matmul(a, a, axes=[(-2, -1), (-1, -2), (0, 1)])
        if d.shape != (4, 4, 3):
            raise AssertionError(f"shape mismatch: {d.shape} != (4, 4, 3)")
        e = np.swapaxes(d, 0, 2)
        assert_array_equal(e, c)
        f = np.matmul(a, np.arange(3), axes=[(1, 0), (0), (0)])
        if f.shape != (4, 5):
            raise AssertionError(f"shape mismatch: {f.shape} != (4, 5)")


```

---

### 26. `test_matmul_exception_multiply` — `torch_np/numpy_tests/core/test_multiarray.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/torch_np/numpy_tests/core/test_multiarray.py:5627–5628` |
| **Classification** | `AS_IS` |
| **Module** | `torch_np` |
| **Description** | The test verifies that the matmul operation raises a TypeError when the input arrays contain objects that lack a __mul__ method.... |
| **LLM Dtype** | bool|np.uint8|np.int8 |
| **Shapes** | (3, 3) |
| **LLM Shape** | (3, 3)|(2, 0)|(0,)|(4, 5)|(5, 4)|(2, 0) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_matmul_exception_multiply(self):
        # test that matmul fails if `__mul__` is missing
```

---

### 27. `test_matmul_bool` — `torch_np/numpy_tests/core/test_multiarray.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/torch_np/numpy_tests/core/test_multiarray.py:5648–5672` |
| **Classification** | `AS_IS` |
| **Module** | `torch_np` |
| **Description** | The test verifies boolean matrix multiplication behavior, including value range, equivalence with dot, and handling of empty dimensions. It checks that the result contains only False/True values and t... |
| **Dtypes** | torch.bool |
| **LLM Dtype** | bool|np.uint8|np.int8 |
| **Shapes** | (0), (2), (2, 0), (4, 5) |
| **LLM Shape** | (3, 3)|(2, 0)|(0,)|(4, 5)|(5, 4)|(2,)|(3,) |
| **Device Classification** | no_device |
| **Match Score** | 25 |

**Source snippet:**
```python
    def test_matmul_bool(self):
        # gh-14439
        a = np.array([[1, 0], [1, 1]], dtype=bool)
        if np.max(a.view(np.uint8)) != 1:
            raise AssertionError(f"max mismatch: {np.max(a.view(np.uint8))} != 1")
        b = np.matmul(a, a)
        # matmul with boolean output should always be 0, 1
        if np.max(b.view(np.uint8)) != 1:
            raise AssertionError(f"max mismatch: {np.max(b.view(np.uint8))} != 1")

        # rg = np.random.default_rng(np.random.PCG64(43))
        # d = rg.integers(2, size=4*5, dtype=np.int8)
        # d = d.reshape(4, 5) > 0
        np.random.seed(1234)
        d = np.random.randint(2, size=(4, 5)) > 0

        out1 = np.matmul(d, d.reshape(5, 4))
        out2 = np.dot(d, d.reshape(5, 4))
        assert_equal(out1, out2)

        c = np.matmul(np.zeros((2, 0), dtype=bool), np.zeros(0, dtype=bool))
        if np.any(c):
            raise AssertionError("c should be all False")


```

---

### 28. `test_matmul_triton_kernel_benchmark` — `inductor/test_kernel_benchmark.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_kernel_benchmark.py:181–188` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test compiles and runs a matrix multiplication followed by ReLU using the Triton backend with autotuning enabled, then checks that the expected compiled kernels were generated.... |
| **Dtypes** | torch.float16 |
| **LLM Dtype** | torch.float16|torch.float32 |
| **Shapes** | (M, K), (N, K) |
| **LLM Shape** | 12544|256|64|(M, K)|(N, K)|(M, N, K)|1000|1000|10|(M, K)|(2, 3)|(1024,)|(2, K_2)|(K_2, 1)|(K, N)|(1, K)|(M, K_2) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_matmul_triton_kernel_benchmark(self):
        M = 12544
        N = 256
        K = 64
        a = torch.rand(M, K, dtype=torch.float16, device=GPU_TYPE)
        b = torch.rand(N, K, dtype=torch.float16, device=GPU_TYPE).t()

        @torch.compile
```

---

### 29. `test_two_inputs_matmul` — `dynamo/test_fx_graph_runnable.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/dynamo/test_fx_graph_runnable.py:313–320` |
| **Classification** | `AS_IS` |
| **Module** | `dynamo` |
| **Description** | The test defines a two‑argument function that performs a matrix multiplication followed by a ReLU, compiles it with torch.compile, runs it on random inputs, and verifies the compiled payload execution... |
| **LLM Dtype** | torch.float16 |
| **Shapes** | (2, 3), (3, 4) |
| **LLM Shape** | (4096,)|(2, 3)|(3, 4)|(5,)|(10, 12)|(5, 1)|(1, 8)|(3, 8)|(16, 12)|(7, 10) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_two_inputs_matmul(self):
        def f(a, b):
            return (a @ b).relu()

        a, b = torch.randn(2, 3), torch.randn(3, 4)
        torch.compile(f)(a, b)
        self._exec_and_verify_payload()

```

---

### 30. `test_constant_fold_transpose_matmul` — `onnx/test_utility_funs.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/onnx/test_utility_funs.py:444–445` |
| **Classification** | `AS_IS` |
| **Module** | `onnx` |
| **Description** | The test defines a simple module that multiplies an input tensor with the transpose of a constant weight matrix, exports it to ONNX, and checks that the transpose operation has been constant‑folded aw... |
| **Shapes** | (2, 3), (5, 3) |
| **LLM Shape** | (5, 3, 7)|(1, 3, 3)|(2, 3)|(5, 3)|(4, 5)|(1, -1, 1, 1) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_constant_fold_transpose_matmul(self):
        class MatMulNet(torch.nn.Module):
```

---

### 31. `test_matmul_layer_norm` — `inductor/test_torchinductor.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_torchinductor.py:6382–6399` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test creates a random 3‑D input tensor and a weight matrix, performs a batched matrix multiplication followed by a LayerNorm, and runs the compiled function through the common test harness.... |
| **LLM Dtype** | torch.complex64|check_lowp=False|check_lowp=False|check_lowp=False|check_lowp=False |
| **Shapes** | (batch_size, seq_length, hidden_size), (hidden_size, hidden_size) |
| **LLM Shape** | [16, 32]|32|[1, 1, 10]|batch_size=32|seq_length=50|hidden_size=256|[batch_size, seq_length, hidden_size]|[hidden_size, hidden_size]|[16, 32]|[32, 16]|[16, 32]|[16, 32] |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_matmul_layer_norm(self):
        batch_size = 32
        seq_length = 50
        hidden_size = 256

        inp = torch.randn(
            batch_size,
            seq_length,
            hidden_size,
            requires_grad=True,
            device=self.device,
        )
        weight = torch.randn(
            hidden_size, hidden_size, requires_grad=True, device=self.device
        )

        layer_norm = torch.nn.LayerNorm(hidden_size, device=self.device)

```

---

### 32. `test_sp24_matmuls_bmm` — `test_sparse_semi_structured.py`

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

### 33. `test_sp24_matmuls_mat_vec` — `test_sparse_semi_structured.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_sparse_semi_structured.py:1016–1033` |
| **Classification** | `AS_IS` |
| **Module** | `sparse_semi_structured` |
| **Description** | The test creates a random 64x128 half‑precision matrix and a 128‑element vector on CUDA, converts the matrix to a sparse semi‑structured format, and verifies that performing a matrix‑vector multiplica... |
| **Dtypes** | torch.float16 |
| **LLM Dtype** | torch.float16 |
| **Shapes** | (128), (64, 128) |
| **LLM Shape** | [64, 128]|[128]|[5, 6, 128] |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_sp24_matmuls_mat_vec(self) -> None:
        a = torch.randn([64, 128], device="cuda", dtype=torch.float16)
        b = torch.randn([128], device="cuda", dtype=torch.float16)
        a_m = sparse24_largest_mask_2d(a)
        a_s = to_sparse_semi_structured(a)

        with pytest.raises(NotImplementedError):
            torch.testing.assert_close(a_s @ b, (a * a_m) @ b, **atol_rtol_kw[a.dtype])

    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        not torch.backends.cusparselt.is_available(),
        "cuSPARSELt not available",
    )
    @unittest.skipIf(
        "RelWithAssert" in torch.__config__.show(),
        "failing in debug build, see https://github.com/pytorch/pytorch/pull/165158 for context",
    )
```

---

### 34. `test_sp24_matmuls` — `test_sparse_semi_structured.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_sparse_semi_structured.py:971–1015` |
| **Classification** | `AS_IS` |
| **Module** | `sparse_semi_structured` |
| **Description** | The test creates random CUDA tensors a (64×1024) and b (1024×256), generates 2:4 sparsity masks, packs them into SparseSemiStructuredTensorCUTLASS objects, and verifies that matrix multiplications inv... |
| **LLM Dtype** | dtype|torch.float16 |
| **Shapes** | (K, N), (M, K) |
| **LLM Shape** | [M, K]|[K, N]|[64, 256, 1024]|[64, 128]|[128]|a.shape|b.shape |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_sp24_matmuls(self, dtype) -> None:
        M, N, K = 64, 256, 1024
        a = torch.randn([M, K], device="cuda", dtype=dtype)
        b = torch.randn([K, N], device="cuda", dtype=dtype)
        a_m = sparse24_largest_mask_2d(a)
        b_m = sparse24_largest_mask_2d(b)
        (packed, meta, packed_t, meta_t, bitmask) = torch._sparse_semi_structured_tile(
            a
        )
        a_s = SparseSemiStructuredTensorCUTLASS(
            a.shape,
            packed=packed,
            meta=meta,
            packed_t=packed_t,
            meta_t=meta_t,
            compressed_swizzled_bitmask=bitmask,
        )
        (packed, meta, packed_t, meta_t, bitmask) = torch._sparse_semi_structured_tile(
            b
        )
        b_s = SparseSemiStructuredTensorCUTLASS(
            b.shape,
            packed=packed,
            meta=meta,
            packed_t=packed_t,
            meta_t=meta_t,
            compressed_swizzled_bitmask=bitmask,
        )

        torch.testing.assert_close(a_s @ b, (a * a_m) @ b, rtol=1e-1, atol=1.5e-1)
        torch.testing.assert_close(a @ b_s, a @ (b * b_m), rtol=1e-1, atol=1.5e-1)
        torch.testing.assert_close(
            a @ a_s.t(), a @ (a * a_m).t(), rtol=1e-1, atol=1.5e-1
        )
        torch.testing.assert_close(a_s.t() @ a, (a * a_m).t() @ a, rtol=1e-1, atol=1e-1)

    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @unittest.skipIf(
        not torch.backends.cusparselt.is_available(),
        "cuSPARSELt not available",
    )
    @unittest.skipIf(
        "RelWithAssert" in torch.__config__.show(),
        "failing in debug build, see https://github.com/pytorch/pytorch/pull/165158 for context",
    )
```

---

### 35. `test_broadcast_fused_matmul` — `test_linalg.py`

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
| **Match Score** | 20 |

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

### 36. `test_matmul_bandwidth_computation` — `inductor/test_kernel_benchmark.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_kernel_benchmark.py:220–226` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test compiles a function that performs a matrix multiplication followed by an elementwise multiplication (squaring) on 1000x1000 intermediate results, then checks that the reported memory bandwidt... |
| **LLM Dtype** | torch.float32 |
| **Shapes** | (K, N), (M, K) |
| **LLM Shape** | M, K|K, N|M, N, K|1000, 1000, 10|M, K|K, N|M, N|5, 1000000|1000, 20, 1000 |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_matmul_bandwidth_computation(self):
        """
        The test does a matmul and then mul. Without max-autotune, we use
        the matmul in aten. So there is a single triton kernel for mul.
        The kernel we generated is like:

            @triton.jit
```

---

### 37. `test_matmul_exception_add` — `torch_np/numpy_tests/core/test_multiarray.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/torch_np/numpy_tests/core/test_multiarray.py:5638–5639` |
| **Classification** | `AS_IS` |
| **Module** | `torch_np` |
| **Description** | The test verifies that the matrix multiplication operation raises a TypeError when the input objects lack an __add__ method.... |
| **LLM Dtype** | bool|np.uint8|np.int8 |
| **Shapes** | (3, 3) |
| **LLM Shape** | (3, 3)|(2, 0)|(0,)|(4, 5)|(5, 4)|(2,) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_matmul_exception_add(self):
        # test that matmul fails if `__add__` is missing
```

---

### 38. `test_matmul` — `inductor/test_padding.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_padding.py:451–465` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test measures and compares the GPU latency of a matrix multiplication using torch.compile for two tensor configurations: one with a contiguous "good" shape and one with a deliberately mis‑aligned ... |
| **Dtypes** | torch.float16 |
| **LLM Dtype** | torch.float16 |
| **Shapes** | (30522, 768), (30528, 768), (8192, 30528), (8192, 768) |
| **LLM Shape** | (8192, 30522)|(30522, 768)|(8192, 768)|(8192, 30528)|(30528, 768)|(30522, 1)|(8192, 30522)|(30528, 1) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_matmul(self):
        """
        Latency with good and bad shapes: 1.705 v.s. 2.625
        """
        x_good_shape = torch.randn(8192, 30528, dtype=torch.float16)
        weight_good_shape = torch.randn(30528, 768, dtype=torch.float16)
        out_good_shape = torch.randn(8192, 768, dtype=torch.float16)

        # Using stride (30522, 1) does not make a difference here.
        x_bad_shape = rand_strided(
            (8192, 30522), (30528, 1), device=GPU_TYPE, dtype=torch.float16
        )
        weight_bad_shape = torch.randn(30522, 768, dtype=torch.float16)
        out_bad_shape = torch.randn(8192, 768, dtype=torch.float16)

```

---

### 39. `test_permute_matmul` — `test_linalg.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_linalg.py:10342–10347` |
| **Classification** | `AS_IS` |
| **Module** | `linalg` |
| **Description** | The test creates two all‑ones tensors, permutes the dimensions of the first, performs a batched matrix multiplication with the second, and checks the resulting tensor's min, max, and sum values.... |
| **Shapes** | (2, 5, 24, 24), (3, 2, 5, 24, 24) |
| **LLM Shape** | [2, 5, 24, 24]|[3, 2, 5, 24, 24] |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_permute_matmul(self):
        a = torch.ones([2, 5, 24, 24])
        b = torch.ones([3, 2, 5, 24, 24])
        c = a.permute(0, 1, 3, 2).matmul(b)
        self.assertEqual([c.min(), c.max(), c.sum()], [24, 24, 414720])

```

---

### 40. `test_matmul` — `inductor/test_nv_universal_gemm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_nv_universal_gemm.py:112–119` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that matrix multiplication works correctly on CUDA for float16 and bfloat16 tensors under a variety of memory layouts, including contiguous, offset‑aligned, view‑based, and padded st... |
| **Dtypes** | torch.float16, torch.bfloat16 |
| **LLM Dtype** | torch.float16|torch.bfloat16 |
| **Shapes** | (rows, cols) |
| **LLM Shape** | (513, 512, 512)|(rows, cols)|(cols, 1)|(rows, cols)|(row_pitch, 1)|512|513|128|520|8 |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_matmul(self, dtype, layout_a, layout_b):
        """Test matmul with various dtypes and tensor layouts.

        M=513 tests that non-divisible M dimension works
        (only N and K must be divisible by 16).
        """
        m, n, k = 513, 512, 512

```

---

### 41. `test_batchmatmul` — `inductor/test_native_matmul.py`

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
| **Match Score** | 20 |

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

### 42. `test_matmul_fp16` — `inductor/test_native_matmul.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_native_matmul.py:80–95` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test defines a matrix multiplication of a float16 tensor with a float32 tensor (cast to float16) and verifies the result against the eager reference with a relaxed tolerance, also checking the gen... |
| **Dtypes** | torch.float16, torch.float32 |
| **LLM Dtype** | torch.float16|torch.float32 |
| **LLM Shape** | (M, K)|(K, 1)|(K, N)|(N, 1)|(M,)|(1,)|(1,)|(0,)|(M, N) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_matmul_fp16(self):
        def f(x, y):
            z = x @ y.to(x.dtype)
            return z

        M, K, N = 128, 128, 128
        x = rand_strided((M, K), (K, 1), dtype=torch.float16, device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), dtype=torch.float32, device=GPU_TYPE)

        # _check_equal calls torch._dynamo.utils.same with kwarg tol=1e-4.
        # For fp16 dtype, torch.allclose() defaults to atol=1e-3 rtol=1e-5,
        # but same() uses the single value to assign both, resulting in
        # Accuracy failed: allclose not within tol=0.0001.
        self._check_equal(f, (x, y), tol=1e-3)
        self._check_code(f, (x, y), 1, 1)

```

---

### 43. `test_matmul` — `inductor/test_native_matmul.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_native_matmul.py:44–55` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test defines a simple function that performs a matrix multiplication (x @ y) on two 128x128 tensors with custom non‑contiguous strides, then verifies that the compiled Inductor output matches the ... |
| **Dtypes** | torch.float16|torch.float32 |
| **LLM Dtype** | torch.float16|torch.float32 |
| **Shapes** | (128, 128)|(128, 128)|(128,)|(1,) |
| **LLM Shape** | (M, K)|(K, 1)|(K, N)|(N, 1)|(M,)|(1,)|(0,)|(1,)|(M, K)|(K, 1)|(K, N)|(N, 1)|(M, K)|(K, 1)|(K, N)|(N, 1)|(M, K)|(K, 1)|(K, N)|(N, 1)|128|128|128|128|128|128|128|128|128|62|62|62 |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_matmul(self):
        def f(x, y):
            z = x @ y
            return z

        M, K, N = 128, 128, 128
        x = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y))
        self._check_code(f, (x, y), 1, 1)

```

---

### 44. `test_matmul_tracing` — `test_fx.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_fx.py:420–422` |
| **Classification** | `AS_IS` |
| **Module** | `fx` |
| **Description** | The test verifies that Torch FX's symbolic tracing correctly captures the matrix multiplication (the @ operator) when a constant vector is multiplied on either the left or right side of an input tenso... |
| **Shapes** | (3) |
| **LLM Shape** | (3)|(10)|(3, 4) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_matmul_tracing(self):
        const = torch.randn(3)

```

---

### 45. `test_evt_reuse_matmul_input` — `inductor/test_cutlass_backend.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_cutlass_backend.py:2281–2282` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test defines a simple model that performs a matrix multiplication, applies ReLU, feeds the result into a parametrized operation (op) with extra arguments, applies ReLU again, and finally adds the ... |
| **Dtypes** | torch.half |
| **LLM Dtype** | half |
| **Shapes** | (1024, 1024)|(512, 512)|(1024, 64)|(128, 256) |
| **LLM Shape** | (1024, 1024)|(512, 512)|(1024, 64)|(128, 256)|(M, N)|(N, N)|(1024, 512) |
| **Device Classification** | device_generic |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_evt_reuse_matmul_input(self, op):
        class TestModel(torch.nn.Module):
```

---

### 46. `test_matmul_lower_precision` — `test_mkldnn.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_mkldnn.py:1594–1599` |
| **Classification** | `AS_IS` |
| **Module** | `mkldnn` |
| **Description** | The test verifies that lower‑precision (float16 and bfloat16) matrix multiplication using MKLDNN produces results numerically consistent with float32 reference computations. It also checks that tensor... |
| **Dtypes** | torch.float16, torch.bfloat16 |
| **LLM Dtype** | torch.float16|torch.bfloat16|torch.float16|torch.bfloat16|dtype|dtype|dtype|dtype|dtype |
| **Shapes** | (64, 1, 33), (64, 33, 256), (shape1), (shape2) |
| **LLM Shape** | [64, 1, 33]|[64, 1, 33]|[33, 3, 1]|64|33|256|(33, 77)|(77, 22)|(128, 256)|(256, 10)|(7, 300)|(300, 3)|(1, 100)|(100, 60)|(100, 1)|(1, 100)|(20, 54, 78)|(20, 78, 10)|(1, 300, 1)|(1, 1, 300) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_matmul_lower_precision(self, dtype):
        support_check = {
            torch.bfloat16: torch.ops.mkldnn._is_mkldnn_bf16_supported,
            torch.float16: torch.ops.mkldnn._is_mkldnn_fp16_supported,
        }

```

---

### 47. `test_scan_downstream_scan_matmul` — `functorch/test_control_flow.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/functorch/test_control_flow.py:2140–2145` |
| **Classification** | `AS_IS` |
| **Module** | `functorch` |
| **Description** | The test verifies that a downstream matrix multiplication applied to the result of a functorch `scan` operation produces the correct values across different compile modes, reversal settings, devices, ... |
| **Shapes** | (2, 5), (3, 10, 2), (3, 2) |
| **LLM Shape** | (3, 10, 2)|(3, 2)|(2, 5)|(1, 10, 2) |
| **Device Classification** | device_generic |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_scan_downstream_scan_matmul(self, compile_mode, reverse, device, autograd):
        inp = torch.randn(3, 10, 2, device=device, requires_grad=autograd)
        init = torch.randn(3, 2, device=device, requires_grad=autograd)

        for ind in range(2):
            # Chain with matmul
```

---

### 48. `test_matmul_inplace_2` — `torch_np/numpy_tests/core/test_multiarray.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/torch_np/numpy_tests/core/test_multiarray.py:5710–5717` |
| **Classification** | `AS_IS` |
| **Module** | `torch_np` |
| **Description** | The test creates two 3x3 identity matrices and verifies that attempting an in‑place matrix multiplication on them raises a TypeError.... |
| **LLM Dtype** | np.int8|np.float64|np.typecodes["AllInteger"]|np.typecodes["AllFloat"]|dt |
| **Shapes** | (3, 3)|(3, 4, 5)|(1, 80)|(2, 2) |
| **LLM Shape** | (3,)|(3, 4, 5)|(3, 4, 4)|(4, 4, 3)|(4, 5)|(1, 80)|(2,)|(1, 2)|(3, 4) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_matmul_inplace_2(self):
        a = np.eye(3)
        b = np.eye(3)

        assert_raises(TypeError, operator.imatmul, a, b)
        assert_raises(TypeError, exec, "a @= b", globals(), locals())

    @xpassIfTorchDynamo_np  # (reason="matmul_axes")
```

---

### 49. `test_matmul_inplace` — `torch_np/numpy_tests/core/test_multiarray.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/torch_np/numpy_tests/core/test_multiarray.py:5701–5709` |
| **Classification** | `AS_IS` |
| **Module** | `torch_np` |
| **Description** | The test verifies that attempting an in-place matrix multiplication (a @= b) on NumPy arrays raises a TypeError.... |
| **Dtypes** | float64 |
| **LLM Dtype** | np.int8|np.float64 |
| **Shapes** | (3,)|(3, 4, 5)|(3, 4, 4)|(4, 4, 3)|(4, 5)|(1, 80) |
| **LLM Shape** | (3,)|(3, 4, 5)|(3, 4, 4)|(4, 4, 3)|(4, 5)|(1, 80)|(2,) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_matmul_inplace(self):
        # It would be nice to support in-place matmul eventually, but for now
        # we don't have a working implementation, so better just to error out
        # and nudge people to writing "a = a @ b".
        a = np.eye(3)
        b = np.eye(3)
        assert_raises(TypeError, a.__imatmul__, b)

    @xfail  # XXX: what's up with exec under Dynamo
```

---

### 50. `test_matmul1` — `dynamo/test_misc.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/dynamo/test_misc.py:779–785` |
| **Classification** | `AS_IS` |
| **Module** | `dynamo` |
| **Description** | The test defines a simple matrix multiplication operation using the @ operator and verifies that TorchDynamo correctly captures it as a single operation.... |
| **Shapes** | (3,) |
| **LLM Shape** | (x, y)|(3,) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_matmul1(self):
        def matmul_op1(a, b):
            return a @ b

        # TODO(jansel): FX doesn't support this, should add upstream support
        torch._dynamo.testing.standard_test(self, matmul_op1, 2, expected_ops=1)

```

---

### 51. `test_matmul_softmax` — `inductor/test_auto_chunker.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_auto_chunker.py:114–116` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test runs a common matrix‑multiplication benchmark with a softmax applied to the result, using the auto‑chunker configuration. It checks that the chunked execution produces the same output as the ... |
| **Dtypes** | torch.bfloat16 |
| **LLM Dtype** | torch.bfloat16 |
| **Shapes** | (32, 1024, 768)|(32, 1024) |
| **LLM Shape** | B=32|T=1024|C=768|V=50257|(B, T, C)|(B, T) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_matmul_softmax(self):
        self.common_matmul_test(has_softmax=True)

```

---

### 52. `test_unbacked_3d_matmul` — `export/test_export.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/export/test_export.py:6062–6063` |
| **Classification** | `AS_IS` |
| **Module** | `export` |
| **Description** | The test defines a simple model that unsqueezes and expands a 2D tensor based on a scalar integer, then performs a batched matrix multiplication with a ones vector. It exports the model with torch.exp... |
| **Dtypes** | torch.int |
| **LLM Dtype** | torch.int32|torch.int |
| **Shapes** | (3), (4, 3) |
| **LLM Shape** | (4, 3)|(2, 5)|(2, 5)|3|x.size(0)|u0 // 2|x.size(-1)|{0: batch}|{0: batch} |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_unbacked_3d_matmul(self):
        class Model(torch.nn.Module):
```

---

### 53. `test_cutlass_backend_matmul_nonzero_offset` — `inductor/test_cutlass_backend.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_cutlass_backend.py:2016–2036` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that the CUTLASS backend correctly handles matrix multiplication on sliced (non‑zero offset) half‑precision CUDA tensors when compiled with torch.compile.... |
| **LLM Dtype** | half |
| **Shapes** | (M) |
| **LLM Shape** | M|M - 1|129|128|1024|M, M|M, M - 1 |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_cutlass_backend_matmul_nonzero_offset(self):
        max_autotune_gemm_backends = "CUTLASS"

        M = 129
        A = torch.randn(M, M - 1).to(GPU_TYPE).half()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cutlass.cutlass_max_profiling_configs": 2,
            }
        ):
            compiled = torch.compile(torch.mm)
            torch.testing.assert_close(
                A[1:, :] @ A[1:, :].t(), compiled(A[1:, :], A[1:, :].t())
            )

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
```

---

### 54. `test_matmul_uneven_chain` — `functorch/test_ac.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/functorch/test_ac.py:127–132` |
| **Classification** | `AS_IS` |
| **Module** | `functorch` |
| **Description** | The test builds a chain of matrix multiplications followed by cosine and summation, then measures the eager memory usage of the whole computation and checks it against expected values derived from the... |
| **Shapes** | (0, 50), (512), (512, 512) |
| **LLM Shape** | [512, 512]|[512, in_dim]|[11, 3, 4, 2]|[512, dim * 512]|[3, 5, 11, 17, 14] |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_matmul_uneven_chain(self):
        # This function is constructed so that we are saving one input of size
        # [512, in_dim] for each w
        # In addition, every matmul has a same ratio of compute to "memory
        # saved", so this test is essentially testing our knapsack solving

```

---

### 55. `test_matmul_autocast_float16_precision` — `test_cpp_extensions_aot.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_cpp_extensions_aot.py:359–371` |
| **Classification** | `AS_IS` |
| **Module** | `cpp_extensions_aot` |
| **Description** | The test creates two float32 tensors on the Maia device, runs a matrix multiplication inside an autocast context set to float16, and checks that autocast is active and the result is cast to float16 wi... |
| **Dtypes** | torch.float, torch.float16 |
| **LLM Dtype** | torch.float|torch.float|torch.float16|torch.float|torch.float|torch.bfloat16|torch.int64|torch.int64|torch.int64 |
| **Shapes** | (2, 4), (4, 2) |
| **LLM Shape** | (2, 4)|4, 2|(4, 2)|(2, 2)|(2, 4)|(4, 2)|(2, 2)|(10,)|(10,) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_matmul_autocast_float16_precision(self):
        # Ensure we can change low precision dtype.
        x = torch.empty((2, 4), dtype=torch.float, device="maia")
        w = torch.empty((4, 2), dtype=torch.float, device="maia")
        with torch.autocast(device_type="maia", dtype=torch.float16):
            self.assertTrue(torch.is_autocast_enabled("maia"))
            y = torch.ops.aten.matmul(x, w)
            self.assertEqual(y.dtype, torch.float16)
            self.assertEqual(y.shape, (2, 2))

    @skipIfTorchDynamo(
        "dynamo cannot handle maia device. Output tensor may have wrong dtype."
    )
```

---

### 56. `test_matmul_py3` — `jit/test_python_builtins.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/jit/test_python_builtins.py:48–50` |
| **Classification** | `AS_IS` |
| **Module** | `jit` |
| **Description** | The test defines a Python function that performs matrix multiplication using the @ operator, writes it to a temporary script, loads it as a TorchScript function, and verifies that the scripted functio... |
| **LLM Dtype** | torch.float |
| **Shapes** | (3, 2), (4, 3) |
| **LLM Shape** | (4, 3)|(3, 2)|(1)|(10) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_matmul_py3(self):
        code = dedent(
            """
```

---

### 57. `test_matmul_autocast_default_precision` — `test_cpp_extensions_aot.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_cpp_extensions_aot.py:372–383` |
| **Classification** | `AS_IS` |
| **Module** | `cpp_extensions_aot` |
| **Description** | The test creates two float tensors on the Maia device, runs a matrix multiplication inside a Maia autocast context, and checks that autocast is active and the result is cast to bfloat16 with the corre... |
| **Dtypes** | torch.bfloat16, torch.float |
| **LLM Dtype** | torch.float|torch.bfloat16|torch.float16|torch.int64 |
| **Shapes** | (2, 4), (4, 2) |
| **LLM Shape** | (2, 4)|(4, 2)|(2, 2)|(10,) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_matmul_autocast_default_precision(self):
        # Use default lower precision dtype, bfloat16.
        x = torch.empty((2, 4), dtype=torch.float, device="maia")
        w = torch.empty((4, 2), dtype=torch.float, device="maia")
        with torch.autocast(device_type="maia"):
            self.assertTrue(torch.is_autocast_enabled("maia"))
            y = torch.ops.aten.matmul(x, w)
            self.assertEqual(y.dtype, torch.bfloat16)
            self.assertEqual(y.shape, (2, 2))


@torch.testing._internal.common_utils.markDynamoStrictTest
```

---

### 58. `test_codegen_structure_parallel_matmuls` — `inductor/test_user_streams.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_user_streams.py:1412–1415` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that torch.compile correctly generates a wrapper graph for two parallel matrix multiplications executed on separate CUDA streams with explicit event synchronization and a final addit... |
| **LLM Dtype** | torch.float32|f32 |
| **Shapes** | (32, 32) |
| **LLM Shape** | (32, 32)|(32, 1) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_codegen_structure_parallel_matmuls(self):
        """Verify wrapper structure for parallel matmuls with join."""
        from torch._inductor.utils import run_and_get_code

```

---

### 59. `test_triton_kernel_matmul_tracking` — `inductor/test_triton_kernels.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_triton_kernels.py:1201–1202` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test defines a Triton JIT kernel that writes ones into a tensor, compiles a function with torch.compile that calls this kernel and then performs a matrix multiplication with a random input, adding... |
| **LLM Dtype** | torch.float32 |
| **Shapes** | (0, BLOCK_SIZE), (4, 4), (x) |
| **LLM Shape** | (10, )|(1, )|(4,)|(4, 4)|(4, 4)|(128, 128)|(32, 16)|(64, 256) |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_triton_kernel_matmul_tracking(self):
        @triton.jit
```

---

### 60. `test_diff_matmul_share_same_kernel` — `inductor/test_cutlass_backend.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_cutlass_backend.py:415–417` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test compiles a model that performs two matrix multiplications sharing the same left operand and verifies that both operations are executed using the same Cutlass kernel. It runs the compiled mode... |
| **LLM Dtype** | half |
| **Shapes** | (128, 16), (512, 16) |
| **LLM Shape** | 128, 16|128, 16|512, 16 |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_diff_matmul_share_same_kernel(self, dynamic):
        max_autotune_gemm_backends = "CUTLASS"

```

---

### 61. `test_matmul_even_chain` — `functorch/test_ac.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/functorch/test_ac.py:99–108` |
| **Classification** | `AS_IS` |
| **Module** | `functorch` |
| **Description** | The test defines a chain of cosine and matrix‑multiply operations on a 512×512 tensor and measures memory usage and FLOP count under varying memory budgets, asserting expected values.... |
| **Shapes** | (512, 512) |
| **LLM Shape** | 512, 512|512, 512|512, 512|512|512|11, 3, 4, 2|512|512 |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_matmul_even_chain(self):
        def f(x, ws):
            x = x.cos()
            for w in ws:
                x = torch.mm(x, w).cos()
            return x.sum()

        x = torch.randn(512, 512, requires_grad=True)
        ws = [torch.randn(512, 512, requires_grad=True) for _ in range(5)]

```

---

### 62. `test_cutlass_backend_matmul_same_tensor` — `inductor/test_cutlass_backend.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_cutlass_backend.py:1996–2015` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that the CUTLASS backend correctly handles a matrix multiplication where the same tensor is used for both operands (one transposed). It compiles torch.mm with torch.compile under CUT... |
| **LLM Dtype** | half |
| **Shapes** | (M, M) |
| **LLM Shape** | M, M|M, M - 1|1:, :|M, M |
| **Device Classification** | no_device |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_cutlass_backend_matmul_same_tensor(self):
        max_autotune_gemm_backends = "CUTLASS"

        M = 128
        A = torch.randn(M, M).to(GPU_TYPE).half()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cutlass.cutlass_max_profiling_configs": 2,
            }
        ):
            compiled = torch.compile(torch.mm)

            torch.testing.assert_close(A @ A.t(), compiled(A, A.t()))

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
```

---

### 63. `test_associative_scan_downstream_scan_matmul` — `functorch/test_control_flow.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/functorch/test_control_flow.py:4337–4339` |
| **Classification** | `AS_IS` |
| **Module** | `functorch` |
| **Description** | The test verifies that an associative scan with a downstream matrix multiplication works correctly across different combine modes, compile settings, reversal directions, devices, and autograd configur... |
| **Shapes** | (2, 5), (3, 10, 2) |
| **LLM Shape** | [3, 10, 2]|f32[3, 10, 2]|(2, 5)|(3, 10, 2) |
| **Device Classification** | device_generic |
| **Match Score** | 20 |

**Source snippet:**
```python
    def test_associative_scan_downstream_scan_matmul(
        self, combine_mode, compile_mode, reverse, device, autograd
    ):
```

---

### 64. `test_flop_counter_variety` — `test_flop_counter.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_flop_counter.py:75–85` |
| **Classification** | `AS_IS` |
| **Module** | `flop_counter` |
| **Description** | The test runs a series of matrix operations (mm, addmm, matmul, einsum) and a Linear layer inside a FlopCounterMode context, then checks that the total counted FLOPs match the expected value of 3012.... |
| **Shapes** | (4,5)|(5,6)|(4,6)|(5,6)|(6,7)|(7,8)|(8,9)|(3,4,5)|(3,5,6)|(4,1)|(6)|(3,4,6)|(2,3,6,6)|(6,3,4,4)|(2,3,6)|(6,3,4) |
| **LLM Shape** | (2, 32, 128, 64)|(2, 8, 128, 64)|(2, 5, 128, 64)|(4, 5)|(5, 6)|(4, 6)|(5, 6)|(6, 7)|(7, 8)|(8, 9)|(3, 4, 5)|(3, 5, 6)|(4, 1)|(3, 4, 6)|(2, 3, 6, 6)|(6, 3, 4, 4)|(2, 3, 6)|(6, 3, 4)|(2, 5, 5)|(4, 4)|(2, 5)|(4) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_flop_counter_variety(self):
        mod = torch.nn.Linear(9, 10)
        with FlopCounterMode() as mode:
            torch.mm(T(4, 5), T(5, 6))
            torch.addmm(T(4, 6), T(4, 5), T(5, 6), beta=0.5, alpha=0.5)
            torch.matmul(T(5, 6), T(6, 7))
            torch.einsum("ab,bc->ac", T(6, 7), T(7, 8))
            mod(T(8, 9))

        self.assertExpectedInline(get_total_flops(mode), """3012""")

```

---

### 65. `test_array_priority_override` — `torch_np/numpy_tests/core/test_multiarray.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/torch_np/numpy_tests/core/test_multiarray.py:5679–5682` |
| **Classification** | `AS_IS` |
| **Module** | `torch_np` |
| **Description** | The test defines a custom class with a high __array_priority__ and overrides the matmul methods, then checks that matmul between this object and a NumPy array always dispatches to the custom implement... |
| **LLM Dtype** | bool|np.uint8|np.int8 |
| **Shapes** | (2) |
| **LLM Shape** | (2, 0)|(0,)|(4, 5)|(5, 4)|(2,)|(3, 4, 5)|(-2, -1)|(-1, -2)|(1, 2)|(0, 1)|(4, 4)|(3, 4, 4)|(4, 4, 3)|(4, 5)|(1, 0)|(0) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_array_priority_override(self):
        class A:
            __array_priority__ = 1000

```

---

### 66. `test_fuse_linear` — `quantization/jit/test_quantize_jit.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/quantization/jit/test_quantize_jit.py:380–381` |
| **Classification** | `AS_IS` |
| **Module** | `quantization` |
| **Description** | The test creates a simple functional linear module implemented with matmul and optional bias addition, traces it with TorchScript, runs the JIT fuse_linear pass, and checks that the matmul pattern is ... |
| **Shapes** | (3), (5), (5, 3), (5, 5), (5, 5, 100), (5, 5, 5), (5, 6, 5) |
| **LLM Shape** | (3)|(5, 3)|(5)|(5, 5)|(5, 5, 5) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_fuse_linear(self):
        class FunctionalLinear(torch.nn.Module):
```

---

### 67. `test_mm_decompose_mm_dde` — `test_decomp.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_decomp.py:1486–1536` |
| **Classification** | `AS_IS` |
| **Module** | `decomp` |
| **Description** | The test builds a synthetic program that performs a series of torch.matmul operations on tensors of varying shapes and types, exercising TorchDynamo's decomposition of matrix multiplication and its dy... |
| **Dtypes** | torch.bool, torch.float64, torch.int64, torch.int8 |
| **LLM Dtype** | float64|torch.float64 |
| **Shapes** | (0, 2), (1), (1.0), (1008), (1152), (1188), (126), (13, 16), (1350), (156, 8), (162), (1728), (2), (720), (729), (8, 9), (819), (864), (891), (9), (9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (9, 1, 11), (9, 1, 2), (9, 1, 8), (9, 10, 6), (9, 11, 12), (9, 13), (9, 15, 2), (9, 15, 7), (9, 16, 15), (9, 16, 5), (9, 2, 1), (9, 3, 8), (9, 5, 10), (9, 7, 1), (90), (936), (990, 2) |
| **LLM Shape** | (9, 9, 9)|(9, 9, 11)|(9, 11, 12)|(9, 12, 8)|(9, 9, 8)|(9, 8, 13)|(9, 13, 7) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_mm_decompose_mm_dde(self):
        def fuzzed_program(
            arg_0,
            arg_1,
            arg_2,
            arg_3,
            arg_4,
            arg_5,
            arg_6,
            arg_7,
            arg_8,
            arg_9,
            arg_10,
            arg_11,
            arg_12,
            arg_13,
            arg_14,
            arg_15,
            arg_16,
            arg_17,
            arg_18,
            sentinel,
        ):
            var_node_6 = (
                arg_0  # size=(9, 9, 9), stride=(81, 9, 1), dtype=float64, device=cuda
            )
            var_node_7 = (
                arg_1  # size=(9, 9, 11), stride=(99, 11, 1), dtype=float64, device=cuda
            )
            var_node_5 = torch.matmul(
                var_node_6.to(torch.float64), var_node_7.to(torch.float64)
            )  # size=(9, 9, 11), stride=(99, 11, 1), dtype=float64, device=cuda
            var_node_9 = torch.full(
                (9, 11, 12), 1.5758497316910556, dtype=torch.float64
            )  # size=(9, 11, 12), stride=(132, 12, 1), dtype=float64, device=cuda
            var_node_10 = (
                arg_2  # size=(9, 12, 8), stride=(96, 8, 1), dtype=float64, device=cuda
            )
            var_node_8 = torch.matmul(
                var_node_9.to(torch.float64), var_node_10.to(torch.float64)
            )  # size=(9, 11, 8), stride=(88, 8, 1), dtype=float64, device=cuda
            var_node_4 = torch.matmul(
                var_node_5.to(torch.float64), var_node_8.to(torch.float64)
            )  # size=(9, 9, 8), stride=(72, 8, 1), dtype=float64, device=cuda
            var_node_13 = arg_3  # size=(9, 8, 13), stride=(104, 13, 1), dtype=float64, device=cuda
            var_node_14 = (
                arg_4  # size=(9, 13, 7), stride=(91, 7, 1), dtype=float64, device=cuda
            )
            var_node_12 = torch.matmul(
                var_node_13.to(torch.float64), var_node_14.to(torch.float64)
            )  # size=(9, 8, 7), stride=(56, 7, 1), dtype=float64, device=cuda
```

---

### 68. `test_lstm_gates_permutations` — `test_jit_fuser_te.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_jit_fuser_te.py:1022–1028` |
| **Classification** | `AS_IS` |
| **Module** | `jit_fuser_te` |
| **Description** | The test iterates over all devices and generates every permutation of the four LSTM gate terms (input‑hidden matmul, hidden‑hidden matmul, input bias, hidden bias). For each permutation it JIT‑compile... |
| **Shapes** | (4, 1) |
| **LLM Shape** | 4|1 |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_lstm_gates_permutations(self):
        for device in self.devices:
            # lstm has gates = x.mm(w_ih.t()) + hx.mm(w_hh.t()) + b_ih + b_hh.
            # Test that any permutation of this will still result in one FusionGroup.
            choices = ["x.mm(w_ih.t())", "hx.mm(w_hh.t())", "b_ih", "b_hh"]
            template = dedent(
                """
```

---

### 69. `test_vector_vector_values` — `torch_np/numpy_tests/core/test_multiarray.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/torch_np/numpy_tests/core/test_multiarray.py:5341–5359` |
| **Classification** | `AS_IS` |
| **Module** | `torch_np` |
| **Description** | The test verifies NumPy's (or a compatible) matmul implementation for vector‑vector and vector‑matrix products across multiple data types, including a boolean case. It checks that the result matches e... |
| **LLM Dtype** | dt|self.types[1:]|? |
| **Shapes** | (2,)|(2, 1)|(1,)|(2, 2)|(2,) |
| **LLM Shape** | (1,)|(2,)|(-1, 1)|(1, -1)|[1, 2]|[3, 4]|[11]|[[3, 6], [4, 8]]|[True, True]|[1, 2]|[[1, 2], [3, 4]]|(2,)|(2, 2)|[7, 10]|[True, False]|[5, 11] |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_vector_vector_values(self):
        vec1 = np.array([1, 2])
        vec2 = np.array([3, 4]).reshape(-1, 1)
        tgt1 = np.array([11])
        tgt2 = np.array([[3, 6], [4, 8]])
        for dt in self.types[1:]:
            v1 = vec1.astype(dt)
            v2 = vec2.astype(dt)
            res = self.matmul(v1, v2)
            assert_equal(res, tgt1)
            # no broadcast, we must make v1 into a 2d ndarray
            res = self.matmul(v2, v1.reshape(1, -1))
            assert_equal(res, tgt2)

        # boolean type
        vec = np.array([True, True], dtype="?")
        res = self.matmul(vec, vec)
        assert_equal(res, True)

```

---

### 70. `test_default_use_parent` — `test_mkldnn.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_mkldnn.py:1717–1734` |
| **Classification** | `AS_IS` |
| **Module** | `mkldnn` |
| **Description** | The test verifies that the MKL-DNN matmul backend correctly inherits and overrides the fp32_precision flag from both its own and the global torch.backends contexts.... |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_default_use_parent(self):
        torch.backends.mkldnn.matmul.fp32_precision = "none"
        with torch.backends.mkldnn.flags(enabled=None, fp32_precision="bf16"):
            self.assertEqual(torch.backends.mkldnn.matmul.fp32_precision, "bf16")
        with torch.backends.mkldnn.flags(enabled=None, fp32_precision="tf32"):
            self.assertEqual(torch.backends.mkldnn.matmul.fp32_precision, "tf32")
        with torch.backends.mkldnn.flags(enabled=None, fp32_precision="none"):
            with torch.backends.flags(fp32_precision="bf16"):
                self.assertEqual(torch.backends.mkldnn.matmul.fp32_precision, "bf16")
            with torch.backends.flags(fp32_precision="tf32"):
                self.assertEqual(torch.backends.mkldnn.matmul.fp32_precision, "tf32")


instantiate_device_type_tests(TestMkldnn, globals(), only_for=('cpu',))

if __name__ == '__main__':
    run_tests()

```

---

### 71. `test_external_calls` — `test_tensorexpr_pybind.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_tensorexpr_pybind.py:51–69` |
| **Classification** | `AS_IS` |
| **Module** | `tensorexpr_pybind` |
| **Description** | The test builds a TensorExpr external call that invokes the ATen matmul operation, generates executable code, runs it on simple one‑filled tensors, and verifies the output matches PyTorch's matmul res... |
| **Dtypes** | torch.float32 |
| **LLM Dtype** | torch.float32|torch.float64|torch.int32|torch.double |
| **Shapes** | (1, 1), (1, 4), (4, 1) |
| **LLM Shape** | [1, 4]|[4, 1]|[1, 1]|[dN]|[dN, dM] |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_external_calls(self):
        dtype = torch.float32

        A = te.BufHandle("A", [1, 4], dtype)
        B = te.BufHandle("B", [4, 1], dtype)
        C = te.BufHandle("C", [1, 1], dtype)

        s = te.ExternalCall(C, "nnc_aten_matmul", [A, B], [])

        loopnest = te.LoopNest(s, [C])
        loopnest.prepare_for_codegen()
        codegen = te.construct_codegen("ir_eval", s, [A, B, C])

        tA = torch.ones(1, 4)
        tB = torch.ones(4, 1)
        tC = torch.empty(1, 1)
        codegen.call([tA, tB, tC])
        torch.testing.assert_close(torch.matmul(tA, tB), tC)

```

---

### 72. `test_vector_matrix_values` — `torch_np/numpy_tests/core/test_multiarray.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/torch_np/numpy_tests/core/test_multiarray.py:5360–5386` |
| **Classification** | `AS_IS` |
| **Module** | `torch_np` |
| **Description** | The test verifies that the matmul implementation correctly computes the product of a 1‑D vector with a 2‑D matrix and with a batch of matrices, for a variety of numeric dtypes and for boolean dtype.... |
| **LLM Dtype** | self.types[1:]|dt|? |
| **Shapes** | (2,)|(2,2)|(2,2,2) |
| **LLM Shape** | (1, -1)|(-1, 1)|(2,)|(2, 2)|(2, 2, 2) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_vector_matrix_values(self):
        vec = np.array([1, 2])
        mat1 = np.array([[1, 2], [3, 4]])
        mat2 = np.stack([mat1] * 2, axis=0)
        tgt1 = np.array([7, 10])
        tgt2 = np.stack([tgt1] * 2, axis=0)
        for dt in self.types[1:]:
            v = vec.astype(dt)
            m1 = mat1.astype(dt)
            m2 = mat2.astype(dt)
            res = self.matmul(v, m1)
            assert_equal(res, tgt1)
            res = self.matmul(v, m2)
            assert_equal(res, tgt2)

        # boolean type
        vec = np.array([True, False])
        mat1 = np.array([[True, False], [False, True]])
        mat2 = np.stack([mat1] * 2, axis=0)
        tgt1 = np.array([True, False])
        tgt2 = np.stack([tgt1] * 2, axis=0)

        res = self.matmul(vec, mat1)
        assert_equal(res, tgt1)
        res = self.matmul(vec, mat2)
        assert_equal(res, tgt2)

```

---

### 73. `test_matrix_vector_values` — `torch_np/numpy_tests/core/test_multiarray.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/torch_np/numpy_tests/core/test_multiarray.py:5387–5413` |
| **Classification** | `AS_IS` |
| **Module** | `torch_np` |
| **Description** | The test verifies that the matmul implementation correctly computes matrix‑vector products for a variety of numeric dtypes and for boolean data, both for a single 2‑D matrix and for a batch of matrice... |
| **LLM Dtype** | self.types[1:]|boolean type |
| **Shapes** | (2,)|(2, 2)|(2, 2, 2) |
| **LLM Shape** | [1, 2]|[[1, 2], [3, 4]]|[7, 10]|[True, False]|[[True, False], [False, True]]|[True, False]|[5, 11]|[[1, 2], [3, 4]]|[1, 0]|[1, 1]|[[7, 10], [15, 22]]|[[3, 2], [7, 4]]|[[1, 2], [4, 6]] |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_matrix_vector_values(self):
        vec = np.array([1, 2])
        mat1 = np.array([[1, 2], [3, 4]])
        mat2 = np.stack([mat1] * 2, axis=0)
        tgt1 = np.array([5, 11])
        tgt2 = np.stack([tgt1] * 2, axis=0)
        for dt in self.types[1:]:
            v = vec.astype(dt)
            m1 = mat1.astype(dt)
            m2 = mat2.astype(dt)
            res = self.matmul(m1, v)
            assert_equal(res, tgt1)
            res = self.matmul(m2, v)
            assert_equal(res, tgt2)

        # boolean type
        vec = np.array([True, False])
        mat1 = np.array([[True, False], [False, True]])
        mat2 = np.stack([mat1] * 2, axis=0)
        tgt1 = np.array([True, False])
        tgt2 = np.stack([tgt1] * 2, axis=0)

        res = self.matmul(vec, mat1)
        assert_equal(res, tgt1)
        res = self.matmul(vec, mat2)
        assert_equal(res, tgt2)

```

---

### 74. `test_shapes` — `torch_np/numpy_tests/core/test_multiarray.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/torch_np/numpy_tests/core/test_multiarray.py:5283–5302` |
| **Classification** | `AS_IS` |
| **Module** | `torch_np` |
| **Description** | The test verifies that the custom matmul implementation produces the correct output shapes when broadcasting matrix stacks and when performing vector‑vector multiplication, across all supported data t... |
| **LLM Dtype** | ?|b|h|i|l|B|e|f|d|F|D|dt|O |
| **Shapes** | (2), (dm1), (dm2) |
| **LLM Shape** | (1, 1)|(2, 1, 1)|(1,)|(2,)|(2, 1)|(1, 2)|(3, 1)|()|(1, 1, 1)|(2, 2, 1)|(3, 1, 2)|(2,)|(1,)|(1, -1)|(1,) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_shapes(self):
        dims = [
            ((1, 1), (2, 1, 1)),  # broadcast first argument
            ((2, 1, 1), (1, 1)),  # broadcast second argument
            ((2, 1, 1), (2, 1, 1)),  # matrix stack sizes match
        ]

        for dt, (dm1, dm2) in itertools.product(self.types, dims):
            a = np.ones(dm1, dtype=dt)
            b = np.ones(dm2, dtype=dt)
            res = self.matmul(a, b)
            assert_(res.shape == (2, 1, 1))

        # vector vector returns scalars.
        for dt in self.types:
            a = np.ones((2,), dtype=dt)
            b = np.ones((2,), dtype=dt)
            c = self.matmul(a, b)
            assert_(np.array(c).shape == ())

```

---

### 75. `test_exceptions` — `torch_np/numpy_tests/core/test_multiarray.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/torch_np/numpy_tests/core/test_multiarray.py:5259–5282` |
| **Classification** | `AS_IS` |
| **Module** | `torch_np` |
| **Description** | The test iterates over a set of mismatched shape pairs and data types, creates NumPy arrays of ones for each pair, and verifies that invoking the matmul operation raises an exception.... |
| **LLM Dtype** | np.float64|np.complex128|np.float32|?|b|h|i|l|B|e|f|d|F|D |
| **Shapes** | (dm1), (dm2) |
| **LLM Shape** | (1,)|(2,)|(2, 1)|(2,)|(2,)|(1, 2)|(1, 2)|(3, 1)|(1,)|()|(1,)|()|(1, 1)|()|(1, 1)|(1, 1)|(2, 2, 1)|(3, 1, 2)|(1, 1)|(2, 1, 1)|(2, 1, 1)|(1, 1)|(2, 1, 1)|(2, 1, 1)|(2, 1, 1)|(2,)|(2,)|()|(1, 1)|(1,) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_exceptions(self):
        dims = [
            ((1,), (2,)),  # mismatched vector vector
            (
                (
                    2,
                    1,
                ),
                (2,),
            ),  # mismatched matrix vector
            ((2,), (1, 2)),  # mismatched vector matrix
            ((1, 2), (3, 1)),  # mismatched matrix matrix
            ((1,), ()),  # vector scalar
            ((), (1)),  # scalar vector
            ((1, 1), ()),  # matrix scalar
            ((), (1, 1)),  # scalar matrix
            ((2, 2, 1), (3, 1, 2)),  # cannot broadcast
        ]

        for dt, (dm1, dm2) in itertools.product(self.types, dims):
            a = np.ones(dm1, dtype=dt)
            b = np.ones(dm2, dtype=dt)
            assert_raises((RuntimeError, ValueError), self.matmul, a, b)

```

---

### 76. `test_matrix_matrix_values` — `torch_np/numpy_tests/core/test_multiarray.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/torch_np/numpy_tests/core/test_multiarray.py:5414–5480` |
| **Classification** | `AS_IS` |
| **Module** | `torch_np` |
| **Description** | The test verifies that the custom matmul implementation produces correct results for 2x2 matrices across multiple data types, including batched (stacked) inputs and boolean tensors.... |
| **Dtypes** | np.bool_ |
| **LLM Dtype** | np.bool_|self.types[1:]|dt |
| **Shapes** | (2,2)|(3,2,2) |
| **LLM Shape** | (2, 2)|(2, 2, 2) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_matrix_matrix_values(self):
        mat1 = np.array([[1, 2], [3, 4]])
        mat2 = np.array([[1, 0], [1, 1]])
        mat12 = np.stack([mat1, mat2], axis=0)
        mat21 = np.stack([mat2, mat1], axis=0)
        tgt11 = np.array([[7, 10], [15, 22]])
        tgt12 = np.array([[3, 2], [7, 4]])
        tgt21 = np.array([[1, 2], [4, 6]])
        tgt12_21 = np.stack([tgt12, tgt21], axis=0)
        tgt11_12 = np.stack((tgt11, tgt12), axis=0)
        tgt11_21 = np.stack((tgt11, tgt21), axis=0)
        for dt in self.types[1:]:
            m1 = mat1.astype(dt)
            m2 = mat2.astype(dt)
            m12 = mat12.astype(dt)
            m21 = mat21.astype(dt)

            # matrix @ matrix
            res = self.matmul(m1, m2)
            assert_equal(res, tgt12)
            res = self.matmul(m2, m1)
            assert_equal(res, tgt21)

            # stacked @ matrix
            res = self.matmul(m12, m1)
            assert_equal(res, tgt11_21)

            # matrix @ stacked
            res = self.matmul(m1, m12)
            assert_equal(res, tgt11_12)

            # stacked @ stacked
            res = self.matmul(m12, m21)
            assert_equal(res, tgt12_21)

        # boolean type
        m1 = np.array([[1, 1], [0, 0]], dtype=np.bool_)
        m2 = np.array([[1, 0], [1, 1]], dtype=np.bool_)
        m12 = np.stack([m1, m2], axis=0)
        m21 = np.stack([m2, m1], axis=0)
        tgt11 = m1
        tgt12 = m1
        tgt21 = np.array([[1, 1], [1, 1]], dtype=np.bool_)
        tgt12_21 = np.stack([tgt12, tgt21], axis=0)
        tgt11_12 = np.stack((tgt11, tgt12), axis=0)
        tgt11_21 = np.stack((tgt11, tgt21), axis=0)

        # matrix @ matrix
        res = self.matmul(m1, m2)
        assert_equal(res, tgt12)
        res = self.matmul(m2, m1)
        assert_equal(res, tgt21)

        # stacked @ matrix
        res = self.matmul(m12, m1)
        assert_equal(res, tgt11_21)

        # matrix @ stacked
        res = self.matmul(m1, m12)
        assert_equal(res, tgt11_12)

        # stacked @ stacked
        res = self.matmul(m12, m21)
        assert_equal(res, tgt12_21)


@instantiate_parametrized_tests
```

---

### 77. `test_out_arg` — `torch_np/numpy_tests/core/test_multiarray.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/torch_np/numpy_tests/core/test_multiarray.py:5486–5514` |
| **Classification** | `AS_IS` |
| **Module** | `torch_np` |
| **Description** | The test verifies that the matmul implementation correctly handles the optional 'out' argument for storing results, both as a positional and keyword parameter, and enforces type safety and casting rul... |
| **Dtypes** | torch.float |
| **LLM Dtype** | float|np.bool_|np.int32|np.complex128 |
| **Shapes** | (5, 2) |
| **LLM Shape** | (5, 2)|(2, 2)|(0, 1, 1)|(1, 1, 1)|(5, 2, 2)|[1, 3]|[5, 7]|(10, 2) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_out_arg(self):
        a = np.ones((5, 2), dtype=float)
        b = np.array([[1, 3], [5, 7]], dtype=float)
        tgt = np.dot(a, b)

        # test as positional argument
        msg = "out positional argument"
        out = np.zeros((5, 2), dtype=float)
        self.matmul(a, b, out)
        assert_array_equal(out, tgt, err_msg=msg)

        # test as keyword argument
        msg = "out keyword argument"
        out = np.zeros((5, 2), dtype=float)
        self.matmul(a, b, out=out)
        assert_array_equal(out, tgt, err_msg=msg)

        # test out with not allowed type cast (safe casting)
        msg = "Cannot cast"
        out = np.zeros((5, 2), dtype=np.int32)
        assert_raises_regex(TypeError, msg, self.matmul, a, b, out=out)

        # test out with type upcast to complex
        out = np.zeros((5, 2), dtype=np.complex128)
        c = self.matmul(a, b, out=out)
        assert_(c is out)
        c = c.astype(tgt.dtype)
        assert_array_equal(c, tgt)

```

---

### 78. `test_no_dgemv_2` — `torch_np/numpy_tests/core/test_multiarray.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/torch_np/numpy_tests/core/test_multiarray.py:2617–2632` |
| **Classification** | `AS_IS` |
| **Module** | `torch_np` |
| **Description** | The test verifies that np.dot and np.matmul produce identical results when operating on unaligned input arrays versus aligned copies, including transposed variants, across several numeric dtypes.... |
| **LLM Dtype** | ifdFD|dtype|np.dtype(dtype)|int16 |
| **Shapes** | (2, 4) |
| **LLM Shape** | 8 * dt.itemsize // 2 + 1|2, 4|(2, 4)|[1:] |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_no_dgemv_2(self, func, dtype):
        # check for unaligned data
        dt = np.dtype(dtype)
        a = np.zeros(8 * dt.itemsize // 2 + 1, dtype="int16")[1:].view(dtype)
        a = a.reshape(2, 4)
        b = a[0]
        # make sure it is not aligned
        assert_(a.__array_interface__["data"][0] % dt.itemsize != 0)
        ret1 = func(a, b)
        ret2 = func(a.copy(), b.copy())
        assert_equal(ret1, ret2)

        ret1 = func(b.T, a.T)
        ret2 = func(b.T.copy(), a.T.copy())
        assert_equal(ret1, ret2)

```

---

### 79. `test_no_dgemv` — `torch_np/numpy_tests/core/test_multiarray.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/torch_np/numpy_tests/core/test_multiarray.py:2601–2616` |
| **Classification** | `AS_IS` |
| **Module** | `torch_np` |
| **Description** | The test verifies that NumPy's dot and matmul functions produce identical results when one operand is a non‑contiguous broadcasted vector, both in regular and transposed forms.... |
| **LLM Dtype** | ifdFD|np.float32|np.float64|np.complex64|np.complex128|dtype|int16 |
| **Shapes** | (8.0) |
| **LLM Shape** | (2, 4)|(4, 1) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_no_dgemv(self, func, dtype):
        # check vector arg for contiguous before gemv
        # gh-12156
        a = np.arange(8.0, dtype=dtype).reshape(2, 4)
        b = np.broadcast_to(1.0, (4, 1))
        ret1 = func(a, b)
        ret2 = func(a, b.copy())
        assert_equal(ret1, ret2)

        ret1 = func(b.T, a.T)
        ret2 = func(b.T.copy(), a.T)
        assert_equal(ret1, ret2)

    @skip(reason="__array_interface__")
    @parametrize("func", (np.dot, np.matmul))
    @parametrize("dtype", "ifdFD")
```

---

### 80. `test_arr_mult_2` — `torch_np/numpy_tests/core/test_multiarray.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/torch_np/numpy_tests/core/test_multiarray.py:2586–2600` |
| **Classification** | `AS_IS` |
| **Module** | `torch_np` |
| **Description** | The test verifies that np.dot and np.matmul produce identical results when operating on non‑contiguous (negative‑stride) views of a 2‑D array versus their contiguous copies, across several floating‑po... |
| **Dtypes** | float32|float64|complex64|complex128 |
| **LLM Dtype** | np.float32|np.float64|np.complex64|np.complex128|dtype (parametrized as 'ifdFD')|int16 |
| **LLM Shape** | (2, 4)|(4, 1)|(2, 4) [reshape]|(8 * dt.itemsize // 2 + 1)|[1:] [slice]|(2, 4) [reshape] |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_arr_mult_2(self, func):
        # syrk - different shape, stride, and view validations
        for et in [np.float32, np.float64, np.complex64, np.complex128]:
            edf = d.astype(et)
            assert_equal(
                func(edf[::-1, :], edf.T), func(edf[::-1, :].copy(), edf.T.copy())
            )
            assert_equal(
                func(edf[:, ::-1], edf.T), func(edf[:, ::-1].copy(), edf.T.copy())
            )
            assert_equal(func(edf, edf[::-1, :].T), func(edf, edf[::-1, :].T.copy()))
            assert_equal(func(edf, edf[:, ::-1].T), func(edf, edf[:, ::-1].T.copy()))

    @parametrize("func", (np.dot, np.matmul))
    @parametrize("dtype", "ifdFD")
```

---

### 81. `test_arr_mult` — `torch_np/numpy_tests/core/test_multiarray.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/torch_np/numpy_tests/core/test_multiarray.py:2525–2585` |
| **Classification** | `AS_IS` |
| **Module** | `torch_np` |
| **Description** | The test verifies that the NumPy (or PyTorch) matrix multiplication functions (np.dot and np.matmul) produce correct results across multiple data types, transpositions, and memory layouts. It checks i... |
| **LLM Dtype** | np.float32|np.float64|np.complex64|np.complex128 |
| **Shapes** | (24) |
| **LLM Shape** | (2, 2)|[[1, 0], [0, 1]]|[[0, 1], [1, 0]]|(4, 6)|[4, 6]|[55, 145, 235, 325]|[145, 451, 757, 1063]|[235, 757, 1279, 1801]|[325, 1063, 1801, 2539]|[504, 540, 576, 612, 648, 684]|[540, 580, 620, 660, 700, 740]|[576, 620, 664, 708, 752, 796]|[612, 660, 708, 756, 804, 852]|[648, 700, 752, 804, 856, 908]|[684, 740, 796, 852, 908, 964] |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_arr_mult(self, func):
        a = np.array([[1, 0], [0, 1]])
        b = np.array([[0, 1], [1, 0]])
        d = np.arange(24).reshape(4, 6)
        ddt = np.array(
            [
                [55, 145, 235, 325],
                [145, 451, 757, 1063],
                [235, 757, 1279, 1801],
                [325, 1063, 1801, 2539],
            ]
        )
        dtd = np.array(
            [
                [504, 540, 576, 612, 648, 684],
                [540, 580, 620, 660, 700, 740],
                [576, 620, 664, 708, 752, 796],
                [612, 660, 708, 756, 804, 852],
                [648, 700, 752, 804, 856, 908],
                [684, 740, 796, 852, 908, 964],
            ]
        )

        # gemm vs syrk optimizations
        for et in [np.float32, np.float64, np.complex64, np.complex128]:
            eaf = a.astype(et)
            assert_equal(func(eaf, eaf), eaf)
            assert_equal(func(eaf.T, eaf), eaf)
            assert_equal(func(eaf, eaf.T), eaf)
            assert_equal(func(eaf.T, eaf.T), eaf)
            assert_equal(func(eaf.T.copy(), eaf), eaf)
            assert_equal(func(eaf, eaf.T.copy()), eaf)
            assert_equal(func(eaf.T.copy(), eaf.T.copy()), eaf)

        # syrk validations
        for et in [np.float32, np.float64, np.complex64, np.complex128]:
            eaf = a.astype(et)
            ebf = b.astype(et)
            assert_equal(func(ebf, ebf), eaf)
            assert_equal(func(ebf.T, ebf), eaf)
            assert_equal(func(ebf, ebf.T), eaf)
            assert_equal(func(ebf.T, ebf.T), eaf)
        # syrk - different shape
        for et in [np.float32, np.float64, np.complex64, np.complex128]:
            edf = d.astype(et)
            eddtf = ddt.astype(et)
            edtdf = dtd.astype(et)
            assert_equal(func(edf, edf.T), eddtf)
            assert_equal(func(edf.T, edf), edtdf)

            assert_equal(
                func(edf[: edf.shape[0] // 2, :], edf[::2, :].T),
                func(edf[: edf.shape[0] // 2, :].copy(), edf[::2, :].T.copy()),
            )
            assert_equal(
                func(edf[::2, :], edf[: edf.shape[0] // 2, :].T),
                func(edf[::2, :].copy(), edf[: edf.shape[0] // 2, :].T.copy()),
            )

    @skip(reason="dot/matmul with negative strides")
    @parametrize("func", (np.dot, np.matmul))
```

---

### 82. `test_decomposed_linear` — `test_xnnpack_integration.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_xnnpack_integration.py:1175–1179` |
| **Classification** | `AS_IS` |
| **Module** | `xnnpack_integration` |
| **Description** | The test defines three simple linear‑like modules (using addmm, matmul+add_, and pure matmul) with random weight and bias tensors, then runs them through the XNNPACK rewrite pass to verify pattern det... |
| **Shapes** | (weight_output_dim), (weight_shape) |
| **LLM Shape** | [2, 32]|[24, 32]|(weight_output_dim, data_shape[-1])|(16, linear_weight_shape[1]) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_decomposed_linear(self):
        data_shape = [2, 32]
        weight_output_dim = 24
        weight_shape = (weight_output_dim, data_shape[-1])

```

---

### 83. `test_cslt_sparse_mm_search` — `test_sparse_semi_structured.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_sparse_semi_structured.py:1482–1495` |
| **Classification** | `NEEDS_REFACTOR` |
| **Module** | `sparse_semi_structured` |
| **Description** | The test creates a random semi-structured sparse mask, compresses it, searches for the optimal sparse matrix multiplication algorithm, performs the multiplication with a dense matrix, and verifies the... |
| **Dtypes** | torch.float32 |
| **LLM Dtype** | dtype|torch.float16|torch.bfloat16|torch.int32|torch.int8|torch.int64|torch.float32 |
| **Shapes** | (128, 128) |
| **LLM Shape** | 256, 128|128, 128|128|256|128 |
| **Device Classification** | device_generic |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_cslt_sparse_mm_search(self, device, dtype):
        A = rand_sparse_semi_structured_mask(256, 128, dtype=dtype)
        A_compressed = torch._cslt_compress(A)
        B = torch.ones((128, 128), device=device).to(dtype)

        A_compressed = torch._cslt_compress(A)
        alg_id = torch._cslt_sparse_mm_search(A_compressed, B.t())
        sparse_result = torch._cslt_sparse_mm(A_compressed, B.t(), alg_id=alg_id)
        dense_result = torch.mm(A.to(torch.float32), B.to(torch.float32))
        dense_result = dense_result.to(dtype)
        torch.testing.assert_close(sparse_result, dense_result, rtol=1e-3, atol=1e-3)

    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    @inference_dtypes
```

---

### 84. `test_dot_equivalent` — `torch_np/numpy_tests/core/test_multiarray.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/torch_np/numpy_tests/core/test_multiarray.py:5618–5626` |
| **Classification** | `AS_IS` |
| **Module** | `torch_np` |
| **Description** | The test iterates over many pairs of tensors (matrix‑matrix, matrix‑vector, vector‑matrix) with various memory layouts—including transposed, sliced, and zero‑size tensors—and checks that the PyTorch m... |
| **Dtypes** | bool|uint8|int8 |
| **LLM Dtype** | bool|np.uint8|np.int8 |
| **Shapes** | (3, 3)|(2, 0)|(0,)|(2, 2)|(4, 5)|(5, 4)|(2, 0) |
| **LLM Shape** | (3, 3)|[1, 0], [1, 1]|(4, 5)|4|5 |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_dot_equivalent(self, args):
        r1 = np.matmul(*args)
        r2 = np.dot(*args)
        assert_equal(r1, r2)

        r3 = np.matmul(args[0].copy(), args[1].copy())
        assert_equal(r1, r3)

    @skip(reason="object arrays")
```

---

### 85. `test_empty_out` — `torch_np/numpy_tests/core/test_multiarray.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/torch_np/numpy_tests/core/test_multiarray.py:5515–5527` |
| **Classification** | `AS_IS` |
| **Module** | `torch_np` |
| **Description** | The test verifies that a matmul operation involving an empty (zero‑size) tensor returns an empty result with the correct shape and that providing an explicit out tensor with an incompatible broadcast ... |
| **LLM Dtype** | float|np.int32|np.complex128|float |
| **Shapes** | (0, 1, 1), (1, 1, 1) |
| **LLM Shape** | (5, 2)|(5, 2)|(5, 2)|(5, 2)|(5, 2)|(1, 1, 1)|(0, 1, 1)|(1, 1, 1)|(0, 1, 1)|(5, 2)|(5, 2)|(1, 3)|(5, 7)|(5, 2, 2)|(5, 2)|(10, 2)|(5, 2)|(5, 2)|(5, 2, 2)|(5, 3)|(3, 7)|(5, 6) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_empty_out(self):
        # Check that the output cannot be broadcast, so that it cannot be
        # size zero when the outer dimensions (iterator size) has size zero.
        arr = np.ones((0, 1, 1))
        out = np.ones((1, 1, 1))
        if self.matmul(arr, arr).shape != (0, 1, 1):
            raise AssertionError(
                f"shape mismatch: {self.matmul(arr, arr).shape} != (0, 1, 1)"
            )

        with pytest.raises((RuntimeError, ValueError)):
            self.matmul(arr, arr, out=out)

```

---

### 86. `test_out_contiguous` — `torch_np/numpy_tests/core/test_multiarray.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/torch_np/numpy_tests/core/test_multiarray.py:5528–5554` |
| **Classification** | `AS_IS` |
| **Module** | `torch_np` |
| **Description** | The test verifies that the matmul implementation correctly writes results into provided output buffers, handling both non‑contiguous and contiguous memory layouts as well as transposed views. It compa... |
| **Dtypes** | torch.float |
| **LLM Dtype** | float|np.int32|np.complex128|float|float|float|float|float|float|float|float|float|float|float|float |
| **Shapes** | (10, 2), (5, 2), (5, 2, 2) |
| **LLM Shape** | (5, 2)|(5, 2)|(5, 2)|(0, 1, 1)|(1, 1, 1)|(0, 1, 1)|(5, 2)|(5, 2)|(5, 2, 2)|(5, 2)|(10, 2)|(5, 2)|(5, 2)|(5, 2)|..., 0|[:, 0, 0]|[:, 0, 0]|[::2, :]|.T|(5, 2)|(5, 3)|(3, 7)|(5, 6)|[:, ::2]|(10,)|(6,)|(3, 0) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_out_contiguous(self):
        a = np.ones((5, 2), dtype=float)
        b = np.array([[1, 3], [5, 7]], dtype=float)
        v = np.array([1, 3], dtype=float)
        tgt = np.dot(a, b)
        tgt_mv = np.dot(a, v)

        # test out non-contiguous
        out = np.ones((5, 2, 2), dtype=float)
        c = self.matmul(a, b, out=out[..., 0])
        assert_array_equal(c, tgt)
        c = self.matmul(a, v, out=out[:, 0, 0])
        assert_array_equal(c, tgt_mv)
        c = self.matmul(v, a.T, out=out[:, 0, 0])
        assert_array_equal(c, tgt_mv)

        # test out contiguous in only last dim
        out = np.ones((10, 2), dtype=float)
        c = self.matmul(a, b, out=out[::2, :])
        assert_array_equal(c, tgt)

        # test transposes of out, args
        out = np.ones((5, 2), dtype=float)
        c = self.matmul(b.T, a.T, out=out.T)
        assert_array_equal(out, tgt)

    @xfailIfTorchDynamo
```

---

### 87. `test_blockwise_mxfp8_nvfp4_error_messages` — `test_scaled_matmul_cuda.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_scaled_matmul_cuda.py:2285–2377` |
| **Classification** | `AS_IS` |
| **Module** | `scaled_matmul_cuda` |
| **Description** | The test verifies that block‑wise scaling for FP8 matmul raises informative ValueErrors when the provided scale tensors have incorrect sizes or dtypes. It constructs full‑precision matrices, converts ... |
| **Dtypes** | torch.bfloat16, torch.float8_e4m3fn |
| **LLM Dtype** | torch.float8_e4m3fn|torch.float8_e8m0fnu|e4m3_type|scale_dtype |
| **Shapes** | (M, K), (N, K), (expected_a_size), (expected_b_size) |
| **LLM Shape** | (M, K)|(N, K)|BLOCK_SIZE_K|BLOCK_SIZE_MN|K|M|N|1024|512|2048|16|32|128|num_k_blocks|padded_num_k_blocks|expected_a_size|expected_b_size|ceil_div(M, BLOCK_SIZE_MN)|ceil_div(N, BLOCK_SIZE_MN)|ceil_div(num_k_blocks, 4)|expected_a_size - 1 |
| **Device Classification** | device_generic |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_blockwise_mxfp8_nvfp4_error_messages(self, device, recipe) -> None:
        if recipe == "mxfp4" and SM120OrLater:
            raise unittest.SkipTest("MXFP4 on CUDA only supported on B200/B300")
        M, K, N = (1024, 512, 2048)
        BLOCK_SIZE_K = 16 if recipe == "nvfp4" else 32
        BLOCK_SIZE_MN = 128
        fill_value = 0.5
        scale_dtype = torch.float8_e4m3fn if recipe == "nvfp4" else torch.float8_e8m0fnu

        x = torch.full((M, K), fill_value, device=device)
        y = torch.full((N, K), fill_value, device=device)

        if recipe == "mxfp8":
            x_lowp = x.to(e4m3_type)
            y_lowp = y.to(e4m3_type).t()
        else:  # nvfp4 #mxfp4
            x_lowp = _bfloat16_to_float4_e2m1fn_x2(x.bfloat16())
            y_lowp = _bfloat16_to_float4_e2m1fn_x2(y.bfloat16()).t()

        num_k_blocks = ceil_div(K, BLOCK_SIZE_K)
        padded_num_k_blocks = ceil_div(num_k_blocks, 4) * 4
        expected_a_size = BLOCK_SIZE_MN * ceil_div(M, BLOCK_SIZE_MN) * padded_num_k_blocks
        expected_b_size = BLOCK_SIZE_MN * ceil_div(N, BLOCK_SIZE_MN) * padded_num_k_blocks

        block = (
            ScalingType.BlockWise1x16
            if recipe == "nvfp4"
            else ScalingType.BlockWise1x32
        )
        if torch.version.hip:
            swizzle = SwizzleType.NO_SWIZZLE
        else:
            swizzle = SwizzleType.SWIZZLE_32_4_4

        # Test wrong scale tensor size for scale_a with correct dtype
        with self.assertRaisesRegex(
            ValueError,
            f".*For Block[W,w]ise.*scaling.*scale_a should have {expected_a_size} "
            f"elements.*"
            ,
        ):
            incorrect_size_a = torch.ones(expected_a_size - 1, device=device, dtype=scale_dtype)
            correct_size_b = torch.ones(expected_b_size, device=device, dtype=scale_dtype)

            scaled_mm_wrap(
                x_lowp,
                y_lowp,
                scale_a=incorrect_size_a,
                scale_recipe_a=block,
                scale_b=correct_size_b,
                scale_recipe_b=block,
                swizzle_a=swizzle,
                swizzle_b=swizzle,
                out_dtype=torch.bfloat16,
            )

        # Test wrong scale tensor size for scale_b with correct dtype
        with self.assertRaisesRegex(
            ValueError,
            f"For Block[W,w]ise.*scaling.*scale_b should have {expected_b_size} "
            f"elements.*"
            ,
        ):
            correct_size_a = torch.ones(expected_a_size, device=device, dtype=scale_dtype)
            incorrect_size_b = torch.ones(expected_b_size + 1, device=device, dtype=scale_dtype)
            scaled_mm_wrap(
                x_lowp,
                y_lowp,
                scale_a=correct_size_a,
                scale_recipe_a=block,
                scale_b=incorrect_size_b,
                scale_recipe_b=block,
                swizzle_a=swizzle,
                swizzle_b=swizzle,
                out_dtype=torch.bfloat16,
            )

        # Test non-contiguous scale tensors with correct dtype
        with self.assertRaisesRegex(
            ValueError,
            "For Block[W,w]ise.*scaling.*both scales should be contiguous"
            ,
        ):
            non_contiguous_a = torch.ones(expected_a_size * 2, device=device, dtype=scale_dtype)[::2]
            contiguous_b = torch.ones(expected_b_size, device=device, dtype=scale_dtype)
            scaled_mm_wrap(
                x_lowp,
                y_lowp,
                scale_a=non_contiguous_a,
                scale_b=contiguous_b,
                out_dtype=torch.bfloat16,
            )

```

---

### 88. `test_honor_sm_carveout` — `test_scaled_matmul_cuda.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_scaled_matmul_cuda.py:1755–1826` |
| **Classification** | `AS_IS` |
| **Module** | `scaled_matmul_cuda` |
| **Description** | The test verifies that the experimental SM carveout setting is correctly honored by the scaled FP8 matrix multiplication kernel on CUDA. It toggles the carveout value, runs a large FP8 matmul, and che... |
| **Dtypes** | torch.bfloat16, torch.float32 |
| **LLM Dtype** | torch.float32|torch.bfloat16|e4m3_type |
| **Shapes** | (8192, 2048) |
| **LLM Shape** | 8192, 2048|8192, 2048 |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_honor_sm_carveout(self) -> None:
        torch.manual_seed(42)

        x = torch.randn(8192, 2048, device="cuda", dtype=torch.float32)
        y = torch.randn(8192, 2048, device="cuda", dtype=torch.float32).t()
        x_scales = tensor_to_scale(x, e4m3_type, dim=1).reciprocal()
        y_scales = tensor_to_scale(y, e4m3_type, dim=0).reciprocal()
        x_fp8 = to_fp8_saturated(x / x_scales, e4m3_type)
        y_fp8 = to_fp8_saturated(y / y_scales, e4m3_type)

        cu_count = torch.cuda.get_device_properties().multi_processor_count
        carveout = 66 if torch.version.cuda else cu_count // 8

        with tempfile.NamedTemporaryFile() as f:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                self.assertIsNone(torch._C._get_sm_carveout_experimental())
                scaled_mm_wrap(x_fp8, y_fp8, scale_a=x_scales, scale_b=y_scales, out_dtype=torch.bfloat16)
                torch._C._set_sm_carveout_experimental(0)
                self.assertEqual(torch._C._get_sm_carveout_experimental(), 0)
                scaled_mm_wrap(x_fp8, y_fp8, scale_a=x_scales, scale_b=y_scales, out_dtype=torch.bfloat16)
                torch._C._set_sm_carveout_experimental(66)
                self.assertEqual(torch._C._get_sm_carveout_experimental(), 66)
                scaled_mm_wrap(x_fp8, y_fp8, scale_a=x_scales, scale_b=y_scales, out_dtype=torch.bfloat16)
                torch._C._set_sm_carveout_experimental(None)
                self.assertIsNone(torch._C._get_sm_carveout_experimental())
                scaled_mm_wrap(x_fp8, y_fp8, scale_a=x_scales, scale_b=y_scales, out_dtype=torch.bfloat16)

            prof.export_chrome_trace(f.name)
            if torch.version.hip:
                with open(f.name) as file:
                    events = [evt for evt in json.load(file)["traceEvents"] if evt.get("cat", "") == "kernel"]
                # events were returned out of order; need to be sorted on "ts" timestamp
                events = sorted(events, key=lambda x: x['ts'])
                # ROCm carveout is invisible except for kernels running slower on fewer CUs
                no_carveout, carveout_0, carveout, no_carveout_again = [float(evt.get("dur", "0.0")) for evt in events]
                if True or not (no_carveout < carveout and carveout_0 < carveout and no_carveout_again < carveout):  # noqa: SIM222
                    # something went wrong, print more info to help debug flaky test
                    print("ROCm debug info for test_honor_sm_carveout")
                    print("cu_count", cu_count)
                    print("no_carveout", no_carveout)
                    print("carveout_0", carveout_0)
                    print("carveout", carveout)
                    print("no_carveout_again", no_carveout_again)
                self.assertTrue(no_carveout < carveout)
                self.assertTrue(carveout_0 < carveout)
                self.assertTrue(no_carveout_again < carveout)
                # ROCm carveout will create new streams when enabled, and go back to the original stream when disabled
                no_carveout, carveout_0, carveout, no_carveout_again = [int(evt.get("tid", "0")) for evt in events]
                self.assertTrue(no_carveout == no_carveout_again)
                self.assertTrue(no_carveout == carveout_0)
                self.assertTrue(no_carveout != carveout)
                self.assertTrue(carveout_0 != carveout)
            else:
                with open(f.name) as file:
                    no_carveout, carveout_0, carveout_66, no_carveout_again = [
                        math.prod(evt.get("args", {}).get("grid", []))
                        for evt in json.load(file)["traceEvents"]
                        if evt.get("cat", "") == "kernel"
                    ]

                self.assertEqual(no_carveout, no_carveout_again)
                if SM100OrLater:
                    # expected failure
                    # CUTLASS only supports SM carveout via green contexts on SM100
                    self.assertEqual(no_carveout, carveout_66)
                    self.assertEqual(carveout_66, carveout_0)
                else:
                    # correct behavior
                    self.assertNotEqual(no_carveout, carveout_66)
                    self.assertNotEqual(carveout_66, carveout_0)

    @skipXPU
```

---

### 89. `test_scalar_output` — `torch_np/numpy_tests/core/test_multiarray.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/torch_np/numpy_tests/core/test_multiarray.py:5324–5340` |
| **Classification** | `AS_IS` |
| **Module** | `torch_np` |
| **Description** | The test verifies that the matmul implementation correctly handles inputs that produce a scalar (or 1‑D) output, including type casting across multiple dtypes and boolean inputs. It checks both left‑h... |
| **LLM Dtype** | self.types[1:]|dt|"O"|"?" |
| **Shapes** | (1,)|(1, 2)|(1, -1)|(2,)|(1, 2)|(1, -1)|(2, 2) |
| **LLM Shape** | (2, 1, 1)|(2,)|()|1|(1, 1)|(1,)|(1, -1)|(-1, 1)|(2,)|(3, 4)|(1, -1)|(2,)|(True, True)|(1, -1)|[True, True]|(1, -1)|[1, 2]|(-1, 1)|(1, -1)|[True, True]|(1, 2)|[[1, 2], [3, 4]]|2|(7, 10)|[True, False]|[[True, False], [False, True]]|(2,) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_scalar_output(self):
        vec1 = np.array([2])
        vec2 = np.array([3, 4]).reshape(1, -1)
        tgt = np.array([6, 8])
        for dt in self.types[1:]:
            v1 = vec1.astype(dt)
            v2 = vec2.astype(dt)
            res = self.matmul(v1, v2)
            assert_equal(res, tgt)
            res = self.matmul(v2.T, v1)
            assert_equal(res, tgt)

        # boolean type
        vec = np.array([True, True], dtype="?").reshape(1, -1)
        res = self.matmul(vec[:, 0], vec)
        assert_equal(res, True)

```

---

### 90. `test_scaled_mm_block_wise_numerics` — `test_scaled_matmul_cuda.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_scaled_matmul_cuda.py:1390–1412` |
| **Classification** | `AS_IS` |
| **Module** | `scaled_matmul_cuda` |
| **Description** | The test verifies the numerical correctness of block‑wise scaled matrix multiplication (FP8) on CUDA across multiple block configurations, output data types, and scaling strategies. It generates A and... |
| **Dtypes** | torch.bfloat16, torch.float32, torch.float16, torch.float8_e4m3fn, torch.uint8 |
| **LLM Dtype** | torch.bfloat16|torch.float32 |
| **Shapes** | (0, 255), (M, K), (N, K), (scale_shape, val) |
| **LLM Shape** | (1, 1)|(128, 1)|(1, 128)|(256, 768, 512)|(384, 128, 1280)|(512, 512, 512)|[M, K]|[N, K]|[M, K // 128]|[1, M]|[1, N]|[L4, N // 128]|[L4, M // 128]|[M, K // 128]|[1, M]|[N, K // 128]|[1, N]|[M, K]|[M, K // 128]|[M]|[K // 128]|[0]|[1]|[M // 128, L4]|[L4, M // 128] |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_scaled_mm_block_wise_numerics(self, output_dtype, lhs_block, rhs_block, M, N, K, test_case):
        """
        subsume test_scaled_mm_vs_emulated_block_wise for random inputs, random scales,
        do some other functional tests as well.

        # Inputs (as generated are):
        #   A: [M, K]
        #   B: [N, K]
        # then scales are, for the 3 combinations:
        #   1x128 x 1x128:
        #     As: [M, K // 128], stride: [1, M] -> scale.t().contiguous().t()
        #     Bs: [N, K // 128], stride: [1, N] -> scale.t().contiguous().t()
        #   1x128 x 128x128
        #     L4 = round_up(K // 128, 4)
        #     As: [M, K // 128], stride: [1, M]   -> scale.t().contiguous().t()
        #     Bs: [L4, N // 128], stride: [1, L4] -> scale.t()
        #   128x128 x 1x128
        #     L4 = round_up(K // 128, 4)
        #     As: [L4, M // 128], stride: [1, L4]
        #     Bs: [N, K // 128], stride: [1, N]
        """
        torch.manual_seed(42)

```

---

### 91. `test_scaled_mm_vs_emulated_row_wise` — `test_scaled_matmul_cuda.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_scaled_matmul_cuda.py:1293–1326` |
| **Classification** | `NEEDS_REFACTOR` |
| **Module** | `scaled_matmul_cuda` |
| **Description** | The test verifies that the row‑wise scaled matrix multiplication implementation (`_scaled_mm`) produces the same results as an explicit emulation that dequantizes FP8 inputs, applies per‑row/column sc... |
| **Dtypes** | torch.bfloat16, torch.float16, torch.float32, torch.float |
| **LLM Dtype** | torch.bfloat16|torch.float16|torch.float32|e4m3_type|base_dtype|torch.ones_like|input_dtype|output_dtype|e4m3_type|e4m3_type |
| **Shapes** | (128, 512, 256), (M, K), (N), (N, K) |
| **LLM Shape** | (128, 512, 256)|(N,)|(M, K)|(N, K) |
| **Device Classification** | device_generic |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_scaled_mm_vs_emulated_row_wise(self, base_dtype, shapes, device):
        M, K, N = shapes
        # Fp32 out_dtype is only supported by cuBLAS, which however only started
        # shipping row-wise kernels in CUDA 12.9, and only for sm90+.
        if base_dtype is torch.float32:
            if torch.version.hip:
                raise unittest.SkipTest("hipblaslt rowwise _scaled_mm only supports BFloat16")
            if torch.cuda.is_available() and _get_torch_cuda_version() < (12, 9):
                raise unittest.SkipTest("Need CUDA 12.9+ for row-wise fp8 w/ cuBLAS")
            if torch.cuda.is_available() and torch.cuda.get_device_capability() < (9, 0):
                raise unittest.SkipTest("Need sm90+ for row-wise fp8 w/ cuBLAS")

        if base_dtype is torch.float16:
            if torch.version.hip:
                raise unittest.SkipTest("hipblaslt rowwise _scaled_mm only supports BFloat16")
            if torch.cuda.is_available() and torch.cuda.get_device_capability() < (9, 0):
                raise unittest.SkipTest("Need sm90+ for row-wise fp8 w/ cuBLAS")

        torch.manual_seed(42)
        input_dtype = e4m3_type
        output_dtype = base_dtype

        x = random_matrix_with_scaled_reduction_dim(M, K, dtype=base_dtype, device=device, reduction_dim=-1)
        y = random_matrix_with_scaled_reduction_dim(N, K, dtype=base_dtype, device=device, reduction_dim=-1).t()
        bias = None
        if base_dtype in {torch.bfloat16, torch.float16}:
            bias = torch.randn((N,), device=device, dtype=base_dtype)

        x_scales = tensor_to_scale(x, input_dtype, dim=1).float()
        y_scales = tensor_to_scale(y, input_dtype, dim=0).float()

        x_fp8 = to_fp8_saturated(x * x_scales, e4m3_type)
        y_fp8 = to_fp8_saturated(y * y_scales, e4m3_type)

```

---

### 92. `test_scaled_mm_vs_emulated` — `test_scaled_matmul_cuda.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_scaled_matmul_cuda.py:965–1019` |
| **Classification** | `NEEDS_REFACTOR` |
| **Module** | `scaled_matmul_cuda` |
| **Description** | The test compares the result of the native scaled FP8 matrix multiplication (scaled_mm_wrap) against an emulated FP8 matmul implementation (mm_float8_emulated) across different base dtypes and tensor ... |
| **Dtypes** | torch.float16, torch.bfloat16, torch.float32 |
| **LLM Dtype** | torch.float16|torch.bfloat16|torch.float32|e4m3_type|torch.float32|input_dtype|base_dtype|output_dtype|compare_type |
| **Shapes** | (K, M), (K, N), (M, K), (N, K) |
| **LLM Shape** | 16|32|16|M|K|N|K|M|N|K |
| **Device Classification** | device_generic |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_scaled_mm_vs_emulated(self, base_dtype, x_cm, y_cm, device):
        # Blackwell (SM_10) supports all possible layout permutations, while Hopper only TN
        if torch.cuda.is_available():
            if (x_cm, y_cm) != (True, False) and torch.cuda.get_device_properties(0).major != 10:
                raise unittest.SkipTest("Unsupported layout on the architecture")
        torch.manual_seed(42)
        input_dtype = e4m3_type
        output_dtype = base_dtype
        compare_type = torch.float32

        M, N, K = 16, 32, 16
        x = torch.randn(M, K, device=device, dtype=base_dtype) if x_cm else torch.randn(K, M, device=device, dtype=base_dtype).t()
        y = torch.randn(K, N, device=device, dtype=base_dtype) if y_cm else torch.randn(N, K, device=device, dtype=base_dtype).t()

        x_scale = tensor_to_scale(x, input_dtype).float()
        y_scale = tensor_to_scale(y, input_dtype).float()


        x_fp8 = to_fp8_saturated(x * x_scale, input_dtype)
        y_fp8 = to_fp8_saturated(y * y_scale, input_dtype)

        # Calculate actual F8 mm
        out_scaled_mm = scaled_mm_wrap(
            x_fp8,
            y_fp8,
            scale_a=x_scale.reciprocal(),
            scale_b=y_scale.reciprocal(),
            out_dtype=output_dtype
        )

        # Calculate emulated F8 mm
        out_emulated = mm_float8_emulated(
            x_fp8,
            x_scale,
            y_fp8,
            y_scale,
            output_dtype
        )

        if output_dtype != base_dtype:
            out_scaled_mm = out_scaled_mm.to(compare_type)
            out_scaled_mm = out_scaled_mm / tensor_to_scale(out_scaled_mm, input_dtype)

            out_emulated = out_emulated.to(compare_type)
            out_emulated = out_emulated / tensor_to_scale(out_emulated, input_dtype)

        if base_dtype in {torch.bfloat16, torch.float16}:
            atol, rtol = 7e-2, 7e-2
        else:
            atol, rtol = 3e-3, 3e-3

        torch.testing.assert_close(out_scaled_mm, out_emulated, atol=atol, rtol=rtol)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize("base_dtype", [torch.float16, torch.bfloat16, torch.float32])
```

---

### 93. `test_register_fallthrough` — `test_python_dispatch.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_python_dispatch.py:799–818` |
| **Classification** | `AS_IS` |
| **Module** | `python_dispatch` |
| **Description** | The test registers a fallthrough kernel for the ATen "mm" operation, then verifies that under CPU autocast to bfloat16 the "mm" op still produces float32 results while other ops like "matmul" follow t... |
| **Dtypes** | torch.bfloat16, torch.float32 |
| **LLM Dtype** | torch.float32|torch.bfloat16 |
| **Shapes** | (2, 3), (3, 2) |
| **LLM Shape** | 2, 3|3, 2 |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_register_fallthrough(self):
        with _scoped_library("aten", "IMPL") as my_lib:
            my_lib.impl("mm", fallthrough_kernel, "AutocastCPU")

            a = torch.randn(2, 3, device="cpu", dtype=torch.float32)
            b = torch.randn(3, 2, device="cpu", dtype=torch.float32)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                # dtype for mm should be float32 since we registered a fallthrough
                self.assertEqual(torch.mm(a, b).dtype, torch.float32)
                # ops that don't have a fallthrough registered should not be affected
                self.assertEqual(torch.matmul(a, b).dtype, torch.bfloat16)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            # default behavior should have been restored
            self.assertEqual(torch.mm(a, b).dtype, torch.bfloat16)


instantiate_parametrized_tests(TestPythonRegistration)


```

---

### 94. `test_pre_dispatch_functionalization_view_op` — `test_proxy_tensor.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_proxy_tensor.py:502–521` |
| **Classification** | `AS_IS` |
| **Module** | `proxy_tensor` |
| **Description** | The test traces a function that functionalizes a tensor, performs matmul, transpose, addition, and a view reshape, then unwraps the functional tensor. It verifies that the traced graph contains the ex... |
| **Shapes** | (4, 4) |
| **LLM Shape** | (4, 4)|[2, 8]|(3,)|(1, 3)|(3) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_pre_dispatch_functionalization_view_op(self):
        def f(x):
            a = FunctionalTensorMode(pre_dispatch=True, export=True)
            with a:
                x_unwrapped = FunctionalTensor.to_functional(x)
                y = torch.matmul(x_unwrapped, x_unwrapped)
                x_unwrapped = x_unwrapped.transpose(1, 0)
                y = y + x_unwrapped
                y = y.view(2, 8)
                y_unwrapped = torch._from_functional_tensor(y.elem)
                return y_unwrapped

        from torch._dispatch.python import enable_python_dispatcher

        with enable_python_dispatcher():
            inp = torch.randn(4, 4)
            gm = make_fx(f, pre_dispatch=True)(inp)

        # TODO actually not decompose
        self.assertExpectedInline(gm.code.strip(), """\
```

---

### 95. `test_pre_dispatch_mode_stack` — `test_proxy_tensor.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_proxy_tensor.py:161–175` |
| **Classification** | `AS_IS` |
| **Module** | `proxy_tensor` |
| **Description** | The test defines a simple function that multiplies an input tensor by a 4x4 tensor of ones, runs it under the Python dispatcher, and then traces it with make_fx(pre_dispatch=True). It asserts that the... |
| **Shapes** | (4, 4) |
| **LLM Shape** | [4, 4]|(4, 4)|(4) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_pre_dispatch_mode_stack(self):
        def f(a):
            b = torch.ones(4, 4)
            return torch.matmul(a, b)
        # We expect to see matmul in the trace - it should NOT be decomposed into mm.
        # Also, torch.ones() doesn't show up in the trace.
        # This is annoying but expected: ones() never dispatches to the Autograd dispatch key,
        # so our mode never sees it - it goes directly to the BackendSelect key.
        inp = torch.ones(4, 4)
        # Test that make_fx(pre_dispatch=True) clears caches properly.
        from torch._dispatch.python import enable_python_dispatcher
        with enable_python_dispatcher():
            out1 = f(inp)
        fx_g = make_fx(f, pre_dispatch=True)(inp)
        self.assertExpectedInline(fx_g.code.strip(), """\
```

---

### 96. `test_zero_dim_tensorwise` — `test_scaled_matmul_cuda.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/test_scaled_matmul_cuda.py:1729–1754` |
| **Classification** | `NEEDS_REFACTOR` |
| **Module** | `scaled_matmul_cuda` |
| **Description** | The test verifies that the scaled FP8 matrix multiplication wrapper correctly handles tensors with a zero-sized dimension, across different zero-dimension choices and with/without torch.compile. It ch... |
| **Dtypes** | torch.bfloat16, torch.float |
| **LLM Dtype** | e4m3_type|e4m3_type|torch.bfloat16|torch.float|torch.float|torch.float32|torch.float32|torch.float32|e4m3_type|e4m3_type|e4m3_type|e4m3_type |
| **Shapes** | (M, K), (N, K) |
| **LLM Shape** | 32, 32, 32|M, K|N, K|8192, 2048|8192, 2048 |
| **Device Classification** | device_generic |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_zero_dim_tensorwise(self, which_dim_zero, use_torch_compile, device) -> None:
        x_dtype, y_dtype = e4m3_type, e4m3_type
        out_dtype = torch.bfloat16
        M, K, N = 32, 32, 32
        if which_dim_zero == 0:
            M = 0
        elif which_dim_zero == 1:
            K = 0
        elif which_dim_zero == 2:
            N = 0

        x_fp8 = torch.zeros(M, K, device=device).to(x_dtype)
        y_fp8 = torch.zeros(N, K, device=device, dtype=y_dtype).t()
        out_fp32 = torch.mm(x_fp8.to(torch.float), y_fp8.to(torch.float))
        scale_a = torch.tensor(float('-inf'), device=device)
        scale_b = torch.tensor(float('-inf'), device=device)
        f = scaled_mm_wrap
        if use_torch_compile:
            f = torch.compile(scaled_mm_wrap)
        out_fp8 = f(x_fp8, y_fp8, scale_a, scale_b, out_dtype=out_dtype)
        self.assertEqual(out_dtype, out_fp8.dtype)
        self.assertEqual(out_fp32, out_fp8.to(torch.float))

    @unittest.skipIf(IS_WINDOWS, "Windows doesn't support row-wise scaling")
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @unittest.skipIf(not SM90OrLater, "sm89 kernel isn't opted into carveout yet")
```

---

### 97. `test_profiler_elapsed_time` — `cpp_extensions/open_registration_extension/torch_openreg/tests/test_profiler.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_profiler.py:224–248` |
| **Classification** | `AS_IS` |
| **Module** | `cpp_extensions` |
| **Description** | The test runs a series of 100x100 matrix multiplications on the OpenReg device within a custom stream, profiles the execution, and verifies that the recorded CPU time for the matmul operations is grea... |
| **Shapes** | (100, 100) |
| **LLM Shape** | 100, 100|100, 100|10, 10|10, 10|10, 10|10, 10|10, 10 |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_profiler_elapsed_time(self):
        """Test that event elapsed time is correctly calculated."""
        with autograd_profile(use_device="openreg") as prof:
            stream = torch.Stream(device="openreg")

            with stream:
                x = torch.randn(100, 100, device="openreg")
                y = torch.randn(100, 100, device="openreg")
                # Multiple matmuls to ensure measurable time
                for _ in range(10):
                    z = x @ y
                    x = z

            stream.synchronize()

        events = prof.function_events
        # Check that operations have non-zero duration
        compute_events = [
            e for e in events if "aten::mm" in e.name or "aten::matmul" in e.name
        ]
        if compute_events:
            total_time = sum(e.cpu_time_total for e in compute_events)
            self.assertGreater(total_time, 0)

    @skipIfTorchDynamo()
```

---

### 98. `test_scalar_type` — `onnx/test_pytorch_onnx_onnxruntime.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/onnx/test_pytorch_onnx_onnxruntime.py:7127–7128` |
| **Classification** | `AS_IS` |
| **Module** | `onnx` |
| **Description** | The test exports several small models to ONNX and runs them with ONNX Runtime, checking that scalar tensors and mixed‑type operations (size, arithmetic, comparisons, matmul, full, and concatenation) a... |
| **Dtypes** | torch.float32, torch.int32 |
| **LLM Dtype** | torch.float32|torch.int32|torch.float32|torch.float16 |
| **Shapes** | (12.0), (2, 3), (3, 3), (3, 4) |
| **LLM Shape** | (2, 3)|(2, 3)|(2, 3)|(3, 3)|(3, 3)|(3, 4)|[0.5]|[1.5]|(2, 3) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_scalar_type(self):
        class ArithmeticModel(torch.nn.Module):
```

---

### 99. `test_wrap_with_multiple_ops` — `dynamo/test_wrap_inductor_compiled_regions.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/dynamo/test_wrap_inductor_compiled_regions.py:220–227` |
| **Classification** | `AS_IS` |
| **Module** | `dynamo` |
| **Description** | The test compiles a function containing a matmul, ReLU, and addition using torch.compile with the Inductor backend and region wrapping enabled, then verifies that the compiled region is reported and t... |
| **Shapes** | (4, 4) |
| **LLM Shape** | 4, 4 | 4 |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_wrap_with_multiple_ops(self):
        """Test wrapping with a function that has multiple operations"""

        @torch.compile(
            backend="inductor",
            options={"wrap_inductor_compiled_regions": True},
            fullgraph=True,
        )
```

---

### 100. `test_mm_padding` — `inductor/test_deterministic.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_deterministic.py:51–54` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test compiles a simple matrix multiplication with torch.compile, runs it on two 2049x2049 random GPU tensors, and verifies the result matches the eager matmul. It also checks whether the padding o... |
| **Shapes** | (2049, 2049) |
| **LLM Shape** | [2049, 2049]|[2048, 2048]|2048|(2048, 2048) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_mm_padding(self, deterministic):
        with inductor_config.patch(deterministic=deterministic):

            @torch.compile()
```

---

### 101. `test_max_autotune_cutlass_backend_no_fusion_dtype_mismatch` — `inductor/test_cutlass_backend.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_cutlass_backend.py:1134–1144` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test defines a matrix‑multiply function that casts the fp16 result to float32 and scales it, then verifies that the Cutlass backend does not fuse this operation when the output dtype differs from ... |
| **Dtypes** | torch.float32 |
| **LLM Dtype** | torch.float32|torch.int8 |
| **LLM Shape** | (100, 16)|(32, 16) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_max_autotune_cutlass_backend_no_fusion_dtype_mismatch(self):
        def mm(a, b):
            # this should not be fused, since the output dtype is different from the matmul dtype
            return (a @ b).to(torch.float32) * 0.00001

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            fp16=True, expected_fuse_count=0, mm=mm
        )

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
```

---

### 102. `test_partial_input_gen_fns` — `inductor/test_custom_op_autotune.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_custom_op_autotune.py:1390–1396` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that custom operator autotuning works correctly when the input generation functions are provided for only a subset of the operator's inputs. It registers a fake custom op that perfor... |
| **LLM Dtype** | self.dtype | dtype=self.dtype | dtype=self.dtype | dtype=self.dtype | x.dtype |
| **Shapes** | (256, 128), (64, 256), (t) |
| **LLM Shape** | (8, 32, 64) | (16, 32, 64) | (32, 32, 64) | (64, 32, 64) | (128, 32, 64) | (64, 256) | (256, 128) | (x.shape[0], weight.shape[1]) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_partial_input_gen_fns(self):
        """Test autotuning when input_gen_fns covers only some inputs.

        The uncovered inputs should fall back to ir_node_to_tensor with concrete hints.
        """
        test_op_name = f"test_lib::partial_gen_{id(self)}"

```

---

### 103. `test_torch_cond_with_shape_accessing_implementations` — `inductor/test_custom_op_autotune.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_custom_op_autotune.py:918–926` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test registers a custom operator with two implementations—one simple matmul and one that accesses tensor shapes, reshapes, permutes, and uses batched matrix multiplication. It then autotunes the o... |
| **LLM Dtype** | dtype=mat1.dtype |
| **Shapes** | (64, 32), (8, 64), (t) |
| **LLM Shape** | mat1.shape|mat2.shape|mat2.shape[1]|m, k = mat1.shape|n = mat2.shape[1]|k_splits = 4|k % k_splits|k // k_splits|m, k_splits, k_parts|(1, 0, 2)|k_splits, k_parts, n|mat1.shape[0]|mat2.shape[1]|dim: 0|split_points=[4, 16] |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_torch_cond_with_shape_accessing_implementations(self):
        """Test torch.cond dispatch with implementations that access tensor shapes.

        Validates that implementations like decompose_k that access tensor shapes
        (e.g., `m, k = mat1.shape`) work correctly with torch.cond dispatch.
        The fix uses _build_cond_dispatch_graph to pre-trace each implementation.
        """
        test_op_name = f"test_lib::shape_access_cond_{id(self)}"

```

---

### 104. `test_min_speedup_threshold_api` — `inductor/test_custom_op_autotune.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_custom_op_autotune.py:648–651` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test registers a custom matrix‑multiplication op with an autotuning configuration that includes a min_speedup_threshold, compiles a model using torch.compile, runs it, and checks the output agains... |
| **LLM Dtype** | self.dtype|x.dtype |
| **Shapes** | (256, 128), (64, 256), (t) |
| **LLM Shape** | (256, 128)|(64, 256)|(256, 128)|x.shape[0]|weight.shape[1] |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_min_speedup_threshold_api(self):
        """Test that min_speedup_threshold parameter is accepted and compilation works."""
        test_op_name = f"test_lib::min_speedup_{id(self)}"

```

---

### 105. `test_in_out_buffer` — `inductor/test_cpu_repro.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_cpu_repro.py:4366–4376` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test compiles a simple matmul‑transpose function with inplace buffer optimization enabled, checks that the generated C++ code contains the in‑out buffer marker, and verifies that the compiled func... |
| **LLM Dtype** | torch.float32 |
| **Shapes** | (1, 2, 8, 4) |
| **LLM Shape** | (1, 3, 10, 10)|(1, 2, 8, 4)|(1, 2, 8, 4)|(1024, 32, 128)|(4096, 1, 32)|(64, 128, 16, 32)|(65536, 512, 32, 1)|(2, 32, 4, 4)|[0, 2, 1, 3]|[1024, -1, 32]|[0, 2, 1] |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_in_out_buffer(self):
        def fn(x, y):
            z = torch.matmul(x, y.transpose(-1, -2)) / 8.0
            return z

        inps = [torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4)]
        fn_opt = torch.compile(fn, backend="inductor")
        _, code = run_and_get_cpp_code(fn_opt, *inps)
        self.assertTrue("in_out_ptr" in code)
        self.assertEqual(fn_opt(*inps), fn(*inps))

```

---

### 106. `test_max_autotune_precompile_non_contiguous` — `inductor/test_ck_backend.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_ck_backend.py:207–235` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that a matrix multiplication with non‑contiguous input tensors correctly falls back when using the CK backend under autotuning. It compiles the matmul with torch.compile, runs it on ... |
| **Dtypes** | torch.float16 |
| **LLM Dtype** | torch.float16 |
| **LLM Shape** | (50257, 32768)|(1, 50304)|(32768, 768)|(768, 1) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_max_autotune_precompile_non_contiguous(self, max_autotune_gemm_backends):
        """
        Make sure the matmul with non-contiguous inputs can fallback
        """

        tensor_options = {"device": "cuda", "dtype": torch.float16}

        a = torch.empty_strided((50257, 32768), (1, 50304), **tensor_options)
        b = torch.empty_strided((32768, 768), (768, 1), **tensor_options)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "autotune_in_subproc": True,
                    "max_autotune_gemm_backends": max_autotune_gemm_backends,
                    "compile_threads": 16,
                    "rocm.ck_dir": self.ck_dir,
                    "rocm.ck_max_profiling_configs": 8,
                    "rocm.ck_tile_max_profiling_configs": 8,
                }
            ),
            tf32_off(),
        ):

            @torch.compile(dynamic=False)
```

---

### 107. `test_triton_autotuning` — `inductor/test_aot_inductor.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_aot_inductor.py:7759–7762` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that Triton's autotuning mechanism selects a valid grid configuration for a custom matmul kernel when using AOT compilation with sample inputs. It runs the model on a GPU, extracts t... |
| **Dtypes** | torch.float32, torch.int32 |
| **LLM Dtype** | torch.int32|torch.float32 |
| **Shapes** | (1024, 2048), (4096), (4096, 1024), (_M, N) |
| **LLM Shape** | (4096, 1024)|(1024, 2048)|(_M, N)|[4096]|4096 * 2046 |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_triton_autotuning(self):
        if self.device != GPU_TYPE:
            raise unittest.SkipTest("requires GPU")

```

---

### 108. `test_fx_annotations` — `higher_order_ops/test_local_map.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/higher_order_ops/test_local_map.py:897–908` |
| **Classification** | `AS_IS` |
| **Module** | `higher_order_ops` |
| **Description** | The test verifies that FX tracing correctly propagates custom annotations and placement metadata through a locally mapped function that includes a matrix multiplication and checkpointing. It captures ... |
| **LLM Dtype** | torch.int32 |
| **Shapes** | (80, 80) |
| **LLM Shape** | (8, 80)|(1, 8)|(80, 80) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_fx_annotations(self):
        @local_map(
            out_placements=((Replicate(), Replicate(), Replicate()),),
            in_placements=(
                (Replicate(), Replicate(), Replicate()),
                (Replicate(), Replicate(), Replicate()),
                None,
            ),
            redistribute_inputs=True,
            in_grad_placements=None,
            device_mesh=self.mesh,
        )
```

---

### 109. `test_serialize_infinite_sym_int` — `export/test_serialize.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/export/test_serialize.py:442–443` |
| **Classification** | `AS_IS` |
| **Module** | `export` |
| **Description** | The test defines a simple module performing matmul, addition, division, reshape, and concatenation, exports it with symbolic dynamic dimensions, serializes the ExportedProgram, and checks that the ser... |
| **Shapes** | (2, 4), (2, 7), (4, 7) |
| **LLM Shape** | (2, 4)|(4, 7)|(2, 7)|(1, symint, 3)|(10) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_serialize_infinite_sym_int(self) -> None:
        class DynamicShapeSimpleModel(torch.nn.Module):
```

---

### 110. `test_split_const_gm_with_lifted_constants` — `export/test_export.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/export/test_export.py:16119–16120` |
| **Classification** | `AS_IS` |
| **Module** | `export` |
| **Description** | The test builds a simple model with a transposed weight, ReLU, bias addition, matmul, and an arange constant, exports it, and then splits the exported graph to isolate constant sub‑graphs. It checks t... |
| **Dtypes** | torch.float16 |
| **LLM Dtype** | torch.float16 |
| **Shapes** | (4), (4, 4) |
| **LLM Shape** | (4, 4)|(4,)|(4, 4)|4 |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_split_const_gm_with_lifted_constants(self):
        class Model(torch.nn.Module):
```

---

### 111. `test_wrap_no_dispatch_mode_no_hop_invoked` — `dynamo/test_wrap_inductor_compiled_regions.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/dynamo/test_wrap_inductor_compiled_regions.py:1007–1029` |
| **Classification** | `AS_IS` |
| **Module** | `dynamo` |
| **Description** | The test verifies that when TorchDispatchMode is not active, the higher‑order op (HOP) wrapper for inductor‑compiled regions is not invoked, even with wrap_inductor_compiled_regions=True. It compiles ... |
| **Shapes** | (4, 4) |
| **LLM Shape** | (4, 4) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_wrap_no_dispatch_mode_no_hop_invoked(self):
        """
        Test that without TorchDispatchMode, the HOP is NOT invoked.

        Even when wrap_inductor_compiled_regions=True, if there's no active
        TorchDispatchMode, the wrapper should not invoke the HOP (optimization).
        This verifies that we're not paying the HOP overhead unnecessarily.
        """
        from unittest.mock import patch

        from torch._higher_order_ops.wrap import inductor_compiled_code

        # Patch it in the output_code module where it's imported and used
        patch_path = "torch._inductor.output_code.inductor_compiled_code"

        # Test WITHOUT dispatch mode - HOP should not route through a mode
        with patch(patch_path, wraps=inductor_compiled_code) as mock_hop:

            @torch.compile(
                backend="inductor",
                options={"wrap_inductor_compiled_regions": True},
                fullgraph=True,
            )
```

---

### 112. `test_function_with_kwargs` — `dynamo/test_higher_order_ops.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/dynamo/test_higher_order_ops.py:6878–6881` |
| **Classification** | `AS_IS` |
| **Module** | `dynamo` |
| **Description** | The test defines a simple function that applies a sigmoid of a matrix multiplication, wraps it with torch.utils.checkpoint using keyword arguments, and runs it through an AOT autograd backend that cou... |
| **Shapes** | (4, 4) |
| **LLM Shape** | 4, 4|4|4 |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_function_with_kwargs(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

```

---

### 113. `test_attention_export_qk_output_modes` — `onnx/ops/test_ops.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/onnx/ops/test_ops.py:1335–1346` |
| **Classification** | `AS_IS` |
| **Module** | `onnx` |
| **Description** | The test iterates over four qk_matmul_output_mode values, builds a simple module that calls torch.onnx.ops.attention, exports it to ONNX, and verifies the generated Attention node and its attributes.... |
| **LLM Dtype** | ir.DataType.FLOAT |
| **Shapes** | (batch_size, kv_num_heads, kv_seq_len, head_size), (batch_size, q_num_heads, q_seq_len, head_size) |
| **LLM Shape** | [batch_size, q_num_heads, q_seq_len, kv_seq_len]|[q_seq_len, kv_seq_len]|(batch_size, q_num_heads, q_seq_len, head_size)|(batch_size, kv_num_heads, kv_seq_len, head_size)|(batch_size, kv_num_heads, kv_seq_len, head_size)|2|4|6|8|8|64 |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_attention_export_qk_output_modes(self):
        """Test export with different QK output modes."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        for mode in [0, 1, 2, 3]:

```

---

### 114. `test_wrap_inductor_compiled_regions_option` — `dynamo/test_higher_order_ops.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/dynamo/test_higher_order_ops.py:3389–3401` |
| **Classification** | `AS_IS` |
| **Module** | `dynamo` |
| **Description** | The test verifies that the `wrap_inductor_compiled_regions` compile option causes compiled regions to be wrapped in an `inductor_compiled_code` higher‑order operation (HOP) that is visible to `DebugMo... |
| **Shapes** | (4, 4) |
| **LLM Shape** | (2, 4)|(4)|(4, 4)|(4, 4) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_wrap_inductor_compiled_regions_option(self):
        """
        Test that wrap_inductor_compiled_regions option wraps compiled regions
        in inductor_compiled_code HOP, making them visible to DebugMode.
        """
        from torch.utils._debug_mode import DebugMode

        # Test with wrapping enabled
        @torch.compile(
            backend="inductor",
            options={"wrap_inductor_compiled_regions": True},
            fullgraph=True,
        )
```

---

### 115. `test_cublas_allow_tf32` — `dynamo/test_functions.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/dynamo/test_functions.py:1202–1208` |
| **Classification** | `AS_IS` |
| **Module** | `dynamo` |
| **Description** | The test checks the behavior of a function depending on the CUDA matmul TF32 flag, returning either sin(x)+1 or cos(x)-1.... |
| **Shapes** | (2, 2) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_cublas_allow_tf32(x):
        if torch.backends.cuda.matmul.allow_tf32:
            return x.sin() + 1

        return x.cos() - 1

    @make_test
```

---

### 116. `test_compile_selective_checkpoint_outplace_op` — `dynamo/test_activation_checkpointing.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/dynamo/test_activation_checkpointing.py:1444–1453` |
| **Classification** | `AS_IS` |
| **Module** | `dynamo` |
| **Description** | The test verifies that selective activation checkpointing works with out-of-place operations when compiled with torch.compile. It defines a composite function involving matmul, selu, sigmoid, and relu... |
| **Shapes** | (4, 4) |
| **LLM Shape** | 4|4 |
| **Device Classification** | device_generic |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_compile_selective_checkpoint_outplace_op(self, device, partition_fn):
        def selective_checkpointing_context_fn():
            no_recompute_list = [
                torch.ops.aten.mm.default,
                torch.ops.aten.sigmoid.default,
            ]
            return create_selective_checkpoint_contexts(
                _get_custom_policy(no_recompute_list=no_recompute_list),
            )

```

---

### 117. `test_compile_selective_checkpoint_partial_ctx_fn` — `dynamo/test_activation_checkpointing.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/dynamo/test_activation_checkpointing.py:1391–1396` |
| **Classification** | `AS_IS` |
| **Module** | `dynamo` |
| **Description** | The test verifies that selective activation checkpointing works when a custom context function excludes specific operations (matrix multiplication) from recomputation. It compiles a function with torc... |
| **Shapes** | (4, 4) |
| **LLM Shape** | (4, 4) |
| **Device Classification** | device_generic |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_compile_selective_checkpoint_partial_ctx_fn(self, device, partition_fn):
        def selective_checkpointing_context_fn(no_recompute_list):
            return create_selective_checkpoint_contexts(
                _get_custom_policy(no_recompute_list=no_recompute_list)
            )

```

---

### 118. `test_compile_selective_checkpoint_custom_rule` — `dynamo/test_activation_checkpointing.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/dynamo/test_activation_checkpointing.py:1319–1324` |
| **Classification** | `AS_IS` |
| **Module** | `dynamo` |
| **Description** | The test verifies that selective activation checkpointing with a user‑defined recompute policy works when compiling a model with torch.compile. It runs a small graph containing multiple matrix multipl... |
| **Shapes** | (4, 4) |
| **LLM Shape** | (4, 4) |
| **Device Classification** | device_generic |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_compile_selective_checkpoint_custom_rule(self, device, partition_fn):
        def _get_custom_policy(meta):
            no_recompute_list = [
                torch.ops.aten.mm.default,
            ]

```

---

### 119. `test_compile_selective_checkpoint_tensor_subclass` — `dynamo/test_activation_checkpointing.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/dynamo/test_activation_checkpointing.py:1262–1270` |
| **Classification** | `AS_IS` |
| **Module** | `dynamo` |
| **Description** | The test verifies that selective activation checkpointing works correctly when the inputs are instances of a custom tensor subclass. It compiles a function that performs nested matrix multiplications ... |
| **Shapes** | (4, 4) |
| **LLM Shape** | 4, 4|(4, 4) |
| **Device Classification** | device_generic |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_compile_selective_checkpoint_tensor_subclass(self, device, partition_fn):
        def selective_checkpointing_context_fn():
            no_recompute_list = [
                torch.ops.aten.mm.default,
            ]
            return create_selective_checkpoint_contexts(
                _get_custom_policy(no_recompute_list=no_recompute_list)
            )

```

---

### 120. `test_compile_selective_checkpoint_triton_kernel` — `dynamo/test_activation_checkpointing.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/dynamo/test_activation_checkpointing.py:1188–1190` |
| **Classification** | `AS_IS` |
| **Module** | `dynamo` |
| **Description** | The test verifies that a selective activation checkpointing region containing a custom Triton kernel can be compiled with torch.compile without raising errors. It defines a custom autograd Function th... |
| **Shapes** | (4, 4), (x) |
| **LLM Shape** | (4, 4)|(n_elements,)|BLOCK_SIZE=4 |
| **Device Classification** | device_generic |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_compile_selective_checkpoint_triton_kernel(self, device, partition_fn):
        # Copy of the above test, but make sure that having a triton kernel in the
        # region does not error.
```

---

### 121. `test_compile_selective_checkpoint_must_recompute` — `dynamo/test_activation_checkpointing.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/dynamo/test_activation_checkpointing.py:968–978` |
| **Classification** | `AS_IS` |
| **Module** | `dynamo` |
| **Description** | The test verifies that selective activation checkpointing correctly forces a matrix multiplication (aten.mm) to be recomputed during the backward pass when it is listed in the must‑recompute policy. I... |
| **Shapes** | (4, 4) |
| **LLM Shape** | 4, 4|5, 5|4, 4 |
| **Device Classification** | device_generic |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_compile_selective_checkpoint_must_recompute(self, device, partition_fn):
        def context_fn_must_recompute_mm():
            must_recompute_list = [
                torch.ops.aten.mm.default,
            ]
            return create_selective_checkpoint_contexts(
                _get_custom_policy(
                    must_recompute_list=must_recompute_list,
                ),
            )

```

---

### 122. `test_symints_location` — `dynamo/test_activation_checkpointing.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/dynamo/test_activation_checkpointing.py:930–933` |
| **Classification** | `AS_IS` |
| **Module** | `dynamo` |
| **Description** | The test verifies that a function using activation checkpointing (matmul followed by dropout) can be compiled with torch.compile and correctly handles inputs of different symbolic sizes. It checks tha... |
| **Shapes** | (4, 4), (5, 5) |
| **LLM Shape** | (4, 4)|(4, 4)|(4, 4)|(5, 5)|(5, 5) |
| **Device Classification** | device_generic |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_symints_location(self, device):
        def gn(x, y):
            return torch.matmul(x, torch.nn.functional.dropout(y, 0.5))

```

---

### 123. `test_kwargs` — `dynamo/test_activation_checkpointing.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/dynamo/test_activation_checkpointing.py:896–902` |
| **Classification** | `AS_IS` |
| **Module** | `dynamo` |
| **Description** | The test verifies that activation checkpointing correctly handles keyword arguments when compiled with torch.compile. It runs a function that performs a matmul (optionally a second matmul) wrapped in ... |
| **Shapes** | (4, 4) |
| **LLM Shape** | (4, 4)|(5, 5) |
| **Device Classification** | device_generic |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_kwargs(self, device):
        def gn(x, y, z=None):
            a = torch.matmul(x, y)
            if z is not None:
                return torch.matmul(a, z)
            return a

```

---

### 124. `test_tags_multiple_checkpoints` — `dynamo/test_activation_checkpointing.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/dynamo/test_activation_checkpointing.py:601–604` |
| **Classification** | `AS_IS` |
| **Module** | `dynamo` |
| **Description** | The test defines a function that applies sin, then two checkpointed matrix‑multiply‑sigmoid operations, and finally another sin. It runs this function under AOT autograd with a custom forward and back... |
| **Shapes** | (4, 4) |
| **LLM Shape** | (4, 4)|(10, 10) |
| **Device Classification** | device_generic |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_tags_multiple_checkpoints(self, device, partition_fn):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

```

---

### 125. `test_tags_function_with_kwargs` — `dynamo/test_activation_checkpointing.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/dynamo/test_activation_checkpointing.py:367–370` |
| **Classification** | `AS_IS` |
| **Module** | `dynamo` |
| **Description** | The test defines a simple function that applies a sine, a matrix multiplication, and a sigmoid, wrapped with torch.utils.checkpoint.checkpoint (non‑reentrant). It then runs AOT Autograd with custom fo... |
| **Shapes** | (4, 4) |
| **LLM Shape** | (4, 4)|(4, 4)|(4, 4)|(4, 4)|(4, 4)|(4, 4) |
| **Device Classification** | device_generic |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_tags_function_with_kwargs(self, device, partition_fn):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

```

---

### 126. `test_max_autotune` — `inductor/test_deterministic.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_deterministic.py:70–73` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test compiles a simple matrix multiplication with torch.compile under the inductor backend, runs it on two large random GPU tensors, and checks that the result matches the eager matmul. It then ve... |
| **Shapes** | (2048, 2048) |
| **LLM Shape** | [2049, 2049]|[2048, 2048] |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_max_autotune(self, deterministic):
        with inductor_config.patch(deterministic=deterministic):

            @torch.compile()
```

---

### 127. `test_mx_fusion` — `inductor/test_fp8.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_fp8.py:1108–1123` |
| **Classification** | `NEEDS_REFACTOR` |
| **Module** | `inductor` |
| **Description** | The test registers a fake scaled matrix‑multiply custom operator (fake_scaled_mm) for CUDA/XPU, runs a small model that uses this op, and checks that the Inductor compiler correctly fuses the matmul w... |
| **Dtypes** | torch.bfloat16, torch.float32, torch.float8_e4m3fn, torch.int32, torch.uint8 |
| **LLM Dtype** | torch.bfloat16|torch.float32 |
| **Shapes** | (K, N), (M, K), (M, N) |
| **Device Classification** | device_generic |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_mx_fusion(self, device):
        # use a device key for library registration
        device_type = torch.device(device).type
        device_dispatch_key = "CUDA" if device_type == "cuda" else "XPU"
        # Register fake_scaled_mm custom op scoped to this test
        with torch.library._scoped_library("test_fp8", "FRAGMENT") as lib:
            # Define the op schema
            lib.define(
                "fake_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scale_a, Tensor scale_b, "
                "Tensor? bias=None, Tensor? scale_result=None, ScalarType? out_dtype=None, "
                "bool use_fast_accum=False) -> Tensor"
            )
            input_values = []

            # Register CUDA/XPU implementation
            @torch.library.impl(lib, "fake_scaled_mm", device_dispatch_key)
```

---

### 128. `test_rowwise_scaling_acceptable_input_dims` — `inductor/test_fp8.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_fp8.py:1351–1370` |
| **Classification** | `NEEDS_REFACTOR` |
| **Module** | `inductor` |
| **Description** | The test creates random bfloat16 matrices x (M×K) and w (N×K), quantizes them row‑wise to FP8, and runs a scaled matrix multiplication with bias using torch._scaled_mm. It then checks that the result ... |
| **Dtypes** | torch.bfloat16, torch.float8_e4m3fn |
| **LLM Dtype** | torch.bfloat16|torch.float8_e4m3fn|torch.bfloat16 |
| **Shapes** | (M, K), (N), (N, K) |
| **LLM Shape** | M, K|K, N|N, K|M, K|N|N|1, N |
| **Device Classification** | device_generic |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_rowwise_scaling_acceptable_input_dims(
        self, M: int, K: int, N: int, persistent_matmul: bool, device
    ):
        dtype: torch.dtype = torch.bfloat16
        use_fast_accum = True
        # xpu does not support fast_accum now
        if "xpu" in device:
            use_fast_accum = False
        dtype_float8 = torch.float8_e4m3fn
        dtype_float8 = _fix_fp8_dtype_for_rocm(dtype_float8, device)

        x = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(N, K, dtype=dtype, device=device)
        bias = torch.randn(N, device=device, dtype=torch.bfloat16)

        w_fp8, w_inverse_scale = _quantize_rowwise(w, dtype_float8)
        w_t_fp8 = w_fp8.t()
        w_inverse_scale = w_inverse_scale.t()  # scale_b should be (1, N)
        x_fp8, x_inverse_scale = _quantize_rowwise(x, dtype_float8)

```

---

### 129. `test_permute_bmm_fusion` — `inductor/test_fx_fusion.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_fx_fusion.py:133–134` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test defines a module that permutes a 3‑D tensor and then performs a batched matrix multiplication with a stored tensor. It applies a FX fusion pass that should replace the permute + torch.bmm pat... |
| **Shapes** | (batch, k, m), (batch, k, n) |
| **LLM Shape** | batch, k, n | 6, 16, 8, 4 | (batch, k, n) | (batch, k, m) | 0, 2, 1 |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_permute_bmm_fusion(self):
        class TestModule(torch.nn.Module):
```

---

### 130. `test_attention_qk_output_modes` — `onnx/ops/test_ops.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/onnx/ops/test_ops.py:853–879` |
| **Classification** | `AS_IS` |
| **Module** | `onnx` |
| **Description** | The test iterates over four QK matmul output modes, runs the ONNX attention operator with random Q, K, V tensors, and verifies that the produced attention output and QK score tensor have the expected ... |
| **Shapes** | (batch_size, kv_num_heads, kv_seq_len, head_size), (batch_size, q_num_heads, q_seq_len, head_size) |
| **LLM Shape** | (batch_size, kv_num_heads, expected_total_seq_len, head_size)|(batch_size, q_num_heads, q_seq_len, head_size)|(batch_size, kv_num_heads, kv_seq_len, head_size)|(batch_size, q_num_heads, q_seq_len, kv_seq_len)|2|4|6|8|64 |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_attention_qk_output_modes(self):
        """Test different QK matmul output modes."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        for mode in [0, 1, 2, 3]:
            torch.library.opcheck(
                _impl.attention_23,
                (Q, K, V),
                dict(qk_matmul_output_mode=mode),
            )
            output, _, _, qk_output = torch.onnx.ops.attention(
                Q, K, V, qk_matmul_output_mode=mode
            )

            self.assertEqual(
                output.shape, (batch_size, q_num_heads, q_seq_len, head_size)
            )
            self.assertEqual(
                qk_output.shape, (batch_size, q_num_heads, q_seq_len, kv_seq_len)
            )

```

---

### 131. `test_shape_function_includes` — `jit/test_symbolic_shape_analysis.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/jit/test_symbolic_shape_analysis.py:722–739` |
| **Classification** | `AS_IS` |
| **Module** | `jit` |
| **Description** | The test verifies that the symbolic shape functions for conv2d and matmul in torch.jit correctly compute output shapes given input shape specifications.... |
| **Shapes** | (1, 16, 5, 10)|(33, 16, 3, 3)|(1, 33, 2, 4)|(10, 20)|(20, 10)|(10, 10) |
| **LLM Shape** | [1, 16, 5, 10]|[33, 16, 3, 3]|[2, 2]|[0, 0]|[1, 1]|[1, 33, 2, 4]|[10, 20]|[20, 10]|[10, 10]|[1, 10]|[20, 10]|[15, 1]|[5, 1]|[20, 1]|[None, None] |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_shape_function_includes(self):
        inp_shape = [1, 16, 5, 10]
        weight_shape = [33, 16, 3, 3]
        bias = None
        stride = [2, 2]
        padding = [0, 0]
        dilation = [1, 1]
        groups = 1
        res = torch.jit._shapes.conv2d(
            inp_shape, weight_shape, bias, stride, padding, dilation, groups
        )
        self.assertEqual(res, [1, 33, 2, 4])

        m1_shape = [10, 20]
        m2_shape = [20, 10]
        res = torch.jit._shapes.matmul(m1_shape, m2_shape)
        self.assertEqual(res, [10, 10])

```

---

### 132. `test_fuse_linear` — `jit/test_graph_rewrite_passes.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/jit/test_graph_rewrite_passes.py:11–12` |
| **Classification** | `AS_IS` |
| **Module** | `jit` |
| **Description** | The test verifies that the JIT graph rewrite pass _jit_pass_fuse_linear correctly fuses a matmul (with optional bias addition) into a single aten::linear node and preserves source location information... |
| **Shapes** | (3), (5), (5, 3), (5, 5, 100), (5, 6, 5) |
| **LLM Shape** | (3)|(5, 3)|(5)|(5, 6, 5)|(5, 5, 100) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_fuse_linear(self):
        class FunctionalLinear(torch.nn.Module):
```

---

### 133. `test_allow_reuse_disable_if_exceed_peak` — `inductor/test_torchinductor.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_torchinductor.py:15933–15934` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test patches configuration to disable Triton matmul and then compiles a function that performs mean, subtraction, squaring, a matrix multiplication, and another mean on a (100, 100) tensor. It ver... |
| **Shapes** | (100, 100) |
| **LLM Shape** | [2] * nd|[100, 100]|[100, 100]|[100, 100]|[100, 100]|[20]|[4]|[20]|[4] |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_allow_reuse_disable_if_exceed_peak(self):
        @torch.compile
```

---

### 134. `test_graph_partition_refcount` — `inductor/test_torchinductor.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_torchinductor.py:15518–15531` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that after graph partitioning and compilation, the original input tensors are released (reference count drops to zero) before the matmul operation is executed. It does this by weak‑r... |
| **Shapes** | (5, 5) |
| **LLM Shape** | [5, 5]|[5, 5] |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_graph_partition_refcount(self):
        contexts = [
            contextlib.nullcontext,
            lambda: torch._inductor.config.patch({"triton.cudagraphs": True}),
        ]

        for context in contexts:
            with context():
                inps = [
                    torch.rand([5, 5]).to(self.device),
                    torch.rand([5, 5]).to(self.device),
                ]
                inp_refs = [weakref.ref(inp) for inp in inps]

```

---

### 135. `test_list_clearing` — `inductor/test_torchinductor.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_torchinductor.py:11912–11928` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that TorchInductor correctly clears references to input tensors after a compiled matmul operation, allowing them to be garbage‑collected. It runs the compiled function under differen... |
| **LLM Dtype** | torch.float32|dtype |
| **Shapes** | (5, 5) |
| **LLM Shape** | (2, 3)|(3, 1)|()|()|[5, 5] |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_list_clearing(self):
        if self.device == "cpu":
            contexts = [contextlib.nullcontext]
        else:
            contexts = [
                contextlib.nullcontext,
                lambda: config.patch({"triton.cudagraphs": True}),
            ]

        for context in contexts:
            with context():
                inps = [
                    torch.rand([5, 5]).to(self.device),
                    torch.rand([5, 5]).to(self.device),
                ]
                inp_refs = [weakref.ref(inp) for inp in inps]

```

---

### 136. `test_deterministic_codegen_with_suffix` — `inductor/test_torchinductor.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_torchinductor.py:7800–7806` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test compiles two functions with torch.compile: one performing cos‑>sin‑>softmax, and another performing the same sequence followed by a matmul. It extracts the generated kernel code for each and ... |
| **LLM Dtype** | torch.float32 |
| **Shapes** | (16, 256), (256, 256) |
| **LLM Shape** | (16, 256)|(256, 256)|[1, 2, 6, 6]|[1, 2, 6, 6]|[1, 2, 6, 6]|[1, 2, 6, 6]|10|() |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_deterministic_codegen_with_suffix(self):
        if "cpu" in str(self.device) and config.cpp_wrapper:
            raise unittest.SkipTest(
                "run_and_get_kernels can't extract kernels from CPU cpp_wrapper code"
            )

        @torch.compile(fullgraph=True)
```

---

### 137. `test_subgraph_decompose_k` — `inductor/test_subgraph_choice.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_subgraph_choice.py:37–43` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test registers a custom matmul operation that can be lowered to a DecomposeKSugraphTemplate with a K‑partition of 256, runs Inductor's autotuning to select this subgraph, compiles the function wit... |
| **Dtypes** | torch.float16 |
| **Shapes** | (mat1_shape), (mat2_shape) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_subgraph_decompose_k(self):
        from torch._inductor.kernel.mm import aten_mm
        from torch._inductor.kernel.mm_common import mm_args

        mat1_shape, mat2_shape = (32, 4096), (4096, 32)

        @torch.library.custom_op("mylib::matmul_decompose", mutates_args={})
```

---

### 138. `test_match_equivalent_function_invocations3` — `inductor/test_pattern_matcher.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_pattern_matcher.py:1628–1637` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test registers a graph pattern that matches aten.addmm calls with an explicit beta keyword argument, replaces the matched subgraph with a matmul plus addition, and verifies that the pattern is app... |
| **Shapes** | (10, 15), (15, 20), (20) |
| **LLM Shape** | (20,)|(10, 15)|(15, 20)|(3, 3) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_match_equivalent_function_invocations3(self):
        counter = 0
        test_pass = PatternMatcherPass()

        args = [
            torch.randn(20, device=GPU_TYPE),
            torch.randn(10, 15, device=GPU_TYPE),
            torch.randn(15, 20, device=GPU_TYPE),
        ]

```

---

### 139. `test_replace_by_example_in_pre_grad` — `inductor/test_pattern_matcher.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_pattern_matcher.py:1470–1479` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test registers a graph pattern for the aten.addmm operation, replaces it with an equivalent explicit matmul expression using replace_by_example, and verifies that the custom pass fires during comp... |
| **LLM Dtype** | torch.float16 |
| **Shapes** | (10, 15), (15, 20), (20) |
| **LLM Shape** | 20|10, 15|15, 20|1024, 1024|1024, 1024 |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_replace_by_example_in_pre_grad(self):
        counter = 0
        test_pass = PatternMatcherPass()

        args = [
            torch.randn(20, device=GPU_TYPE),
            torch.randn(10, 15, device=GPU_TYPE),
            torch.randn(15, 20, device=GPU_TYPE),
        ]

```

---

### 140. `test_match_equivalent_function_invocations1` — `inductor/test_pattern_matcher.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_pattern_matcher.py:1413–1422` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test registers a graph pattern that matches calls to torch.ops.aten.addmm with optional beta and alpha keyword arguments, replaces them with an equivalent matmul‑plus‑scale expression, and verifie... |
| **Shapes** | (10, 15), (15, 20), (20) |
| **LLM Shape** | 20|10, 15|15, 20 |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_match_equivalent_function_invocations1(self):
        counter = 0
        test_pass = PatternMatcherPass()

        args = [
            torch.randn(20, device=GPU_TYPE),
            torch.randn(10, 15, device=GPU_TYPE),
            torch.randn(15, 20, device=GPU_TYPE),
        ]

```

---

### 141. `test_pad_single_cat` — `inductor/test_pad_mm.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_pad_mm.py:354–355` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test compiles a simple matrix multiplication function with shape padding enabled, runs it on two random 5x5 GPU tensors, and checks that the compiled result matches the eager matmul.... |
| **LLM Dtype** | torch.float32|torch.float16 |
| **Shapes** | (5, 5) |
| **LLM Shape** | [5, 5]|[4, 5]|[5, 6]|[a, b]|[1, 4]|[1, 6]|[4, 5]|[5, 6]|[a]|[4, 5]|[5, 6]|(batch_size, m, k)|(batch_size, k, n)|(3, 6, 11)|(3, 11, 9) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_pad_single_cat(self):
        @torch.compile()
```

---

### 142. `test_fusion_acc_large_reads` — `inductor/test_memory.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_memory.py:340–355` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test defines a small kernel that repeatedly computes a matrix multiplication plus a bias and accumulates the result. It injects a custom fusion decision rule based on the total size of read tensor... |
| **Dtypes** | torch.float32 |
| **LLM Dtype** | torch.float32|torch.float32|torch.float32 |
| **Shapes** | (N, N) |
| **LLM Shape** | N|N|N|N|N|N|N|N |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_fusion_acc_large_reads(self):
        def f(x, y, z):
            res = torch.zeros_like(x[0])
            for _ in range(4):
                temp = torch.matmul(x, y) + z
                res = res + temp
            return res

        N = 128
        x = torch.rand(N, N, dtype=torch.float32, device=GPU_TYPE)
        y = torch.rand(N, N, dtype=torch.float32, device=GPU_TYPE)
        z = torch.rand(N, N, dtype=torch.float32, device=GPU_TYPE)

        from torch._inductor.choices import InductorChoices
        from torch._inductor.scheduler import BaseSchedulerNode, Scheduler

```

---

### 143. `test_template_epilogue_fusion_static_analysis` — `inductor/test_max_autotune.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_max_autotune.py:5136–5203` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies the scheduler's static analysis logic that decides whether to fuse a matmul epilogue based on estimated register spillage and relative runtime of Triton vs. ATen implementations.... |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_template_epilogue_fusion_static_analysis(
        self, test_case: str, use_async_compile: bool
    ):
        """
        Test static analysis decisions for matmul epilogue fusions.

        Tests the scheduler logic that decides whether to fuse epilogues without
        benchmarking, based on:
        1. Register spillage (n_spills <= 8 required for fusion)
        2. Runtime comparison (epilogue_runtime + ms_min_choice > choice_timings[choice])
        """
        if test_case == "spills_reject":
            mock_n_spills = 100
            triton_time = 0.1
            aten_time = float("inf")
            expect_fusion = False
        elif test_case == "timing_reject":
            mock_n_spills = 0
            triton_time = 100.0
            aten_time = 0.001
            expect_fusion = False
        elif test_case == "accept_with_triton_faster":
            mock_n_spills = 0
            triton_time = 0.1
            aten_time = float("inf")
            expect_fusion = True
        elif test_case == "accept_with_aten_faster":
            mock_n_spills = 0
            triton_time = 0.1
            aten_time = 0.09999
            expect_fusion = True
        else:
            raise RuntimeError("Invalid test case")

        f = self._get_mm_with_epilogue_fn()
        a, b = self._get_mm_inputs()

        with self._setup_mm_heuristic(use_async_compile):
            with self.get_common_patches(
                use_async_compile,
                False,
                aten_time=aten_time,
                triton_time=triton_time,
                mock_n_spills=mock_n_spills,
            ):
                compiled_f = torch.compile(f)
                _, code = run_and_get_code(compiled_f, a, b)

                if expect_fusion:
                    FileCheck().check("triton_tem_fused__to_copy_add_mm_0").run(code[0])
                elif triton_time < aten_time:
                    FileCheck().check("triton_tem_fused_mm").check(
                        "triton_poi_fused__to_copy"
                    ).run(code[0])
                else:
                    FileCheck().check_not("triton_tem_fused_mm").check(
                        "triton_poi_fused__to_copy"
                    ).run(code[0])

    @unittest.skipIf(
        not HAS_CUDA_AND_TRITON, "Scheduler static analysis only tested on cuda"
    )
    @unittest.skipIf(
        config.cpp_wrapper, "Skip static analysis codegen checks on cpp_wrapper"
    )
    @skipIfRocm(msg="Scheduler static analysis needs investigation on ROCm")
    @parametrize("fuse_epilogue", (True, False))
    @parametrize("use_async_compile", (True, False))
```

---

### 144. `test_low_precision` — `inductor/test_max_autotune.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_max_autotune.py:4449–4454` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that low‑precision matmul (FP8 × BF16) is performed without up‑casting to float32 when possible, and that adding a scalar forces higher‑precision handling. It compiles a simple funct... |
| **Dtypes** | torch.bfloat16, torch.float8_e4m3fn |
| **LLM Dtype** | torch.float16|torch.float16|torch.float8_e4m3fn|torch.bfloat16|torch.float|torch.float16 |
| **Shapes** | (K, N), (M, K) |
| **LLM Shape** | [M, K]|[K, N]|M|[M, K]|[K, N]|[M, K]|[K, N]|64, 128, 256|64, 64, 64|64, 120, 64 |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_low_precision(self):
        M = K = N = 128

        x = torch.rand([M, K], device=GPU_TYPE).to(torch.float8_e4m3fn)
        y = torch.rand([K, N], dtype=torch.bfloat16, device=GPU_TYPE)

```

---

### 145. `test_triton_template_generated_code_caching_mm_plus_mm` — `inductor/test_max_autotune.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_max_autotune.py:2544–2557` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that the Triton template-based matmul code generation cache works correctly when the same matmul operations are invoked multiple times within a compiled function. It runs a compiled ... |
| **LLM Dtype** | torch.float |
| **Shapes** | (10, 40), (40, 30) |
| **LLM Shape** | (10, 10, 22)|(10, 22, 30)|(10, 40)|(40, 30)|(32, 32, 32768)|(32, 32, 256)|(M, K) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_triton_template_generated_code_caching_mm_plus_mm(self):
        def func_test1(x, y, z, m):
            a = torch.mm(x, y)
            b = torch.mm(z, m)
            sum1 = a + b

            c = torch.mm(x, y)
            d = torch.mm(z, m)
            sum2 = c + d
            return sum1, sum2

        a = torch.rand(10, 40, device=GPU_TYPE)
        b = torch.rand(40, 30, device=GPU_TYPE)

```

---

### 146. `test_max_autotune_decompose_k` — `inductor/test_max_autotune.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_max_autotune.py:1696–1727` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that the max‑autotune path correctly generates and selects kernels that use the decompose_k transformation for large‑K matrix multiplications on GPU. It creates random M×K and K×N te... |
| **Dtypes** | torch.float16, torch.bfloat16 |
| **LLM Dtype** | bfloat16|torch.float16|torch.bfloat16|dtype |
| **LLM Shape** | B * T|C|B * T|V|32, 32, 32768|64, 128, 200000|64, 64, 177147|M|N|K |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_max_autotune_decompose_k(self, sizes, dtype, dynamic):
        # UT specific change to force testing decompose K feature on ROCm until
        # enabled by default, same strategy as #169948
        with config.patch(_DECOMPOSE_K_PATCH_ROCM):
            fp16_red_setting = (
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            )
            bf16_red_setting = (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            )
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

            M, N, K = sizes

            atol = 1e-4
            rtol = 1e-4
            # K can be huge huge, this is why the data distribution is set to iid N(0, K ** 0.5),
            # which makes the result of reductions distributed as N(0, 1).
            a, b = self._make_matrices(
                M,
                K,
                N,
                dtype=dtype,
                device=GPU_TYPE,
                requires_grad=True,
            )

            possible_splits = range(2, min(K // M, K // N) + 1)

            divisors = {split for split in possible_splits if K % split == 0}

```

---

### 147. `test_baddmm` — `inductor/test_max_autotune.py`

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

### 148. `test_honor_sm_carveout_with_triton_tma` — `inductor/test_max_autotune.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_max_autotune.py:1048–1051` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that the SM carveout setting is correctly honored when performing large matrix multiplications with Triton persistent TMA enabled, for both regular and scaled matmul operations.... |
| **Dtypes** | torch.bfloat16, torch.float32, torch.float8_e4m3fn |
| **LLM Dtype** | torch.bfloat16|torch.bfloat16|torch.float32|torch.float32|torch.float8_e4m3fn|torch.float8_e4m3fn |
| **Shapes** | (1), (size, size) |
| **LLM Shape** | 2560|2560|2560|2560 |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_honor_sm_carveout_with_triton_tma(self, carveout, op: str):
        def mm_func(a, b):
            return torch.mm(a, b)

```

---

### 149. `test_max_autotune_addmm_tma_dynamic_outer_dim` — `inductor/test_max_autotune.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_max_autotune.py:999–1047` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test checks that TorchInductor's max_autotune mechanism correctly selects the persistent TMA matmul kernel for an addmm operation when the outer dimension of the input matrix is marked dynamic. It... |
| **Dtypes** | torch.float16 |
| **LLM Dtype** | torch.float16 |
| **Shapes** | (K, N), (M, K), (N) |
| **LLM Shape** | M, K|K, N|N|8|8, 8|8, 8 |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_max_autotune_addmm_tma_dynamic_outer_dim(self):
        def addmm(x, a, b):
            return torch.addmm(x, a, b)

        M, N, K = 21, 31, 11
        a = torch.randn(M, K).to(torch.float16).to(GPU_TYPE)
        b = torch.randn(K, N).to(torch.float16).to(GPU_TYPE)
        x = torch.randn(N).to(torch.float16).to(GPU_TYPE)

        # TMA requires 16-byte alignment: here we repeat the dims
        # by the factor of 8, as float16 is 2-byte. All dims are
        # repeated due to the possible transpositions below.
        x = x.repeat(8)
        a = a.repeat(8, 8)
        b = b.repeat(8, 8)

        torch._dynamo.maybe_mark_dynamic(a, 0)

        choice_name_regex = (
            "blackwell_ws_persistent_device_tma"
            if has_datacenter_blackwell_tma_device()
            else "mm_persistent_tma"
        )

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": "1",
                "triton.native_matmul": False,
                "test_configs.autotune_choice_name_regex": choice_name_regex,
            }
        ):
            c_actual = torch.compile(addmm)(x, a, b)
            c_expected = addmm(x, a, b)

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)

    @fresh_cache()
    @skipIfXpu(msg="XPU doesn't support sm carveout")
    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support sm carveout")
    @unittest.skipIf(IS_WINDOWS, "Windows doesn't support persistent TMA")
    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @unittest.skipIf(
        has_datacenter_blackwell_tma_device(), "B200 doesn't support sm carveout"
    )
    @parametrize("carveout", (None, 0, 27))
    @parametrize("op", ("mm", "scaled_mm"))
```

---

### 150. `test_max_autotune_addmm_persistent_tma_illegal_alignment` — `inductor/test_max_autotune.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_max_autotune.py:969–998` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that enabling max_autotune with persistent TMA matmul fails when the inner dimensions of the inputs are not 16‑byte aligned, causing the Triton persistent TMA template to be skipped ... |
| **Dtypes** | torch.float16 |
| **LLM Dtype** | torch.float16|torch.float16|torch.float16|torch.float16|torch.float16|torch.float16 |
| **Shapes** | (K, N), (M, K), (N) |
| **LLM Shape** | M, N, K = 21, 31, 11|M, K|K, N|N|M, N, K = 21, 31, 11|M, K|K, N|N |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_max_autotune_addmm_persistent_tma_illegal_alignment(self, dynamic):
        def addmm(x, a, b):
            return torch.addmm(x, a, b)

        M, N, K = 21, 31, 11
        a = torch.randn(M, K).to(torch.float16).to(GPU_TYPE)
        b = torch.randn(K, N).to(torch.float16).to(GPU_TYPE)
        x = torch.randn(N).to(torch.float16).to(GPU_TYPE)

        with (
            self.assertRaises(BackendCompilerFailed) as context,
            config.patch(
                {
                    "max_autotune": True,
                    "triton.enable_persistent_tma_matmul": "1",
                    "triton.native_matmul": False,
                    "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
                }
            ),
        ):
            torch.compile(addmm, dynamic=dynamic)(x, a, b)

        # Lowering to the persistent+TMA Triton template should be skipped
        # if any of the input inner dims are not 16-byte aligned. As a result,
        # given the config flags above, we should have no choices left.
        self.assertIn("NoValidChoicesError", str(context.exception))

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
```

---

### 151. `test_max_autotune_regular_mm_tma_dynamic_outer_dim` — `inductor/test_max_autotune.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_max_autotune.py:761–802` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that TorchInductor's max_autotune mechanism correctly selects a persistent TMA matmul kernel for a regular matrix multiplication when the outer dimension is marked dynamic. It compil... |
| **Dtypes** | torch.float16 |
| **LLM Dtype** | torch.float16|torch.float16|torch.float16|torch.float16|torch.float16|torch.float16 |
| **Shapes** | (K, N), (M, K) |
| **LLM Shape** | (M, K)|(K, 1)|(M, K)|(K, N)|(1, K)|(K, N)|(M, N)|(N, 1)|(M, N)|21|31|11|8|8|8|8|(0, 10)|(10, 100) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_max_autotune_regular_mm_tma_dynamic_outer_dim(self):
        def mm(a, b):
            return torch.mm(a, b)

        M, N, K = 21, 31, 11
        a = torch.randn(M, K).to(torch.float16).to(GPU_TYPE)
        b = torch.randn(K, N).to(torch.float16).to(GPU_TYPE)

        # TMA requires 16-byte alignment: here we repeat the dims
        # by the factor of 8, as float16 is 2-byte. All dims are
        # repeated due to the possible transpositions below.
        a = a.repeat(8, 8)
        b = b.repeat(8, 8)

        torch._dynamo.maybe_mark_dynamic(a, 0)

        choice_name_regex = (
            "blackwell_ws_persistent_device_tma"
            if has_datacenter_blackwell_tma_device()
            else "mm_persistent_tma"
        )

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": "1",
                "triton.native_matmul": False,
                "test_configs.autotune_choice_name_regex": choice_name_regex,
            }
        ):
            c_actual = torch.compile(mm)(a, b)
            c_expected = mm(a, b)

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @unittest.skipIf(
        has_datacenter_blackwell_tma_device(),
        "Hopper-style mm_persistent_tma template is shadowed by the Blackwell warp-specialized TMA template on data-center Blackwell.",
    )
```

---

### 152. `test_max_autotune_regular_mm_persistent_tma_illegal_output_alignment` — `inductor/test_max_autotune.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_max_autotune.py:725–727` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that max‑autotune fails when a persistent‑TMA matmul is attempted with an output tensor whose stride alignment does not satisfy TMA requirements. It expects a BackendCompilerFailed e... |
| **Dtypes** | torch.float16 |
| **LLM Dtype** | torch.float16|torch.float16|torch.float16|torch.float16|torch.float16|torch.float16 |
| **Shapes** | (K, N), (M, K) |
| **LLM Shape** | (M, K)|(K, N)|(M, K)|(K, N)|(M, N)|(K, 1)|(1, K)|(N, 1) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_max_autotune_regular_mm_persistent_tma_illegal_output_alignment(
        self, dynamic
    ):
```

---

### 153. `test_max_autotune_regular_mm_persistent_tma_illegal_alignment` — `inductor/test_max_autotune.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_max_autotune.py:695–724` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that enabling max_autotune with persistent TMA matmul fails when the inner dimensions of the matrices are not 16‑byte aligned, raising a BackendCompilerFailed exception.... |
| **Dtypes** | torch.float16 |
| **LLM Dtype** | torch.float16 |
| **Shapes** | (K, N), (M, K) |
| **LLM Shape** | M, N, K|21, 31, 11|(M, K)|(K, N)|(M, K)|(K, N) |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_max_autotune_regular_mm_persistent_tma_illegal_alignment(self, dynamic):
        def mm(a, b):
            return torch.mm(a, b)

        M, N, K = 21, 31, 11
        a = torch.randn(M, K).to(torch.float16).to(GPU_TYPE)
        b = torch.randn(K, N).to(torch.float16).to(GPU_TYPE)

        with (
            self.assertRaises(BackendCompilerFailed) as context,
            config.patch(
                {
                    "max_autotune": True,
                    "triton.enable_persistent_tma_matmul": "1",
                    "triton.native_matmul": False,
                    "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
                }
            ),
        ):
            torch.compile(mm, dynamic=dynamic)(a, b)

        # Lowering to the persistent+TMA Triton template should be skipped
        # if any of the input inner dims are not 16-byte aligned. As a result,
        # given the config flags above, we should have no choices left.
        self.assertIn("NoValidChoicesError", str(context.exception))

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @parametrize("dynamic", (False, True))
```

---

### 154. `test_for_reordering_reindex` — `inductor/test_loop_ordering.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_loop_ordering.py:246–255` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test verifies that removing ComputedBuffer.iter_reordering_reindex enables fusion of two Triton kernels (a reduction and a matmul) into a single kernel when compiling with Inductor.... |
| **LLM Dtype** | torch.float8_e5m2|torch.float8_e4m3fn|torch.float32 |
| **Shapes** | (A, A) |
| **LLM Shape** | [A, A, B]|[B, B * A + 300, 1]|[A, A]|[A * A * B]|[A * A]|[A * A * 3]|[M, N]|[N, M]|[N, N * 2]|[N]|1024|2048|20|30 |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_for_reordering_reindex(self):
        """
        ComputedBuffer.iter_reoredering_reindex can cause some fusion
        opportunitiies being skipped.

        In this test case, Inductor generates 2 triton kernels before.
        By removing ComputedBuffer.iter_reoredering_reindex, we can fuse those
        two kernels into a single one.
        """

```

---

### 155. `test_match_equivalent_function_invocations2` — `inductor/test_pattern_matcher.py`

| Field | Value |
|---|---|
| **File** | `/Users/thanmaiboddoju/Library/CloudStorage/Box-Box/My Box Notes/Backup/Documents/Work/ci_cd/pytorch/test/inductor/test_pattern_matcher.py:1547–1556` |
| **Classification** | `AS_IS` |
| **Module** | `inductor` |
| **Description** | The test registers a graph pattern that matches a plain aten.addmm call and replaces it with an equivalent matmul+add expression. It then compiles three variants of addmm (with and without beta/alpha ... |
| **LLM Dtype** | torch.float16|torch.float16 |
| **Shapes** | (10, 15), (15, 20), (20) |
| **LLM Shape** | 1024, 1024|1024, 1024|1024, 1024|256|256|20|10, 15|15, 20|3, 3|3, 3 |
| **Device Classification** | no_device |
| **Match Score** | 5 |

**Source snippet:**
```python
    def test_match_equivalent_function_invocations2(self):
        counter = 0
        test_pass = PatternMatcherPass()

        args = [
            torch.randn(20, device=GPU_TYPE),
            torch.randn(10, 15, device=GPU_TYPE),
            torch.randn(15, 20, device=GPU_TYPE),
        ]

```

---

