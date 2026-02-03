# DTensor Internals

This document explains the core architecture of PyTorch's Distributed Tensor (DTensor) system.

## Architecture Overview

```
                              DTensor Architecture
================================================================================

    User Code                          DTensor API Layer
    ---------                          -----------------
        |
        v
+------------------+     +-------------------+     +-------------------+
|  distribute_     |     |  DTensor.         |     |  Factory funcs    |
|  tensor()        |     |  from_local()     |     |  ones/zeros/rand  |
+------------------+     +-------------------+     +-------------------+
        |                        |                         |
        +------------------------+-------------------------+
                                 |
                                 v
================================================================================
                          DTensor Core
================================================================================
        +------------------------------------------------------------+
        |                       DTensor                               |
        |  (torch.Tensor subclass)                                   |
        |                                                            |
        |   +------------------+    +-----------------------------+  |
        |   | _local_tensor    |    | _spec: DTensorSpec          |  |
        |   | (torch.Tensor)   |    |   - mesh: DeviceMesh        |  |
        |   |                  |    |   - placements: tuple       |  |
        |   | The actual data  |    |   - tensor_meta: TensorMeta |  |
        |   | on this rank     |    |                             |  |
        |   +------------------+    +-----------------------------+  |
        +------------------------------------------------------------+
                    |                           |
                    v                           v
================================================================================
                    Supporting Components
================================================================================

    +-------------------+                 +----------------------------+
    |    DeviceMesh     |                 |     Placement Types        |
    |-------------------|                 |----------------------------|
    | - device_type     |                 |                            |
    | - mesh (tensor)   |                 |  +--------+  +----------+  |
    | - mesh_dim_names  |                 |  | Shard  |  | Replicate|  |
    | - process groups  |                 |  | (dim)  |  |    ()    |  |
    |                   |                 |  +--------+  +----------+  |
    | Topology of       |                 |                            |
    | devices across    |                 |  +--------+  +-----------+ |
    | mesh dimensions   |                 |  |Partial |  |_StridedSh | |
    +-------------------+                 |  |(op)    |  | ard(dim,  | |
                                          |  +--------+  | sf)       | |
                                          |              +-----------+ |
                                          +----------------------------+

================================================================================
                    Operator Dispatch Flow
================================================================================

   DTensor op call (e.g., dtensor + dtensor)
              |
              v
    +-------------------+
    | __torch_dispatch__|  (C++ fast path, not Python)
    +-------------------+
              |
              v
    +-------------------+
    |  OpDispatcher     |  torch/distributed/tensor/_dispatch.py
    |-------------------|
    | 1. Extract specs  |
    | 2. Sharding prop  |  --> Determines output placements
    | 3. Local compute  |  --> Run op on _local_tensor
    | 4. Wrap result    |  --> Create new DTensor with new spec
    +-------------------+
              |
              v
    +-------------------+
    | New DTensor       |
    | with propagated   |
    | placements        |
    +-------------------+
```

## Core Components

### 1. DTensor (`_api.py:269`)

`DTensor` is a `torch.Tensor` subclass that represents a distributed tensor. It stores:

- **`_local_tensor`**: The actual tensor data held by this rank (a regular `torch.Tensor`)
- **`_spec`**: A `DTensorSpec` describing how the tensor is distributed

```python
class DTensor(torch.Tensor):
    _local_tensor: torch.Tensor
    _spec: DTensorSpec
```

DTensor is **not** constructed directly. Use these APIs instead:

| API | Use Case |
|-----|----------|
| `DTensor.from_local()` | Wrap existing local tensors (mid-computation) |
| `distribute_tensor()` | Distribute a "global" leaf tensor (init time) |
| `dtensor.ones/zeros/rand()` | Create distributed tensors from scratch |

### 2. DeviceMesh (`device_mesh.py:130`)

A `DeviceMesh` represents an N-dimensional arrangement of devices. Each dimension can be used for different parallelism strategies.

```python
# 2D mesh: 2 nodes x 4 GPUs per node
mesh_2d = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))

# mesh.mesh tensor:
# [[0, 1, 2, 3],
#  [4, 5, 6, 7]]
#
# Dimension 0 ("dp"): ranks across nodes (data parallel)
# Dimension 1 ("tp"): ranks within node (tensor parallel)
```

Key properties:
- **`device_type`**: "cuda", "cpu", etc.
- **`mesh`**: Tensor of global rank IDs arranged in N-D
- **`mesh_dim_names`**: Optional names for each dimension
- **Process groups**: One per mesh dimension for collective communication

### 3. Placement Types (`placement_types.py`)

Placements describe how a tensor is distributed along each mesh dimension.

#### `Shard(dim)` - Tensor is sharded along `dim`

```
Global tensor shape: [8, 4]
Placement: [Shard(0)]  # shard along dim 0
Mesh: [0, 1, 2, 3]     # 4 ranks

Rank 0: rows 0-1   [2, 4]
Rank 1: rows 2-3   [2, 4]
Rank 2: rows 4-5   [2, 4]
Rank 3: rows 6-7   [2, 4]
```

#### `Replicate()` - Full copy on each rank

```
Global tensor shape: [8, 4]
Placement: [Replicate()]
Mesh: [0, 1, 2, 3]

All ranks hold the full [8, 4] tensor
```

#### `Partial(reduce_op)` - Pending reduction

```
After a matmul with sharded inputs, the result may be partial:
Each rank holds a partial sum that needs all-reduce to get final value.

Partial("sum")  -> all-reduce with sum
Partial("max")  -> all-reduce with max
```

#### Multi-dimensional example

```
2D Mesh: [[0, 1], [2, 3]]  shape (2, 2), names=("dp", "tp")
Global tensor: [8, 8]
Placements: [Shard(0), Shard(1)]  # shard dim 0 on dp, dim 1 on tp

Rank 0 (dp=0, tp=0): rows 0-3, cols 0-3  -> [4, 4]
Rank 1 (dp=0, tp=1): rows 0-3, cols 4-7  -> [4, 4]
Rank 2 (dp=1, tp=0): rows 4-7, cols 0-3  -> [4, 4]
Rank 3 (dp=1, tp=1): rows 4-7, cols 4-7  -> [4, 4]
```

### 4. DTensorSpec (`_dtensor_spec.py:69`)

Internal dataclass that fully describes a DTensor's distribution:

```python
@dataclass
class DTensorSpec:
    mesh: DeviceMesh              # The device topology
    placements: tuple[Placement, ...]  # One placement per mesh dim
    tensor_meta: TensorMeta | None     # shape, stride, dtype of global tensor
    shard_order: ShardOrder            # For multi-dim sharding order
```

The spec is used for:
1. **Sharding propagation**: Determining output placements from input placements
2. **Redistribution**: Converting between different placements
3. **Caching**: Memoizing propagation results

## Key Operations

### Creating DTensors

#### `distribute_tensor()` - For leaf tensors at init time

```python
# On rank 0: tensor = torch.randn(8, 8)
# On all ranks: scatter/broadcast from rank 0
dtensor = distribute_tensor(tensor, mesh, [Shard(0)])
```

This broadcasts/scatters from `src_data_rank` (default 0) to ensure all ranks have consistent data.

#### `DTensor.from_local()` - For wrapping existing shards

```python
# Each rank already has its local shard
local_shard = torch.randn(2, 8)  # different on each rank
dtensor = DTensor.from_local(local_shard, mesh, [Shard(0)], shape=(8, 8))
```

No communication by default (`run_check=False`). User is responsible for correctness.

### Converting back to local tensor

#### `dtensor.to_local()` - Get local shard

```python
local = dtensor.to_local()  # Returns _local_tensor (or view of it)
```

#### `dtensor.full_tensor()` - Gather to full tensor

```python
full = dtensor.full_tensor()  # All-gather sharded dims, returns full tensor
```

### Redistribution

```python
# Change sharding: Shard(0) -> Shard(1) requires all-to-all
new_dtensor = dtensor.redistribute(placements=[Shard(1)])

# Gather to replicate: Shard(0) -> Replicate() requires all-gather
replicated = dtensor.redistribute(placements=[Replicate()])
```

Redistribution rules:
| From | To | Collective |
|------|-----|------------|
| `Shard(dim)` | `Replicate()` | all-gather |
| `Shard(src)` | `Shard(dst)` | all-to-all |
| `Replicate()` | `Shard(dim)` | local chunk (no comm) |
| `Partial()` | `Replicate()` | all-reduce |
| `Partial()` | `Shard(dim)` | reduce-scatter |

## Autograd Integration

DTensor operations are differentiable. Backward ops are themselves DTensor ops that go through
the same dispatch and sharding propagation as forward ops.

### Worked Example: Tensor Parallel MLP Backward

Consider a column-parallel linear layer in tensor parallelism:

```
Forward:
  X: [batch, in_features], Replicate      # activations replicated across TP ranks
  W: [in_features, out_features], Shard(1) # weights sharded on output dim (column parallel)
  Y = X @ W: [batch, out_features], Shard(1)

  Each rank computes: [batch, in_features] @ [in_features, out_features/n] -> [batch, out_features/n]
```

In backward, we need `grad_X`:

```
Backward:
  grad_Y: [batch, out_features], Shard(1)  # cotangent has same placement as Y
  W.T: [out_features, in_features], Shard(0)  # transpose flips shard dim

  grad_X = grad_Y @ W.T
```

**Sharding propagation for this matmul:**
- `grad_Y` is `Shard(1)` - sharded on dim 1 (the contracting dimension)
- `W.T` is `Shard(0)` - sharded on dim 0 (the contracting dimension)
- Both inputs sharded on contraction dim → output is **`Partial("sum")`**

```
Each rank computes: [batch, out_features/n] @ [out_features/n, in_features] -> [batch, in_features]
But this is only a partial contribution! True grad_X = sum of all ranks' contributions.
```

**The result:** The primal `X` was `Replicate`, but the computed `grad_X` is `Partial`.

### Eager Mode: Gradients Can Be Partial

In eager mode, **there is no automatic coercion of gradient placements**. Gradients have
whatever placement sharding propagation determines. This is confirmed by tests:

```python
# From test/distributed/tensor/parallel/test_tp_style.py:383-384
self.assertEqual(sp_norm.weight.grad.placements, (_Partial(),))
self.assertEqual(sp_norm.bias.grad.placements, (_Partial(),))
```

For leaf DTensors created via `distribute_tensor()` or factory functions:
1. There's no autograd function in the computation graph
2. Gradients are accumulated via standard PyTorch AccumulateGrad
3. The gradient is a DTensor with whatever placement backward ops produced
4. **`Partial` gradients are allowed and expected**

**Why this is useful:** `Partial` gradients enable more efficient collectives:
- FSDP2 can do reduce-scatter (`Partial → Shard`) instead of all-reduce (`Partial → Replicate`)
- Reduces communication volume when optimizer only needs sharded gradients

**User responsibility:** If you need a non-Partial gradient, explicitly redistribute:
```python
if any(isinstance(p, Partial) for p in param.grad.placements):
    param.grad = param.grad.redistribute(placements=[Replicate()])
```

### Compiled Mode: Automatic Coercion via Hooks

Under `torch.compile`, AOT autograd traces both forward and backward ahead of time.
DTensor implements tensor subclass hooks that AOT autograd calls to normalize cotangents:

#### `__coerce_tangent_metadata__` (`_api.py:370-376`)

Called during tracing to normalize cotangents with "problematic" placements:

```python
def __coerce_tangent_metadata__(self):
    if not any(isinstance(p, Partial) for p in self.placements):
        return self
    # Convert Partial -> Replicate via all-reduce
    placements = [
        Replicate() if isinstance(p, Partial) else p for p in self.placements
    ]
    return self.redistribute(device_mesh=self.device_mesh, placements=placements)
```

**Why `Partial` is "problematic" for compile:** AOT autograd needs to trace a single
backward graph that works for any runtime cotangent placement. You cannot convert a
`Replicate` or `Shard` tensor *into* a `Partial` tensor (that would require "un-reducing").
By ensuring traced cotangents are never `Partial`, we guarantee the graph can handle
any runtime placement.

#### `__coerce_same_metadata_as_tangent__` (`_api.py:378-386`)

Called at runtime when the actual cotangent placement doesn't match what was traced:

```python
def __coerce_same_metadata_as_tangent__(self, flatten_spec, expected_type=None):
    if expected_type is not None:
        return None
    (spec, _) = flatten_spec  # Expected placement from primal
    return self.redistribute(
        device_mesh=self.device_mesh,
        placements=spec.placements,
    )
```

#### Where These Hooks Are Called

In `torch/_functorch/_aot_autograd/collect_metadata_analysis.py:125-126`:

```python
if is_subclass and hasattr(out, "__coerce_tangent_metadata__"):
    out = out.__coerce_tangent_metadata__()
```

And in `torch/_functorch/_aot_autograd/runtime_wrappers.py:2271-2273`:

```python
if same_type:
    return x.__coerce_same_metadata_as_tangent__(expected_meta)
return x.__coerce_same_metadata_as_tangent__(expected_meta, expected_type)
```

### Summary: Eager vs Compiled

| Aspect | Eager Mode | Compiled Mode |
|--------|------------|---------------|
| Partial gradients | Allowed | Coerced to Replicate |
| Coercion mechanism | Only at from_local/to_local boundary | `__coerce_tangent_metadata__` hooks |
| When coercion happens | When exiting DTensor | During AOT tracing |
| FSDP2 reduce-scatter | Possible (Partial → Shard) | Requires workarounds |

The eager mode behavior is more flexible but requires users/systems to handle Partial
gradients appropriately. Compiled mode is more restrictive but guarantees gradients
are never Partial.

## File Structure

```
torch/distributed/tensor/
├── _api.py              # DTensor class, from_local, distribute_tensor, factory funcs
├── _dtensor_spec.py     # DTensorSpec, TensorMeta, ShardOrder
├── placement_types.py   # Shard, Replicate, Partial, _StridedShard
├── device_mesh.py       # DeviceMesh, init_device_mesh (re-exported)
├── _dispatch.py         # OpDispatcher, operator dispatch logic
├── _sharding_prop.py    # Sharding propagation rules
├── _redistribute.py     # Redistribution logic
├── _collective_utils.py # Collective communication helpers
├── _utils.py            # Utility functions (shape computation, etc.)
└── _random.py           # RNG state management for reproducibility
```

## Next Topics

Topics for deeper dives (future docs):

1. **Sharding Propagation** (`_sharding_prop.py`): How output placements are inferred from inputs
2. **Op Registration** (`_op_schema.py`): How operators register their sharding strategies
3. **Redistribute** (`_redistribute.py`): The mechanics of placement conversion
4. **Compile/Dynamo Integration**: How DTensor works with `torch.compile`
5. **FSDP2/TP Integration**: How DTensor powers composable parallelism
