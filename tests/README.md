# Tests Directory

## 📁 Structure

```
tests/
├── unit/                  # Unit tests (component-level)
│   ├── loadstore/        # Memory access operations
│   ├── ptx/             # PTX inline assembly tests
│   ├── tensor/           # Tensor Core tests
│   ├── container/        # Container utilities (tuple, etc.)
│   └── layout/           # Layout and shape utilities
│
└── integration/           # Integration tests (multi-component)
    └── test_flash_attention.cu
```

## 🎯 Test Types

### Unit Tests (`unit/`)
Test individual components in isolation:
- **loadstore**: Memory loading/storing with different access patterns
- **ptx**: PTX inline assembly and Tensor Core operations
- **tensor**: Tensor Core fragment operations
- **container**: Utility containers and data structures
- **layout**: Shape and layout transformations

### Integration Tests (`integration/`)
Test multiple components working together:
- `test_flash_attention.cu`: End-to-end Flash Attention test

## 🚀 Running Tests

### Run All Unit Tests
```bash
# Using gtest (if configured)
cd tests/unit
./run_all_unit_tests

# Or manually compile individual tests
nvcc -O3 -arch=sm_80 unit/loadstore/test_copy_atom.cu -o test_atom
./test_atom
```

### Run Integration Tests
```bash
cd tests/integration
nvcc -O3 -arch=sm_80 \
    -I../../src \
    test_flash_attention.cu \
    -o test_flash_attn
./test_flash_attn
```

## 📝 Notes

- Unit tests verify component correctness before integration
- Use gtest for automated test running
- Performance tests are in `../benchmark/` directory
- Correctness tests for Flash Attention are in `../benchmark/cpp/`

## 🔗 Related

- `benchmark/` - Performance benchmarks and correctness tests for Flash Attention
- `src/` - Source code for Flash Attention kernels
