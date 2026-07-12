# CLAUDE.md - Stationarity Testing Project

## Guidelines & Principles

### 1. Code Quality & Engineering Standards
* **Senior Engineer Persona**: Write code to the standard of a Senior Software Engineer at top-tier firms like Jane Street or Google. Focus on clarity, performance, mathematical correctness, strict typing, and comprehensive docstrings.
* **Architecture**: Prefer clean, modular designs. Separate statistical logic, data transformation, and visualization layers.
* **Type Hinting**: Use strict static typing via Python's `typing` module (`List`, `Dict`, `Tuple`, `Optional`, `Callable`) and type aliases for complex types.
* **Vectorization**: Utilize NumPy and SciPy vectorization fully; avoid explicit `for` loops over data rows/arrays wherever possible.

### 2. Theoretical Rigor & Verification
* **Zero Hallucination Policy**: Never hallucinate mathematical formulas, statistical assumptions, or academic references. Every test implemention or interpretation must exactly align with established econometric theory (e.g., Dickey & Fuller (1979), Kwiatkowski et al. (1992)).
* **Hypothesis Testing Clarity**: Clearly document the null ($H_0$) and alternative ($H_1$) hypotheses for every test (e.g., ADF, KPSS, Phillips-Perron) to avoid standard misinterpretations (such as confusing a null of stationarity with a null of a unit root).
* **Edge Cases**: Explicitly handle and test for mathematical edge cases, such as constant variance violations, near-unit roots, and deterministic trends.

### 3. Token & IO Constraints
* **Token Efficiency**: Keep code concise, focused, and free of redundant or overly verbose boilerplate comments. Optimize prompts and code context to stay well within token limits.
* **Data Storage Constraints**: **Do not attempt to read, parse, or process `.csv`, `.xlsx`, or other tabular/database data storage formats** within this framework.
* **Allowed IO**: Reading `.json` files for configuration/metadata or `.pdf` files for academic source paper extraction is permitted if strictly required. Mock or synthetically generate timeseries data using NumPy/SciPy for testing purposes.

---

## Development Workflow

### Build & Run Commands
* **Run Tests**: `pytest`
* **Type Checking**: `mypy src/`
* **Linting & Formatting**: `black src/ tests/ && isort src/ tests/`

### Core Project Structure
```text
src/mht/
    testing/
        kernel_test.py        # KernelTest, Simulator, TestPlotter
        hypothesis.py         # MultipleHypTest, UnitRootTest, LaTeXTable
        leybourne_mccabe.py   # Leybourne-McCabe test
    models/
        processes.py          # BivariateOUProcess, BivariateCorrelatedBM, ...
    io/
        reader.py             # Reader class for simulation CSV files
    viz/                      # TestPlotter re-exported here
    utils/
        decorators.py
simulations/                  # Pre-computed CSV simulation results
notebooks/
    example.ipynb
tests/
    test_processes.py
    test_kernel_test.py
CLAUDE.md
```