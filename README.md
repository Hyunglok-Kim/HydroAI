# HydroAI

This module, HydroAI, has been designed to aid in conducting research within the HydroAI Laboratory at the Gwangju Institute of Science and Technology (GIST) in Korea.

> [!WARNING]
> **⚠️ Import paths have changed (2026-07).**
> The library moved from a flat layout to a `src/hydroai` package.
>
> | Before | After |
> |---|---|
> | `sys.path.append(...)` + `import HydroAI.Data as hData` | `pip install -e .` + `import hydroai.Data as hData` |
>
> Old `HydroAI.*` imports still work for now through a compatibility shim, but they print a
> `DeprecationWarning` and will be removed in a future release. Please migrate your scripts
> and notebooks to `hydroai.*`.
>
> **(한국어)** 라이브러리 구조가 `src/hydroai` 패키지로 바뀌었습니다. 기존
> `import HydroAI.Data` 방식은 당분간 호환 shim으로 동작하지만 경고가 출력되며, 추후
> 제거될 예정입니다. `pip install -e .` 후 `import hydroai.Data` 형태로 바꿔주세요.

## Installation

```bash
git clone <this repository>
cd HydroAI
pip install -e .            # core dependencies
pip install -e ".[ml]"      # + Shallow_ML dependencies (sklearn, xgboost, ...)
pip install -e ".[bayes]"   # + WBM Bayesian fitting (pymc, arviz, ...)
```

Or with conda:

```bash
conda env create -f hydroai_environment.yml
conda activate hydroai
```

Note: GPU acceleration in `Shallow_ML` uses NVIDIA RAPIDS (`cuml`), which must be
installed separately via conda; the module works without it.

## Usage

```python
import hydroai.Data as hData
import hydroai.Plot as hPlot
import hydroai.Grid as hGrid
```

Example notebooks live in `examples/`. The HydroAI module can be utilized in academic research as well as in commercial applications that involve hydrological data analysis, modeling, and interpretation.

## Citation

When using this code for research or any form of commercialization, kindly cite our work appropriately. Detailed citation information will be provided soon.

## Contact

For additional information or inquiries, feel free to reach out to Dr. Hyunglok Kim at hyunglokkim@gmail.com.
