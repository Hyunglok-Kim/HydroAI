"""
hydroai: hydrological data analysis toolkit of the HydroAI Laboratory (GIST).

Typical usage:

    import hydroai.Data as hData
    import hydroai.Plot as hPlot
    import hydroai.Grid as hGrid

Modules are imported individually (not eagerly here) because several of them
pull in heavy optional dependencies (cartopy, pymc, xgboost, ...).
"""

__version__ = "0.2.0"
