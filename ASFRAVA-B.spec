# -*- mode: python ; coding: utf-8 -*-
block_cipher = None

from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs,
)

# 1) Hidden imports for lazy-loads
hiddenimports = (
    collect_submodules("seaborn") +
    collect_submodules("statsmodels") +
    collect_submodules("scipy") +
    [
        'openseespy.opensees',
        'openseespywin',
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'sklearn',
        'customtkinter',
    ]
)

# 2) Data files - INCLUDE ALL YOUR PACKAGE FOLDERS
datas = (
    collect_data_files("matplotlib") +
    [
        ("modules", "modules"),           # Analysis modules
        ("gui", "gui"),                   # GUI components
        ("utils", "utils"),               # Utility functions
        ("input", "input"),               # Input data/templates
        ("assets", "assets"),             # Logo and other assets
    ]
)

# 3) Native binaries (OpenSeesPy .pyd/.dll)
binaries = collect_dynamic_libs("openseespy")

# 4) Analysis step
a = Analysis(
    ["main.py"],
    pathex=["."],
    hiddenimports=hiddenimports,
    datas=datas,
    binaries=binaries,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# 5) Build the EXE
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ASFRAVA-B",
    icon=["assets\\Logo.ico"],
    console=False,
    upx=True,
)

# 6) Bundle it all in one folder
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ASFRAVA-B",
)