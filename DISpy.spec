# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, copy_metadata

datas = collect_data_files('obspy') + copy_metadata('obspy')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'cupy_backends.cuda.api._runtime_enum',
        'cupy_backends.cuda.stream',
        'cupy._core._carray',
        'fastrlock',
        'fastrlock.rlock',
        'cupy_backends.cuda.api._driver_enum',
        'cupy._core._ufuncs',
        'cupy._core._cub_reduction',
        'cupy._core._routines_sorting',
        'cupy._core.flags',
        'cupy_backends.cuda._softlink',
        'cupy.cuda.common',
        'cupy._core.new_fusion',
        'cupy._core._fusion_trace',
        'cupy._core._fusion_variable',
        'cupy._core._fusion_op',
        'cupy._core._fusion_optimization',
        'cupy._core._fusion_kernel',
    ],
    hookspath=['.'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DISpy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DISpy',
)
