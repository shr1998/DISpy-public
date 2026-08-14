# hook-cupy.py
from PyInstaller.utils.hooks import collect_data_files

# 收集 CuPy 的所有数据文件（包括头文件）
datas = collect_data_files('cupy', include_py_files=False)