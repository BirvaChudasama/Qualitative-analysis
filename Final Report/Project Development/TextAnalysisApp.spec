from PyInstaller.utils.hooks import collect_data_files
import os

# Correct path to _sqlite3.dll based on your environment
sqlite_dll_path = r"C:\Users\ashok\anaconda3\Library\bin\sqlite3.dll"

binaries = [
    (sqlite_dll_path, os.path.join('DLLs', 'sqlite3.dll')),
]

# Updated .spec file code
a = Analysis(
    ['main.py', 'gui.py'],
    pathex=[r"C:\Users\ashok\OneDrive\Desktop\Project Development"],
    binaries=binaries,
    datas = [
    ('src/*.py', 'src'),
    ('images/*', 'images'),
    ],
    hiddenimports=['pandas', 'sklearn', 'nltk'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    block_cipher=None,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)
