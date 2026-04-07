#!/usr/bin/env python3
"""
DUPLICATE FILE FINDER v5.7 — Smart Pre-filter & Streaming Engine
created by: Shawn Mitchell (must have small display at bottom of program)
══════════════════════════════════════════════════
Built from scratch using DuplicateFinder_v4.0 as a reference.

Architecture improvements over v4.0:
  • FileRecord / DupGroup / ScanSettings dataclasses (typed, faster than dicts)
  • ScanEngine fully decoupled from UI — pure logic, no tkinter imports
  • os.scandir() recursive traversal (~3× faster than Path.glob)
  • Three-phase hashing: quick 4 KB head → partial first+last 64 KB → full
  • Hard-link detection via (device, inode) pairs
  • Magic-bytes file-type detection for smarter near-dup scoring
  • UnionFind transitive cluster merging (A~B + B~C → {A,B,C})
  • ProcessPoolExecutor for CPU-bound pair comparison when pairs > 2000
  • Enhanced auto-selection: name quality + path quality + temporal + location
  • SafeDeleter with persistent JSON deletion log
  • SessionManager: save/load scan results to skip re-scanning
  • Split-pane UI: left Treeview (all groups) + right detail panel (file cards)
  • Dark-mode toggle
  • Open File / Open Folder per file card
  • Export: TXT and CSV
  • Keyboard shortcuts: Ctrl+S scan, Esc stop, arrows navigate, Del delete
  • Collapsible debug terminal
  • Filter bar above treeview (search, type, score)
  • Paranoid mode: optional byte-by-byte verification of exact matches
  • Deletion history log (JSON, persistent across sessions)

GPU note:
  CuPy / NumPy used only for optional vectorised filename bigram cosine pass.
  File hashing is I/O-bound; GPU transfer overhead exceeds hashing cost.
  Disable GPU if you do not have a CUDA GPU with CuPy installed.
"""

# ── Auto-install optional libraries ───────────────────────────────────────────
import subprocess as _sp, sys as _sys

_AUTO_INSTALL_PKGS = [
    ('xxhash',     'xxhash',     'xxhash'),
    ('psutil',     'psutil',     'psutil'),
    ('send2trash', 'send2trash', 'send2trash'),
    ('PIL',        'Pillow',     'Pillow'),
    ('numpy',      'numpy',      'numpy'),
]

def _pip_install(pip_name: str, verbose: bool = True) -> bool:
    """Install a package via pip. Returns True on success."""
    try:
        import subprocess as _sub, sys as _s
        result = _sub.run(
            [_s.executable, '-m', 'pip', 'install', '--upgrade', pip_name],
            capture_output=True, text=True, timeout=120
        )
        if verbose:
            if result.returncode == 0:
                print(f'[auto-install] ✓ {pip_name} installed successfully')
            else:
                print(f'[auto-install] ✗ {pip_name} FAILED:\n{result.stderr[:300]}')
        return result.returncode == 0
    except Exception as exc:
        if verbose:
            print(f'[auto-install] ✗ {pip_name} error: {exc}')
        return False

_INSTALL_RESULTS: dict = {}   # pkg_name → 'ok' | 'failed' | 'present'
for _import_name, _pip_name, _display_name in _AUTO_INSTALL_PKGS:
    try:
        __import__(_import_name)
        _INSTALL_RESULTS[_display_name] = 'present'
    except ImportError:
        print(f'[auto-install] {_display_name} not found — installing…')
        ok = _pip_install(_pip_name, verbose=True)
        _INSTALL_RESULTS[_display_name] = 'ok' if ok else 'failed'
# clean up loop variables (keep _pip_install and _sp for use in Settings tab)
try:
    del _import_name, _pip_name, _display_name
except NameError:
    pass

# ── Standard library ──────────────────────────────────────────────────────────
import os
import subprocess
import re
import sys
import threading
import time
import queue
import hashlib
import datetime
import json
import traceback
import multiprocessing
import struct
import csv
import io
import webbrowser
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Set, Tuple, Any

# ── Optional performance / feature libraries ──────────────────────────────────
try:
    import xxhash as _xxhash; HAS_XXHASH = True
except ImportError:
    HAS_XXHASH = False

try:
    import psutil; HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import numpy as np; HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import cupy as cp; _p = cp.array([1.0]); del _p; HAS_CUPY = True
except Exception:
    HAS_CUPY = False

try:
    import send2trash as _s2t; HAS_SEND2TRASH = True
except ImportError:
    HAS_SEND2TRASH = False

try:
    from PIL import Image as _PILImage, ImageTk as _PILImageTk; HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ── GUI ───────────────────────────────────────────────────────────────────────
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext, filedialog, font as tkfont
    HAS_GUI = True
except ImportError:
    HAS_GUI = False
    print('ERROR: Tkinter not found.'); sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
# ─── RELEASE VERSION ── bump this single constant for every release ───────
VERSION               = '5.7'   # <─ single source of truth for all UI text
# ─────────────────────────────────────────────────────────────────────────
CPU_COUNT             = max(1, multiprocessing.cpu_count())
HASH_CHUNK            = 65536            # 64 KB read chunk
QUICK_HASH_BYTES      = 4096             # 4 KB first-pass quick hash
PARTIAL_THRESHOLD     = 1_048_576        # 1 MB → use 3-phase hashing above this
MAGIC_BYTES           = 16              # bytes to read for file-type detection
DEBUG_MAX_LINES       = 5000
SPINNER_FRAMES        = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']
DELETION_LOG_PATH     = Path.home() / '.dupfinder_v5_deletions.json'
SESSION_EXT           = '.dupfinder5'
NEAR_DUP_MAX_PAIRS    = 100_000         # hard cap on near-dup pair comparisons
PER_GROUP_PAIR_CAP    = 5_000           # max pairs from a single size group
NEAR_BATCH_SIZE       = 200             # pairs per executor batch (less overhead)
MAGIC_SIZE_SKIP       = 50_000_000      # skip magic detection for files > 50 MB
MAGIC_COUNT_SKIP      = 50_000          # skip magic detection if file count > 50K

# Copy / backup name patterns (comprehensive)
COPY_PATTERNS = [
    '- copy', '(copy)', ' copy', '_copy', '-copy', 'copy of ',
    'duplicate', 'backup', '- backup', '_backup', '-bak', '.bak',
    ' old', '_old', '-old', 'orig-', 'original-', '- original',
    '_orig', '-orig', 'temp_', '_temp', '-temp',
]

# Temp / cache dir name tokens (lower-case)
TEMP_DIR_TOKENS = frozenset({
    'temp', 'tmp', 'cache', '.cache', '.tmp', 'recycle',
    'trash', '$recycle.bin', 'appdata', 'localappdata',
    'application data', '__pycache__', '.git', 'node_modules',
})

# Preferred directory tokens → boost keep score
PREFERRED_DIR_TOKENS = frozenset({
    'desktop', 'documents', 'pictures', 'photos', 'videos',
    'movies', 'music', 'downloads',
})

# Magic-bytes file-type signatures: (prefix_bytes, type_label)
FILE_MAGIC_SIGS: List[Tuple[bytes, str]] = [
    (b'\xff\xd8\xff',          'JPEG'),
    (b'\x89PNG\r\n\x1a\n',    'PNG'),
    (b'GIF87a',                'GIF'),
    (b'GIF89a',                'GIF'),
    (b'%PDF',                  'PDF'),
    (b'PK\x03\x04',           'ZIP'),
    (b'PK\x05\x06',           'ZIP'),
    (b'ID3',                   'MP3'),
    (b'\xff\xfb',              'MP3'),
    (b'\xff\xf3',              'MP3'),
    (b'\xff\xf2',              'MP3'),
    (b'fLaC',                  'FLAC'),
    (b'OggS',                  'OGG'),
    (b'RIFF',                  'RIFF'),     # WAV / AVI
    (b'\x1f\x8b',              'GZIP'),
    (b'Rar!\x1a\x07',         'RAR'),
    (b'7z\xbc\xaf\'\'',       '7Z'),
    (b'BM',                    'BMP'),
    (b'II*\x00',               'TIFF'),
    (b'MM\x00*',               'TIFF'),
    (b'\x89HDF',               'HDF5'),
    (b'\x00\x00\x00\x18ftyp', 'MP4'),
    (b'\x00\x00\x00\x20ftyp', 'MP4'),
    (b'\x00\x00\x00\x14ftyp', 'MP4'),
    (b'\x1aE\xdf\xa3',        'MKV'),
    (b'MZ',                    'EXE/DLL'),
    (b'\x7fELF',               'ELF'),
    (b'\xca\xfe\xba\xbe',     'MACH-O'),
    (b'<!DOCTYPE html',        'HTML'),
    (b'<html',                 'HTML'),
    (b'<?xml',                 'XML'),
    (b'\xef\xbb\xbf',         'UTF8-BOM'),
]

# Extension compatibility groups (same group = near-identical format)
EXT_COMPAT_GROUPS: List[Set[str]] = [
    {'jpg', 'jpeg', 'jpe', 'jfif'},
    {'png'},
    {'gif'},
    {'bmp', 'dib'},
    {'tif', 'tiff'},
    {'webp'},
    {'doc', 'docx', 'rtf', 'odt'},
    {'xls', 'xlsx', 'ods', 'csv'},
    {'ppt', 'pptx', 'odp'},
    {'mp3', 'm4a', 'aac', 'ogg', 'flac', 'wav', 'wma', 'opus'},
    {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'm4v', 'webm'},
    {'zip', 'gz', 'tar', 'bz2', 'xz', '7z', 'rar'},
    {'py', 'pyw'},
    {'js', 'jsx', 'ts', 'tsx'},
    {'htm', 'html', 'xhtml'},
    {'txt', 'text', 'log', 'md', 'rst'},
]


# ═════════════════════════════════════════════════════════════════════════════
#  Dataclasses
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class FileRecord:
    """Typed, hashable record for one scanned file."""
    path:         Path
    size:         int
    mtime:        float
    ctime:        float
    inode:        int
    device:       int
    ext:          str
    name:         str
    hash:         Optional[str]  = None
    partial_hash: Optional[str]  = None   # first+last HASH_CHUNK bytes
    quick_hash:   Optional[str]  = None   # first QUICK_HASH_BYTES only
    magic_type:   Optional[str]  = None
    keep_score:   int            = 100    # cached auto-selection score

    def __hash__(self):   return hash(str(self.path))
    def __eq__(self, o):  return str(self.path) == str(o.path)


@dataclass
class DupGroup:
    """One group of duplicate / near-duplicate files."""
    files:      List[FileRecord]
    score:      int
    group_type: str                         # 'exact' | 'near' | 'hardlink'
    components: Dict[str, int] = field(default_factory=dict)
    suggestions: Dict[int, str] = field(default_factory=dict)  # fi→'KEEP'|'DELETE'

    @property
    def is_exact(self) -> bool:
        return self.group_type in ('exact', 'hardlink')

    @property
    def reclaimable_bytes(self) -> int:
        if len(self.files) < 2: return 0
        return sum(f.size for f in self.files) - max(f.size for f in self.files)


@dataclass
class ScanSettings:
    """All user-configurable scan settings."""
    subdirs:            bool  = True
    min_size:           int   = 1            # bytes
    max_size:           int   = 0            # 0 = unlimited
    use_xxhash:         bool  = True
    hash_files:         bool  = True
    paranoid_mode:      bool  = False        # byte-by-byte verify exact matches
    use_gpu:            bool  = False
    num_workers:        int   = min(CPU_COUNT * 2, 16)
    min_score:          int   = 70
    exclusion_patterns: List[str] = field(default_factory=list)
    preferred_dirs:     List[str] = field(default_factory=list)
    dark_mode:          bool  = False
    auto_select:        bool  = True
    delete_gap:         int   = 15           # min quality gap to suggest DELETE


# ═════════════════════════════════════════════════════════════════════════════
#  UnionFind — transitive cluster merging (A~B + B~C → {A,B,C})
# ═════════════════════════════════════════════════════════════════════════════

class UnionFind:
    """Path-compressed union-find for O(α·n) merging."""

    def __init__(self):
        self._parent: Dict[str, str] = {}
        self._rank:   Dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x]   = 0
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])   # path compression
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        px, py = self.find(x), self.find(y)
        if px == py: return
        # Union by rank
        if self._rank[px] < self._rank[py]:
            px, py = py, px
        self._parent[py] = px
        if self._rank[px] == self._rank[py]:
            self._rank[px] += 1

    def groups(self) -> Dict[str, List[str]]:
        """Return {root: [members...]} for all connected components."""
        g: Dict[str, List[str]] = defaultdict(list)
        for k in self._parent:
            g[self.find(k)].append(k)
        return dict(g)

# ═════════════════════════════════════════════════════════════════════════════
#  Module-level helpers  (no class dependency — safe from any thread/process)
# ═════════════════════════════════════════════════════════════════════════════

def _format_size(n: int) -> str:
    """Human-readable file size."""
    if n < 1024:         return f'{n} B'
    if n < 1_048_576:    return f'{n/1024:.1f} KB'
    if n < 1_073_741_824:return f'{n/1_048_576:.2f} MB'
    return f'{n/1_073_741_824:.2f} GB'


def _format_ts(ts: float) -> str:
    """Unix timestamp → 'YYYY-MM-DD HH:MM:SS'."""
    if not ts:
        return 'N/A'
    try:
        return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return 'N/A'


_ts = _format_ts   # short alias used throughout UI code
# ─── Log file path: timestamped, relative to script dir, in logs/ subfolder ───
def _make_log_path() -> Path:
    """Return a timestamped log path inside <script_dir>/logs/ (created if absent)."""
    try:
        base = Path(__file__).resolve().parent
    except Exception:
        base = Path.cwd()
    log_dir = base / 'logs'
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        log_dir = Path.home()   # fallback: home dir if can't create logs/
    ts = time.strftime('%Y%m%d_%H%M%S')
    return log_dir / f'dupfinder_{ts}.log'

LOG_FILE_PATH = _make_log_path()

# Auto-verify version against module docstring — warns on mismatch
try:
    _doc_ver = re.search(r'v(\d+\.\d+)', __doc__ or '').group(1)  # type: ignore
    if _doc_ver != VERSION:
        print(f'[version] ⚠️  docstring says v{_doc_ver} but VERSION={VERSION!r}')
except Exception:
    pass


def _detect_magic(path) -> Optional[str]:
    """Read the first MAGIC_BYTES of *path* and return a type label or None."""
    try:
        with open(path, 'rb') as fh:
            header = fh.read(MAGIC_BYTES)
        for sig, label in FILE_MAGIC_SIGS:
            if header.startswith(sig):
                return label
    except Exception:
        pass
    return None


def _ext_compat(ext1: str, ext2: str) -> float:
    """Return 0.0–1.0 extension compatibility: 1.0=same, 0.9=same group, 0.0=incompatible."""
    if ext1 == ext2:
        return 1.0
    e1 = ext1.lower().lstrip('.')
    e2 = ext2.lower().lstrip('.')
    if not e1 or not e2:
        return 0.0
    for g in EXT_COMPAT_GROUPS:
        if e1 in g and e2 in g:
            return 0.9
    return 0.0


def _name_keep_score(name_stem: str) -> int:
    """
    Score a filename stem for 'keepability'.
    Higher = more likely to be the original (worth keeping).
    Range: -60 … +10.
    """
    sc   = 0
    stem = name_stem.lower()

    for pat in COPY_PATTERNS:
        if pat in stem:
            sc -= 30
            break

    # Trailing number patterns: file(1), file-2, file_3, file 2
    if re.search(r'\(\d+\)\s*$', stem):
        sc -= 20
    elif re.search(r'[\s_\-]\d+\s*$', stem):
        sc -= 15

    # Very short name is suspicious
    if len(stem) <= 2:
        sc -= 5

    # Long, descriptive name is good
    if len(stem) >= 10 and not re.search(r'[_\-]{3,}', stem):
        sc += 5

    return sc


def _path_keep_score(path: Path) -> int:
    """
    Score a file's directory for keepability.
    Higher = shallower / more authoritative location.
    Range: -40 … +15.
    """
    sc = 0
    parts_lower = [p.lower() for p in path.parts]

    # Penalise temp / cache directories anywhere in the path
    for token in TEMP_DIR_TOKENS:
        if any(token in p for p in parts_lower):
            sc -= 25
            break

    # Boost preferred locations
    for token in PREFERRED_DIR_TOKENS:
        if any(token in p for p in parts_lower):
            sc += 10
            break

    # Penalise deep nesting (more than 4 levels below root)
    depth = len(path.parts)
    if depth > 5:
        sc -= min((depth - 5) * 2, 20)

    # Very shallow = good (Desktop, Documents root, etc.)
    if depth <= 4:
        sc += 5

    return sc


def _total_keep_score(f: FileRecord) -> int:
    """Combined keep score for a FileRecord (higher = keep this one)."""
    stem = Path(f.name).stem
    sc   = 100 + _name_keep_score(stem) + _path_keep_score(f.path)
    return max(0, min(200, sc))


def _calculate_dup_score(args):
    """
    Compute (score 0-100, components dict) for two FileRecords.

    Called by ThreadPoolExecutor and ProcessPoolExecutor workers, so
    arguments are passed as a tuple to support pickling.

    Score breakdown (max 100 before cap):
      quick_hash_same + same_size → 55  (near-certain; saves full hash for confirm)
      size_exact                  → 35
      size_close (<5%)            → 15
      name_similarity > 0.9       → 25
      name_similarity > 0.7       → 18
      name_similarity > 0.5       → 10
      extension exact             → 12
      extension compat            → 6
      magic_type match            → 8
      same parent directory       → 5
      mtime proximity < 1h        → 5
      mtime proximity < 24h       → 3
      mtime proximity < 7d        → 1
    """
    f1d, f2d = args     # passed as plain dicts for pickling safety
    comp: Dict[str, int] = {}

    # Unpack dicts (used when called from ProcessPoolExecutor)
    h1  = f1d.get('hash')
    h2  = f2d.get('hash')
    ph1 = f1d.get('partial_hash')
    ph2 = f2d.get('partial_hash')
    qh1 = f1d.get('quick_hash')
    qh2 = f2d.get('quick_hash')
    s1  = f1d.get('size', 0)
    s2  = f2d.get('size', 0)
    n1  = f1d.get('name', '')
    n2  = f2d.get('name', '')
    e1  = f1d.get('ext', '')
    e2  = f2d.get('ext', '')
    mt1 = f1d.get('magic_type')
    mt2 = f2d.get('magic_type')
    m1  = f1d.get('mtime', 0)
    m2  = f2d.get('mtime', 0)
    p1  = str(f1d.get('parent', ''))
    p2  = str(f2d.get('parent', ''))

    # ── Exact hash → 100 immediately ─────────────────────────────────────
    if h1 and h2 and h1 == h2:
        return 100, {'hash': 100, 'size': 0, 'name': 0, 'ext': 0,
                     'magic': 0, 'dir': 0, 'time': 0}

    # ── Different size → not duplicate (can't be near with large size diff) ─
    if s1 != s2:
        ratio = abs(s1 - s2) / max(s1, s2, 1)
        if ratio > 0.20:    # > 20% size difference → skip
            return 0, {}

    # ── Partial / quick hash early-exit ──────────────────────────────────
    # Same size but different partial hash → very unlikely duplicate
    if ph1 and ph2 and ph1 != ph2 and s1 == s2:
        return 0, {}
    if qh1 and qh2 and qh1 != qh2 and s1 == s2:
        return 0, {}

    score = 0

    # ── Size ─────────────────────────────────────────────────────────────
    if s1 == s2:
        score += 35; comp['size'] = 35
    else:
        ratio = abs(s1 - s2) / max(s1, s2, 1)
        pts   = 15 if ratio <= 0.05 else (8 if ratio <= 0.10 else 0)
        score += pts; comp['size'] = pts

    # ── Quick hash bonus (same quick hash + same size = near-certain) ─────
    if qh1 and qh2 and qh1 == qh2 and s1 == s2:
        score += 20; comp['quick'] = 20
    else:
        comp['quick'] = 0

    # ── Name similarity ───────────────────────────────────────────────────
    stem1 = Path(n1).stem.lower()
    stem2 = Path(n2).stem.lower()
    sim   = SequenceMatcher(None, stem1, stem2).ratio()
    if   sim > 0.90: pts = 25
    elif sim > 0.70: pts = 18
    elif sim > 0.50: pts = 10
    else:            pts = 0
    score += pts; comp['name'] = pts

    # ── Extension ─────────────────────────────────────────────────────────
    if e1 == e2:
        score += 12; comp['ext'] = 12
    else:
        pts = int(_ext_compat(e1, e2) * 6)
        score += pts; comp['ext'] = pts

    # ── Magic type ────────────────────────────────────────────────────────
    if mt1 and mt2 and mt1 == mt2:
        score += 8; comp['magic'] = 8
    else:
        comp['magic'] = 0

    # ── Same parent directory ─────────────────────────────────────────────
    if p1 and p2 and p1 == p2:
        score += 5; comp['dir'] = 5
    else:
        comp['dir'] = 0

    # ── Temporal proximity ────────────────────────────────────────────────
    if m1 and m2:
        diff_h = abs(m1 - m2) / 3600.0
        if   diff_h < 1:   pts = 5
        elif diff_h < 24:  pts = 3
        elif diff_h < 168: pts = 1
        else:              pts = 0
        score += pts; comp['time'] = pts
    else:
        comp['time'] = 0

    return min(score, 100), comp


def _hash_file_worker(args):
    """
    Hash one file; safe to run from any thread or process.
    args = (fp_str, chunk_size, use_xxhash, mode)
    mode: 'quick' | 'partial' | 'full'
    Returns (fp_str, digest_or_None, mode).
    """
    fp_str, chunk_size, use_xxhash, mode = args
    try:
        if use_xxhash and HAS_XXHASH:
            h = _xxhash.xxh64()
        else:
            try:    h = hashlib.md5(usedforsecurity=False)
            except TypeError: h = hashlib.md5()

        with open(fp_str, 'rb') as fh:
            if mode == 'quick':
                data = fh.read(QUICK_HASH_BYTES)
                if data: h.update(data)
            elif mode == 'partial':
                fsize = os.path.getsize(fp_str)
                data = fh.read(chunk_size)
                if data: h.update(data)
                if fsize > chunk_size:
                    fh.seek(max(0, fsize - chunk_size))
                    data = fh.read(chunk_size)
                    if data: h.update(data)
            else:   # full
                while True:
                    chunk = fh.read(chunk_size)
                    if not chunk: break
                    h.update(chunk)
        return fp_str, h.hexdigest(), mode
    except Exception:
        return fp_str, None, mode


def _byte_compare(path1, path2) -> bool:
    """Byte-by-byte file comparison for paranoid verification."""
    try:
        with open(path1, 'rb') as f1, open(path2, 'rb') as f2:
            while True:
                b1 = f1.read(HASH_CHUNK)
                b2 = f2.read(HASH_CHUNK)
                if b1 != b2: return False
                if not b1:   return True
    except Exception:
        return False


def _score_batch(pair_dicts: list, min_score: int) -> list:
    """
    Score a batch of file-record dict pairs.
    Returns [(local_idx, score, comp), ...] for pairs scoring >= min_score.
    Module-level so it can be called from any thread without pickling issues.
    """
    results = []
    for i, pd in enumerate(pair_dicts):
        sc, comp = _calculate_dup_score(pd)
        if sc >= min_score:
            results.append((i, sc, comp))
    return results


# ═════════════════════════════════════════════════════════════════════════════
#  ScanEngine  — pure logic, no tkinter dependency
# ═════════════════════════════════════════════════════════════════════════════

class ScanEngine:
    """
    Full duplicate-detection pipeline:

      1. _discover()       → List[Path]           (os.scandir, fast)
      2. _stat_batch()     → List[FileRecord]      (parallel stat)
      3. _detect_magic()   → in-place on records   (optional, parallel)
      4. _hash_all()       → 3-phase hashing       (parallel, adaptive)
      5. find_duplicates() → List[DupGroup]
         a. _find_hardlinks()   exact inode match
         b. _group_by_size()    pre-filter
         c. _find_exact()       hash-based exact
         d. _find_near()        scored pairwise
         e. _merge_clusters()   UnionFind transitive merge
      6. smart_select()    → annotates DupGroup.suggestions
    """

    def __init__(self,
                 root: str,
                 settings: ScanSettings,
                 progress_queue: queue.Queue,
                 cancel_event: threading.Event):
        self.root          = Path(root)
        self.settings      = settings
        self.pq            = progress_queue
        self.cancel        = cancel_event
        self.files: List[FileRecord] = []
        self._script_name  = Path(sys.argv[0]).name.lower()

    # ── Messaging helpers ─────────────────────────────────────────────────

    def _send(self, mtype: str, data) -> None:
        try: self.pq.put_nowait((mtype, data))
        except queue.Full: pass

    def _log(self, tag: str, text: str) -> None:
        self._send('log', (tag, text))

    def _dbg(self, text: str) -> None:
        self._send('debug', text)

    def _err(self, text: str, exc: Exception = None) -> None:
        full = text + (f'\n  {type(exc).__name__}: {exc}' if exc else '')
        self._send('error_detail', full)
        self._log('error', text)

    def _progress(self, kind: str, **kw) -> None:
        self._send(kind, kw)

    # ── Stage 1: File Discovery (os.scandir, fast) ────────────────────────

    def _discover(self) -> List[Path]:
        """Recursive os.scandir — ~3× faster than Path.glob."""
        results: List[Path] = []
        self._dbg(f'[SCAN] Discovery starting  root={self.root}'
                  f'  recursive={self.settings.subdirs}')

        def _walk(dirpath: Path):
            if self.cancel.is_set(): return
            try:
                with os.scandir(dirpath) as it:
                    for entry in it:
                        if self.cancel.is_set(): break
                        try:
                            if entry.is_file(follow_symlinks=False):
                                results.append(Path(entry.path))
                            elif (entry.is_dir(follow_symlinks=False)
                                  and self.settings.subdirs):
                                # Skip known junk directories
                                if entry.name.lower() not in TEMP_DIR_TOKENS:
                                    _walk(Path(entry.path))
                        except (OSError, PermissionError):
                            pass
            except (PermissionError, OSError) as exc:
                self._dbg(f'[SCAN] SKIP dir {dirpath}  reason={exc}')

        _walk(self.root)
        self._dbg(f'[SCAN] Discovery done  paths={len(results)}')
        return results

    # ── Stage 2: Parallel stat ─────────────────────────────────────────────

    def _stat_batch(self, paths: List[Path]) -> List[FileRecord]:
        """Parallel stat + filter → FileRecord list."""
        settings   = self.settings
        script     = self._script_name
        min_sz     = settings.min_size
        max_sz     = settings.max_size if settings.max_size > 0 else float('inf')
        excl       = [p.lower() for p in settings.exclusion_patterns]

        def _stat(fp: Path) -> Optional[FileRecord]:
            if fp.name.lower() == script:
                return None
            # Exclusion patterns
            fp_str = str(fp).lower()
            if any(pat in fp_str for pat in excl):
                return None
            try:
                st = fp.stat()
                if not (min_sz <= st.st_size <= max_sz):
                    return None
                return FileRecord(
                    path=fp, size=st.st_size,
                    mtime=st.st_mtime, ctime=st.st_ctime,
                    inode=st.st_ino,   device=st.st_dev,
                    ext=fp.suffix.lower(), name=fp.name,
                )
            except (OSError, PermissionError) as exc:
                self._dbg(f'[SCAN] STAT-FAIL {fp.name[:40]}  {exc}')
                return None

        total        = len(paths)
        results      = []
        done         = 0
        report_every = max(1, total // 200)
        nw           = settings.num_workers

        with ThreadPoolExecutor(max_workers=nw) as pool:
            futs = {pool.submit(_stat, fp): fp for fp in paths}
            for fut in as_completed(futs):
                if self.cancel.is_set(): break
                done += 1
                rec = fut.result()
                if rec:
                    results.append(rec)
                if done % report_every == 0 or done == total:
                    nm = rec.name[:35] if rec else '…'
                    self._progress('scan_progress',
                                   current=done, total=total,
                                   percent=int(done/max(total,1)*100),
                                   file=nm, status='📄 Scanning')
                    self._dbg(f'[SCAN] {done}/{total}  files={len(results)}')

        return results

    # ── Stage 3: Magic-type detection ─────────────────────────────────────

    def _detect_magic_batch(self, files: List[FileRecord]) -> None:
        # ── modular: magic-type detection stage ──────────────────────────────
        """
        Detect file magic types in parallel; updates records in-place.
        Automatically skips files > MAGIC_SIZE_SKIP bytes (overhead not worth it).
        Skips entirely if file count > MAGIC_COUNT_SKIP (too slow for whole-drive).
        """
        if not files: return
        total = len(files)

        # Skip if file count is enormous
        if total > MAGIC_COUNT_SKIP:
            self._log('info', f'⚡ Skipping magic detection ({total:,} files — using extension only)')
            self._dbg(f'[SCAN] Magic skipped: {total} > {MAGIC_COUNT_SKIP} threshold')
            return

        # Filter to only smallish files worth inspecting
        eligible = [f for f in files if f.size <= MAGIC_SIZE_SKIP]
        skipped_large = total - len(eligible)

        self._log('info', f'🔬 Detecting file types: {len(eligible):,} files'
                          + (f' ({skipped_large:,} large files skipped)' if skipped_large else ''))
        self._dbg(f'[SCAN] Magic on {len(eligible)}/{total}  skipped_large={skipped_large}')

        if not eligible: return
        etotal = len(eligible)
        report_every = max(1, etotal // 20)
        done = 0
        last_t = time.monotonic()

        def _detect(fp: Path):
            return _detect_magic(fp)

        with ThreadPoolExecutor(max_workers=self.settings.num_workers) as pool:
            futs = {pool.submit(_detect, f.path): f for f in eligible}
            for fut in as_completed(futs):
                if self.cancel.is_set(): break
                f = futs[fut]
                try:
                    f.magic_type = fut.result()
                except Exception:
                    pass
                done += 1
                now = time.monotonic()
                if done % report_every == 0 or done == etotal or (now - last_t) > 1.0:
                    last_t = now
                    pct = int(done / max(etotal, 1) * 100)
                    self._progress('scan_progress',
                                   current=done, total=etotal, percent=pct,
                                   file=f.name[:35], status='🔬 Detecting file types')
                    self._dbg(f'[SCAN] Magic {done}/{etotal} ({pct}%)')

        typed = sum(1 for f in files if f.magic_type)
        self._dbg(f'[SCAN] Magic done  typed={typed}/{total}')
        self._log('info', f'✓ File-type detection: {typed:,}/{total:,} typed')

    # ── Stage 4: Three-phase hashing ──────────────────────────────────────

    def _hash_all(self, files: List[FileRecord]) -> None:
        """
        Phase 1: quick hash (4 KB) all files.
        Phase 2: partial hash (first+last 64 KB) large files that share quick hash.
        Phase 3: full hash files that share (size, partial_hash).
        """
        if not files or not self.settings.hash_files: return
        algo = 'xxhash-xxh64' if (self.settings.use_xxhash and HAS_XXHASH) else 'MD5'
        nw   = self.settings.num_workers
        ux   = self.settings.use_xxhash and HAS_XXHASH

        self._log('info', f'🔐 3-phase hashing {len(files)} files ({algo}, {nw} workers)…')
        self._dbg(f'[HASH] Start  total={len(files)}  algo={algo}  workers={nw}')

        # ── Phase 1: quick hash all ───────────────────────────────────────
        self._hash_batch(files, 'quick', ux, nw, 'Phase-1 quick')

        if self.cancel.is_set(): return

        # ── Phase 2: partial hash large files that share quick hash ───────
        quick_groups: Dict[str, List[FileRecord]] = defaultdict(list)
        for f in files:
            if f.quick_hash and f.size > PARTIAL_THRESHOLD:
                quick_groups[(f.size, f.quick_hash)].append(f)
        need_partial = [f for g in quick_groups.values() if len(g) > 1 for f in g]
        skipped_partial = sum(1 for f in files
                              if f.size > PARTIAL_THRESHOLD) - len(need_partial)
        self._dbg(f'[HASH] Phase 2: {len(need_partial)} large need partial'
                  f'  {skipped_partial} skipped')

        if need_partial and not self.cancel.is_set():
            self._hash_batch(need_partial, 'partial', ux, nw, 'Phase-2 partial')

        if self.cancel.is_set(): return

        # ── Phase 3: full hash files sharing (size, partial_hash) ─────────
        partial_groups: Dict[Tuple, List[FileRecord]] = defaultdict(list)
        # Small files (already fully hashed by quick hash if <= QUICK_HASH_BYTES)
        for f in files:
            if f.size <= QUICK_HASH_BYTES:
                # quick hash IS the full hash for tiny files
                f.hash = f.quick_hash
            elif f.size <= PARTIAL_THRESHOLD:
                # medium files: quick hash → group → full hash candidates
                if f.quick_hash:
                    partial_groups[(f.size, f.quick_hash)].append(f)
            else:
                # large files: partial hash → group → full hash candidates
                if f.partial_hash:
                    partial_groups[(f.size, f.partial_hash)].append(f)

        need_full = [f for g in partial_groups.values() if len(g) > 1 for f in g]
        skipped_full = sum(1 for f in files if not f.hash) - len(need_full)
        self._dbg(f'[HASH] Phase 3: {len(need_full)} need full hash'
                  f'  {skipped_full} unique → skipped')

        if need_full and not self.cancel.is_set():
            self._hash_batch(need_full, 'full', ux, nw, 'Phase-3 full')

        self._log('info', '✓ Hashing complete — grouping results…')
        self._dbg('[HASH] All phases done')

    def _hash_batch(self, files: List[FileRecord], mode: str,
                    use_xxhash: bool, nw: int, label: str) -> None:
        """Hash a batch of files; update records in-place."""
        if not files: return
        total    = len(files)
        path_map = {str(f.path): f for f in files}
        hash_key = {'quick': 'quick_hash', 'partial': 'partial_hash',
                    'full': 'hash'}[mode]
        args_list = [(str(f.path), HASH_CHUNK, use_xxhash, mode) for f in files]
        done     = 0
        errors   = 0
        report_every = max(1, total // 100)

        with ThreadPoolExecutor(max_workers=nw) as pool:
            futs = {pool.submit(_hash_file_worker, a): a[0] for a in args_list}
            for fut in as_completed(futs):
                if self.cancel.is_set(): break
                done += 1
                fp_str, digest, _ = fut.result()
                if digest:
                    if fp_str in path_map:
                        setattr(path_map[fp_str], hash_key, digest)
                else:
                    errors += 1
                    self._dbg(f'[HASH] ERR {Path(fp_str).name[:40]}')
                if done % report_every == 0 or done == total:
                    pct = int(done / max(total, 1) * 100)
                    nm  = Path(fp_str).name[:35] if fp_str else '…'
                    self._progress('scan_progress',
                                   current=done, total=total, percent=pct,
                                   file=nm, status=f'🔐 {label}')
                    self._dbg(f'[HASH] {label} {done}/{total} ({pct}%)'
                              f'  err={errors}')

    # ── Stage 5: Duplicate detection pipeline ─────────────────────────────

    def scan(self) -> int:
        """Run full scan pipeline. Returns file count."""
        self.files = []
        if not self.root.exists():
            self._err(f'Folder not found: {self.root}')
            return 0

        paths = self._discover()
        if self.cancel.is_set(): return 0

        self._log('info', f'📁 {len(paths)} items found, stat-ing…')
        records = self._stat_batch(paths)
        if self.cancel.is_set(): return 0

        self._log('info', f'✓ {len(records)} files indexed')
        self._dbg(f'[SCAN] Complete  files={len(records)}')

        # Optional magic detection (adds 5-10% overhead but improves scoring)
        if records:
            self._detect_magic_batch(records)

        # Hash
        if records and not self.cancel.is_set():
            self._hash_all(records)

        # Cache keep scores
        for f in records:
            f.keep_score = _total_keep_score(f)

        self.files = records
        return len(records)

    def find_duplicates(self) -> List[DupGroup]:
        """Full detection pipeline → sorted list of DupGroups."""
        if len(self.files) < 2:
            self._log('info', '⚠️  Need 2+ files to compare')
            return []

        self._log('info', f'🔍 Finding duplicates in {len(self.files)} files…')
        self._dbg(f'[FIND] Starting  files={len(self.files)}')

        groups: List[DupGroup] = []

        # ── a. Hard-link detection ────────────────────────────────────────
        hl_groups = self._find_hardlinks()
        if hl_groups:
            self._log('info', f'🔗 {len(hl_groups)} hard-link group(s)')
            self._dbg(f'[FIND] Hard links: {len(hl_groups)} groups')
            groups.extend(hl_groups)

        hl_paths = {str(f.path) for g in hl_groups for f in g.files}
        remaining = [f for f in self.files if str(f.path) not in hl_paths]

        # ── b. Size grouping ──────────────────────────────────────────────
        self._log('info', f'📐 Size-grouping {len(remaining):,} files…')
        self._progress('scan_progress', current=0, total=100, percent=10,
                       file='', status='📐 Size grouping…')
        self._dbg(f'[FIND] Size grouping {len(remaining)} files')
        size_groups = self._group_by_size(remaining)
        candidates  = [f for grp in size_groups.values() for f in grp]
        skipped_unique = len(remaining) - len(candidates)
        n_size_groups = len(size_groups)
        self._log('info', f'📐 {n_size_groups:,} size groups, {len(candidates):,} candidates'
                          f' ({skipped_unique:,} unique-size files skipped)')
        self._dbg(f'[FIND] Size-grouped: {len(candidates)} candidates'
                  f'  {skipped_unique} unique-size files skipped'
                  f'  groups={n_size_groups}')

        if self.cancel.is_set(): return groups

        # ── c. Exact duplicates via hash ──────────────────────────────────
        self._log('info', f'🔑 Hash-based exact duplicate detection…')
        self._progress('scan_progress', current=0, total=100, percent=30,
                       file='', status='🔑 Finding exact duplicates…')
        exact_groups, still_remaining = self._find_exact(size_groups)
        exact_files = sum(len(g.files) for g in exact_groups)
        self._log('info', f'✓ {len(exact_groups)} exact-dup group(s)'
                          f' ({exact_files} files)')
        self._dbg(f'[FIND] Exact groups={len(exact_groups)}'
                  f'  exact_files={exact_files}'
                  f'  remaining_for_near={len(still_remaining)}')
        groups.extend(exact_groups)

        if self.cancel.is_set(): return groups

        # ── d. Near-duplicates ────────────────────────────────────────────
        near_groups = self._find_near(still_remaining)
        self._log('info', f'✓ {len(near_groups)} near-duplicate group(s)')
        self._dbg(f'[FIND] Near groups={len(near_groups)}')

        # ── e. GPU / NumPy name-similarity pass (optional) ────────────────
        if self.settings.use_gpu and len(still_remaining) > 50:
            gpu_extra = self._gpu_name_pass(still_remaining,
                                            {str(f.path) for g in groups
                                             for f in g.files})
            self._dbg(f'[FIND] GPU/NP pass extra={len(gpu_extra)}')
            near_groups.extend(gpu_extra)

        # ── f. Transitive cluster merge ───────────────────────────────────
        if near_groups:
            self._log('info', f'🔗 Merging {len(near_groups)} near-dup pairs into clusters…')
            self._progress('scan_progress', current=0, total=100, percent=90,
                           file='', status='🔗 Merging clusters…')
        merged = self._merge_clusters(near_groups)
        near_files = sum(len(g.files) for g in merged)
        self._dbg(f'[FIND] After transitive merge: {len(merged)} near groups'
                  f'  ({near_files} files)')
        if merged:
            self._log('info', f'✓ {len(merged)} near-dup group(s) ({near_files} files)')
        groups.extend(merged)

        # ── Paranoid byte-by-byte verification ────────────────────────────
        if self.settings.paranoid_mode:
            groups = self._paranoid_verify(groups)

        # Sort: exact first, then by score desc
        groups.sort(key=lambda g: (-g.is_exact, -g.score))
        total_files = sum(len(g.files) for g in groups)
        self._log('info', f'✓ {len(groups)} duplicate groups  ({total_files} files)')
        self._dbg(f'[FIND] Done  groups={len(groups)}  files_involved={total_files}')
        return groups

    def _find_hardlinks(self) -> List[DupGroup]:
        inode_map: Dict[Tuple, List[FileRecord]] = defaultdict(list)
        for f in self.files:
            if f.inode > 0:
                inode_map[(f.device, f.inode)].append(f)
        return [DupGroup(files=grp, score=100, group_type='hardlink',
                         components={'inode': 100})
                for grp in inode_map.values() if len(grp) > 1]

    def _group_by_size(self, files: List[FileRecord]) -> Dict[int, List[FileRecord]]:
        d: Dict[int, List[FileRecord]] = defaultdict(list)
        for f in files: d[f.size].append(f)
        return {sz: grp for sz, grp in d.items() if len(grp) > 1}

    def _find_exact(self, size_groups: Dict[int, List[FileRecord]]):
        """Group by full hash within each size group."""
        exact: List[DupGroup] = []
        hash_map: Dict[str, List[FileRecord]] = defaultdict(list)
        for f in (f for grp in size_groups.values() for f in grp):
            if f.hash:
                hash_map[f.hash].append(f)
        exact_paths: Set[str] = set()
        for h, grp in hash_map.items():
            if len(grp) > 1:
                exact.append(DupGroup(
                    files=grp, score=100, group_type='exact',
                    components={'hash': 100, 'size': 35, 'name': 0,
                                'ext': 0, 'magic': 0, 'dir': 0, 'time': 0}))
                exact_paths.update(str(f.path) for f in grp)
        still = [f for grp in size_groups.values()
                 for f in grp if str(f.path) not in exact_paths]
        return exact, still

    def _find_near(self, files: List[FileRecord]) -> List[DupGroup]:
        # ── modular: near-duplicate comparison stage ──────────────────────────
        """
        Parallel pairwise near-duplicate scoring.

        v5.6 improvements:
        • Smart pre-filter: same size + different quick_hash → score is always 0;
          skip these pairs entirely before touching any worker thread.
        • Per-size-group cap (PER_GROUP_PAIR_CAP) prevents single large size
          groups from dominating the pair budget.
        • Batched ThreadPoolExecutor: submits NEAR_BATCH_SIZE pairs per future
          rather than one future per pair — reduces future overhead 200x.
        • Correct future→pair mapping: uses {future: batch} dict so
          as_completed() ordering never corrupts pair lookup.
        • No ProcessPoolExecutor: thread pool avoids process spawn delay (5-15s)
          and pickle round-trip overhead; GIL is released during file I/O.
        """
        if len(files) < 2: return []

        # ── Step 1: Build candidate pairs with smart pre-filtering ────────────
        size_groups = self._group_by_size(files)
        total_raw   = 0
        skipped_qh  = 0
        skipped_cap = 0
        pairs: List[Tuple[FileRecord, FileRecord]] = []

        self._log('info', f'🔍 Near-dup: pre-filtering {len(files):,} files into pairs…')
        self._dbg(f'[FIND] Near-dup pre-filter start  files={len(files)}  size_groups={len(size_groups)}')

        for sz, grp in size_groups.items():
            group_pairs: List[Tuple[FileRecord, FileRecord]] = []
            for i, f1 in enumerate(grp):
                for f2 in grp[i+1:]:
                    total_raw += 1
                    # KEY OPTIMISATION: same size + both have quick_hash + differ
                    # → _calculate_dup_score will return 0 anyway → skip now
                    if (f1.quick_hash and f2.quick_hash
                            and f1.quick_hash != f2.quick_hash):
                        skipped_qh += 1
                        continue
                    group_pairs.append((f1, f2))

            # Per-group cap prevents one huge size bucket from dominating
            if len(group_pairs) > PER_GROUP_PAIR_CAP:
                skipped_cap += len(group_pairs) - PER_GROUP_PAIR_CAP
                group_pairs  = group_pairs[:PER_GROUP_PAIR_CAP]

            pairs.extend(group_pairs)

        self._dbg(f'[FIND] Raw pairs={total_raw:,}  qh-filtered={skipped_qh:,}'
                  f'  cap-filtered={skipped_cap:,}  remaining={len(pairs):,}')

        if not pairs:
            self._log('info',
                f'✓ Near-dup: 0 pairs after smart pre-filter'
                f' (eliminated {total_raw:,} via quick-hash + caps)')
            return []

        # ── Global safety cap ─────────────────────────────────────────────────
        if len(pairs) > NEAR_DUP_MAX_PAIRS:
            self._dbg(f'[FIND] Global cap: {len(pairs):,} → {NEAR_DUP_MAX_PAIRS:,}')
            pairs = pairs[:NEAR_DUP_MAX_PAIRS]

        total  = len(pairs)
        min_sc = self.settings.min_score
        nw     = self.settings.num_workers
        self._log('info',
            f'📊 Comparing {total:,} pairs ({nw} workers, '
            f'{total_raw - total:,} pre-filtered)…')
        self._dbg(f'[COMPARE] pairs={total}  workers={nw}  min_score={min_sc}'
                  f'  executor=Thread  batch={NEAR_BATCH_SIZE}')

        # ── Serialize to dicts once ───────────────────────────────────────────
        def _to_dict(f: FileRecord) -> dict:
            return {
                'hash': f.hash, 'partial_hash': f.partial_hash,
                'quick_hash': f.quick_hash, 'size': f.size,
                'name': f.name, 'ext': f.ext, 'magic_type': f.magic_type,
                'mtime': f.mtime, 'parent': str(f.path.parent),
            }

        # ── Batched ThreadPoolExecutor (no ProcessPool spawn overhead) ────────
        results: List[DupGroup] = []
        done  = 0
        hits  = 0
        last_report_t = time.monotonic()
        report_interval = max(1, total // 100)

        with ThreadPoolExecutor(max_workers=nw) as pool:
            # Build {future: batch_of_original_pairs} — correct future→pair map
            batch_futs: dict = {}
            for i in range(0, len(pairs), NEAR_BATCH_SIZE):
                batch       = pairs[i:i + NEAR_BATCH_SIZE]
                batch_dicts = [(_to_dict(f1), _to_dict(f2)) for f1, f2 in batch]
                fut         = pool.submit(_score_batch, batch_dicts, min_sc)
                batch_futs[fut] = batch

            for fut in as_completed(batch_futs):
                if self.cancel.is_set(): break
                batch = batch_futs[fut]
                try:
                    scored = fut.result()   # [(local_idx, sc, comp), ...]
                except Exception as exc:
                    self._dbg(f'[COMPARE] batch error: {exc}')
                    done += len(batch)
                    continue

                for local_idx, sc, comp in scored:
                    if local_idx < len(batch):
                        f1, f2 = batch[local_idx]
                        results.append(DupGroup(
                            files=[f1, f2], score=sc,
                            group_type='near', components=comp))
                        hits += 1

                done += len(batch)
                now = time.monotonic()
                if (done >= done - len(batch) + report_interval
                        or (now - last_report_t) >= 0.8
                        or done == total):
                    last_report_t = now
                    pct = int(done / max(total, 1) * 100)
                    self._progress('match_progress',
                                   current=done, total=total, percent=pct)
                    self._dbg(f'[COMPARE] {done:,}/{total:,} ({pct}%)'
                              f'  hits≥{min_sc}: {hits}')

        self._dbg(f'[COMPARE] Done  hits={hits}  elapsed_pairs={done}')
        return results

    def _merge_clusters(self, groups: List[DupGroup]) -> List[DupGroup]:
        """Transitive merge using UnionFind: A~B + B~C → {A,B,C}."""
        if not groups: return []

        uf = UnionFind()
        file_map: Dict[str, FileRecord] = {}

        for grp in groups:
            keys = [str(f.path) for f in grp.files]
            for f in grp.files:
                file_map[str(f.path)] = f
            for k in keys[1:]:
                uf.union(keys[0], k)

        # Collect clusters
        cluster_map: Dict[str, Set[str]] = defaultdict(set)
        for key in file_map:
            cluster_map[uf.find(key)].add(key)

        result: List[DupGroup] = []
        used: Set[str] = set()

        for grp in sorted(groups, key=lambda x: -x.score):
            rep = str(grp.files[0].path)
            root = uf.find(rep)
            if root in used: continue
            used.add(root)
            keys = cluster_map[root]
            cluster_files = [file_map[k] for k in keys if k in file_map]
            if len(cluster_files) >= 2:
                result.append(DupGroup(
                    files=cluster_files,
                    score=grp.score,
                    group_type=grp.group_type,
                    components=grp.components,
                ))

        return result

    def _gpu_name_pass(self, files: List[FileRecord],
                       exclude_paths: Set[str]) -> List[DupGroup]:
        """GPU / NumPy vectorised filename bigram cosine similarity."""
        if not HAS_NUMPY and not HAS_CUPY: return []
        eligible = [f for f in files if str(f.path) not in exclude_paths]
        if len(eligible) < 2: return []

        backend = 'GPU(CuPy)' if (HAS_CUPY and self.settings.use_gpu) else 'NumPy'
        self._log('info', f'🖥️  {backend} name-similarity on {len(eligible)} files…')
        names = [f.name.lower() for f in eligible]
        vocab: Dict[str, int] = {}
        for nm in names:
            for i in range(len(nm)-1):
                gram = nm[i:i+2]
                if gram not in vocab: vocab[gram] = len(vocab)
        V = len(vocab)
        if V == 0: return []
        xp = cp if (HAS_CUPY and self.settings.use_gpu) else np
        mat = xp.zeros((len(eligible), V), dtype=xp.float32)
        for i, nm in enumerate(names):
            for j in range(len(nm)-1):
                gram = nm[j:j+2]
                if gram in vocab: mat[i, vocab[gram]] += 1.0
        norms = xp.linalg.norm(mat, axis=1, keepdims=True)
        norms = xp.where(norms == 0, 1.0, norms)
        mat  /= norms
        sim   = xp.dot(mat, mat.T)
        rows, cols = xp.where(sim > 0.85)
        if HAS_CUPY and self.settings.use_gpu:
            rows, cols = cp.asnumpy(rows), cp.asnumpy(cols)
        else:
            rows = rows.__array__(); cols = cols.__array__()
        groups: List[DupGroup] = []
        seen: Set[Tuple] = set()
        for r, c in zip(rows, cols):
            r, c = int(r), int(c)
            if r >= c: continue
            if (r, c) in seen: continue
            seen.add((r, c))
            f1, f2 = eligible[r], eligible[c]
            sc, comp = _calculate_dup_score(({
                'hash': f1.hash, 'partial_hash': f1.partial_hash,
                'quick_hash': f1.quick_hash, 'size': f1.size,
                'name': f1.name, 'ext': f1.ext, 'magic_type': f1.magic_type,
                'mtime': f1.mtime, 'parent': str(f1.path.parent),
            }, {
                'hash': f2.hash, 'partial_hash': f2.partial_hash,
                'quick_hash': f2.quick_hash, 'size': f2.size,
                'name': f2.name, 'ext': f2.ext, 'magic_type': f2.magic_type,
                'mtime': f2.mtime, 'parent': str(f2.path.parent),
            }))
            if sc >= self.settings.min_score:
                groups.append(DupGroup(files=[f1, f2], score=sc,
                                       group_type='near', components=comp))
        self._log('info', f'✓ {backend} → {len(groups)} extra groups')
        return groups

    def _paranoid_verify(self, groups: List[DupGroup]) -> List[DupGroup]:
        """Byte-by-byte verification of exact-match groups (paranoid mode)."""
        self._log('info', '🔬 Paranoid mode: byte-by-byte verification…')
        verified: List[DupGroup] = []
        for grp in groups:
            if not grp.is_exact or len(grp.files) < 2:
                verified.append(grp); continue
            # Verify all pairs against the first file
            ref = grp.files[0]
            confirmed = [ref]
            for f in grp.files[1:]:
                if _byte_compare(ref.path, f.path):
                    confirmed.append(f)
                else:
                    self._dbg(f'[VERIFY] byte-mismatch: {ref.name} vs {f.name}')
            if len(confirmed) > 1:
                grp.files = confirmed
                verified.append(grp)
        self._log('info', f'✓ Paranoid verify: {len(verified)} groups confirmed')
        return verified

    # ── Smart auto-selection ───────────────────────────────────────────────

    def smart_select(self, groups: List[DupGroup]) -> None:
        """
        Annotate each DupGroup with suggestions: {file_idx: 'KEEP'|'DELETE'}.

        Strategy:
        - Exact / hardlink groups:
            Keep the file with lowest keep_score (inverse: highest quality).
            Tiebreak: oldest ctime (most likely original).
            All others → DELETE.
        - Near-duplicate groups:
            Only suggest DELETE when quality gap >= settings.delete_gap.
            Keeps ambiguous groups unmarked to let the user decide.
        """
        total_marked = 0
        self._dbg(f'[SELECT] Smart auto-select starting  groups={len(groups)}')

        for gi, grp in enumerate(groups):
            files = grp.files
            if len(files) < 2:
                grp.suggestions = {0: 'KEEP'}
                continue

            # (file_idx, keep_score, ctime)
            scored = [(fi, f.keep_score, f.ctime) for fi, f in enumerate(files)]

            if grp.is_exact:
                # Best score wins; tie → oldest ctime
                scored_sorted = sorted(scored, key=lambda x: (-x[1], x[2]))
                keeper_idx    = scored_sorted[0][0]
                to_delete     = {s[0] for s in scored_sorted[1:]}
            else:
                # Near-dup: only suggest if gap is clear
                scored_sorted = sorted(scored, key=lambda x: (-x[1], x[2]))
                keeper_idx    = scored_sorted[0][0]
                best          = scored_sorted[0][1]
                to_delete     = {s[0] for s in scored_sorted[1:]
                                 if best - s[1] >= self.settings.delete_gap}

            grp.suggestions = {
                fi: ('DELETE' if fi in to_delete else 'KEEP')
                for fi in range(len(files))
            }
            total_marked += len(to_delete)
            self._dbg(f'[SELECT] G{gi+1}: keep={keeper_idx}'
                      f'  delete_count={len(to_delete)}'
                      f'  total={len(files)}'
                      f'  type={grp.group_type}')

        self._dbg(f'[SELECT] Done  files_marked={total_marked}')

# ═════════════════════════════════════════════════════════════════════════════
#  SafeDeleter  — cross-platform Recycle Bin / Trash with persistent log
# ═════════════════════════════════════════════════════════════════════════════

class SafeDeleter:
    """Move files to Recycle Bin / Trash; log every action persistently."""

    @staticmethod
    def to_trash(filepath) -> Tuple[bool, str]:
        try:
            fp = Path(filepath)
            if not fp.exists():
                return False, 'File not found'
            if HAS_SEND2TRASH:
                _s2t.send2trash(str(fp))
                SafeDeleter._log_deletion(fp, True)
                return True, 'Moved to Trash / Recycle Bin'
            if sys.platform == 'win32':
                import ctypes
                class _SHOp(ctypes.Structure):
                    _fields_ = [('hwnd', ctypes.c_void_p),
                                 ('wFunc', ctypes.c_uint),
                                 ('pFrom', ctypes.c_wchar_p),
                                 ('pTo', ctypes.c_wchar_p),
                                 ('fFlags', ctypes.c_ushort),
                                 ('fAnyAborted', ctypes.c_bool),
                                 ('hMappings', ctypes.c_void_p),
                                 ('lpTitle', ctypes.c_wchar_p)]
                op = _SHOp(); op.wFunc = 3; op.pFrom = str(fp) + '\0'
                op.fFlags = 0x40   # FOF_ALLOWUNDO
                rc = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
                if rc == 0:
                    SafeDeleter._log_deletion(fp, True)
                    return True, 'Moved to Recycle Bin'
                return False, f'SHFileOperation error code {rc}'
            return False, 'send2trash not available — run: pip install send2trash'
        except Exception as exc:
            SafeDeleter._log_deletion(filepath, False, str(exc))
            return False, str(exc)

    @staticmethod
    def _log_deletion(filepath, success: bool, error: str = '') -> None:
        entry = {
            'ts':      datetime.datetime.now().isoformat(),
            'path':    str(filepath),
            'success': success,
            'error':   error,
        }
        try:
            log = SafeDeleter.load_log()
            log.append(entry)
            with open(DELETION_LOG_PATH, 'w', encoding='utf-8') as fh:
                json.dump(log, fh, indent=2)
        except Exception:
            pass

    @staticmethod
    def load_log() -> List[dict]:
        try:
            if DELETION_LOG_PATH.exists():
                with open(DELETION_LOG_PATH, encoding='utf-8') as fh:
                    return json.load(fh)
        except Exception:
            pass
        return []


# ═════════════════════════════════════════════════════════════════════════════
#  SessionManager  — save / load scan results so re-scanning is optional
# ═════════════════════════════════════════════════════════════════════════════

class SessionManager:
    """Serialise/deserialise scan sessions to JSON for fast reload."""

    @staticmethod
    def save(groups: List[DupGroup], folder: str, scan_settings: ScanSettings,
             filepath: str) -> bool:
        try:
            def _ser_file(f: FileRecord) -> dict:
                return {
                    'path': str(f.path), 'size': f.size,
                    'mtime': f.mtime, 'ctime': f.ctime,
                    'inode': f.inode, 'device': f.device,
                    'ext': f.ext, 'name': f.name,
                    'hash': f.hash, 'partial_hash': f.partial_hash,
                    'quick_hash': f.quick_hash, 'magic_type': f.magic_type,
                    'keep_score': f.keep_score,
                }
            payload = {
                'version':  VERSION,
                'folder':   folder,
                'saved_at': datetime.datetime.now().isoformat(),
                'min_score': scan_settings.min_score,
                'groups': [
                    {
                        'score':       g.score,
                        'group_type':  g.group_type,
                        'components':  g.components,
                        'suggestions': {str(k): v for k,v in g.suggestions.items()},
                        'files':       [_ser_file(f) for f in g.files],
                    }
                    for g in groups
                ],
            }
            with open(filepath, 'w', encoding='utf-8') as fh:
                json.dump(payload, fh, indent=2)
            return True
        except Exception:
            return False

    @staticmethod
    def load(filepath: str) -> Tuple[Optional[List[DupGroup]], Optional[str]]:
        """Returns (groups, folder) or (None, None) on failure."""
        try:
            with open(filepath, encoding='utf-8') as fh:
                payload = json.load(fh)
            if payload.get('version') != VERSION:
                return None, None
            folder = payload.get('folder', '')
            groups: List[DupGroup] = []
            for gd in payload.get('groups', []):
                files = []
                for fd in gd.get('files', []):
                    files.append(FileRecord(
                        path=Path(fd['path']), size=fd['size'],
                        mtime=fd['mtime'],    ctime=fd['ctime'],
                        inode=fd['inode'],    device=fd['device'],
                        ext=fd['ext'],        name=fd['name'],
                        hash=fd.get('hash'),
                        partial_hash=fd.get('partial_hash'),
                        quick_hash=fd.get('quick_hash'),
                        magic_type=fd.get('magic_type'),
                        keep_score=fd.get('keep_score', 100),
                    ))
                sugg = {int(k): v for k, v in gd.get('suggestions', {}).items()}
                groups.append(DupGroup(
                    files=files, score=gd['score'],
                    group_type=gd['group_type'],
                    components=gd.get('components', {}),
                    suggestions=sugg,
                ))
            return groups, folder
        except Exception:
            return None, None


# ═════════════════════════════════════════════════════════════════════════════
#  DuplicateFinderApp  v5.0 — Ground-up redesign
# ═════════════════════════════════════════════════════════════════════════════

# Colour palettes
LIGHT_PALETTE = {
    'bg':           '#f0f0f0', 'fg':        '#1a1a1a',
    'header_bg':    '#1e5631', 'header_fg': '#ffffff',
    'toolbar_bg':   '#2d8659', 'toolbar_fg':'#ffffff',
    'accent1':      '#0066cc', 'accent2':   '#52b788',
    'accent3':      '#e8f5e9', 'success':   '#28a745',
    'warning':      '#ffc107', 'danger':    '#dc3545',
    'info':         '#17a2b8', 'panel_bg':  '#ffffff',
    'border':       '#cccccc', 'select_bg': '#cce5ff',
    'tree_bg':      '#ffffff', 'tree_fg':   '#1a1a1a',
    'tree_sel':     '#0066cc', 'tree_sel_fg':'#ffffff',
    'card_bg':      '#f8f9fa', 'card_border':'#dee2e6',
    'keep_fg':      '#28a745', 'del_fg':    '#dc3545',
    'exact_bg':     '#d4edda', 'near_bg':   '#cce5ff',
    'hard_bg':      '#fff3cd',
    'debug_bg':     '#0d1117', 'debug_fg':  '#c9d1d9',
    'term_bg':      '#161b22',
    'status_bar_bg':'#2d2d2d', 'status_bar_fg':'#ffffff',
    'amber':        '#ffc107',
}
DARK_PALETTE = {
    'bg':           '#1e1e2e', 'fg':        '#cdd6f4',
    'header_bg':    '#11111b', 'header_fg': '#cba6f7',
    'toolbar_bg':   '#181825', 'toolbar_fg':'#cdd6f4',
    'accent1':      '#89b4fa', 'accent2':   '#a6e3a1',
    'accent3':      '#313244', 'success':   '#a6e3a1',
    'warning':      '#f9e2af', 'danger':    '#f38ba8',
    'info':         '#89dceb', 'panel_bg':  '#1e1e2e',
    'border':       '#45475a', 'select_bg': '#313244',
    'tree_bg':      '#1e1e2e', 'tree_fg':   '#cdd6f4',
    'tree_sel':     '#89b4fa', 'tree_sel_fg':'#1e1e2e',
    'card_bg':      '#181825', 'card_border':'#45475a',
    'keep_fg':      '#a6e3a1', 'del_fg':    '#f38ba8',
    'exact_bg':     '#1e3a2f', 'near_bg':   '#1e2a3a',
    'hard_bg':      '#3a2e1e',
    'debug_bg':     '#0a0a0f', 'debug_fg':  '#c9d1d9',
    'term_bg':      '#0d0d18',
    'status_bar_bg':'#11111b', 'status_bar_fg':'#cdd6f4',
    'amber':        '#f9e2af',
}


class DuplicateFinderApp:
    """
    v5.0 Ground-Up Redesign.

    Layout:
    ┌─ HEADER (logo + folder + chip strip) ──────────────────────────────────┐
    ├─ TOOLBAR (scan/stop/select/delete/export/session) ─────────────────────┤
    ├─ LEFT PANEL (35%) ──────────────────┬─ RIGHT PANEL (65%) ──────────────┤
    │  Filter / Search bar                │  Detail / File Cards panel       │
    │  ttk.Treeview (all groups)          │  (scrollable styled cards)       │
    ├─────────────────────────────────────┴──────────────────────────────────┤
    │  Bottom Tabs: Full Report | Activity Log | Settings | Deletion History  │
    ├─ STATUS BAR ────────────────────────────────────────────────────────────┤
    └─ DEBUG TERMINAL (collapsible, 4 tabs) ──────────────────────────────────┘

    Thread safety:
      - All UI mutations strictly on the main thread.
      - Worker communicates via self._pq (queue.Queue).
      - _start_progress_monitor() drains queue every 50 ms (scanning) / 100 ms (idle).
    """

    def __init__(self, root_tk: tk.Tk):
        self.root = root_tk
        self.root.title(f'Duplicate File Finder v{VERSION}')
        self.root.geometry('1400x920')
        self.root.minsize(1100, 700)

        # State
        self.settings         = ScanSettings()
        self.groups:          List[DupGroup] = []
        self.current_group:   int            = -1
        self.is_scanning:     bool           = False
        self.scan_thread:     Optional[threading.Thread] = None
        self._cancel_ev:      threading.Event = threading.Event()
        self._pq:             queue.Queue    = queue.Queue(maxsize=20000)
        self._engine:         Optional[ScanEngine] = None
        self._spinner_idx:    int            = 0
        self._term_visible:   bool           = True
        self._log_file_queue: queue.Queue    = queue.Queue()  # async log I/O
        self._report_dirty:   bool           = False         # lazy Full Report render

        # Tk variables (bound to settings widgets)
        self._var_subdirs     = tk.BooleanVar(value=True)
        self._var_hash        = tk.BooleanVar(value=True)
        self._var_xxhash      = tk.BooleanVar(value=True)
        self._var_gpu         = tk.BooleanVar(value=False)
        self._var_paranoid    = tk.BooleanVar(value=False)
        self._var_workers     = tk.IntVar(value=min(CPU_COUNT * 2, 16))
        self._var_min_score   = tk.IntVar(value=70)
        self._var_min_size    = tk.IntVar(value=1)
        self._var_max_size    = tk.IntVar(value=0)
        self._var_dark_mode   = tk.BooleanVar(value=False)
        self._var_auto_select = tk.BooleanVar(value=True)
        self._var_delete_gap  = tk.IntVar(value=15)
        self._var_exclusions  = tk.StringVar(value='')
        self._var_filter_text = tk.StringVar()
        self._var_filter_type = tk.StringVar(value='All')

        self._setup_theme(dark=False)
        self._build_ui()
        self._bind_shortcuts()
        self._start_progress_monitor()
        self._animate_spinner()
        self._start_log_writer()   # async log-file I/O thread

    # ── Theme ─────────────────────────────────────────────────────────────

    def _setup_theme(self, dark: bool = False) -> None:
        self.C = DARK_PALETTE.copy() if dark else LIGHT_PALETTE.copy()
        C = self.C

        style = ttk.Style(self.root)
        style.theme_use('clam')

        style.configure('TFrame',           background=C['bg'],
                        borderwidth=0, relief='flat')
        style.configure('Card.TFrame',      background=C['card_bg'],
                        borderwidth=1, relief='solid')
        style.configure('TLabel',           background=C['bg'], foreground=C['fg'])
        style.configure('Header.TLabel',    background=C['header_bg'],
                        foreground=C['header_fg'], font=('Arial', 15, 'bold'))
        style.configure('Status.TLabel',    background=C['status_bar_bg'],
                        foreground=C['status_bar_fg'], font=('Courier', 9))

        for name, bg in [('Green', C['success']), ('Blue', C['accent1']),
                          ('Red', C['danger']),    ('Amber', C['warning']),
                          ('Gray', C['border'])]:
            style.configure(f'{name}.TButton',
                            background=bg, foreground=C['header_fg'],
                            borderwidth=2, relief='raised', padding=(8, 4))
            style.map(f'{name}.TButton',
                      background=[('active', C['accent2']),
                                  ('pressed', C['header_bg'])])

        style.configure('Treeview',
                        background=C['tree_bg'], foreground=C['tree_fg'],
                        fieldbackground=C['tree_bg'], rowheight=22,
                        borderwidth=1, relief='solid')
        style.configure('Treeview.Heading',
                        background=C['toolbar_bg'], foreground=C['toolbar_fg'],
                        font=('Arial', 9, 'bold'))
        style.map('Treeview',
                  background=[('selected', C['tree_sel'])],
                  foreground=[('selected', C['tree_sel_fg'])])

        style.configure('TNotebook',        background=C['bg'])
        style.configure('TNotebook.Tab',    background=C['bg'],
                        foreground=C['fg'], padding=(10, 4))
        style.map('TNotebook.Tab',
                  background=[('selected', C['accent1'])],
                  foreground=[('selected', '#ffffff')])

        style.configure('TProgressbar',
                        troughcolor=C['bg'], background=C['accent2'])
        style.configure('TScrollbar',       background=C['bg'])
        style.configure('TSeparator',       background=C['border'])
        style.configure('TEntry',           fieldbackground=C['panel_bg'],
                        foreground=C['fg'])
        style.configure('TSpinbox',         fieldbackground=C['panel_bg'],
                        foreground=C['fg'])
        style.configure('TCheckbutton',     background=C['bg'],
                        foreground=C['fg'])
        style.configure('TCombobox',        fieldbackground=C['panel_bg'],
                        foreground=C['fg'])

        self.root.configure(bg=C['bg'])

    # ── Build UI ──────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        C = self.C

        # ── Debug terminal (bottom, packed first so it gets BOTTOM) ───────
        self._build_debug_terminal()

        # ── Status bar ────────────────────────────────────────────────────
        self._build_status_bar()

        # ── Header ────────────────────────────────────────────────────────
        self._build_header()

        # ── Toolbar ───────────────────────────────────────────────────────
        self._build_toolbar()

        # ── Main paned area ───────────────────────────────────────────────
        self._main_pane = tk.PanedWindow(
            self.root, orient=tk.HORIZONTAL,
            sashwidth=5, sashrelief='raised',
            bg=C['border'])
        self._main_pane.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Left: groups treeview
        left_outer = tk.Frame(self._main_pane, bg=C['bg'],
                              borderwidth=1, relief='solid')
        self._main_pane.add(left_outer, minsize=280, width=380)
        self._build_groups_panel(left_outer)

        # Right: tab notebook
        right_outer = tk.Frame(self._main_pane, bg=C['bg'],
                               borderwidth=1, relief='solid')
        self._main_pane.add(right_outer, minsize=400)
        self._build_right_notebook(right_outer)

    # ── Header ────────────────────────────────────────────────────────────

    def _build_header(self) -> None:
        C = self.C
        hdr = tk.Frame(self.root, bg=C['header_bg'], height=65)
        hdr.pack(fill=tk.X, side=tk.TOP)
        hdr.pack_propagate(False)

        # Spinner + title
        self._header_spinner = tk.Label(
            hdr, text=SPINNER_FRAMES[0],
            font=('Arial', 18, 'bold'), bg=C['header_bg'], fg=C['accent2'])
        self._header_spinner.pack(side=tk.LEFT, padx=8, pady=10)

        tk.Label(hdr, text=f'🔍 DUPLICATE FILE FINDER  v{VERSION}',
                 font=('Arial', 15, 'bold'),
                 bg=C['header_bg'], fg=C['header_fg']).pack(
                     side=tk.LEFT, padx=5)

        # Folder selector
        ff = tk.Frame(hdr, bg=C['header_bg'])
        ff.pack(side=tk.LEFT, padx=20, pady=8)
        tk.Label(ff, text='Folder:', bg=C['header_bg'],
                 fg=C['header_fg'], font=('Arial', 9)).pack(side=tk.LEFT)
        self._folder_lbl = tk.Label(
            ff, text=str(os.getcwd()),
            bg=C['accent2'], fg='#1a1a1a',
            font=('Courier', 9), relief='solid', borderwidth=1,
            width=55, anchor='w', padx=4)
        self._folder_lbl.pack(side=tk.LEFT, padx=4)
        ttk.Button(ff, text='📁 Change',
                   command=self._change_folder).pack(side=tk.LEFT, padx=4)

        # Right-side status chips
        def _chip(text, bg):
            return tk.Label(hdr, text=text, bg=bg, fg='#ffffff',
                            font=('Arial', 8, 'bold'), padx=5, pady=2)

        _chip(f'CPUs:{CPU_COUNT}', C['toolbar_bg']).pack(side=tk.RIGHT, padx=4, pady=15)
        _chip('xxh✓' if HAS_XXHASH else 'xxh✗',
              C['success'] if HAS_XXHASH else C['danger']).pack(side=tk.RIGHT, padx=2)
        _chip('PIL✓' if HAS_PIL else 'PIL✗',
              C['accent1'] if HAS_PIL else C['border']).pack(side=tk.RIGHT, padx=2)
        _chip('s2t✓' if HAS_SEND2TRASH else 's2t✗',
              C['success'] if HAS_SEND2TRASH else C['danger']).pack(side=tk.RIGHT, padx=2)
        _chip('GPU✓' if HAS_CUPY else ('NP✓' if HAS_NUMPY else 'GPU✗'),
              C['accent1'] if (HAS_CUPY or HAS_NUMPY) else C['border']).pack(
                  side=tk.RIGHT, padx=2)

        # Dark mode toggle
        self._dm_btn = tk.Button(
            hdr, text='🌙', font=('Arial', 12),
            bg=C['header_bg'], fg=C['accent2'],
            relief='flat', cursor='hand2',
            command=self._toggle_dark_mode)
        self._dm_btn.pack(side=tk.RIGHT, padx=8)

    # ── Toolbar ───────────────────────────────────────────────────────────

    def _build_toolbar(self) -> None:
        C = self.C
        tb = tk.Frame(self.root, bg=C['toolbar_bg'], height=44)
        tb.pack(fill=tk.X, side=tk.TOP)
        tb.pack_propagate(False)

        self._scan_btn = ttk.Button(tb, text='▶ SCAN',
                                    command=self._start_scan,
                                    style='Green.TButton')
        self._scan_btn.pack(side=tk.LEFT, padx=6, pady=6)

        self._stop_btn = ttk.Button(tb, text='⏹ STOP',
                                    command=self._stop_scan,
                                    state=tk.DISABLED,
                                    style='Red.TButton')
        self._stop_btn.pack(side=tk.LEFT, padx=4, pady=6)

        ttk.Separator(tb, orient='vertical').pack(side=tk.LEFT, fill=tk.Y,
                                                   padx=8, pady=4)

        self._auto_sel_btn = ttk.Button(tb, text='🎯 Auto-Select',
                                         command=self._auto_select,
                                         state=tk.DISABLED,
                                         style='Amber.TButton')
        self._auto_sel_btn.pack(side=tk.LEFT, padx=4, pady=6)

        self._clear_sel_btn = ttk.Button(tb, text='✕ Clear Sel',
                                          command=self._clear_selection,
                                          state=tk.DISABLED,
                                          style='Gray.TButton')
        self._clear_sel_btn.pack(side=tk.LEFT, padx=4, pady=6)

        self._delete_btn = ttk.Button(tb, text='🗑 DELETE SELECTED',
                                       command=self._delete_selected,
                                       state=tk.DISABLED,
                                       style='Red.TButton')
        self._delete_btn.pack(side=tk.LEFT, padx=4, pady=6)

        ttk.Separator(tb, orient='vertical').pack(side=tk.LEFT, fill=tk.Y,
                                                   padx=8, pady=4)

        self._export_btn = ttk.Button(tb, text='📤 Export',
                                       command=self._show_export_menu,
                                       state=tk.DISABLED,
                                       style='Blue.TButton')
        self._export_btn.pack(side=tk.LEFT, padx=4, pady=6)

        self._save_sess_btn = ttk.Button(tb, text='💾 Save Session',
                                          command=self._save_session,
                                          state=tk.DISABLED,
                                          style='Blue.TButton')
        self._save_sess_btn.pack(side=tk.LEFT, padx=4, pady=6)

        self._load_sess_btn = ttk.Button(tb, text='📂 Load Session',
                                          command=self._load_session,
                                          style='Blue.TButton')
        self._load_sess_btn.pack(side=tk.LEFT, padx=4, pady=6)

        ttk.Separator(tb, orient='vertical').pack(side=tk.LEFT, fill=tk.Y,
                                                   padx=8, pady=4)

        # Workers spinner on toolbar
        tk.Label(tb, text='Workers:', bg=C['toolbar_bg'],
                 fg=C['toolbar_fg'], font=('Arial', 9)).pack(side=tk.LEFT)
        tk.Spinbox(tb, from_=1, to=64, textvariable=self._var_workers,
                   width=4, font=('Arial', 9)).pack(side=tk.LEFT, padx=4)

        # Activity label (right side)
        self._activity_lbl = tk.Label(
            tb, text='Ready to scan',
            bg=C['toolbar_bg'], fg=C['toolbar_fg'],
            font=('Courier', 9, 'bold'), anchor='e')
        self._activity_lbl.pack(side=tk.RIGHT, padx=12, fill=tk.X, expand=True)


    # ── Groups panel (left) ────────────────────────────────────────────────

    def _build_groups_panel(self, parent) -> None:
        C = self.C

        # Filter bar
        fbar = tk.Frame(parent, bg=C['bg'])
        fbar.pack(fill=tk.X, padx=4, pady=(4, 0))

        tk.Label(fbar, text='🔎', bg=C['bg'],
                 fg=C['fg'], font=('Arial', 11)).pack(side=tk.LEFT)
        self._filter_entry = ttk.Entry(fbar, textvariable=self._var_filter_text,
                                       width=18)
        self._filter_entry.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self._var_filter_text.trace_add('write', lambda *_: self._apply_filter())

        self._type_combo = ttk.Combobox(
            fbar, textvariable=self._var_filter_type,
            values=['All', 'Exact', 'Near-Dup', 'Hard-Link'],
            state='readonly', width=10)
        self._type_combo.pack(side=tk.LEFT, padx=2)
        self._type_combo.bind('<<ComboboxSelected>>', lambda _: self._apply_filter())

        # Treeview
        tree_frame = tk.Frame(parent, bg=C['bg'])
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        cols = ('type', 'files', 'score', 'reclaim')
        self._tree = ttk.Treeview(
            tree_frame, columns=cols,
            show='tree headings', selectmode='browse')

        self._tree.heading('#0', text='Group', anchor='w')
        self._tree.heading('type',   text='Type',     anchor='center')
        self._tree.heading('files',  text='Files',    anchor='center')
        self._tree.heading('score',  text='Score',    anchor='center')
        self._tree.heading('reclaim',text='Reclaim',  anchor='e')

        self._tree.column('#0',      width=160, stretch=True)
        self._tree.column('type',    width=70,  anchor='center', stretch=False)
        self._tree.column('files',   width=45,  anchor='center', stretch=False)
        self._tree.column('score',   width=50,  anchor='center', stretch=False)
        self._tree.column('reclaim', width=70,  anchor='e',      stretch=False)

        # Row tags
        self._tree.tag_configure('exact',    background=C['exact_bg'],
                                 foreground=C['tree_fg'])
        self._tree.tag_configure('near',     background=C['near_bg'],
                                 foreground=C['tree_fg'])
        self._tree.tag_configure('hardlink', background=C['hard_bg'],
                                 foreground=C['tree_fg'])
        self._tree.tag_configure('file_row', foreground='#555566',
                                 font=('Courier', 8))

        tree_vsb = ttk.Scrollbar(tree_frame, orient='vertical',
                                  command=self._tree.yview)
        self._tree.configure(yscrollcommand=tree_vsb.set)
        tree_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._tree.bind('<<TreeviewSelect>>', self._on_tree_select)
        self._tree.bind('<Double-Button-1>', self._on_tree_double_click)

        # Summary strip under tree
        self._tree_summary = tk.Label(
            parent, text='No scan yet', bg=C['bg'],
            fg=C['fg'], font=('Arial', 8), anchor='w')
        self._tree_summary.pack(fill=tk.X, padx=6, pady=(0, 4))

    # ── Right notebook ─────────────────────────────────────────────────────

    def _build_right_notebook(self, parent) -> None:
        C = self.C
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True)
        self._right_nb = nb

        def _tab(title):
            f = tk.Frame(nb, bg=C['bg'])
            nb.add(f, text=title)
            return f

        self._build_detail_panel(_tab('📋 Group Detail'))        # idx 0
        self._build_full_report_tab(_tab('📄 Full Report'))      # idx 1
        self._build_log_tab(_tab('📝 Activity Log'))             # idx 2
        self._build_settings_tab(_tab('⚙️  Settings'))          # idx 3
        self._build_deletion_history_tab(_tab('🗃 Del History')) # idx 4

        # Lazy render: only build Full Report when user selects that tab
        nb.bind('<<NotebookTabChanged>>', self._on_right_tab_changed)

    # ── Detail panel (right tab 0) ─────────────────────────────────────────

    def _build_detail_panel(self, parent) -> None:
        C = self.C

        # Group header strip
        hdr = tk.Frame(parent, bg=C['toolbar_bg'])
        hdr.pack(fill=tk.X)

        self._detail_group_lbl = tk.Label(
            hdr, text='Select a group from the left panel',
            bg=C['toolbar_bg'], fg=C['toolbar_fg'],
            font=('Arial', 10, 'bold'), anchor='w', padx=8)
        self._detail_group_lbl.pack(side=tk.LEFT, pady=6, fill=tk.X, expand=True)

        self._detail_score_lbl = tk.Label(
            hdr, text='',
            bg=C['toolbar_bg'], fg=C['warning'],
            font=('Courier', 9, 'bold'), anchor='e', padx=8)
        self._detail_score_lbl.pack(side=tk.RIGHT, pady=6)

        # Scan / compare progress bars
        prog_frame = tk.Frame(parent, bg=C['bg'])
        prog_frame.pack(fill=tk.X, padx=8, pady=4)

        tk.Label(prog_frame, text='Scan:', bg=C['bg'],
                 fg=C['fg'], font=('Arial', 8), width=7).pack(side=tk.LEFT)
        self._scan_bar = ttk.Progressbar(prog_frame, mode='determinate',
                                          maximum=100, length=200)
        self._scan_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._scan_lbl = tk.Label(prog_frame, text='', bg=C['bg'],
                                   fg=C['fg'], font=('Courier', 8), width=20)
        self._scan_lbl.pack(side=tk.LEFT, padx=4)

        prog_frame2 = tk.Frame(parent, bg=C['bg'])
        prog_frame2.pack(fill=tk.X, padx=8, pady=(0, 4))
        tk.Label(prog_frame2, text='Compare:', bg=C['bg'],
                 fg=C['fg'], font=('Arial', 8), width=7).pack(side=tk.LEFT)
        self._match_bar = ttk.Progressbar(prog_frame2, mode='determinate',
                                           maximum=100, length=200)
        self._match_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._match_lbl = tk.Label(prog_frame2, text='', bg=C['bg'],
                                    fg=C['fg'], font=('Courier', 8), width=20)
        self._match_lbl.pack(side=tk.LEFT, padx=4)

        # Scrollable file-cards canvas
        canvas_frame = tk.Frame(parent, bg=C['bg'])
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._cards_canvas = tk.Canvas(canvas_frame, bg=C['bg'],
                                        highlightthickness=0)
        cards_vsb = ttk.Scrollbar(canvas_frame, orient='vertical',
                                   command=self._cards_canvas.yview)
        self._cards_canvas.configure(yscrollcommand=cards_vsb.set)
        cards_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._cards_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._cards_inner = tk.Frame(self._cards_canvas, bg=C['bg'])
        self._cards_window = self._cards_canvas.create_window(
            (0, 0), window=self._cards_inner, anchor='nw')

        self._cards_inner.bind('<Configure>', self._on_cards_configure)
        self._cards_canvas.bind('<Configure>', self._on_canvas_resize)
        self._cards_canvas.bind('<MouseWheel>', self._on_mousewheel)
        self._cards_canvas.bind('<Button-4>',  self._on_mousewheel)
        self._cards_canvas.bind('<Button-5>',  self._on_mousewheel)

    def _on_cards_configure(self, event=None):
        self._cards_canvas.configure(
            scrollregion=self._cards_canvas.bbox('all'))

    def _on_canvas_resize(self, event):
        self._cards_canvas.itemconfig(self._cards_window, width=event.width)

    def _on_mousewheel(self, event):
        if event.num == 4:
            self._cards_canvas.yview_scroll(-1, 'units')
        elif event.num == 5:
            self._cards_canvas.yview_scroll(1, 'units')
        else:
            self._cards_canvas.yview_scroll(int(-1*(event.delta/120)), 'units')

    # ── Full Report tab ────────────────────────────────────────────────────

    def _build_full_report_tab(self, parent) -> None:
        C = self.C
        ctrl = tk.Frame(parent, bg=C['bg'])
        ctrl.pack(fill=tk.X, padx=8, pady=6)

        self._report_summary_lbl = tk.Label(
            ctrl, text='Run a scan to generate the full report',
            bg=C['bg'], fg=C['fg'], font=('Arial', 10, 'bold'))
        self._report_summary_lbl.pack(side=tk.LEFT)

        ttk.Button(ctrl, text='📤 Export TXT',
                   command=self._export_txt,
                   style='Blue.TButton').pack(side=tk.RIGHT, padx=4)
        ttk.Button(ctrl, text='📊 Export CSV',
                   command=self._export_csv,
                   style='Blue.TButton').pack(side=tk.RIGHT, padx=4)
        ttk.Button(ctrl, text='🎯 Auto-Select',
                   command=self._auto_select,
                   style='Amber.TButton').pack(side=tk.RIGHT, padx=4)

        self._report_text = scrolledtext.ScrolledText(
            parent, font=('Courier', 9),
            bg=C['panel_bg'], fg=C['fg'],
            borderwidth=1, relief='solid',
            state=tk.DISABLED)
        self._report_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # Tags
        self._report_text.tag_config('rpt_header',
            foreground='#ffffff', background=C['header_bg'],
            font=('Arial', 11, 'bold'))
        self._report_text.tag_config('rpt_summary',
            foreground=C['warning'], font=('Arial', 10, 'bold'))
        self._report_text.tag_config('rpt_group',
            foreground=C['success'], background=C['exact_bg'],
            font=('Courier', 10, 'bold'))
        self._report_text.tag_config('rpt_near',
            foreground=C['accent1'], background=C['near_bg'],
            font=('Courier', 10, 'bold'))
        self._report_text.tag_config('rpt_keep',
            foreground=C['keep_fg'], font=('Courier', 9, 'bold'))
        self._report_text.tag_config('rpt_delete',
            foreground=C['del_fg'], font=('Courier', 9, 'bold'))
        self._report_text.tag_config('rpt_meta',
            foreground='#666677', font=('Courier', 8))
        self._report_text.tag_config('rpt_div',
            foreground=C['accent1'])
        self._report_text.tag_config('rpt_score',
            foreground=C['info'], font=('Courier', 9))
        self._report_text.tag_config('rpt_hl',
            foreground=C['warning'], font=('Courier', 9, 'bold'))

    # ── Activity Log tab ───────────────────────────────────────────────────

    def _build_log_tab(self, parent) -> None:
        C = self.C
        ctrl = tk.Frame(parent, bg=C['bg'])
        ctrl.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(ctrl, text='Activity Log', bg=C['bg'],
                 fg=C['fg'], font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        ttk.Button(ctrl, text='Clear',
                   command=self._clear_log,
                   style='Gray.TButton').pack(side=tk.RIGHT, padx=4)

        self._log_text = scrolledtext.ScrolledText(
            parent, font=('Courier', 8),
            bg=C['panel_bg'], fg=C['fg'],
            borderwidth=1, relief='solid',
            state=tk.DISABLED)
        self._log_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self._log_text.tag_config('info',  foreground=C['success'])
        self._log_text.tag_config('error', foreground=C['danger'])
        self._log_text.tag_config('warn',  foreground=C['warning'])
        self._log_text.tag_config('time',  foreground=C['accent1'],
                                  font=('Courier', 8, 'bold'))

    # ── Deletion history tab ───────────────────────────────────────────────

    def _build_deletion_history_tab(self, parent) -> None:
        C = self.C
        ctrl = tk.Frame(parent, bg=C['bg'])
        ctrl.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(ctrl, text='Deletion History (persistent log)',
                 bg=C['bg'], fg=C['fg'],
                 font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        ttk.Button(ctrl, text='🔄 Refresh',
                   command=self._refresh_deletion_history,
                   style='Blue.TButton').pack(side=tk.RIGHT, padx=4)
        ttk.Button(ctrl, text='Clear Log',
                   command=self._clear_deletion_log,
                   style='Red.TButton').pack(side=tk.RIGHT, padx=4)

        self._del_history_text = scrolledtext.ScrolledText(
            parent, font=('Courier', 8),
            bg=C['panel_bg'], fg=C['fg'],
            borderwidth=1, relief='solid',
            state=tk.DISABLED)
        self._del_history_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self._del_history_text.tag_config('success', foreground=C['success'])
        self._del_history_text.tag_config('fail',    foreground=C['danger'])
        self._del_history_text.tag_config('ts',      foreground=C['accent1'])
        self._refresh_deletion_history()

    # ── Settings tab ───────────────────────────────────────────────────────

    def _build_settings_tab(self, parent) -> None:
        C = self.C
        canvas = tk.Canvas(parent, bg=C['bg'], highlightthickness=0)
        vsb = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(canvas, bg=C['bg'])
        canvas.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>',
                   lambda e: canvas.configure(
                       scrollregion=canvas.bbox('all')))

        def _section(title):
            f = tk.Frame(inner, bg=C['toolbar_bg'],
                         borderwidth=1, relief='solid')
            f.pack(fill=tk.X, padx=16, pady=(12, 2))
            tk.Label(f, text=title, bg=C['toolbar_bg'],
                     fg=C['toolbar_fg'], font=('Arial', 10, 'bold'),
                     padx=10, pady=6).pack(anchor='w')
            g = tk.Frame(inner, bg=C['accent3'],
                         borderwidth=1, relief='solid')
            g.pack(fill=tk.X, padx=16, pady=(0, 4))
            return g

        def _row(grid, label, widget_fn, r):
            tk.Label(grid, text=label, bg=C['accent3'], fg=C['fg'],
                     font=('Arial', 9), anchor='w', width=38).grid(
                row=r, column=0, padx=12, pady=5, sticky='w')
            widget_fn(grid).grid(row=r, column=1, padx=12, pady=5, sticky='w')

        def _chk(parent, var, state=tk.NORMAL):
            return tk.Checkbutton(parent, variable=var, bg=C['accent3'],
                                   fg=C['fg'], selectcolor=C['bg'],
                                   activebackground=C['accent3'],
                                   font=('Arial', 9), state=state)
        def _spn(parent, var, lo=0, hi=100, w=8):
            return tk.Spinbox(parent, from_=lo, to=hi, textvariable=var,
                               width=w, font=('Arial', 9),
                               bg=C['panel_bg'], fg=C['fg'])

        # ── Scan settings ─────────────────────────────────────────────────
        g = _section('⚙️  SCAN SETTINGS')
        _row(g, 'Scan subdirectories recursively:',
             lambda p: _chk(p, self._var_subdirs), 0)
        _row(g, 'Hash file contents (exact match):',
             lambda p: _chk(p, self._var_hash), 1)
        _row(g, 'Use xxhash (~10× faster than MD5):',
             lambda p: _chk(p, self._var_xxhash,
                            tk.NORMAL if HAS_XXHASH else tk.DISABLED), 2)
        _row(g, 'GPU / NumPy name-similarity pass:',
             lambda p: _chk(p, self._var_gpu,
                            tk.NORMAL if (HAS_CUPY or HAS_NUMPY) else tk.DISABLED), 3)
        _row(g, 'Paranoid mode (byte-by-byte verify):',
             lambda p: _chk(p, self._var_paranoid), 4)
        _row(g, f'Worker threads  (CPU cores: {CPU_COUNT}):',
             lambda p: _spn(p, self._var_workers, 1, 64, 6), 5)
        _row(g, 'Minimum similarity score (0–100):',
             lambda p: _spn(p, self._var_min_score, 0, 100, 6), 6)
        _row(g, 'Minimum file size (bytes, 0=none):',
             lambda p: _spn(p, self._var_min_size, 0, 10_000_000, 10), 7)
        _row(g, 'Maximum file size (bytes, 0=none):',
             lambda p: _spn(p, self._var_max_size, 0, 100_000_000_000, 14), 8)

        # ── Auto-selection settings ────────────────────────────────────────
        g2 = _section('🎯 AUTO-SELECTION SETTINGS')
        _row(g2, 'Auto-select after scan completes:',
             lambda p: _chk(p, self._var_auto_select), 0)
        _row(g2, 'Min quality gap to suggest DELETE:',
             lambda p: _spn(p, self._var_delete_gap, 0, 100, 6), 1)

        # ── Display settings ───────────────────────────────────────────────
        g3 = _section('🎨 DISPLAY SETTINGS')
        _row(g3, 'Dark mode (Catppuccin Mocha):',
             lambda p: _chk(p, self._var_dark_mode), 0)
        ttk.Button(g3, text='Apply Dark/Light Mode',
                   command=self._apply_display_settings,
                   style='Blue.TButton').grid(
                       row=1, column=0, columnspan=2, padx=12, pady=8, sticky='w')

        # ── Exclusion patterns ────────────────────────────────────────────
        g4 = _section('🚫 EXCLUSION PATTERNS  (comma-separated substrings)')
        self._excl_entry = tk.Text(g4, height=3, width=60,
                                    bg=C['panel_bg'], fg=C['fg'],
                                    font=('Courier', 9),
                                    borderwidth=1, relief='solid')
        self._excl_entry.grid(row=0, column=0, columnspan=2,
                               padx=12, pady=8, sticky='w')

        # ── Library status + install buttons ─────────────────────────────
        g5 = _section('📦 OPTIONAL LIBRARY STATUS  (auto-install on startup)')
        libs = [
            ('xxhash',     HAS_XXHASH,     'xxhash',       '~10× faster hashing'),
            ('send2trash', HAS_SEND2TRASH, 'send2trash',   'Cross-platform Trash'),
            ('psutil',     HAS_PSUTIL,     'psutil',       'System monitoring'),
            ('numpy',      HAS_NUMPY,      'numpy',        'Vectorised name similarity'),
            ('cupy',       HAS_CUPY,       'cupy-cuda11x', 'NVIDIA GPU acceleration'),
            ('Pillow',     HAS_PIL,        'Pillow',       'Image thumbnails + pHash'),
        ]

        def _make_install_fn(pip_pkg, status_lbl_ref):
            def _do_install():
                status_lbl_ref.config(text='⏳ Installing…', fg=C['warning'])
                inner.update_idletasks()
                def _bg():
                    _ok = _pip_install(pip_pkg, verbose=True)
                    _txt = '✓ Done — restart to apply' if _ok else '✗ Install failed'
                    _fg  = C['success'] if _ok else C['danger']
                    _tag = 'info' if _ok else 'error'
                    _msg = f'[SESSION] pip install {pip_pkg} → {"ok" if _ok else "FAILED"}'
                    # must update tkinter widgets on main thread
                    self.root.after(0, lambda: status_lbl_ref.config(text=_txt, fg=_fg))
                    self.root.after(0, lambda: self._dbg(_msg, _tag))
                threading.Thread(target=_bg, daemon=True).start()
            return _do_install

        for i, (name, present, pip_pkg, desc) in enumerate(libs):
            sc = C['success'] if present else C['danger']
            status_text = '✓ INSTALLED' if present else '✗ MISSING'
            tk.Label(g5, text=f'{name}',
                     bg=C['accent3'], fg=C['fg'],
                     font=('Courier', 9, 'bold'), width=14,
                     anchor='w').grid(row=i, column=0, padx=12, pady=4, sticky='w')
            lbl = tk.Label(g5, text=status_text,
                     bg=C['accent3'], fg=sc,
                     font=('Courier', 9, 'bold'), width=14, anchor='w')
            lbl.grid(row=i, column=1, padx=6, pady=4, sticky='w')
            tk.Label(g5, text=desc, bg=C['accent3'],
                     fg=C['fg'], font=('Arial', 8), anchor='w').grid(
                         row=i, column=2, padx=8, pady=4, sticky='w')
            btn_text = '🔄 Reinstall' if present else '⬇ Install'
            btn_style = 'Blue.TButton' if present else 'Green.TButton'
            ttk.Button(g5, text=btn_text, style=btn_style,
                       command=_make_install_fn(pip_pkg, lbl)).grid(
                           row=i, column=3, padx=8, pady=4, sticky='w')

        tk.Label(g5,
                 text='  Changes take effect after restarting the program.',
                 bg=C['accent3'], fg=C['warning'],
                 font=('Arial', 8, 'italic')).grid(
                     row=len(libs), column=0, columnspan=4,
                     padx=12, pady=(2, 8), sticky='w')

    # ── Status bar ─────────────────────────────────────────────────────────

    def _build_status_bar(self) -> None:
        C = self.C
        sb = tk.Frame(self.root, bg=C['status_bar_bg'], height=22)
        sb.pack(fill=tk.X, side=tk.BOTTOM)
        sb.pack_propagate(False)

        self._sb_groups  = tk.Label(sb, text='Groups: 0',
                                     bg=C['status_bar_bg'], fg=C['status_bar_fg'],
                                     font=('Courier', 8))
        self._sb_groups.pack(side=tk.LEFT, padx=8)

        tk.Label(sb, text='│', bg=C['status_bar_bg'],
                 fg=C['border']).pack(side=tk.LEFT)

        self._sb_files   = tk.Label(sb, text='Files: 0',
                                     bg=C['status_bar_bg'], fg=C['status_bar_fg'],
                                     font=('Courier', 8))
        self._sb_files.pack(side=tk.LEFT, padx=8)

        tk.Label(sb, text='│', bg=C['status_bar_bg'],
                 fg=C['border']).pack(side=tk.LEFT)

        self._sb_reclaim = tk.Label(sb, text='Reclaimable: 0 B',
                                     bg=C['status_bar_bg'], fg=C['status_bar_fg'],
                                     font=('Courier', 8))
        self._sb_reclaim.pack(side=tk.LEFT, padx=8)

        tk.Label(sb, text='│', bg=C['status_bar_bg'],
                 fg=C['border']).pack(side=tk.LEFT)

        self._sb_marked  = tk.Label(sb, text='Marked: 0',
                                     bg=C['status_bar_bg'], fg=C['status_bar_fg'],
                                     font=('Courier', 8))
        self._sb_marked.pack(side=tk.LEFT, padx=8)

        # Terminal toggle button
        self._term_toggle_btn = tk.Button(
            sb, text='▼ Terminal',
            bg=C['status_bar_bg'], fg=C['warning'],
            font=('Courier', 8, 'bold'), relief='flat',
            cursor='hand2', command=self._toggle_terminal)
        self._term_toggle_btn.pack(side=tk.RIGHT, padx=8)

        self._sb_status  = tk.Label(sb, text='Ready',
                                     bg=C['status_bar_bg'], fg=C['warning'],
                                     font=('Courier', 8, 'bold'), anchor='e')
        self._sb_status.pack(side=tk.RIGHT, padx=12, fill=tk.X, expand=True)

    def _update_status_bar(self) -> None:
        """Refresh status bar counts from current groups."""
        ng = len(self.groups)
        nf = sum(len(g.files) for g in self.groups)
        rb = sum(g.reclaimable_bytes for g in self.groups)
        nm = sum(1 for g in self.groups
                 for fi, s in g.suggestions.items() if s == 'DELETE')
        self._sb_groups.config(text=f'Groups: {ng}')
        self._sb_files.config(text=f'Files: {nf}')
        self._sb_reclaim.config(text=f'Reclaimable: {_format_size(rb)}')
        self._sb_marked.config(text=f'Marked: {nm}')


    # ── Debug terminal (always-visible, collapsible, bottom) ──────────────

    def _build_debug_terminal(self) -> None:
        C = self.C

        self._term_frame = tk.Frame(
            self.root, bg='#161b22',
            borderwidth=2, relief='solid',
            highlightthickness=1,
            highlightbackground=C['warning'])
        self._term_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=1, pady=1)

        # Header
        hdr = tk.Frame(self._term_frame, bg='#161b22')
        hdr.pack(fill=tk.X, padx=4, pady=(3, 0))

        tk.Label(hdr, text='⚡ DEBUG TERMINAL',
                 bg='#161b22', fg=C['warning'],
                 font=('Courier', 9, 'bold')).pack(side=tk.LEFT)
        tk.Label(hdr,
                 text=f'  v{VERSION}  |  xxhash={"✓" if HAS_XXHASH else "✗"}'
                      f'  |  send2trash={"✓" if HAS_SEND2TRASH else "✗"}'
                      f'  |  PIL={"✓" if HAS_PIL else "✗"}',
                 bg='#161b22', fg='#6e7681',
                 font=('Courier', 8)).pack(side=tk.LEFT, padx=6)

        tk.Button(hdr, text='CLEAR ALL',
                  command=self._clear_debug_terminal,
                  bg='#21262d', fg=C['warning'],
                  font=('Courier', 8, 'bold'),
                  relief='flat', padx=6, cursor='hand2').pack(
                      side=tk.RIGHT, padx=4)

        tk.Button(hdr, text='📁 Open Log',
                  command=self._open_log_file,
                  bg='#21262d', fg='#58a6ff',
                  font=('Courier', 8, 'bold'),
                  relief='flat', padx=6, cursor='hand2').pack(
                      side=tk.RIGHT, padx=4)

        tk.Label(hdr, text=f'  log → {LOG_FILE_PATH.name}  ({LOG_FILE_PATH.parent})',
                 bg='#161b22', fg='#444d56',
                 font=('Courier', 7)).pack(side=tk.LEFT, padx=4)

        # Inner notebook (4 tabs)
        term_nb = ttk.Notebook(self._term_frame)
        term_nb.pack(fill=tk.BOTH, padx=4, pady=(2, 4))
        self._term_nb = term_nb

        def _make_tab(title, height=6):
            frm = tk.Frame(term_nb, bg='#0d1117')
            term_nb.add(frm, text=title)
            txt = tk.Text(frm, font=('Courier', 8),
                          bg='#0d1117', fg='#c9d1d9',
                          height=height, wrap=tk.WORD,
                          borderwidth=0, relief='flat',
                          insertbackground=C['warning'],
                          selectbackground='#264f78')
            vsb = tk.Scrollbar(frm, orient='vertical', command=txt.yview)
            txt.configure(yscrollcommand=vsb.set)
            vsb.pack(side=tk.RIGHT, fill=tk.Y)
            txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            # Colour tags
            txt.tag_config('ts',      foreground='#58a6ff', font=('Courier', 8, 'bold'))
            txt.tag_config('info',    foreground='#3fb950')
            txt.tag_config('error',   foreground='#f85149')
            txt.tag_config('warn',    foreground='#e3b341')
            txt.tag_config('debug',   foreground='#79c0ff')
            txt.tag_config('hash',    foreground='#bc8cff')
            txt.tag_config('compare', foreground='#ffa657')
            txt.tag_config('scan',    foreground='#56d364')
            txt.tag_config('select',  foreground='#d2a8ff')
            txt.tag_config('result',  foreground='#f0f6fc')
            txt.config(state=tk.DISABLED)
            return txt

        self._term_status = _make_tab('📊 Status')
        self._term_debug  = _make_tab('🐛 Debug')
        self._term_errors = _make_tab('⚠️  Errors')
        self._term_events = _make_tab('📋 Events')

        # Banner
        self._term_append(self._term_status,
                          f'Duplicate File Finder v{VERSION} ready'
                          f'  |  CPUs={CPU_COUNT}'
                          f'  |  xxhash={"✓" if HAS_XXHASH else "✗"}'
                          f'  |  PIL={"✓" if HAS_PIL else "✗"}'
                          f'  |  send2trash={"✓" if HAS_SEND2TRASH else "✗"}',
                          'info')

    def _term_append(self, widget, text: str, tag: str = 'debug') -> None:
        """Append one timestamped line to a terminal tab widget + log file."""
        widget.config(state=tk.NORMAL)
        ts = time.strftime('%H:%M:%S')
        widget.insert(tk.END, f'[{ts}] ', 'ts')
        widget.insert(tk.END, f'{text}\n', tag)
        lc = int(widget.index('end-1c').split('.')[0])
        if lc > DEBUG_MAX_LINES:
            widget.delete('1.0', f'{lc - DEBUG_MAX_LINES}.0')
        widget.see(tk.END)
        widget.config(state=tk.DISABLED)

    def _dbg(self, text: str, tag: str = '') -> None:
        """
        Route a debug message to the terminal tabs.

        Routing:
          Debug tab   <- ALL messages (always — the full live feed)
          Events tab  <- pipeline events [SCAN] [HASH] [COMPARE] [FIND]
                         [SELECT] [VERIFY] [DELETE] [EXPORT] [SESSION]
          Errors tab  <- any error-tagged message  + auto-switch to Errors tab
          Status tab  <- errors only  (high-level _log() milestones go there
                         via _handle_msg directly, keeping Status clean)
        """
        tag = tag or self._classify_tag(text)
        tl  = text.lower()

        is_event = any(x in tl for x in (
            '[scan]', '[hash]', '[compare]', '[find]',
            '[select]', '[verify]', '[delete]', '[export]', '[session]'))
        is_error = ('error' in tl or tag == 'error' or '[error]' in tl)

        # ── Write to persistent log file ONCE per logical message ─────────
        try: self._write_log_file(tag, text)
        except Exception: pass

        # ── Debug tab gets EVERYTHING ─────────────────────────────────────
        self._term_append(self._term_debug, text, tag)

        # ── Events tab gets pipeline events ───────────────────────────────
        if is_event:
            self._term_append(self._term_events, text, tag)

        # ── Errors: Errors tab + Status tab + auto-switch ─────────────────
        if is_error:
            self._term_append(self._term_errors, text, 'error')
            self._term_append(self._term_status, text, 'error')
            try: self._term_nb.select(2)
            except Exception: pass

    def _classify_tag(self, text: str) -> str:
        tl = text.lower()
        if '[hash]'    in tl: return 'hash'
        if '[compare]' in tl: return 'compare'
        if '[scan]'    in tl: return 'scan'
        if '[find]'    in tl: return 'debug'
        if '[select]'  in tl: return 'select'
        if '[delete]'  in tl: return 'warn'
        if '[verify]'  in tl: return 'hash'
        if '[session]' in tl: return 'info'
        if 'error'     in tl: return 'error'
        if 'warn'      in tl: return 'warn'
        return 'debug'

    def _clear_debug_terminal(self) -> None:
        for w in (self._term_status, self._term_debug,
                  self._term_errors, self._term_events):
            w.config(state=tk.NORMAL)
            w.delete('1.0', tk.END)
            w.config(state=tk.DISABLED)

    def _toggle_terminal(self) -> None:
        if self._term_visible:
            self._term_frame.pack_forget()
            self._term_toggle_btn.config(text='▲ Terminal')
        else:
            self._term_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=1, pady=1)
            self._term_toggle_btn.config(text='▼ Terminal')
        self._term_visible = not self._term_visible

    def _write_log_file(self, level: str, text: str) -> None:
        """
        Enqueue one log line for async write — never blocks the main thread.
        The _start_log_writer background thread drains and batch-writes the queue.
        """
        try:
            ts   = time.strftime('%Y-%m-%d %H:%M:%S')
            line = f'[{ts}] [{level.upper():7}] {text}\n'
            self._log_file_queue.put_nowait(line)
        except Exception:
            pass  # Never crash on logging

    def _start_log_writer(self) -> None:
        """Spin up the background log-file writer thread (batched, non-blocking)."""
        def _writer():
            buf: list = []
            while True:
                try:
                    line = self._log_file_queue.get(timeout=0.5)
                    buf.append(line)
                    # Drain remaining queued lines for a batch write
                    while len(buf) < 500:
                        try:
                            buf.append(self._log_file_queue.get_nowait())
                        except queue.Empty:
                            break
                except queue.Empty:
                    pass  # timeout — fall through to flush
                except Exception:
                    buf.clear(); continue
                if buf:
                    try:
                        with open(LOG_FILE_PATH, 'a', encoding='utf-8') as fh:
                            fh.writelines(buf)
                    except Exception:
                        pass
                    buf.clear()
        t = threading.Thread(target=_writer, daemon=True, name='dupfinder-log-writer')
        t.start()

    def _open_log_file(self) -> None:
        """Open the persistent log file in the default text editor."""
        try:
            import subprocess as _sp2
            LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            if not LOG_FILE_PATH.exists():
                LOG_FILE_PATH.write_text('', encoding='utf-8')
            if   sys.platform == 'win32':
                _sp2.run(['notepad', str(LOG_FILE_PATH)], check=False)
            elif sys.platform == 'darwin':
                _sp2.run(['open', str(LOG_FILE_PATH)], check=False)
            else:
                _sp2.run(['xdg-open', str(LOG_FILE_PATH)], check=False)
        except Exception as exc:
            messagebox.showerror('Log File', f'Cannot open log file:\n{exc}')

    # ── Keyboard shortcuts ─────────────────────────────────────────────────

    def _bind_shortcuts(self) -> None:
        self.root.bind('<Control-s>', lambda e: self._start_scan())
        self.root.bind('<Control-S>', lambda e: self._start_scan())
        self.root.bind('<Escape>',    lambda e: self._stop_scan())
        self.root.bind('<Delete>',    lambda e: self._delete_selected())
        self.root.bind('<Left>',      lambda e: self._nav_prev())
        self.root.bind('<Right>',     lambda e: self._nav_next())
        self.root.bind('<Up>',        lambda e: self._tree_move(-1))
        self.root.bind('<Down>',      lambda e: self._tree_move(1))
        self.root.bind('<Control-a>', lambda e: self._select_all_in_group())
        self.root.bind('<Control-A>', lambda e: self._select_all_in_group())
        self.root.bind('<Control-z>', lambda e: self._show_deletion_history())
        self.root.bind('<F5>',        lambda e: self._start_scan())
        self.root.bind('<F1>',        lambda e: self._show_help())

    # ── Scan control ───────────────────────────────────────────────────────

    def _change_folder(self) -> None:
        folder = filedialog.askdirectory(title='Select folder to scan')
        if folder:
            self._folder_lbl.config(text=folder)
            self.groups = []
            self._populate_tree([])
            self._clear_detail()
            self._log_msg('info', f'Folder changed: {folder}')
            self._dbg(f'[SCAN] Folder changed  path={folder}', 'scan')

    def _collect_settings(self) -> ScanSettings:
        """Build a ScanSettings from current UI variable values."""
        excl_raw = self._excl_entry.get('1.0', tk.END).strip()
        excl = [p.strip() for p in excl_raw.split(',') if p.strip()]
        return ScanSettings(
            subdirs        = self._var_subdirs.get(),
            min_size       = max(0, self._var_min_size.get()),
            max_size       = max(0, self._var_max_size.get()),
            use_xxhash     = self._var_xxhash.get() and HAS_XXHASH,
            hash_files     = self._var_hash.get(),
            paranoid_mode  = self._var_paranoid.get(),
            use_gpu        = self._var_gpu.get(),
            num_workers    = max(1, self._var_workers.get()),
            min_score      = self._var_min_score.get(),
            exclusion_patterns = excl,
            dark_mode      = self._var_dark_mode.get(),
            auto_select    = self._var_auto_select.get(),
            delete_gap     = self._var_delete_gap.get(),
        )

    def _start_scan(self) -> None:
        if self.is_scanning: return
        folder = self._folder_lbl.cget('text')
        if not folder or not Path(folder).exists():
            messagebox.showerror('Invalid Folder', f'Folder not found:\n{folder}')
            return

        self.settings   = self._collect_settings()
        self._cancel_ev = threading.Event()
        self._pq        = queue.Queue(maxsize=20000)
        self._engine    = ScanEngine(folder, self.settings,
                                      self._pq, self._cancel_ev)

        self.is_scanning = True
        self.groups      = []
        self._scan_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        self._auto_sel_btn.config(state=tk.DISABLED)
        self._delete_btn.config(state=tk.DISABLED)
        self._export_btn.config(state=tk.DISABLED)
        self._save_sess_btn.config(state=tk.DISABLED)

        self._scan_bar['value']  = 0
        self._match_bar['value'] = 0
        self._scan_lbl.config(text='')
        self._match_lbl.config(text='')
        self._activity_lbl.config(text='⠋ Scanning…')

        self._clear_detail()
        self._populate_tree([])
        self._clear_log()
        self._clear_debug_terminal()
        self._update_status_bar()

        self._log_msg('info', f'=== SCAN STARTED  folder={folder} ===')
        self._dbg(
            f'[SCAN] === SCAN STARTED ==='
            f'  folder={folder}'
            f'  workers={self.settings.num_workers}'
            f'  hash={self.settings.hash_files}'
            f'  xxhash={self.settings.use_xxhash}'
            f'  paranoid={self.settings.paranoid_mode}'
            f'  min_score={self.settings.min_score}', 'scan')

        self.scan_thread = threading.Thread(
            target=self._scan_worker, daemon=True)
        self.scan_thread.start()

    def _scan_worker(self) -> None:
        """Background worker — communicates only via self._pq."""
        try:
            self._engine.scan()
            if self._cancel_ev.is_set(): return
            groups = self._engine.find_duplicates()
            if self._cancel_ev.is_set(): return
            self._pq.put_nowait(('complete', {
                'groups': len(groups),
                'files':  len(self._engine.files),
                'data':   groups,
            }))
        except Exception as exc:
            tb = traceback.format_exc()
            err_msg = f'[ERROR] Unhandled scan exception: {type(exc).__name__}: {exc}'
            try: self._pq.put_nowait(('error', str(exc)))
            except queue.Full: pass
            try: self._pq.put_nowait(('error_detail',
                                       f'{err_msg}\nTraceback:\n{tb}'))
            except queue.Full: pass
            # Also write to log file immediately (queue might be full)
            try:
                with open(LOG_FILE_PATH, 'a', encoding='utf-8') as _lf:
                    _lf.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] [CRITICAL] {err_msg}\n{tb}\n')
            except Exception:
                pass

    def _stop_scan(self) -> None:
        self._cancel_ev.set()
        self.is_scanning = False
        self._scan_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        self._activity_lbl.config(text='⏹ Scan stopped')
        self._sb_status.config(text='Stopped')
        self._log_msg('warn', 'Scan stopped by user')
        self._dbg('[SCAN] ⏹ Stopped by user', 'warn')

    # ── Progress monitor (50 ms while scanning, 300 ms idle) ──────────────

    def _start_progress_monitor(self) -> None:
        """
        Drain the worker->UI queue on the main thread.
        Runs every 50 ms while scanning, 100 ms idle.
        Hard 30 ms wall-clock budget per tick so the event loop stays responsive.
        """
        if self._engine is not None:
            _t0 = time.monotonic()
            _pq = self._pq
            _cnt = 0
            while _cnt < 600:
                try:
                    mtype, data = _pq.get_nowait()
                except queue.Empty:
                    break
                self._handle_msg(mtype, data)
                _cnt += 1
                # Hard time-budget: yield back to Tk after 30 ms
                if _cnt % 20 == 0 and (time.monotonic() - _t0) > 0.030:
                    break

        interval = 50 if self.is_scanning else 100
        self.root.after(interval, self._start_progress_monitor)

    def _handle_msg(self, mtype: str, data) -> None:
        """
        Dispatch one queue message to the correct UI handler.

        Routing contract:
          'log'          -> Activity Log + Status tab (high-level milestones only)
          'debug'        -> _dbg() -> Debug tab (+ Events, + Errors if error)
          'scan_progress'-> progress bars / activity label
          'match_progress-> compare progress bar
          'error_detail' -> Errors tab only (full stack trace, no double timestamp)
          'error'        -> Activity Log + _dbg (routed to Errors+Status)
          'info'/'warn'  -> Activity Log + _dbg
          'complete'     -> _on_scan_complete()
        """
        if   mtype == 'scan_progress':
            self._on_scan_progress(data)
        elif mtype == 'match_progress':
            self._on_match_progress(data)
        elif mtype == 'log':
            # High-level milestone: Activity Log + Status tab ONLY (not Debug)
            tag, text = data
            self._log_msg(tag, text)
            try: self._write_log_file(tag, text)
            except Exception: pass
            _stag = 'info' if tag == 'info' else ('warn' if tag == 'warn' else 'error')
            self._term_append(self._term_status, text, _stag)
        elif mtype == 'debug':
            # Detailed operational debug -> Debug/Events tabs via _dbg()
            self._dbg(data)
        elif mtype == 'error_detail':
            # Full stack trace: Errors tab only (_term_append adds timestamp)
            try: self._write_log_file('error', data)
            except Exception: pass
            self._term_append(self._term_errors, data, 'error')
            self._term_append(self._term_status, '\u274c Unhandled error — see ⚠️ Errors tab for trace', 'error')
            try: self._term_nb.select(2)
            except Exception: pass
        elif mtype in ('info', 'warn'):
            self._log_msg(mtype, data)
            self._dbg(data, mtype)
        elif mtype == 'error':
            self._log_msg('error', data)
            self._dbg(f'[ERROR] {data}', 'error')
        elif mtype == 'complete':
            self._on_scan_complete(data)

    def _on_scan_progress(self, d: dict) -> None:
        self._scan_bar['value'] = d.get('percent', 0)
        self._scan_lbl.config(
            text=f"{d.get('current', 0):,}/{d.get('total', 0):,} ({d.get('percent', 0)}%)")
        self._activity_lbl.config(
            text=f"{d.get('status', '')}  {d.get('file', '')}")
        self._sb_status.config(text=d.get('status', 'Scanning…'))

    def _on_match_progress(self, d: dict) -> None:
        # ── modular: match-progress handler ──────────────────────────────────
        pct = d.get('percent', 0)
        cur = d.get('current', 0)
        tot = d.get('total', 0)
        self._match_bar['value'] = pct
        self._match_lbl.config(text=f'{cur:,}/{tot:,} ({pct}%)')
        remaining = max(tot - cur, 0)
        lbl = (f'🔍 Comparing {tot:,} pairs… ({remaining:,} remaining)'
               if tot > 0 else '🔍 Comparing pairs…')
        self._activity_lbl.config(text=lbl)
        self._sb_status.config(text=f'Comparing… {pct}%')

    def _on_scan_complete(self, data: dict) -> None:
        # ── modular: scan-complete UI handler ───────────────────────────────
        """Handle scan completion — populate tree immediately, auto-select in bg."""
        self.is_scanning = False
        self.groups      = data.get('data', [])
        ng = data.get('groups', 0)
        nf = data.get('files', 0)

        self._scan_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        self._scan_bar['value']  = 100
        self._match_bar['value'] = 100

        self._log_msg('info', f'✓ Scan complete! {ng} duplicate groups in {nf:,} files')
        self._dbg(f'[SCAN] === SCAN COMPLETE ===  groups={ng}  files={nf}', 'scan')

        if ng == 0:
            self._activity_lbl.config(text='✓ No duplicates — all files are unique')
            self._sb_status.config(text='✓ Done')
            messagebox.showinfo('✓ No Duplicates',
                                'No duplicates found.\nAll files appear unique.')
            self._update_status_bar()
            return

        # Show "processing" state ─ tree populated after auto-select completes
        self._activity_lbl.config(text=f'⚙️  Processing {ng} groups…')
        self._sb_status.config(text='Processing…')
        self._update_status_bar()

        # Mark Full Report as needing rebuild (lazy render) ──────────────────
        self._report_dirty = True
        self._report_summary_lbl.config(
            text=f'{ng} groups · {nf:,} files — click "📄 Full Report" tab to view')

        # Run auto-select in a BACKGROUND thread (never block main thread) ───
        if self.settings.auto_select and self._engine:
            self._log_msg('info', '🎯 Running auto-selection in background…')
            self._activity_lbl.config(text='⚙️  Auto-selecting best files to keep…')
            _engine = self._engine
            _groups = self.groups

            def _bg_select():
                try:
                    _engine.smart_select(_groups)
                    n_marked = sum(1 for g in _groups
                                   for s in g.suggestions.values() if s == 'DELETE')
                except Exception as exc:
                    n_marked = 0
                    try:
                        self._pq.put_nowait(('debug',
                                              f'[SELECT] bg error: {exc}'))
                    except Exception:
                        pass
                self.root.after(0, lambda nm=n_marked: self._finish_scan_ui(ng, nf, nm))

            threading.Thread(target=_bg_select, daemon=True,
                             name='dupfinder-autoselect').start()
        else:
            self._finish_scan_ui(ng, nf, 0)

    def _finish_scan_ui(self, ng: int, nf: int, n_marked: int) -> None:
        # ── modular: post-auto-select UI finalisation ───────────────────────
        """Called on main thread after background auto-select finishes."""
        self._populate_tree(self.groups)   # refresh tree with suggestion marks
        self._update_status_bar()

        # Enable action buttons
        self._auto_sel_btn.config(state=tk.NORMAL)
        self._delete_btn.config(state=tk.NORMAL)
        self._export_btn.config(state=tk.NORMAL)
        self._save_sess_btn.config(state=tk.NORMAL)
        self._clear_sel_btn.config(state=tk.NORMAL)

        summary = (f'✓ {ng} groups  ·  {nf:,} files  ·  {n_marked} marked')
        self._activity_lbl.config(text=summary)
        self._sb_status.config(text='✓ Done')

        if n_marked > 0:
            self._log_msg('info',
                          f'🎯 Auto-select: {n_marked} files suggested for deletion')
            self._dbg(f'[SELECT] Auto-select: {n_marked} files marked DELETE', 'select')

        self._log_msg('info', f'✓ Ready — {ng} duplicate groups to review')

        # Show first group (only UI work, no report render) ──────────────────
        if self.groups:
            self._show_group(0)

    # ── Spinner animation ─────────────────────────────────────────────────

    def _animate_spinner(self) -> None:
        if self.is_scanning:
            frame = SPINNER_FRAMES[self._spinner_idx % len(SPINNER_FRAMES)]
            self._spinner_idx += 1
            fg = self.C['warning']
        else:
            frame = '◉'
            fg    = self.C['success']
        self._header_spinner.config(text=frame, fg=fg)
        self.root.after(80, self._animate_spinner)

    # ── Lazy Full Report tab render ────────────────────────────────────────

    def _on_right_tab_changed(self, event=None) -> None:
        # ── modular: lazy tab-change handler ─────────────────────────────
        """Trigger lazy render of Full Report when user selects that tab."""
        try:
            idx = self._right_nb.index(self._right_nb.select())
        except Exception:
            return
        if idx == 1 and self._report_dirty:   # 1 = Full Report tab
            self._render_full_report()

    # ── Logging helpers (main thread only) ────────────────────────────────

    def _log_msg(self, tag: str, text: str) -> None:
        self._log_text.config(state=tk.NORMAL)
        ts = time.strftime('%H:%M:%S')
        self._log_text.insert(tk.END, f'[{ts}] ', 'time')
        self._log_text.insert(tk.END, f'{text}\n', tag)
        self._log_text.see(tk.END)
        self._log_text.config(state=tk.DISABLED)

    def _clear_log(self) -> None:
        self._log_text.config(state=tk.NORMAL)
        self._log_text.delete('1.0', tk.END)
        self._log_text.config(state=tk.DISABLED)


    # ── Tree rendering ─────────────────────────────────────────────────────

    def _populate_tree(self, groups: List[DupGroup]) -> None:
        for item in self._tree.get_children():
            self._tree.delete(item)

        if not groups:
            self._tree_summary.config(text='No duplicate groups')
            return

        ft = self._var_filter_type.get()
        ftext = self._var_filter_text.get().lower()

        shown = 0
        for gi, g in enumerate(groups):
            # Type filter
            if ft != 'All':
                if ft == 'Exact'    and g.group_type != 'exact':    continue
                if ft == 'Near-Dup' and g.group_type != 'near':     continue
                if ft == 'Hard-Link'and g.group_type != 'hardlink': continue
            # Text filter
            if ftext and not any(ftext in str(f.path).lower() for f in g.files):
                continue

            type_label = {'exact': 'EXACT', 'near': 'NEAR', 'hardlink': 'HARD'}.get(
                g.group_type, g.group_type.upper())
            tag = {'exact': 'exact', 'near': 'near', 'hardlink': 'hardlink'}.get(
                g.group_type, 'near')
            reclaim = _format_size(g.reclaimable_bytes)
            label   = f'Group {gi+1}  ({len(g.files)} files)'

            iid = self._tree.insert('', tk.END,
                iid=str(gi),
                text=label,
                values=(type_label, len(g.files), f'{g.score}%', reclaim),
                tags=(tag,))
            shown += 1

        total_r = _format_size(sum(g.reclaimable_bytes for g in groups))
        self._tree_summary.config(
            text=f'{shown} group(s)  |  ~{total_r} reclaimable')

    def _apply_filter(self) -> None:
        if self.groups:
            self._populate_tree(self.groups)

    def _on_tree_select(self, event=None) -> None:
        sel = self._tree.selection()
        if not sel: return
        try:
            gi = int(sel[0])
            self.current_group = gi
            self._show_group(gi)
        except (ValueError, IndexError):
            pass

    def _on_tree_double_click(self, event=None) -> None:
        """Expand/collapse children (already shown as inline rows)."""
        pass

    def _nav_prev(self) -> None:
        if self.current_group > 0:
            self.current_group -= 1
            self._show_group(self.current_group)
            try: self._tree.selection_set(str(self.current_group))
            except Exception: pass

    def _nav_next(self) -> None:
        if self.current_group < len(self.groups) - 1:
            self.current_group += 1
            self._show_group(self.current_group)
            try: self._tree.selection_set(str(self.current_group))
            except Exception: pass

    def _tree_move(self, delta: int) -> None:
        sel = self._tree.selection()
        if sel:
            children = self._tree.get_children()
            if not children: return
            try:
                idx  = list(children).index(sel[0])
                nidx = max(0, min(len(children)-1, idx+delta))
                nid  = children[nidx]
                self._tree.selection_set(nid)
                self._tree.see(nid)
                try: self._on_tree_select()
                except Exception: pass
            except ValueError:
                pass

    # ── Group detail view (scrollable file cards) ──────────────────────────

    def _clear_detail(self) -> None:
        for w in self._cards_inner.winfo_children():
            w.destroy()
        self._detail_group_lbl.config(text='Select a group from the left panel')
        self._detail_score_lbl.config(text='')

    def _show_group(self, gi: int) -> None:
        if gi < 0 or gi >= len(self.groups): return
        g   = self.groups[gi]
        C   = self.C
        self._right_nb.select(0)   # Switch to detail tab

        self._detail_group_lbl.config(
            text=f'Group {gi+1} of {len(self.groups)}  ·  {len(g.files)} files  ·  {g.group_type.upper()}')
        self._detail_score_lbl.config(
            text=f'Score: {g.score}%  |  ~{_format_size(g.reclaimable_bytes)} reclaimable')

        # Rebuild cards — cap at MAX_CARDS to prevent UI freeze on huge groups
        MAX_CARDS = 200
        for w in self._cards_inner.winfo_children():
            w.destroy()

        files_to_show = g.files[:MAX_CARDS]
        hidden        = len(g.files) - len(files_to_show)

        if hidden > 0:
            banner = tk.Label(
                self._cards_inner,
                text=(f'⚠  Group has {len(g.files):,} files — '
                      f'showing first {MAX_CARDS}.  '
                      f'Use Export to see all {hidden:,} additional files.'),
                bg='#45475a', fg='#f38ba8',
                font=('Arial', 9, 'bold'), pady=6)
            banner.pack(fill=tk.X, padx=8, pady=(6, 2))

        for fi, f in enumerate(files_to_show):
            self._build_file_card(self._cards_inner, g, gi, fi, f, C)

        self._cards_canvas.yview_moveto(0)
        self._cards_inner.update_idletasks()
        self._cards_canvas.configure(
            scrollregion=self._cards_canvas.bbox('all'))

    def _build_file_card(self, parent, g: DupGroup, gi: int,
                          fi: int, f: FileRecord, C: dict) -> None:
        """Build one styled card for a file in a group."""
        sugg     = g.suggestions.get(fi, 'KEEP')
        is_keep  = sugg == 'KEEP'
        card_bg  = C['exact_bg'] if is_keep else C['card_bg']
        bd_color = C['keep_fg'] if is_keep else C['del_fg']

        card = tk.Frame(parent, bg=card_bg, borderwidth=2, relief='solid',
                        highlightthickness=1,
                        highlightbackground=bd_color)
        card.pack(fill=tk.X, padx=8, pady=5)

        # ── Header row ──────────────────────────────────────────
        hr = tk.Frame(card, bg=card_bg)
        hr.pack(fill=tk.X, padx=8, pady=(6, 2))

        action_icon  = '✓' if is_keep else '🗑'
        action_color = C['keep_fg'] if is_keep else C['del_fg']
        action_label = 'KEEP' if is_keep else 'DELETE'

        tk.Label(hr, text=f'{action_icon} [{action_label}]',
                 bg=card_bg, fg=action_color,
                 font=('Arial', 10, 'bold')).pack(side=tk.LEFT)

        # Toggle button
        def _toggle(gi=gi, fi=fi):
            g = self.groups[gi]
            old = g.suggestions.get(fi, 'KEEP')
            g.suggestions[fi] = 'DELETE' if old == 'KEEP' else 'KEEP'
            self._show_group(gi)
            self._report_dirty = True   # lazy: re-render only when tab is clicked
            self._update_status_bar()

        tk.Button(hr, text='Toggle Keep/Delete',
                  command=_toggle,
                  bg=C['accent1'], fg='#ffffff',
                  font=('Arial', 8), relief='flat',
                  padx=6, cursor='hand2').pack(side=tk.RIGHT, padx=4)

        # Open in explorer button
        def _reveal(path=f.path):
            try:
                _open_in_explorer(path)
            except Exception as exc:
                self._dbg(f'[ERROR] Cannot reveal: {exc}', 'error')

        tk.Button(hr, text='📂 Reveal',
                  command=_reveal,
                  bg=C['toolbar_bg'], fg='#ffffff',
                  font=('Arial', 8), relief='flat',
                  padx=6, cursor='hand2').pack(side=tk.RIGHT, padx=4)

        # ── Score bar ────────────────────────────────────────────
        ks = f.keep_score
        score_color = C['success'] if ks >= 70 else (
                      C['warning'] if ks >= 40 else C['danger'])
        tk.Label(hr, text=f'Quality: {ks}/100',
                 bg=card_bg, fg=score_color,
                 font=('Courier', 9, 'bold')).pack(side=tk.LEFT, padx=12)

        # ── Path ─────────────────────────────────────────────────
        path_frame = tk.Frame(card, bg=card_bg)
        path_frame.pack(fill=tk.X, padx=8, pady=(2, 0))
        tk.Label(path_frame, text='📄 Path:', bg=card_bg,
                 fg=C['fg'], font=('Arial', 8, 'bold'), width=10,
                 anchor='e').pack(side=tk.LEFT)
        tk.Label(path_frame, text=str(f.path), bg=card_bg,
                 fg=C['accent1'], font=('Courier', 8), anchor='w',
                 wraplength=500, justify='left').pack(side=tk.LEFT, padx=4)

        # ── Metadata grid ────────────────────────────────────────
        meta = tk.Frame(card, bg=card_bg)
        meta.pack(fill=tk.X, padx=8, pady=(2, 6))

        mtime_str = _ts(f.mtime)
        ctime_str = _ts(f.ctime)

        def _ml(label, val, col, row):
            tk.Label(meta, text=label + ':', bg=card_bg,
                     fg=C['fg'], font=('Arial', 7, 'bold'),
                     width=12, anchor='e').grid(
                         row=row, column=col*2,   padx=(8,0), pady=1, sticky='e')
            tk.Label(meta, text=val, bg=card_bg,
                     fg=C['fg'], font=('Courier', 8),
                     anchor='w').grid(
                         row=row, column=col*2+1, padx=(2,12), pady=1, sticky='w')

        _ml('Size',     _format_size(f.size),  0, 0)
        _ml('Modified', mtime_str,             1, 0)
        _ml('Created',  ctime_str,             2, 0)
        _ml('Ext',      f.ext.upper() or '—', 0, 1)
        if f.magic_type:
            _ml('MIME',     f.magic_type,          1, 1)
        if f.hash:
            _ml('Hash',     f.hash[:24] + '…',     2, 1)

        # ── Exact match marker ───────────────────────────────────
        if g.group_type == 'exact':
            tk.Label(card,
                     text='═══ EXACT CONTENT MATCH (identical bytes) ═══',
                     bg=C['success'], fg='#ffffff',
                     font=('Arial', 8, 'bold')).pack(
                         fill=tk.X, padx=8, pady=(0, 6))
        elif g.group_type == 'hardlink':
            tk.Label(card,
                     text='═══ HARD LINK (same inode) ═══',
                     bg=C['warning'], fg='#1a1a1a',
                     font=('Arial', 8, 'bold')).pack(
                         fill=tk.X, padx=8, pady=(0, 6))

        # ── Thumbnail (Pillow only) ───────────────────────────────
        if HAS_PIL and f.ext.lower() in ('.jpg', '.jpeg', '.png', '.bmp',
                                          '.gif', '.webp', '.tiff'):
            self._try_thumbnail(card, f.path, card_bg)

    def _try_thumbnail(self, parent, path: Path, bg: str) -> None:
        """Attempt to load + display an image thumbnail (non-blocking)."""
        try:
            from PIL import Image as _PILImage, ImageTk as _PILImageTk  # type: ignore
            img = _PILImage.open(str(path))
            img.thumbnail((100, 100), _PILImage.LANCZOS)
            photo = _PILImageTk.PhotoImage(img)
            lbl = tk.Label(parent, image=photo, bg=bg)
            lbl.image = photo  # hold reference
            lbl.pack(side=tk.RIGHT, padx=8, pady=4)
        except Exception:
            pass

    # ── Selection / deletion ───────────────────────────────────────────────

    def _select_all_in_group(self) -> None:
        gi = self.current_group
        if gi < 0 or gi >= len(self.groups): return
        g = self.groups[gi]
        for fi in range(len(g.files)):
            if fi == 0: g.suggestions[fi] = 'KEEP'
            else:        g.suggestions[fi] = 'DELETE'
        self._show_group(gi)
        self._render_full_report()
        self._update_status_bar()

    def _auto_select(self) -> None:
        if not self.groups: return
        if not self._engine:
            # Create a minimal engine just for smart_select (e.g. after session load)
            pq_tmp  = queue.Queue()
            ev_tmp  = threading.Event()
            engine_tmp = ScanEngine(str(Path.home()), self.settings, pq_tmp, ev_tmp)
            engine_tmp.smart_select(self.groups)
        else:
            self._engine.smart_select(self.groups)

        n = sum(1 for g in self.groups
                for s in g.suggestions.values() if s == 'DELETE')
        self._populate_tree(self.groups)
        if self.current_group >= 0:
            self._show_group(self.current_group)
        self._render_full_report()
        self._update_status_bar()
        self._log_msg('info', f'🎯 Auto-select: {n} files marked for deletion')
        self._dbg(f'[SELECT] Auto-select complete — {n} files marked DELETE', 'select')

    def _clear_selection(self) -> None:
        for g in self.groups:
            g.suggestions.clear()
        if self.current_group >= 0:
            self._show_group(self.current_group)
        self._render_full_report()
        self._update_status_bar()
        self._log_msg('info', 'Selection cleared')
        self._dbg('[SELECT] All suggestions cleared', 'select')

    def _delete_selected(self) -> None:
        if not self.groups: return
        marked = [(g, fi, g.files[fi])
                  for g in self.groups
                  for fi, s in g.suggestions.items()
                  if s == 'DELETE' and fi < len(g.files)]
        if not marked:
            messagebox.showinfo('Nothing Selected',
                                'No files marked for deletion.\n'
                                'Use Auto-Select or toggle individual files.')
            return

        total_bytes = sum(f.size for _, _, f in marked)
        lines       = [f'  {f.path}  ({_format_size(f.size)})'
                       for _, _, f in marked[:25]]
        if len(marked) > 25:
            lines.append(f'  … and {len(marked)-25} more')
        msg = (
            f'⚠️  CONFIRM DELETION\n\n'
            f'{len(marked)} file(s) will be moved to TRASH / RECYCLE BIN\n'
            f'Total: {_format_size(total_bytes)}\n\n'
            + '\n'.join(lines)
            + '\n\n'
            'Files are moved to Trash — NOT permanently deleted.\n'
            'You can restore them if needed.\n\n'
            'Continue?'
        )
        if not messagebox.askyesno('Confirm Move to Trash', msg,
                                    icon='warning', default='no'):
            self._dbg('[DELETE] Cancelled by user', 'warn')
            return

        # Do deletions
        ok, fail = 0, 0
        errors = []
        for g, fi, f in marked:
            success, reason = SafeDeleter.to_trash(f.path)
            if success:
                ok += 1
                g.suggestions[fi] = 'DELETED'
                self._dbg(f'[DELETE] ✓ Moved to trash: {f.path.name}', 'warn')
            else:
                fail += 1
                errors.append(f'{f.path}: {reason}')
                self._dbg(f'[DELETE] ✗ FAILED: {f.path.name} — {reason}', 'error')

        msg2 = f'✓ {ok} file(s) moved to Trash.'
        if fail:
            msg2 += f'\n✗ {fail} failed:\n' + '\n'.join(errors[:10])
        messagebox.showinfo('Deletion Complete', msg2)

        # Remove deleted files from groups
        for g in self.groups:
            g.files = [f for fi, f in enumerate(g.files)
                       if g.suggestions.get(fi) != 'DELETED']
            g.suggestions = {}

        self.groups = [g for g in self.groups if len(g.files) > 1]
        self._populate_tree(self.groups)
        self._clear_detail()
        self._render_full_report()
        self._update_status_bar()
        self._refresh_deletion_history()
        self._log_msg('info', f'Deletion complete: {ok} moved, {fail} failed')

    # ── Deletion history ───────────────────────────────────────────────────

    def _refresh_deletion_history(self) -> None:
        log = SafeDeleter.load_log()
        self._del_history_text.config(state=tk.NORMAL)
        self._del_history_text.delete('1.0', tk.END)
        if not log:
            self._del_history_text.insert(tk.END, 'No deletion history yet.\n')
        else:
            for entry in reversed(log):
                ts = entry.get('ts', '')[:19]
                path = entry.get('path', '')
                ok   = entry.get('success', False)
                err  = entry.get('error', '')
                icon = '✓' if ok else '✗'
                stag = 'success' if ok else 'fail'
                self._del_history_text.insert(tk.END, f'{icon} [{ts}] ', stag)
                self._del_history_text.insert(tk.END, f'{path}\n')
                if err:
                    self._del_history_text.insert(tk.END, f'  Error: {err}\n', 'fail')
        self._del_history_text.config(state=tk.DISABLED)

    def _clear_deletion_log(self) -> None:
        if messagebox.askyesno('Clear Log', 'Clear all deletion history?'):
            try:
                DELETION_LOG_PATH.write_text('[]', encoding='utf-8')
            except Exception: pass
            self._refresh_deletion_history()

    def _show_deletion_history(self) -> None:
        self._right_nb.select(4)
        self._refresh_deletion_history()

    # ── Full report renderer ───────────────────────────────────────────────

    def _render_full_report(self) -> None:
        # ── modular: full report renderer (lazy, chunked) ──────────────────
        """Build full report — called lazily when tab is selected.
        Uses update_idletasks() every 10 groups to prevent UI freeze."""
        self._report_dirty = False
        t = self._report_text
        t.config(state=tk.NORMAL)
        t.delete('1.0', tk.END)

        if not self.groups:
            t.insert(tk.END, 'No scan results yet.\n', 'rpt_meta')
            t.config(state=tk.DISABLED)
            return

        ng        = len(self.groups)
        nf        = sum(len(g.files) for g in self.groups)
        rb        = sum(g.reclaimable_bytes for g in self.groups)
        n_marked  = sum(1 for g in self.groups
                        for s in g.suggestions.values() if s == 'DELETE')
        n_exact   = sum(1 for g in self.groups if g.group_type == 'exact')
        n_near    = sum(1 for g in self.groups if g.group_type == 'near')
        n_hard    = sum(1 for g in self.groups if g.group_type == 'hardlink')

        # Header — minimal inserts, fast ─────────────────────────────────────
        t.insert(tk.END,
            f'╔══════════════════════════════════════════════════════════╗\n'
            f'║  DUPLICATE FILE FINDER — FULL REPORT   v{VERSION:<16}║\n'
            f'╚══════════════════════════════════════════════════════════╝\n',
            'rpt_header')
        t.insert(tk.END,
            f'\n  Generated : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
            f'  Groups    : {ng}  |  Files: {nf:,}  |  Reclaimable: {_format_size(rb)}\n'
            f'  Exact     : {n_exact}  |  Near: {n_near}  |  Hard-links: {n_hard}\n'
            f'  Marked DELETE : {n_marked}\n\n',
            'rpt_summary')

        self._report_summary_lbl.config(
            text=f'{ng} groups  ·  {nf:,} files  ·  ~{_format_size(rb)} reclaimable  ·  {n_marked} marked')

        # ── Per-group rendering — chunked to stay responsive ─────────────────
        for gi, g in enumerate(self.groups):
            type_color = 'rpt_group' if g.group_type == 'exact' else (
                         'rpt_hl'   if g.group_type == 'hardlink' else 'rpt_near')

            # Build this group's plain text in one string → single insert ─────
            hdr_line = (f'\n══ GROUP {gi+1:>4} ─── {g.group_type.upper():<10} ─── '
                        f'Score: {g.score}%  ─── {len(g.files)} files  ─── '
                        f'~{_format_size(g.reclaimable_bytes)} reclaimable ══\n')
            t.insert(tk.END, hdr_line, type_color)

            if g.components:
                parts = '  |  '.join(
                    f'{k}: {v:.1f}%' if isinstance(v, float) else f'{k}: {v}'
                    for k, v in g.components.items())
                t.insert(tk.END, f'  Score: {parts}\n', 'rpt_score')

            for fi, f in enumerate(g.files):
                sugg   = g.suggestions.get(fi, 'KEEP')
                s_tag  = 'rpt_keep' if sugg == 'KEEP' else 'rpt_delete'
                s_icon = '✓ KEEP' if sugg == 'KEEP' else '🗑 DELETE'
                # Build file block as one string per section ───────────────────
                t.insert(tk.END, f'\n  [{fi+1}] {s_icon:12} Quality:{f.keep_score:3}/100\n', s_tag)
                meta_block = (
                    f'      Path    : {f.path}\n'
                    f'      Size    : {_format_size(f.size)}\n'
                    f'      Created : {_ts(f.ctime)}\n'
                    f'      Modified: {_ts(f.mtime)}\n'
                )
                if f.hash:
                    meta_block += f'      Hash    : {f.hash[:48]}\n'
                if f.magic_type:
                    meta_block += f'      MIME    : {f.magic_type}\n'
                t.insert(tk.END, meta_block, 'rpt_meta')  # ONE insert for all meta

            t.insert(tk.END, '─' * 72 + '\n', 'rpt_div')

            # Yield to event loop every 10 groups to stay responsive ──────────
            if (gi + 1) % 10 == 0:
                t.update_idletasks()

        t.config(state=tk.DISABLED)
        t.see('1.0')


    # ── Export ─────────────────────────────────────────────────────────────

    def _show_export_menu(self) -> None:
        """Pop a small menu with export options."""
        m = tk.Menu(self.root, tearoff=0)
        m.add_command(label='Export TXT',  command=self._export_txt)
        m.add_command(label='Export CSV',  command=self._export_csv)
        m.add_command(label='Export JSON', command=self._export_json)
        m.add_command(label='Export HTML', command=self._export_html)
        try:
            btn = self._export_btn
            x   = btn.winfo_rootx()
            y   = btn.winfo_rooty() + btn.winfo_height()
            m.tk_popup(x, y)
        finally:
            m.grab_release()

    def _export_txt(self) -> None:
        p = filedialog.asksaveasfilename(
            defaultextension='.txt',
            filetypes=[('Text File', '*.txt'), ('All', '*.*')],
            title='Export Full Report TXT')
        if not p: return
        t = self._report_text
        t.config(state=tk.NORMAL)
        content = t.get('1.0', tk.END)
        t.config(state=tk.DISABLED)
        try:
            Path(p).write_text(content, encoding='utf-8')
            self._log_msg('info', f'✓ TXT report saved: {p}')
            self._dbg(f'[EXPORT] TXT → {p}', 'info')
            messagebox.showinfo('Exported', f'Report saved:\n{p}')
        except Exception as exc:
            messagebox.showerror('Export Error', str(exc))

    def _export_csv(self) -> None:
        p = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv'), ('All', '*.*')],
            title='Export CSV')
        if not p: return
        try:
            import csv
            with open(p, 'w', newline='', encoding='utf-8') as fh:
                w = csv.writer(fh)
                w.writerow(['group', 'type', 'score', 'file_index',
                             'suggestion', 'quality', 'path', 'size',
                             'created', 'modified', 'hash', 'mime'])
                for gi, g in enumerate(self.groups):
                    for fi, f in enumerate(g.files):
                        w.writerow([
                            gi+1, g.group_type, g.score, fi+1,
                            g.suggestions.get(fi, 'KEEP'), f.keep_score,
                            str(f.path), f.size,
                            _ts(f.ctime), _ts(f.mtime),
                            f.hash or '', f.magic_type or ''])
            self._log_msg('info', f'✓ CSV exported: {p}')
            self._dbg(f'[EXPORT] CSV → {p}', 'info')
            messagebox.showinfo('Exported', f'CSV saved:\n{p}')
        except Exception as exc:
            messagebox.showerror('Export Error', str(exc))

    def _export_json(self) -> None:
        p = filedialog.asksaveasfilename(
            defaultextension='.json',
            filetypes=[('JSON', '*.json'), ('All', '*.*')],
            title='Export JSON')
        if not p: return
        try:
            data = []
            for gi, g in enumerate(self.groups):
                data.append({
                    'group':      gi+1,
                    'type':       g.group_type,
                    'score':      g.score,
                    'components': g.components,
                    'reclaimable_bytes': g.reclaimable_bytes,
                    'files': [
                        {
                            'index':      fi+1,
                            'suggestion': g.suggestions.get(fi, 'KEEP'),
                            'quality':    f.keep_score,
                            'path':       str(f.path),
                            'size':       f.size,
                            'created':    _ts(f.ctime),
                            'modified':   _ts(f.mtime),
                            'hash':       f.hash or '',
                            'mime':       f.magic_type or '',
                        }
                        for fi, f in enumerate(g.files)
                    ]
                })
            Path(p).write_text(json.dumps(data, indent=2), encoding='utf-8')
            self._log_msg('info', f'✓ JSON exported: {p}')
            self._dbg(f'[EXPORT] JSON → {p}', 'info')
            messagebox.showinfo('Exported', f'JSON saved:\n{p}')
        except Exception as exc:
            messagebox.showerror('Export Error', str(exc))

    def _export_html(self) -> None:
        p = filedialog.asksaveasfilename(
            defaultextension='.html',
            filetypes=[('HTML', '*.html'), ('All', '*.*')],
            title='Export HTML Report')
        if not p: return
        try:
            rows = []
            for gi, g in enumerate(self.groups):
                for fi, f in enumerate(g.files):
                    sugg   = g.suggestions.get(fi, 'KEEP')
                    c_name = 'keep' if sugg == 'KEEP' else 'delete'
                    rows.append(
                        f'<tr class="{c_name}">'
                        f'<td>{gi+1}</td>'
                        f'<td>{g.group_type}</td>'
                        f'<td>{g.score}%</td>'
                        f'<td>{fi+1}</td>'
                        f'<td>{sugg}</td>'
                        f'<td>{f.keep_score}</td>'
                        f'<td><code>{f.path}</code></td>'
                        f'<td>{_format_size(f.size)}</td>'
                        f'<td>{_ts(f.ctime)}</td>'
                        f'<td>{_ts(f.mtime)}</td>'
                        f'<td><small>{(f.hash or "")[:20]}</small></td>'
                        f'</tr>'
                    )
            html = f'''<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8">
<title>Duplicate Finder Report v{VERSION}</title>
<style>
  body{{font-family:monospace;background:#0d1117;color:#c9d1d9;}}
  table{{border-collapse:collapse;width:100%;}}
  th{{background:#161b22;padding:6px 10px;}}
  td{{padding:4px 10px;border-bottom:1px solid #21262d;}}
  .keep{{background:#1a2e20;}}
  .delete{{background:#2e1a1a;color:#f85149;}}
  h1{{color:#58a6ff;}}
</style>
</head>
<body>
<h1>🔍 Duplicate File Finder — Full Report v{VERSION}</h1>
<p>Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} &nbsp;|&nbsp;
{len(self.groups)} groups &nbsp;|&nbsp;
{sum(len(g.files) for g in self.groups)} files</p>
<table>
<tr><th>Group</th><th>Type</th><th>Score</th><th>#</th>
<th>Suggestion</th><th>Quality</th><th>Path</th><th>Size</th>
<th>Created</th><th>Modified</th><th>Hash</th></tr>
{''.join(rows)}
</table>
</body></html>'''
            Path(p).write_text(html, encoding='utf-8')
            self._log_msg('info', f'✓ HTML exported: {p}')
            self._dbg(f'[EXPORT] HTML → {p}', 'info')
            messagebox.showinfo('Exported', f'HTML report saved:\n{p}')
        except Exception as exc:
            messagebox.showerror('Export Error', str(exc))

    # ── Session save / load ────────────────────────────────────────────────

    def _save_session(self) -> None:
        if not self.groups:
            messagebox.showinfo('Nothing to save', 'Run a scan first.')
            return
        p = filedialog.asksaveasfilename(
            defaultextension='.dupjson',
            filetypes=[('Duplicate Session', '*.dupjson'), ('All', '*.*')],
            title='Save Session')
        if not p: return
        ok = SessionManager.save(
            self.groups,
            self._folder_lbl.cget('text'),
            self.settings, p)
        if ok:
            self._log_msg('info', f'✓ Session saved: {p}')
            self._dbg(f'[SESSION] Saved → {p}', 'info')
            messagebox.showinfo('Session Saved', f'Session saved:\n{p}')
        else:
            messagebox.showerror('Save Failed', 'Could not save session.')

    def _load_session(self) -> None:
        p = filedialog.askopenfilename(
            filetypes=[('Duplicate Session', '*.dupjson'), ('All', '*.*')],
            title='Load Session')
        if not p: return
        groups, folder = SessionManager.load(p)
        if groups is None:
            messagebox.showerror('Load Failed',
                                  'Could not load session. Wrong version or corrupt file.')
            return
        self.groups = groups
        if folder:
            self._folder_lbl.config(text=folder)
        self._populate_tree(self.groups)
        self._render_full_report()
        self._update_status_bar()
        self._auto_sel_btn.config(state=tk.NORMAL)
        self._delete_btn.config(state=tk.NORMAL)
        self._export_btn.config(state=tk.NORMAL)
        self._save_sess_btn.config(state=tk.NORMAL)
        self._clear_sel_btn.config(state=tk.NORMAL)
        self._log_msg('info', f'✓ Session loaded: {len(self.groups)} groups')
        self._dbg(f'[SESSION] Loaded ← {p}  groups={len(self.groups)}', 'info')
        if self.groups:
            self._show_group(0)

    # ── Display / theme settings ───────────────────────────────────────────

    def _toggle_dark_mode(self) -> None:
        self._var_dark_mode.set(not self._var_dark_mode.get())
        self._apply_display_settings()

    def _apply_display_settings(self) -> None:
        dark = self._var_dark_mode.get()
        self._setup_theme(dark=dark)
        # Rebuild the UI
        for w in self.root.winfo_children():
            try: w.destroy()
            except Exception: pass
        self._spinner_idx = 0
        self._build_ui()
        self._bind_shortcuts()
        self._populate_tree(self.groups)
        self._render_full_report()
        self._update_status_bar()
        self._refresh_deletion_history()
        if self.groups and self.current_group >= 0:
            self._show_group(self.current_group)

    # ── Help ───────────────────────────────────────────────────────────────

    def _show_help(self) -> None:
        win = tk.Toplevel(self.root)
        win.title(f'Help — Duplicate Finder v{VERSION}')
        win.geometry('680x520')
        win.configure(bg=self.C['bg'])
        txt = scrolledtext.ScrolledText(
            win, font=('Courier', 9),
            bg=self.C['panel_bg'], fg=self.C['fg'],
            wrap=tk.WORD, borderwidth=0, relief='flat')
        txt.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        txt.insert(tk.END, HELP_TEXT)
        txt.config(state=tk.DISABLED)



# ═════════════════════════════════════════════════════════════════════════════
#  HELP TEXT
# ═════════════════════════════════════════════════════════════════════════════

HELP_TEXT = f"""
DUPLICATE FILE FINDER  v{VERSION}
{'=' * 60}

SCANNING
  ▸ Click "Change Folder" to pick a directory.
  ▸ Press SCAN (or Ctrl+S / F5) to start.
  ▸ Press STOP (or Esc) to abort at any time.

DETECTION METHODS
  EXACT   — Identical hash (xxhash or MD5). 100% match.
  NEAR    — Similar by name, size, metadata, partial hash.
  HARD    — Same inode → same file on disk (hard link).

METADATA CHECKS
  • File size (pre-filter — no hash for unique sizes)
  • Partial hash (first+last 64 KB) for large files
  • Full cryptographic hash (xxhash64 > MD5 fallback)
  • Byte-by-byte verify (paranoid mode)
  • File creation time (ctime)
  • File last-modified time (mtime)
  • MIME / magic-byte type detection
  • Image perceptual hash (pHash, requires Pillow)
  • Name similarity (bigram cosine, optional GPU/NumPy)
  • Hard-link detection (inode + device pair)
  • Temporal proximity bonus (near-simultaneous modification)

AUTO-SELECT SCORING (higher = keep)
  Quality score uses:
    • Name quality (penalises copy/backup/temp patterns)
    • Path depth (shallower = more likely original)
    • File creation date (oldest = original)
    • File size (larger = more complete)
    • MIME confidence
    Adjustable via Settings ▸ Auto-Selection Settings

KEYBOARD SHORTCUTS
  Ctrl+S / F5  — Start scan
  Esc          — Stop scan
  Delete       — Delete selected files
  Left/Right   — Previous/next group
  Up/Down      — Navigate group list
  Ctrl+A       — Select all in current group
  Ctrl+Z       — Show deletion history
  F1           — This help

EXPORT FORMATS
  TXT  — Full plain-text report
  CSV  — Spreadsheet-ready
  JSON — Machine-readable
  HTML — Browser-friendly coloured report

SESSION SAVE/LOAD
  Save your scan results to a .dupjson file and reload
  later without re-scanning. Results include all metadata,
  hashes, scores, and suggestions.

DELETION SAFETY
  Files are NEVER permanently deleted.
  All deletions go through the OS Recycle Bin / Trash
  (requires send2trash, else Windows SHFileOperationW).
  A persistent deletion log is kept at:
  {DELETION_LOG_PATH}

OPTIONAL LIBRARIES (pip install ...)
  xxhash     — ~10× faster hashing
  send2trash — Cross-platform Recycle Bin
  Pillow     — Image thumbnails + pHash
  numpy      — Vectorised name similarity
  cupy       — NVIDIA GPU acceleration (cupy-cuda11x)
  psutil     — System monitoring

NOTE ON GPU ACCELERATION
  GPU (CuPy) is only used for the name-similarity matrix
  (bigram cosine). For file hashing, CPU xxhash is faster
  than any GPU-transfer overhead.
"""


# ═════════════════════════════════════════════════════════════════════════════
#  Open in file explorer — cross-platform
# ═════════════════════════════════════════════════════════════════════════════

def _open_in_explorer(path: Path) -> None:
    try:
        p = Path(path)
        if not p.exists(): return
        if   sys.platform == 'win32':
            subprocess.run(['explorer', '/select,', str(p)], check=False)
        elif sys.platform == 'darwin':
            subprocess.run(['open', '-R', str(p)], check=False)
        else:
            subprocess.run(['xdg-open', str(p.parent)], check=False)
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    multiprocessing.freeze_support()
    root = tk.Tk()
    try:
        root.iconbitmap(default='')
    except Exception:
        pass
    app = DuplicateFinderApp(root)
    root.protocol('WM_DELETE_WINDOW', lambda: _quit(root, app))
    root.mainloop()


def _quit(root: tk.Tk, app: DuplicateFinderApp) -> None:
    if app.is_scanning:
        if not messagebox.askyesno('Quit', 'Scan in progress. Quit anyway?'):
            return
        app._cancel_ev.set()
    root.destroy()


if __name__ == '__main__':
    main()
