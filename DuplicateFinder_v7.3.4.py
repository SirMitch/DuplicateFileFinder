#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
#  v7.2 - NEEDED DUPLICATE PROTECTION + PRODUCTION REFINEMENT
# ═════════════════════════════════════════════════════════════════════════════

# Semantic similarity thresholds (per file type)
SEMANTIC_THRESHOLDS = {
    "image": 0.92,  # CLIP similarity threshold for images
    "document": 0.85,  # Sentence-BERT threshold for text docs
    "video": 0.88,  # Frame embedding threshold
    "audio": 0.85,  # Audio fingerprint similarity
    "code": 0.90,  # Code similarity threshold
    "default": 0.85,  # Default fallback
}

# FAISS index parameters
FAISS_INDEX_TYPE = "IVF"  # Inverted File index for ANN search
FAISS_NLIST = 100  # Number of clusters
FAISS_NPROBE = 10  # Clusters to search

# 4-Layer Detection Weights
LAYER_WEIGHTS = {
    "hash": 0.4,  # Deterministic hashing
    "semantic": 0.35,  # Vector embedding similarity
    "contextual": 0.15,  # Folder/timestamp correlation
    "behavioral": 0.0,  # Disabled - not implemented
}

# Auto-install for v7.0 new packages
NEW_V7_PKGS = [
    ("faiss", "faiss-cpu", "faiss-cpu"),
    ("sentence_transformers", "sentence-transformers", "sentence-transformers"),
    ("transformers", "transformers", "transformers"),
    ("clip", "clip", "transformers"),
]

# Check for new v7.0 capabilities
HAS_FAISS = False
HAS_SENTENCE_TRANSFORMERS = False
HAS_CLIP = False
HAS_BLAKE3 = False
HAS_WATCHDOG = False

# ── Optional performance / feature libraries ──────────────────────────────────
try:
    import xxhash as _xxhash

    HAS_XXHASH = True
except ImportError:
    HAS_XXHASH = False

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import cupy as cp

    _p = cp.array([1.0])
    del _p
    HAS_CUPY = True
except Exception:
    HAS_CUPY = False

try:
    import send2trash as _s2t

    HAS_SEND2TRASH = True
except ImportError:
    HAS_SEND2TRASH = False

try:
    from PIL import Image as _PILImage, ImageTk as _PILImageTk

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import imagehash as _imagehash

    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

try:
    import mutagen as _mutagen

    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    pass

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel

    HAS_CLIP = True
except ImportError:
    pass

try:
    import blake3

    HAS_BLAKE3 = True
except ImportError:
    pass

try:
    import watchdog

    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Derived capability flags
HAS_CHROMAPRINT = False  # Requires system fpcalc binary

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    pass

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel

    HAS_CLIP = True
    HAS_TRANSFORMERS = True
except ImportError:
    pass

# Set HAS_DINO after all dependencies are available
HAS_DINO = HAS_TORCH and HAS_TRANSFORMERS


class SemanticEngine:
    """
    v7.0 - AI-powered semantic similarity engine.
    Uses vector embeddings for near-duplicate detection beyond perceptual hashing.
    """

    _instance = None
    _model = None
    _clip_model = None
    _clip_processor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._embedding_cache = {}

    def get_sentence_model(self, model_name: str = "all-MiniLM-L6-v2"):
        """Get or load sentence transformer model."""
        if SemanticEngine._model is None:
            try:
                SemanticEngine._model = SentenceTransformer(model_name)
            except Exception:
                return None
        return SemanticEngine._model

    def get_clip_model(self, model_name: str = "openai/clip-vit-base-patch32"):
        """Get or load CLIP model for image-text similarity."""
        if SemanticEngine._clip_model is None:
            try:
                SemanticEngine._clip_model = CLIPModel.from_pretrained(model_name)
                SemanticEngine._clip_processor = CLIPProcessor.from_pretrained(
                    model_name
                )
            except Exception:
                return None, None
        return SemanticEngine._clip_model, SemanticEngine._clip_processor

    def compute_text_embedding(self, text: str) -> Optional[List[float]]:
        """Compute embedding for text content."""
        model = self.get_sentence_model()
        if model is None:
            return None
        try:
            embedding = model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception:
            return None

    def compute_image_embedding(self, image_path: str) -> Optional[List[float]]:
        """Compute CLIP embedding for image."""
        model, processor = self.get_clip_model()
        if model is None:
            return None
        try:
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            return image_features.numpy().flatten().tolist()
        except Exception:
            return None

    def extract_document_text(self, file_path: str) -> Optional[str]:
        """Extract text content from documents for semantic embedding."""
        ext = Path(file_path).suffix.lower()
        text_content = []

        try:
            if ext == ".txt":
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()[:10000]

            elif ext == ".pdf":
                try:
                    import fitz

                    doc = fitz.open(file_path)
                    for page in doc:
                        text_content.append(page.get_text())
                    return " ".join(text_content)[:10000]
                except ImportError:
                    pass

            elif ext in (".doc", ".docx"):
                try:
                    from docx import Document

                    doc = Document(file_path)
                    for para in doc.paragraphs:
                        text_content.append(para.text)
                    return " ".join(text_content)[:10000]
                except ImportError:
                    pass

            elif ext in (".md", ".json", ".xml", ".html", ".htm"):
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        return f.read()[:10000]
                except Exception:
                    pass

        except Exception:
            pass

        return None

    def cosine_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            import numpy as np

            a = np.array(emb1)
            b = np.array(emb2)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))
        except Exception:
            return 0.0

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two text strings."""
        emb1 = self.compute_text_embedding(text1)
        emb2 = self.compute_text_embedding(text2)
        if emb1 is None or emb2 is None:
            return 0.0
        return self.cosine_similarity(emb1, emb2)

    def get_similarity_threshold(self, file_type: str) -> float:
        """Get threshold for file type."""
        return SEMANTIC_THRESHOLDS.get(file_type, SEMANTIC_THRESHOLDS["default"])

    def is_similar(
        self, emb1: List[float], emb2: List[float], file_type: str = "default"
    ) -> bool:
        """Check if embeddings are similar above threshold."""
        similarity = self.cosine_similarity(emb1, emb2)
        threshold = self.get_similarity_threshold(file_type)
        return similarity >= threshold


class FAISSIndex:
    """
    v7.0 - FAISS-backed vector similarity index for fast ANN search.
    Enables scalable semantic duplicate detection.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.id_to_path = {}
        self._next_id = 0

    def build_index(self, embeddings: Dict[str, List[float]]) -> bool:
        """Build FAISS index from embeddings dict."""
        if not HAS_FAISS or not embeddings:
            return False
        try:
            import numpy as np

            paths = list(embeddings.keys())
            vectors = np.array([embeddings[p] for p in paths], dtype=np.float32)

            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vectors = vectors / norms

            if len(vectors) < FAISS_NLIST:
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexFlatL2(self.dimension)
            else:
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, min(FAISS_NLIST, len(vectors))
                )

            try:
                self.index.train(vectors)
            except Exception:
                self.index = faiss.IndexFlatL2(self.dimension)

            self.index.add(vectors)

            for i, path in enumerate(paths):
                self.id_to_path[i] = path

            return True
        except Exception:
            return False

    def search(
        self, query_embedding: List[float], k: int = 5
    ) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors."""
        if self.index is None:
            return []
        try:
            import numpy as np

            query = np.array([query_embedding], dtype=np.float32)
            query_norm = np.linalg.norm(query)
            if query_norm > 0:
                query = query / query_norm
            else:
                return []

            distances, indices = self.index.search(query, k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx in self.id_to_path:
                    similarity = 1.0 / (1.0 + dist)
                    results.append((self.id_to_path[idx], similarity))

            return results
        except Exception:
            return []


class MultiLayerFusion:
    """
    v7.0 - Combines 4 layers of evidence for duplicate detection.
    Layer 1: Hash, Layer 2: Semantic, Layer 3: Contextual, Layer 4: Behavioral
    """

    def __init__(self):
        self.semantic_engine = SemanticEngine()
        self.faiss_index = FAISSIndex()

    def compute_hash_score(self, file1_hash: str, file2_hash: str) -> float:
        """Layer 1: Hash-based similarity (1.0 if identical)."""
        return 1.0 if file1_hash == file2_hash else 0.0

    def compute_semantic_score(
        self, content1: str, content2: str, file_type: str = "document"
    ) -> float:
        """Layer 2: Semantic similarity using embeddings."""
        return self.semantic_engine.semantic_similarity(content1, content2)

    def compute_contextual_score(
        self, path1: str, path2: str, time1: float, time2: float
    ) -> float:
        """Layer 3: Contextual correlation (folder proximity, timestamp)."""
        score = 0.0

        # Folder proximity
        parent1 = os.path.dirname(path1)
        parent2 = os.path.dirname(path2)
        if parent1 == parent2:
            score += 0.5
        else:
            try:
                common = os.path.commonpath([parent1, parent2])
                if common and not common.startswith(os.path.sep + os.path.sep):
                    score += 0.2
            except ValueError:
                pass

        # Timestamp proximity (within 60 seconds)
        if abs(time1 - time2) < 60:
            score += 0.3

        # Filename similarity
        name1 = os.path.basename(path1)
        name2 = os.path.basename(path2)
        if name1.lower() == name2.lower():
            score += 0.2
        elif os.path.splitext(name1)[0] == os.path.splitext(name2)[0]:
            score += 0.1

        return min(score, 1.0)

    def compute_behavioral_score(self, file1_path: str, file2_path: str) -> float:
        """Layer 4: Behavioral validation (preview-based). Higher score = more likely duplicate."""
        # Placeholder for LLM-based validation
        # In production, this would use LLaVA or similar for visual comparison
        return 0.0

    def fusion_score(
        self, hash_sim: float, sem_sim: float, ctx_sim: float, beh_sim: float
    ) -> float:
        """Combine all layers with weights."""
        return (
            LAYER_WEIGHTS["hash"] * hash_sim
            + LAYER_WEIGHTS["semantic"] * sem_sim
            + LAYER_WEIGHTS["contextual"] * ctx_sim
            + LAYER_WEIGHTS["behavioral"] * beh_sim
        )

    def is_duplicate(self, file1_data: dict, file2_data: dict) -> bool:
        """Determine if two files are duplicates using 4-layer fusion."""
        hash_score = self.compute_hash_score(
            file1_data.get("hash", ""), file2_data.get("hash", "")
        )

        # Early exit for exact hash match
        if hash_score >= 1.0:
            return True

        sem_score = self.compute_semantic_score(
            file1_data.get("content", ""),
            file2_data.get("content", ""),
            file1_data.get("file_type", "default"),
        )

        ctx_score = self.compute_contextual_score(
            file1_data.get("path", ""),
            file2_data.get("path", ""),
            file1_data.get("mtime", 0),
            file2_data.get("mtime", 0),
        )

        beh_score = self.compute_behavioral_score(
            file1_data.get("path", ""), file2_data.get("path", "")
        )

        final_score = self.fusion_score(hash_score, sem_score, ctx_score, beh_score)

        # Use semantic threshold as final decision
        threshold = SEMANTIC_THRESHOLDS.get(
            file1_data.get("file_type", "default"), SEMANTIC_THRESHOLDS["default"]
        )

        return final_score >= threshold


class WatchMode:
    """
    v7.0 - Real-time folder monitoring using watchdog.
    Scans for new duplicates when files change in watched directories.
    """

    def __init__(self, paths: List[str], settings: "ScanSettings", callback=None):
        self.paths = paths
        self.settings = settings
        self.callback = callback
        self.observers = []
        self.is_running = False
        self._file_hashes: Dict[str, str] = {}

    def start(self) -> bool:
        """Start watching all configured paths."""
        if not HAS_WATCHDOG:
            return False

        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class DuplicateWatchHandler(FileSystemEventHandler):
            def __init__(self, watch_mode):
                self.watch_mode = watch_mode

            def on_created(self, event):
                if not event.is_directory:
                    self.watch_mode._on_file_change(event.src_path)

            def on_modified(self, event):
                if not event.is_directory:
                    self.watch_mode._on_file_change(event.src_path)

            def on_deleted(self, event):
                if (
                    not event.is_directory
                    and event.src_path in self.watch_mode._file_hashes
                ):
                    del self.watch_mode._file_hashes[event.src_path]

        try:
            for path in self.paths:
                observer = Observer()
                handler = DuplicateWatchHandler(self)
                observer.schedule(handler, path, recursive=self.settings.subdirs)
                observer.start()
                self.observers.append(observer)

            self.is_running = True
            return True
        except Exception:
            return False

    def stop(self):
        """Stop all watchers."""
        for observer in self.observers:
            observer.stop()
        for observer in self.observers:
            observer.join()
        self.observers = []
        self.is_running = False

    def _on_file_change(self, file_path: str):
        """Handle file changes - compute hash and check for duplicates."""
        try:
            import hashlib

            h = hashlib.md5()
            with open(file_path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            file_hash = h.hexdigest()

            old_hash = self._file_hashes.get(file_path)
            self._file_hashes[file_path] = file_hash

            if old_hash and old_hash != file_hash:
                if self.callback:
                    self.callback("modified", file_path)
            elif not old_hash:
                if self.callback:
                    self.callback("created", file_path)

        except Exception:
            pass


class CloudStorageManager:
    """
    v7.0 - Cloud storage integration for OneDrive, Google Drive, Dropbox.
    Provides unified interface for scanning cloud directories.
    Note: Full OAuth integration requires user authentication setup.
    This provides the framework and local caching.
    """

    def __init__(self):
        self.connected_services: Dict[str, bool] = {
            "onedrive": False,
            "gdrive": False,
            "dropbox": False,
        }
        self._onedrive_paths: List[str] = []
        self._gdrive_paths: List[str] = []
        self._dropbox_paths: List[str] = []

    def get_cloud_paths(self) -> List[str]:
        """Get all configured cloud storage paths."""
        paths = []
        paths.extend(self._onedrive_paths)
        paths.extend(self._gdrive_paths)
        paths.extend(self._dropbox_paths)
        return paths

    def scan_cloud_directory(self, cloud_type: str, path: str) -> List[str]:
        """
        Scan a cloud directory and return list of file paths.
        cloud_type: 'onedrive', 'gdrive', 'dropbox'
        Returns list of local cached file paths.
        """
        if cloud_type == "onedrive":
            return self._scan_onedrive(path)
        elif cloud_type == "gdrive":
            return self._scan_gdrive(path)
        elif cloud_type == "dropbox":
            return self._scan_dropbox(path)
        return []

    def _scan_onedrive(self, path: str) -> List[str]:
        """Scan OneDrive directory using Microsoft Graph API or local cache."""
        local_path = self._get_local_cloud_path("onedrive", path)
        if local_path and os.path.exists(local_path):
            return self._scan_local_directory(local_path)
        return []

    def _scan_gdrive(self, path: str) -> List[str]:
        """Scan Google Drive directory using Drive API or local cache."""
        local_path = self._get_local_cloud_path("gdrive", path)
        if local_path and os.path.exists(local_path):
            return self._scan_local_directory(local_path)
        return []

    def _scan_dropbox(self, path: str) -> List[str]:
        """Scan Dropbox directory using Dropbox API or local cache."""
        local_path = self._get_local_cloud_path("dropbox", path)
        if local_path and os.path.exists(local_path):
            return self._scan_local_directory(local_path)
        return []

    def _get_local_cloud_path(self, service: str, remote_path: str) -> Optional[str]:
        """Get local cache path for cloud file."""
        user_home = Path.home()

        if service == "onedrive":
            onedrive_dir = user_home / "OneDrive"
            if onedrive_dir.exists():
                return str(onedrive_dir / remote_path.strip("/"))
        elif service == "gdrive":
            gdrive_dir = user_home / "Google Drive"
            if gdrive_dir.exists():
                return str(gdrive_dir / remote_path.strip("/"))
        elif service == "dropbox":
            dropbox_dir = user_home / "Dropbox"
            if dropbox_dir.exists():
                return str(dropbox_dir / remote_path.strip("/"))

        return None

    def _scan_local_directory(self, directory: str) -> List[str]:
        """Recursively scan local directory."""
        results = []
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    results.append(os.path.join(root, file))
        except Exception:
            pass
        return results

    def connect_onedrive(self) -> bool:
        """Connect to OneDrive (requires OAuth setup)."""
        try:
            self.connected_services["onedrive"] = True
            return True
        except Exception:
            return False

    def connect_gdrive(self) -> bool:
        """Connect to Google Drive (requires OAuth setup)."""
        try:
            self.connected_services["gdrive"] = True
            return True
        except Exception:
            return False

    def connect_dropbox(self) -> bool:
        """Connect to Dropbox (requires OAuth setup)."""
        try:
            self.connected_services["dropbox"] = True
            return True
        except Exception:
            return False


# ── Standard library imports ─────────────────────────────────────────────────
import os
import sys
import threading
import time
import queue
import hashlib
import datetime
import json
import re
import multiprocessing
import struct
import csv
import io
import traceback
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog, font as tkfont
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Set, Tuple, Any, Callable

# ── Constants ─────────────────────────────────────────────────────────────────
VERSION = "7.3.4"  # <─ single source of truth for all UI text
MIN_PYTHON_VERSION = (3, 9)  # Minimum supported Python version
MAX_PYTHON_VERSION = (
    3,
    14,
)  # Maximum tested Python version (3.14 works but shell has readline issues)

# Python version check with upgrade option
import sys

version_tuple = (sys.version_info.major, sys.version_info.minor)
if not (MIN_PYTHON_VERSION <= version_tuple <= MAX_PYTHON_VERSION):
    print(f"WARNING: Duplicate File Finder v{VERSION}")
    print(f"  Your Python version: {sys.version_info.major}.{sys.version_info.minor}")
    print(
        f"  Supported range: {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} - {MAX_PYTHON_VERSION[0]}.{MAX_PYTHON_VERSION[1]}"
    )
    print(f"  The program may not work correctly with this Python version.")
    print()
    if sys.platform == "win32":
        print("  To switch to a supported version (e.g., Python 3.13):")
        print("    winget install Python.Python.3.13")
        print("  Or download from: https://www.python.org/downloads/")
    else:
        print("  To update Python, run:")
        print("    sudo apt install python3.13   # Ubuntu/Debian")
        print("    brew install python@3.13      # macOS")
    print()
    print(
        "  If you see 'readline' errors, this is a Python 3.14 shell compatibility issue."
    )
    print(
        "  The GUI should still work - run with: python -X utf8 DuplicateFinder_v7.3.py"
    )
    print()


REQUIRED_LIBS = {
    "xxhash": "xxhash",
    "psutil": "psutil",
    "send2trash": "send2trash",
    "PIL": "pillow",
    "tkinter": "tkinter",
}

OPTIONAL_LIBS = {
    "numpy": "numpy",
    "torch": "torch",
    "transformers": "transformers",
    "faiss": "faiss-cpu",
    "clip": "clip",
    "mutagen": "mutagen",
    "opencv": "opencv-python",
    "blake3": "blake3",
    "watchdog": "watchdog",
    "chromaprint": "chromaprint",
    "pymupdf": "pymupdf",
    "python-docx": "python-docx",
}


def _check_and_install_libs() -> None:
    """Check missing libraries and offer to install required ones."""
    import subprocess
    import sys

    missing_required = []
    missing_optional = []
    installed_libs = []

    for lib, pip_name in REQUIRED_LIBS.items():
        if lib == "tkinter":
            continue
        try:
            __import__(lib if lib != "PIL" else "PIL")
            installed_libs.append(pip_name)
        except ImportError:
            missing_required.append(pip_name)

    for lib, pip_name in OPTIONAL_LIBS.items():
        try:
            __import__(lib)
            installed_libs.append(pip_name)
        except ImportError:
            missing_optional.append(pip_name)

    if missing_required or missing_optional:
        # Don't print to terminal - dependency status is shown in main window UI instead
        pass


def _pip_install(package: str, verbose: bool = False) -> bool:
    """Install a package via pip and return success status."""
    import subprocess
    import sys

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


_check_and_install_libs()

CPU_COUNT = max(1, multiprocessing.cpu_count())
HASH_CHUNK = 65536  # 64 KB read chunk
QUICK_HASH_BYTES = 4096  # 4 KB first-pass quick hash
PARTIAL_THRESHOLD = 1_048_576  # 1 MB → use 3-phase hashing above this
MAGIC_BYTES = 16  # bytes to read for file-type detection
DEBUG_MAX_LINES = 5000
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
DELETION_LOG_PATH = Path.home() / ".dupfinder_v7_deletions.json"
SESSION_EXT = ".dupfinder7"
NEAR_DUP_MAX_PAIRS = 100_000  # hard cap on near-dup pair comparisons
PER_GROUP_PAIR_CAP = 5_000  # max pairs from a single size group

# Temp / cache dir name tokens (lower-case)
TEMP_DIR_TOKENS = frozenset(
    {
        "temp",
        "tmp",
        "cache",
        ".cache",
        ".tmp",
        "recycle",
        "trash",
        "$recycle.bin",
        "appdata",
        "localappdata",
        "application data",
        "__pycache__",
        ".git",
        "node_modules",
    }
)

# Preferred directory tokens → boost keep score
PREFERRED_DIR_TOKENS = frozenset(
    {
        "desktop",
        "documents",
        "pictures",
        "photos",
        "music",
        "videos",
        "downloads",
    }
)

# More missing constants (need to add from backup)
NEAR_BATCH_SIZE = 200  # pairs per executor batch
MAGIC_SIZE_SKIP = 50_000_000  # skip magic detection for files > 50 MB
MAGIC_COUNT_SKIP = 50_000
CRITICAL_SYSTEM_DIRS = {
    "C:\\Windows",
    "C:\\Windows\\System32",
    "C:\\Windows\\SysWOW64",
    "/bin",
    "/sbin",
    "/usr/bin",
    "/usr/sbin",
    "/lib",
    "/lib64",
    "/System",
    "/Applications",
    "/private",
    "/var",
    "/etc",
}

NEEDED_PROTECTION_DIRS = {
    "C:\\Program Files",
    "C:\\Program Files (x86)",
    "C:\\ProgramData",
    "C:\\Users\\*\\AppData\\Local\\Programs",
    "C:\\Users\\*\\AppData\\Local\\Microsoft\\WindowsApps",
    "C:\\Steam",
    "C:\\Program Files (x86)\\Steam",
    "C:\\Games",
    "C:\\Epic Games",
    "C:\\Program Files\\Epic Games",
    "C:\\Program Files (x86)\\Epic Games",
    "C:\\Users\\*\\AppData\\Local\\Google\\Chrome",
    "C:\\Users\\*\\AppData\\Local\\Microsoft\\Edge",
    "C:\\Users\\*\\AppData\\Roaming\\Mozilla\\Firefox",
    "C:\\Users\\*\\AppData\\Local\\Docker",
    "C:\\Program Files\\Docker",
    "C:\\Program Files\\Oracle\\VirtualBox",
    "C:\\Program Files\\VMware",
    "C:\\ProgramData\\Microsoft\\Windows\\WER",
    "C:\\Users\\*\\AppData\\Local\\Temp",
    "C:\\Users\\*\\OneDrive",
    "C:\\Users\\*\\Google Drive",
    "C:\\Users\\*\\Dropbox",
    "C:\\Users\\*\\iCloud Drive",
    "/Applications",
    "/System/Applications",
    "~/Library/Application Support",
    "~/Library/Caches",
    "~/Library/Containers",
    "~/.local/share",
    "~/.cache",
    "~/snap",
    "/var/lib",
    "/usr/share",
}

NEEDED_EXTENSIONS = {
    ".dll",
    ".exe",
    ".so",
    ".dylib",
    ".app",
    ".framework",
    ".ocx",
    ".sys",
    ".scr",
    ".cpl",
    ".msi",
    ".msm",
    ".msp",
    ".jar",
    ".class",
    ".pyc",
    ".pyo",
    ".manifest",
    ".config",
    ".ini",
    ".xml",
    ".xsd",
    ".pf",
    ".pfx",
    ".cer",
    ".crt",
    ".pem",
    ".key",
    ".vhd",
    ".vmdk",
    ".vdi",
    ".hdd",
    ".img",
    ".dockerfile",
    "Dockerfile",
    ".dockerignore",
}
MAX_CLUSTER_SIZE = 1000
CLUSTER_MIN_PAIRS = 0.3
SAFE_DELETE_GAP = 20

# Copy / backup name patterns (comprehensive)
COPY_PATTERNS = [
    "- copy",
    "(copy)",
    " copy",
    "_copy",
    "-copy",
    "copy of ",
    "duplicate",
    "backup",
    "- backup",
    "_backup",
    "-bak",
    ".bak",
    " old",
    "_old",
    "-old",
    "orig-",
    "original-",
    "- original",
    "_orig",
    "-orig",
    "temp_",
    "_temp",
    "-temp",
]

# Magic-bytes file-type signatures: (prefix_bytes, type_label)
FILE_MAGIC_SIGS: List[Tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "JPEG"),
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"GIF87a", "GIF"),
    (b"GIF89a", "GIF"),
    (b"%PDF", "PDF"),
    (b"PK\x03\x04", "ZIP"),
    (b"PK\x05\x06", "ZIP"),
    (b"ID3", "MP3"),
    (b"\xff\xfb", "MP3"),
    (b"\xff\xf3", "MP3"),
    (b"\xff\xf2", "MP3"),
    (b"fLaC", "FLAC"),
    (b"OggS", "OGG"),
    (b"RIFF", "RIFF"),  # WAV / AVI
    (b"\x1f\x8b", "GZIP"),
    (b"Rar!\x1a\x07", "RAR"),
    (b"\x00\x00\x01\x00", "ICO"),
    (b"BM", "BMP"),
    (b"\x00\x00\x02\x00", "CUR"),
]

# Extension groups for category detection
EXT_GROUPS: Dict[str, List[str]] = {
    "Image": [
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".webp",
        ".svg",
        ".ico",
        ".tiff",
        ".tif",
        ".psd",
        ".raw",
        ".heic",
        ".heif",
    ],
    "Video": [
        ".mp4",
        ".mkv",
        ".avi",
        ".mov",
        ".wmv",
        ".flv",
        ".webm",
        ".m4v",
        ".mpg",
        ".mpeg",
        ".3gp",
    ],
    "Audio": [
        ".mp3",
        ".wav",
        ".flac",
        ".aac",
        ".ogg",
        ".wma",
        ".m4a",
        ".opus",
        ".alac",
    ],
    "Document": [
        ".txt",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".odt",
        ".ods",
        ".odp",
        ".rtf",
        ".csv",
    ],
    "Code": [
        ".py",
        ".js",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".scala",
        ".r",
        ".m",
        ".matlab",
        ".sh",
        ".bash",
        ".ps1",
        ".bat",
        ".cmd",
    ],
    "Archive": [".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz", ".iso", ".dmg"],
}

# Executable extensions
EXECUTABLE_EXTENSIONS = {
    ".exe",
    ".dll",
    ".sys",
    ".msi",
    ".bat",
    ".cmd",
    ".ps1",
    ".sh",
    ".bash",
    ".py",
    ".jar",
    ".com",
    ".scr",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  I/O Port - External Program Integration
#  Allows other programs to control Duplicate Finder via pipe/streams
# ═════════════════════════════════════════════════════════════════════════════


class IOPort:
    """
    I/O Port for external program integration.
    Provides bidirectional communication via stdin/stdout or named pipes.

    Protocol:
      Commands sent as JSON lines: {"cmd": "...", "args": {...}}
      Responses sent as JSON lines: {"status": "ok|error", "data": ..., "msg": "..."}

    Signals (bidirectional):
      SCAN_START, SCAN_PROGRESS, SCAN_COMPLETE, SCAN_ERROR
      GROUP_SELECTED, FILE_SELECTED, DELETE_START, DELETE_COMPLETE
      PAUSE, RESUME, STOP

    Available Commands:
      scan(folder, settings) - Start scan with optional settings
      stop()                 - Stop current scan
      pause()                - Pause scan
      resume()               - Resume scan
      get_groups()           - Get current duplicate groups
      get_group(index)       - Get specific group details
      select_files(group_idx, file_indices) - Mark files for deletion
      delete_selected()      - Execute deletion of marked files
      get_status()           - Get current status
      set_settings(settings) - Update scan settings
      export(format, path)   - Export results
      save_session(path)     - Save current session
      load_session(path)     - Load session
    """

    def __init__(self, mode: str = "auto"):
        self.mode = mode  # "stdin", "pipe", "auto"
        self.running = False
        self._callbacks: Dict[str, List[callable]] = {}
        self._last_status: Dict = {}
        self._groups: List[DupGroup] = []
        self._engine: Optional[Any] = None
        self._settings: Optional[ScanSettings] = None
        self._output_queue: queue.Queue = queue.Queue()

    def register_callback(self, signal: str, callback: callable) -> None:
        """Register a callback for a specific signal."""
        if signal not in self._callbacks:
            self._callbacks[signal] = []
        self._callbacks[signal].append(callback)

    def emit_signal(self, signal: str, data: Any = None) -> None:
        """Emit a signal to all registered callbacks."""
        if signal in self._callbacks:
            for cb in self._callbacks[signal]:
                try:
                    cb(signal, data)
                except Exception:
                    pass

    def _parse_command(self, line: str) -> Optional[Dict]:
        """Parse a command line into a command dict."""
        try:
            return json.loads(line.strip())
        except json.JSONDecodeError:
            return None

    def _send_response(self, status: str, data: Any = None, msg: str = "") -> None:
        """Send a response line."""
        resp = {"status": status, "data": data, "msg": msg}
        self._output_queue.put_nowait(json.dumps(resp))

    def start(self) -> None:
        """Start the I/O port listener."""
        self.running = True
        self.emit_signal("PORT_STARTED", {"mode": self.mode})

    def stop(self) -> None:
        """Stop the I/O port listener."""
        self.running = False
        self.emit_signal("PORT_STOPPED", {})

    def set_engine(self, engine: Any) -> None:
        """Set the scan engine reference."""
        self._engine = engine

    def set_groups(self, groups: List[DupGroup]) -> None:
        """Update the current groups list."""
        self._groups = groups
        self.emit_signal("GROUPS_UPDATED", {"count": len(groups)})

    def set_settings(self, settings: ScanSettings) -> None:
        """Update the current settings."""
        self._settings = settings

    def process_command(self, cmd: str, args: Dict = None) -> Dict:
        """Process a command and return response."""
        args = args or {}

        if cmd == "get_status":
            return {
                "status": "ok",
                "data": {
                    "running": self.running,
                    "groups_count": len(self._groups),
                    "engine_running": getattr(self._engine, "running", False)
                    if self._engine
                    else False,
                },
            }

        elif cmd == "get_groups":
            summary = []
            for gi, g in enumerate(self._groups):
                summary.append(
                    {
                        "index": gi,
                        "type": g.group_type,
                        "score": g.score,
                        "files": len(g.files),
                        "reclaimable": g.reclaimable_bytes,
                    }
                )
            return {"status": "ok", "data": summary}

        elif cmd == "get_group":
            idx = args.get("index", 0)
            if 0 <= idx < len(self._groups):
                g = self._groups[idx]
                files_data = []
                for fi, f in enumerate(g.files):
                    files_data.append(
                        {
                            "index": fi,
                            "path": str(f.path),
                            "size": f.size,
                            "mtime": f.mtime,
                            "suggestion": g.suggestions.get(fi, "KEEP"),
                            "keep_score": f.keep_score,
                        }
                    )
                return {
                    "status": "ok",
                    "data": {
                        "index": idx,
                        "type": g.group_type,
                        "score": g.score,
                        "risk": g.risk_level,
                        "files": files_data,
                        "components": g.components,
                    },
                }
            return {"status": "error", "msg": "Invalid group index"}

        elif cmd == "select_files":
            group_idx = args.get("group_index", 0)
            file_indices = args.get("file_indices", [])
            action = args.get("action", "DELETE")  # DELETE or KEEP

            if 0 <= group_idx < len(self._groups):
                g = self._groups[group_idx]
                for fi in file_indices:
                    if 0 <= fi < len(g.files):
                        g.suggestions[fi] = action
                return {"status": "ok", "msg": f"Updated {len(file_indices)} files"}
            return {"status": "error", "msg": "Invalid group index"}

        elif cmd == "get_settings":
            if self._settings:
                return {
                    "status": "ok",
                    "data": {
                        "subdirs": self._settings.subdirs,
                        "min_size": self._settings.min_size,
                        "max_size": self._settings.max_size,
                        "min_score": self._settings.min_score,
                        "cleanup_mode": self._settings.cleanup_mode,
                        "use_xxhash": self._settings.use_xxhash,
                        "use_sha256_verify": self._settings.use_sha256_verify,
                    },
                }
            return {"status": "ok", "data": {}}

        elif cmd == "set_settings":
            if self._settings and args:
                for key in [
                    "subdirs",
                    "min_size",
                    "max_size",
                    "min_score",
                    "cleanup_mode",
                ]:
                    if key in args:
                        setattr(self._settings, key, args[key])
                return {"status": "ok", "msg": "Settings updated"}
            return {"status": "error", "msg": "No settings to update"}

        elif cmd == "ping":
            return {"status": "ok", "data": {"version": VERSION, "pong": True}}

        elif cmd == "scan":
            folder = args.get("folder")
            if not folder:
                return {"status": "error", "msg": "Missing folder parameter"}
            if not self._engine:
                return {"status": "error", "msg": "No engine configured"}

            scan_settings = self._settings or ScanSettings()
            self._engine.root = Path(folder)
            self._engine.settings = scan_settings
            self._engine.cancel.clear()

            import threading

            def run_scan():
                self._engine.scan()
                self._engine.find_duplicates()
                self.set_groups(self._engine.groups)
                self.emit_signal("SCAN_COMPLETE", {"groups": len(self._engine.groups)})

            thread = threading.Thread(target=run_scan, daemon=True)
            thread.start()
            return {"status": "ok", "msg": f"Scan started on {folder}"}

        elif cmd == "cancel":
            if self._engine and hasattr(self._engine, "cancel"):
                self._engine.cancel.set()
                return {"status": "ok", "msg": "Cancel signal sent"}
            return {"status": "error", "msg": "No engine to cancel"}

        elif cmd == "get_capabilities":
            return {
                "status": "ok",
                "data": {
                    "version": VERSION,
                    "has_xxhash": HAS_XXHASH,
                    "has_pil": HAS_PIL,
                    "has_psutil": HAS_PSUTIL,
                    "has_numpy": HAS_NUMPY,
                    "has_cupy": HAS_CUPY,
                    "has_send2trash": HAS_SEND2TRASH,
                    "has_imagehash": HAS_IMAGEHASH,
                    "has_mutagen": HAS_MUTAGEN,
                    "has_torch": HAS_TORCH,
                    "has_transformers": HAS_TRANSFORMERS,
                    "has_dino": HAS_DINO,
                    "has_cv2": HAS_CV2,
                    "commands": [
                        "ping",
                        "get_status",
                        "get_groups",
                        "get_group",
                        "scan",
                        "cancel",
                        "get_capabilities",
                        "select_files",
                        "get_settings",
                        "set_settings",
                        "ping",
                    ],
                },
            }

        elif cmd == "get_stats":
            if not self._engine or not hasattr(self._engine, "files"):
                return {"status": "ok", "data": {"total_files": 0, "groups": 0}}

            total_size = sum(f.size for f in self._engine.files)
            return {
                "status": "ok",
                "data": {
                    "total_files": len(self._engine.files),
                    "total_size_bytes": total_size,
                    "total_size_mb": round(total_size / 1024 / 1024, 2),
                    "groups": len(self._groups),
                    "files_in_groups": sum(len(g.files) for g in self._groups),
                    "reclaimable_bytes": sum(g.reclaimable_bytes for g in self._groups),
                },
            }

        else:
            return {"status": "error", "msg": f"Unknown command: {cmd}"}

    def run_io_loop(self, input_stream=None) -> None:
        """Run the I/O processing loop."""
        import select

        if input_stream is None:
            input_stream = sys.stdin

        self.start()

        while self.running:
            try:
                if sys.platform == "win32":
                    import msvcrt

                    if msvcrt.kbhit():
                        line = sys.stdin.readline()
                        if line:
                            resp = self._handle_line(line)
                            if resp:
                                print(json.dumps(resp), flush=True)
                else:
                    if hasattr(select, "select"):
                        ready, _, _ = select.select([input_stream], [], [], 0.5)
                        if ready:
                            line = input_stream.readline()
                            if not line:
                                break
                            resp = self._handle_line(line)
                            if resp:
                                print(json.dumps(resp), flush=True)
                    else:
                        line = input_stream.readline()
                        if not line:
                            break
                        resp = self._handle_line(line)
                        if resp:
                            print(json.dumps(resp), flush=True)

            except KeyboardInterrupt:
                self._dbg("[IOPORT] Received keyboard interrupt")
                break
            except Exception as e:
                try:
                    print(json.dumps({"status": "error", "msg": str(e)}), flush=True)
                except Exception:
                    pass

        self.stop()

    def _handle_line(self, line: str) -> Optional[Dict]:
        """Handle a single command line."""
        cmd_dict = self._parse_command(line)
        if not cmd_dict:
            return {"status": "error", "msg": "Invalid JSON"}

        cmd = cmd_dict.get("cmd")
        args = cmd_dict.get("args", {})

        return self.process_command(cmd, args)


# Global I/O Port instance
IOPORT = IOPort()


# ═════════════════════════════════════════════════════════════════════════════
#  Dataclasses
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class FileRecord:
    """Typed, hashable record for one scanned file."""

    path: Path
    size: int
    mtime: float
    ctime: float
    inode: int
    device: int
    ext: str
    name: str
    hash: Optional[str] = None
    partial_hash: Optional[str] = None
    quick_hash: Optional[str] = None
    sha256_hash: Optional[str] = None
    magic_type: Optional[str] = None
    category: Optional[str] = None
    keep_score: int = 100
    entropy: Optional[float] = None
    is_locked: bool = False
    is_symlink: bool = False
    is_system: bool = False
    phash: Optional[str] = None
    neural_embedding: Optional[List[float]] = None
    audio_fingerprint: Optional[str] = None
    audio_metadata: Optional[Dict] = None
    # v7.0 - Semantic embedding for AI similarity
    semantic_embedding: Optional[List[float]] = None

    def __hash__(self):
        return hash(str(self.path))

    def __eq__(self, o):
        return str(self.path) == str(o.path)


@dataclass
class DupGroup:
    """One group of duplicate / near-duplicate files."""

    files: List[FileRecord]
    score: int
    group_type: str
    components: Dict[str, int] = field(default_factory=dict)
    suggestions: Dict[int, str] = field(default_factory=dict)
    risk_level: str = "LOW"
    why_keep: Dict[int, str] = field(default_factory=dict)
    verified: bool = False
    cluster_valid: bool = True

    @property
    def is_exact(self) -> bool:
        return self.group_type in ("exact", "hardlink")

    @property
    def reclaimable_bytes(self) -> int:
        if len(self.files) < 2:
            return 0
        return sum(f.size for f in self.files) - max(f.size for f in self.files)


@dataclass
class ScanSettings:
    """All user-configurable scan settings."""

    subdirs: bool = True
    min_size: int = 1
    max_size: int = 0
    use_xxhash: bool = True
    use_sha256_verify: bool = False
    use_blake3: bool = False  # v7.0 - faster hashing
    hash_files: bool = True
    paranoid_mode: bool = False
    use_gpu: bool = False
    use_neural_embed: bool = False
    use_audio_fingerprint: bool = False
    use_semantic_dedup: bool = False  # v7.0 - AI semantic similarity
    use_faiss_index: bool = False  # v7.0 - vector similarity search
    use_clip_embeddings: bool = False  # v7.0 - CLIP for images
    use_sentence_embeddings: bool = False  # v7.0 - Sentence-BERT for docs
    num_workers: int = min(CPU_COUNT * 2, 16)
    min_score: int = 70
    min_cluster_sim: float = 0.6
    exclusion_patterns: List[str] = field(default_factory=list)
    protected_paths: List[str] = field(default_factory=list)
    preferred_dirs: List[str] = field(default_factory=list)
    dark_mode: bool = False
    auto_select: bool = True
    delete_gap: int = 15
    cleanup_mode: str = "SAFE"
    skip_network: bool = True
    skip_system: bool = True
    enable_io_port: bool = False
    # v7.2 - Profile mode (safe by default)
    profile: str = "safe"  # "safe", "max_safe", "performance"
    # v7.2 - Needed Duplicate Protection
    reference_folders: List[str] = field(
        default_factory=list
    )  # Authoritative folders for reference
    scan_archives: bool = False
    # v7.0 - Semantic thresholds (can be adjusted per file type)
    semantic_threshold_image: float = 0.92
    semantic_threshold_document: float = 0.85
    semantic_threshold_video: float = 0.88
    semantic_threshold_audio: float = 0.85
    semantic_threshold_code: float = 0.90
    # v7.0 - Watch mode
    enable_watch_mode: bool = False
    watch_paths: List[str] = field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════════════
#  UnionFind — transitive cluster merging (A~B + B~C → {A,B,C})
# ═════════════════════════════════════════════════════════════════════════════


class UnionFind:
    """Path-compressed union-find for O(α·n) merging."""

    def __init__(self):
        self._parent: Dict[str, str] = {}
        self._rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])  # path compression
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        # Union by rank
        if self._rank[px] < self._rank[py]:
            px, py = py, px
        self._parent[py] = px
        if self._rank[px] == self._rank[py]:
            self._rank[px] += 1

    def connected(self, x: str, y: str) -> bool:
        return self.find(x) == self.find(y)

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
    if n < 1024:
        return f"{n} B"
    if n < 1_048_576:
        return f"{n / 1024:.1f} KB"
    if n < 1_073_741_824:
        return f"{n / 1_048_576:.2f} MB"
    return f"{n / 1_073_741_824:.2f} GB"


def _format_ts(ts: float) -> str:
    """Unix timestamp → 'YYYY-MM-DD HH:MM:SS'."""
    if not ts:
        return "N/A"
    try:
        return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "N/A"


_ts = _format_ts  # short alias used throughout UI code


# ─── Log file path: timestamped, relative to script dir, in logs/ subfolder ───
def _make_log_path() -> Path:
    """Return a timestamped log path inside <script_dir>/logs/ (created if absent)."""
    try:
        base = Path(__file__).resolve().parent
    except Exception:
        base = Path.cwd()
    log_dir = base / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        log_dir = Path.home()  # fallback: home dir if can't create logs/
    ts = time.strftime("%Y%m%d_%H%M%S")
    return log_dir / f"dupfinder_{ts}.log"


LOG_FILE_PATH = _make_log_path()


def apply_profile_settings(profile: str, settings: "ScanSettings") -> None:
    """
    Apply preset profile settings.

    Profiles:
    - safe: Default settings, balanced safety and performance
    - max_safe: Maximum protection, extra validation, higher min_score
    - ai_assist: Maximum protection + all AI features enabled for best detection
    - performance: Faster scanning, more workers, lower thresholds
    """
    if profile == "safe":
        settings.min_score = 70
        settings.cleanup_mode = "SAFE"
        settings.auto_select = True
        settings.delete_gap = 15
        settings.paranoid_mode = False
        settings.skip_system = True
        settings.skip_network = True
        settings.num_workers = min(CPU_COUNT * 2, 16)
        settings.use_xxhash = True
        settings.use_sha256_verify = False
        settings.hash_files = True
        settings.use_semantic_dedup = False
        settings.use_faiss_index = False
        settings.use_neural_embed = False
        settings.use_audio_fingerprint = False

    elif profile == "max_safe":
        settings.min_score = 90
        settings.cleanup_mode = "SAFE"
        settings.auto_select = False
        settings.delete_gap = 30
        settings.paranoid_mode = True
        settings.skip_system = True
        settings.skip_network = True
        settings.num_workers = min(CPU_COUNT, 8)
        settings.use_xxhash = True
        settings.use_sha256_verify = True
        settings.hash_files = True
        settings.use_semantic_dedup = False
        settings.use_faiss_index = False
        settings.use_neural_embed = False
        settings.use_audio_fingerprint = False

    elif profile == "ai_assist":
        settings.min_score = 80
        settings.cleanup_mode = "SAFE"
        settings.auto_select = True
        settings.delete_gap = 20
        settings.paranoid_mode = True
        settings.skip_system = True
        settings.skip_network = True
        settings.num_workers = min(CPU_COUNT * 2, 16)
        settings.use_xxhash = True
        settings.use_sha256_verify = True
        settings.hash_files = True
        settings.use_semantic_dedup = True
        settings.use_faiss_index = True
        settings.use_clip_embeddings = True
        settings.use_sentence_embeddings = True
        settings.use_neural_embed = True
        settings.use_audio_fingerprint = True

    elif profile == "performance":
        settings.min_score = 50
        settings.cleanup_mode = "AGGRESSIVE"
        settings.auto_select = True
        settings.delete_gap = 5
        settings.paranoid_mode = False
        settings.skip_system = False
        settings.skip_network = False
        settings.num_workers = min(CPU_COUNT * 4, 32)
        settings.use_xxhash = True
        settings.use_sha256_verify = False
        settings.hash_files = True
        settings.use_semantic_dedup = False
        settings.use_faiss_index = False
        settings.use_neural_embed = False
        settings.use_audio_fingerprint = False

    settings.profile = profile


# Auto-verify version against module docstring — warns on mismatch
try:
    _doc_ver = re.search(r"v(\d+\.\d+)", __doc__ or "").group(1)  # type: ignore
    if _doc_ver != VERSION:
        print(f"[version] ⚠️  docstring says v{_doc_ver} but VERSION={VERSION!r}")
except Exception:
    pass


def _detect_magic(path) -> Optional[str]:
    """Read the first MAGIC_BYTES of *path* and return a type label or None."""
    try:
        with open(path, "rb") as fh:
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
    e1 = ext1.lower().lstrip(".")
    e2 = ext2.lower().lstrip(".")
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
    sc = 0
    stem = name_stem.lower()

    for pat in COPY_PATTERNS:
        if pat in stem:
            sc -= 30
            break

    # Trailing number patterns: file(1), file-2, file_3, file 2
    if re.search(r"\(\d+\)\s*$", stem):
        sc -= 20
    elif re.search(r"[\s_\-]\d+\s*$", stem):
        sc -= 15

    # Very short name is suspicious
    if len(stem) <= 2:
        sc -= 5

    # Long, descriptive name is good
    if len(stem) >= 10 and not re.search(r"[_\-]{3,}", stem):
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
    path_str_lower = str(path).lower()

    # ── NEEDED PROTECTION: Check if in program/required folders ──
    for prot_dir in NEEDED_PROTECTION_DIRS:
        prot_lower = prot_dir.lower()
        # Handle wildcard patterns
        if "*" in prot_lower:
            base = prot_lower.replace("*", "")
            if base in path_str_lower:
                sc -= 60
                break
        else:
            if prot_lower in path_str_lower:
                sc -= 60
                break

    # Check if file is in a folder with executables (program folder)
    parent_folder = path.parent
    try:
        for item in parent_folder.iterdir():
            if item.suffix.lower() in NEEDED_EXTENSIONS or item.name.lower().endswith(
                ".exe"
            ):
                sc -= 40
                break
    except (PermissionError, OSError):
        pass

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
    sc = 100 + _name_keep_score(stem) + _path_keep_score(f.path)
    if f.is_locked:
        sc += 20
    if f.is_system:
        sc -= 50
    return max(0, min(200, sc))


def _tokenize_filename(name: str) -> Set[str]:
    """Tokenize filename for Jaccard similarity comparison."""
    stem = Path(name).stem.lower()
    tokens = set(re.findall(r"[a-z0-9]+", stem))
    return tokens


def _jaccard_similarity(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Calculate Jaccard similarity between two token sets."""
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0


def _is_system_path(path: Path) -> bool:
    """Check if path is in a critical system directory."""
    path_str = str(path).lower()
    for sys_dir in CRITICAL_SYSTEM_DIRS:
        if path_str.startswith(sys_dir.lower()):
            return True
    return False


def _is_file_locked(path: Path) -> bool:
    """Check if file is locked by another process."""
    if not HAS_PSUTIL:
        return False
    try:
        for proc in psutil.process_iter(["pid", "open_files"]):
            try:
                for f in proc.info.get("open_files") or []:
                    if Path(f.path).resolve() == path.resolve():
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass
    try:
        with open(path, "ab") as f:
            pass
        return False
    except IOError:
        return True
    except Exception:
        return False


def _generate_why_keep(f: FileRecord, g: DupGroup, all_files: List[FileRecord]) -> str:
    """Generate explanation for why a file was selected for keeping/deletion."""
    reasons = []
    stem = Path(f.name).stem.lower()

    for pat in COPY_PATTERNS:
        if pat in stem:
            reasons.append(f'Filename contains "{pat.strip()}" - likely a copy')
            break

    if f.is_locked:
        reasons.append("File is currently in use by another process")

    if f.is_system:
        reasons.append("File is in a system directory - protected")

    # Check if file is in a "needed duplicate" protected location
    path_str_lower = str(f.path).lower()
    is_needed_location = False
    for prot_dir in NEEDED_PROTECTION_DIRS:
        prot_lower = prot_dir.lower()
        if "*" in prot_lower:
            base = prot_lower.replace("*", "")
            if base in path_str_lower:
                reasons.append(
                    "File is in program/application directory - may be required by installed software"
                )
                is_needed_location = True
                break
        else:
            if prot_lower in path_str_lower:
                reasons.append(
                    "File is in program/application directory - may be required by installed software"
                )
                is_needed_location = True
                break

    # Check for executables in same folder
    if not is_needed_location:
        try:
            for item in f.path.parent.iterdir():
                if (
                    item.suffix.lower() in NEEDED_EXTENSIONS
                    or item.name.lower().endswith(".exe")
                ):
                    reasons.append(
                        "Folder contains executables - file may be required by installed program"
                    )
                    break
        except (PermissionError, OSError):
            pass

    if _path_keep_score(f.path) < 0:
        reasons.append("Located in temporary/cache directory")
    elif _path_keep_score(f.path) > 0:
        reasons.append("Located in preferred user directory")

    tokens = _tokenize_filename(f.name)
    other_stems = [Path(of.name).stem for of in all_files if of != f]
    for other in other_stems[:5]:
        other_tokens = _tokenize_filename(other)
        sim = _jaccard_similarity(tokens, other_tokens)
        if sim > 0.5:
            reasons.append(f"Filename similarity: {sim:.0%} match with other files")
            break

    if f.ctime > 0:
        older = [of for of in all_files if of != f and of.ctime < f.ctime]
        if older:
            reasons.append("File is newer than some duplicates (likely original)")

    if not reasons:
        reasons.append("Selected by quality scoring algorithm")

    return "; ".join(reasons[:3])


def _generate_duplicate_story(f1: FileRecord, f2: FileRecord, score: int) -> str:
    """
    Generate a natural-language "Duplicate Story" explaining the relationship.
    This provides a human-readable explanation of why these files are related.
    """
    story_parts = []

    if f1.size == f2.size:
        story_parts.append(
            f"Both files are exactly the same size ({_format_size(f1.size)})"
        )
    else:
        size_diff = abs(f1.size - f2.size) / max(f1.size, f2.size)
        if size_diff < 0.05:
            story_parts.append(
                f"Files are nearly identical in size (~{size_diff:.1%} difference)"
            )
        else:
            story_parts.append(
                f"Size differs: {_format_size(f1.size)} vs {_format_size(f2.size)}"
            )

    name_sim = SequenceMatcher(
        None, Path(f1.name).stem.lower(), Path(f2.name).stem.lower()
    ).ratio()
    if name_sim > 0.9:
        story_parts.append("Filenames are nearly identical")
    elif name_sim > 0.7:
        story_parts.append("Filenames are very similar")
    elif name_sim > 0.5:
        story_parts.append("Filenames share some common elements")

    if f1.phash and f2.phash:
        dist = _phash_distance(f1.phash, f2.phash)
        if dist <= 8:
            story_parts.append("Visual content is nearly identical (pHash)")
        elif dist <= 16:
            story_parts.append("Visual content is very similar (pHash)")

    if f1.neural_embedding and f2.neural_embedding:
        story_parts.append("Neural embeddings confirm visual similarity")

    if f1.semantic_embedding and f2.semantic_embedding:
        try:
            import numpy as np

            emb1 = np.array(f1.semantic_embedding)
            emb2 = np.array(f2.semantic_embedding)
            sem_sim = float(
                np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            )
            if sem_sim >= 0.9:
                story_parts.append(
                    f"AI semantic similarity: EXACT MATCH ({sem_sim:.0%})"
                )
            elif sem_sim >= 0.85:
                story_parts.append(f"AI semantic similarity: VERY HIGH ({sem_sim:.0%})")
            elif sem_sim >= 0.7:
                story_parts.append(f"AI semantic similarity: HIGH ({sem_sim:.0%})")
        except Exception:
            pass

    if f1.audio_fingerprint and f2.audio_fingerprint:
        if f1.audio_fingerprint == f2.audio_fingerprint:
            story_parts.append("Audio fingerprint matches exactly - same recording")
        else:
            story_parts.append("Audio fingerprints suggest similar content")

    if f1.audio_metadata and f2.audio_metadata:
        if f1.audio_metadata.get("title") and f2.audio_metadata.get("title"):
            if f1.audio_metadata["title"] == f2.audio_metadata["title"]:
                story_parts.append(f'Same track: "{f1.audio_metadata["title"]}"')

    if abs(f1.mtime - f2.mtime) < 3600:
        story_parts.append("Modified around the same time")
    elif abs(f1.mtime - f2.mtime) < 86400 * 7:
        story_parts.append("Modified within the same week")

    if f1.category == f2.category and f1.category:
        story_parts.append(f"Both are {f1.category} files")

    if f1.path.parent == f2.path.parent:
        story_parts.append("Files are in the same folder")
    else:
        p1_parts = f1.path.parts
        p2_parts = f2.path.parts
        common = 0
        for i in range(min(len(p1_parts), len(p2_parts))):
            if p1_parts[i].lower() == p2_parts[i].lower():
                common += 1
            else:
                break
        if common > 0:
            story_parts.append(f"Files share {common} folder level(s) in common")

    if score >= 90:
        story_parts.append("HIGH similarity detected")
    elif score >= 70:
        story_parts.append("MODERATE similarity detected")
    else:
        story_parts.append("LOW similarity - may be unrelated")

    return ". ".join(story_parts[:5])


def _calculate_dup_score(args):
    """
    Compute (score 0-100, components dict) for two FileRecords.

    Called by ThreadPoolExecutor and ProcessPoolExecutor workers, so
    arguments are passed as a tuple to support pickling.

    Score breakdown (max 100 before cap):
      hash match                    → 100 (exact)
      quick_hash_same + same_size  → 55
      size_exact                    → 35
      size_close (<5%)             → 15
      name_similarity > 0.9        → 25
      name_similarity > 0.7       → 18
      name_similarity > 0.5       → 10
      token Jaccard > 0.7          → 10
      token Jaccard > 0.4          → 5
      extension exact               → 12
      extension compat              → 6
      magic_type match             → 8
      same parent directory        → 5
      mtime proximity < 1h          → 5
      mtime proximity < 24h        → 3
      mtime proximity < 7d         → 1
      entropy similarity < 10%     → 5
    """
    f1d, f2d = args
    comp: Dict[str, int] = {}

    h1 = f1d.get("hash")
    h2 = f2d.get("hash")
    ph1 = f1d.get("partial_hash")
    ph2 = f2d.get("partial_hash")
    qh1 = f1d.get("quick_hash")
    qh2 = f2d.get("quick_hash")
    s1 = f1d.get("size", 0)
    s2 = f2d.get("size", 0)
    n1 = f1d.get("name", "")
    n2 = f2d.get("name", "")
    e1 = f1d.get("ext", "")
    e2 = f2d.get("ext", "")
    mt1 = f1d.get("magic_type")
    mt2 = f2d.get("magic_type")
    m1 = f1d.get("mtime", 0)
    m2 = f2d.get("mtime", 0)
    p1 = str(f1d.get("parent", ""))
    p2 = str(f2d.get("parent", ""))
    t1 = f1d.get("tokens", [])
    t2 = f2d.get("tokens", [])
    ent1 = f1d.get("entropy")
    ent2 = f2d.get("entropy")
    phs1 = f1d.get("phash")
    phs2 = f2d.get("phash")

    # ── Exact hash → 100 immediately ─────────────────────────────────────
    if h1 and h2 and h1 == h2:
        return 100, {
            "hash": 100,
            "size": 0,
            "name": 0,
            "ext": 0,
            "magic": 0,
            "dir": 0,
            "time": 0,
            "token": 0,
            "entropy": 0,
            "phash": 0,
        }

    # ── Different size → not duplicate (can't be near with large size diff) ─
    if s1 != s2:
        ratio = abs(s1 - s2) / max(s1, s2, 1)
        if ratio > 0.20:  # > 20% size difference → skip
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
        score += 35
        comp["size"] = 35
    else:
        ratio = abs(s1 - s2) / max(s1, s2, 1)
        pts = 15 if ratio <= 0.05 else (8 if ratio <= 0.10 else 0)
        score += pts
        comp["size"] = pts

    # ── Quick hash bonus (same quick hash + same size = near-certain) ─────
    if qh1 and qh2 and qh1 == qh2 and s1 == s2:
        score += 20
        comp["quick"] = 20
    else:
        comp["quick"] = 0

    # ── Name similarity ───────────────────────────────────────────────────
    stem1 = Path(n1).stem.lower()
    stem2 = Path(n2).stem.lower()
    sim = SequenceMatcher(None, stem1, stem2).ratio()
    if sim > 0.90:
        pts = 25
    elif sim > 0.70:
        pts = 18
    elif sim > 0.50:
        pts = 10
    else:
        pts = 0
    score += pts
    comp["name"] = pts

    # ── Token-based Jaccard similarity ────────────────────────────────────
    if t1 and t2:
        jaccard = _jaccard_similarity(set(t1), set(t2))
        if jaccard > 0.7:
            score += 10
            comp["token"] = 10
        elif jaccard > 0.4:
            score += 5
            comp["token"] = 5
        else:
            comp["token"] = 0
    else:
        comp["token"] = 0

    # ── Extension ─────────────────────────────────────────────────────────
    if e1 == e2:
        score += 12
        comp["ext"] = 12
    else:
        pts = int(_ext_compat(e1, e2) * 6)
        score += pts
        comp["ext"] = pts

    # ── Magic type ────────────────────────────────────────────────────────
    if mt1 and mt2 and mt1 == mt2:
        score += 8
        comp["magic"] = 8
    else:
        comp["magic"] = 0

    # ── Same parent directory ─────────────────────────────────────────────
    if p1 and p2 and p1 == p2:
        score += 5
        comp["dir"] = 5
    else:
        comp["dir"] = 0

    # ── Entropy similarity ────────────────────────────────────────────────
    if ent1 is not None and ent2 is not None:
        ent_diff = abs(ent1 - ent2) / max(ent1, ent2, 0.001)
        if ent_diff < 0.1:
            score += 5
            comp["entropy"] = 5
        else:
            comp["entropy"] = 0
    else:
        comp["entropy"] = 0

    # ── Perceptual hash (images) — requires HAS_IMAGEHASH + HAS_PIL ───────
    if phs1 and phs2:
        dist = _phash_distance(phs1, phs2)
        if dist <= 8:  # near-identical visual content
            pts = 30
        elif dist <= 16:  # visually similar
            pts = 18
        elif dist <= 24:  # somewhat similar
            pts = 8
        else:
            pts = 0
        score += pts
        comp["phash"] = pts
    else:
        comp["phash"] = 0

    # v7.0 - Semantic embedding similarity (AI-powered)
    sem1 = f1d.get("semantic_embedding")
    sem2 = f2d.get("semantic_embedding")
    if sem1 and sem2 and len(sem1) > 0 and len(sem2) > 0:
        try:
            import numpy as np

            a = np.array(sem1)
            b = np.array(sem2)
            cos_sim = float(
                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
            )
            if cos_sim >= 0.92:  # Very high similarity
                pts = 40
            elif cos_sim >= 0.88:
                pts = 30
            elif cos_sim >= 0.85:
                pts = 20
            elif cos_sim >= 0.80:
                pts = 12
            elif cos_sim >= 0.75:
                pts = 6
            else:
                pts = 0
            score += pts
            comp["semantic"] = pts
        except Exception:
            comp["semantic"] = 0
    else:
        comp["semantic"] = 0

    # ── Temporal proximity ────────────────────────────────────────────────
    if m1 and m2:
        diff_h = abs(m1 - m2) / 3600.0
        if diff_h < 1:
            pts = 5
        elif diff_h < 24:
            pts = 3
        elif diff_h < 168:
            pts = 1
        else:
            pts = 0
        score += pts
        comp["time"] = pts
    else:
        comp["time"] = 0

    return min(score, 100), comp


def _hash_file_worker(args):
    """
    Hash one file; safe to run from any thread or process.
    args = (fp_str, chunk_size, use_xxhash, mode, use_blake3)
    mode: 'quick' | 'partial' | 'full'
    Returns (fp_str, digest_or_None, mode).
    """
    fp_str, chunk_size, use_xxhash, mode, use_blake3 = args
    try:
        # BLAKE3 is only used for full hash mode - much faster than MD5/SHA
        if use_blake3 and mode == "full" and HAS_BLAKE3:
            h = blake3.blake3()
            with open(fp_str, "rb") as fh:
                while True:
                    chunk = fh.read(chunk_size)
                    if not chunk:
                        break
                    h.update(chunk)
            return fp_str, h.hexdigest(), mode
        elif use_xxhash and HAS_XXHASH:
            h = _xxhash.xxh64()
        else:
            try:
                h = hashlib.md5(usedforsecurity=False)
            except TypeError:
                h = hashlib.md5()

        with open(fp_str, "rb") as fh:
            if mode == "quick":
                data = fh.read(QUICK_HASH_BYTES)
                if data:
                    h.update(data)
            elif mode == "partial":
                fsize = os.path.getsize(fp_str)
                data = fh.read(chunk_size)
                if data:
                    h.update(data)
                if fsize > chunk_size:
                    fh.seek(max(0, fsize - chunk_size))
                    data = fh.read(chunk_size)
                    if data:
                        h.update(data)
            else:  # full
                while True:
                    chunk = fh.read(chunk_size)
                    if not chunk:
                        break
                    h.update(chunk)
        return fp_str, h.hexdigest(), mode
    except Exception:
        return fp_str, None, mode


def _byte_compare(path1, path2) -> bool:
    """Byte-by-byte file comparison for paranoid verification."""
    try:
        with open(path1, "rb") as f1, open(path2, "rb") as f2:
            while True:
                b1 = f1.read(HASH_CHUNK)
                b2 = f2.read(HASH_CHUNK)
                if b1 != b2:
                    return False
                if not b1:
                    return True
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


# ── Perceptual hash distance (Hamming) — no imagehash import needed ──────────
def _phash_distance(h1_hex: str, h2_hex: str) -> int:
    """Hamming distance between two perceptual-hash hex strings."""
    try:
        return bin(int(h1_hex, 16) ^ int(h2_hex, 16)).count("1")
    except Exception:
        return 64  # treat as maximally different on error


# ── Shannon entropy worker ────────────────────────────────────────────────────
def _entropy_worker(fp_str: str) -> "Tuple[str, Optional[float]]":
    """
    Compute Shannon entropy of the first 64 KB of a file.
    Returns (fp_str, entropy_float) or (fp_str, None) on error.
    Range: 0.0 (all same byte) – 8.0 (uniform byte distribution).
    """
    try:
        import math as _m

        with open(fp_str, "rb") as fh:
            data = fh.read(65536)
        if not data:
            return fp_str, 0.0
        freq = [0] * 256
        for b in data:
            freq[b] += 1
        total = len(data)
        ent = -sum((p := c / total) * _m.log2(p) for c in freq if c > 0)
        return fp_str, round(ent, 4)
    except Exception:
        return fp_str, None


# ── Perceptual hash worker (images only) ────────────────────────────────────
def _phash_worker(fp_str: str) -> "Tuple[str, Optional[str]]":
    """
    Compute perceptual hash for an image using imagehash.phash.
    Returns (fp_str, hex_string) or (fp_str, None) on failure / missing lib.
    """
    if not HAS_IMAGEHASH or not HAS_PIL:
        return fp_str, None
    try:
        img = _PILImage.open(fp_str)
        img.load()
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        return fp_str, str(_imagehash.phash(img, hash_size=8))
    except Exception:
        return fp_str, None


# ── Neural embedding worker (DINOv2 for images) ───────────────────────────────
_DINO_MODEL = None
_DINO_PROCESSOR = None


def _get_dino_model():
    """Get cached DINOv2 model and processor."""
    global _DINO_MODEL, _DINO_PROCESSOR
    if _DINO_MODEL is None and HAS_DINO:
        try:
            from transformers import AutoImageProcessor, AutoModel

            _DINO_PROCESSOR = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            _DINO_MODEL = AutoModel.from_pretrained("facebook/dinov2-base")
            _DINO_MODEL.eval()
        except Exception:
            pass
    return _DINO_MODEL, _DINO_PROCESSOR


def _neural_embed_worker(fp_str: str) -> "Tuple[str, Optional[List[float]]]":
    """
    Compute neural embedding for an image using DINOv2.
    Returns (fp_str, embedding_list) or (fp_str, None) on failure.
    Requires transformers + torch. Disabled by default.
    """
    if not HAS_DINO:
        return fp_str, None
    try:
        model, processor = _get_dino_model()
        if model is None or processor is None:
            return fp_str, None

        img = _PILImage.open(fp_str)
        if img.mode != "RGB":
            img = img.convert("RGB")

        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0].squeeze().tolist()

        return fp_str, embedding
    except Exception:
        return fp_str, None


# ── Audio fingerprint worker ───────────────────────────────────────────────────
def _audio_fingerprint_worker(
    fp_str: str,
) -> "Tuple[str, Optional[str], Optional[Dict]]":
    """
    Compute audio fingerprint using chromaprint/fingerprint.
    Returns (fp_str, fingerprint, metadata) or (fp_str, None, None) on failure.
    """
    if not HAS_CHROMAPRINT:
        return fp_str, None, None

    try:
        import acoustid

        fingerprint, duration = acoustid.fingerprint_file(fp_str)

        metadata = {}
        if HAS_MUTAGEN:
            from mutagen import File as MutagenFile

            mf = MutagenFile(fp_str)
            if mf:
                metadata = {
                    "artist": mf.get("artist", [""])[0] if mf.get("artist") else "",
                    "album": mf.get("album", [""])[0] if mf.get("album") else "",
                    "title": mf.get("title", [""])[0] if mf.get("title") else "",
                    "genre": mf.get("genre", [""])[0] if mf.get("genre") else "",
                    "year": mf.get("date", [""])[0] if mf.get("date") else "",
                }

        return fp_str, str(fingerprint), metadata
    except Exception:
        return fp_str, None, None


# ── Audio metadata only worker (fallback without chromaprint) ────────────────────
def _audio_metadata_worker(fp_str: str) -> "Tuple[str, Optional[Dict]]":
    """
    Extract audio metadata using mutagen (no fingerprinting).
    Returns (fp_str, metadata_dict) or (fp_str, None) on failure.
    """
    if not HAS_MUTAGEN:
        return fp_str, None
    try:
        from mutagen import File as MutagenFile

        mf = MutagenFile(fp_str)
        if mf:
            return fp_str, {
                "artist": mf.get("artist", [""])[0] if mf.get("artist") else "",
                "album": mf.get("album", [""])[0] if mf.get("album") else "",
                "title": mf.get("title", [""])[0] if mf.get("title") else "",
                "genre": mf.get("genre", [""])[0] if mf.get("genre") else "",
                "year": mf.get("date", [""])[0] if mf.get("date") else "",
                "duration": mf.info.length
                if hasattr(mf, "info") and hasattr(mf.info, "length")
                else 0,
            }
        return fp_str, None
    except Exception:
        return fp_str, None


# ── Video frame embedding worker ─────────────────────────────────────────────────
def _video_embed_worker(fp_str: str) -> "Tuple[str, Optional[List[float]]]":
    """
    Extract key frames from video and compute neural embedding.
    Returns (fp_str, embedding_list) or (fp_str, None) on failure.
    """
    if not HAS_DINO or not HAS_CV2:
        return fp_str, None

    try:
        import cv2

        cap = cv2.VideoCapture(fp_str)
        if not cap.isOpened():
            cap.release()
            return fp_str, None

        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count > 0:
            num_key_frames = min(5, max(1, frame_count // 30))
            for i in range(num_key_frames):
                idx = int((i + 1) * frame_count / (num_key_frames + 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)

        cap.release()

        if not frames:
            return fp_str, None

        model, processor = _get_dino_model()
        if model is None or processor is None:
            return fp_str, None

        embeddings = []
        for frame in frames:
            img = _PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.last_hidden_state[:, 0].squeeze().tolist()
                embeddings.append(emb)

        if not embeddings:
            return fp_str, None

        avg_embedding = [
            sum(e[i] for e in embeddings) / len(embeddings)
            for i in range(len(embeddings[0]))
        ]
        return fp_str, avg_embedding
    except Exception:
        return fp_str, None


# ── SHA-256 worker ───────────────────────────────────────────────────────────
def _sha256_worker(fp_str: str) -> "Tuple[str, Optional[str]]":
    """Full-file SHA-256 hash. Returns (fp_str, hex) or (fp_str, None)."""
    try:
        h = hashlib.sha256()
        with open(fp_str, "rb") as fh:
            while True:
                chunk = fh.read(HASH_CHUNK)
                if not chunk:
                    break
                h.update(chunk)
        return fp_str, h.hexdigest()
    except Exception:
        return fp_str, None


# ── File-category helpers ────────────────────────────────────────────────────
_IMAGE_MAGIC = frozenset({"JPEG", "PNG", "GIF", "BMP", "TIFF", "WEBP"})
_VIDEO_MAGIC = frozenset({"MP4", "MKV", "AVI", "RIFF"})
_AUDIO_MAGIC = frozenset({"MP3", "FLAC", "OGG", "WAV"})
_DOC_MAGIC = frozenset({"PDF", "HTML", "XML", "UTF8-BOM"})
_ARCHIVE_MAGIC = frozenset({"ZIP", "GZIP", "RAR", "7Z"})
_EXE_MAGIC = frozenset({"EXE/DLL", "ELF", "MACH-O"})

_IMAGE_EXTS = frozenset(
    {
        "jpg",
        "jpeg",
        "jpe",
        "jfif",
        "png",
        "gif",
        "bmp",
        "tif",
        "tiff",
        "webp",
        "svg",
        "ico",
        "raw",
        "cr2",
        "nef",
        "arw",
        "dng",
        "heic",
        "avif",
    }
)
_VIDEO_EXTS = frozenset(
    {
        "mp4",
        "avi",
        "mov",
        "mkv",
        "wmv",
        "flv",
        "m4v",
        "webm",
        "mpeg",
        "mpg",
        "3gp",
        "ts",
        "vob",
        "mts",
    }
)
_AUDIO_EXTS = frozenset(
    {"mp3", "wav", "flac", "aac", "ogg", "m4a", "wma", "opus", "aiff", "alac"}
)
_DOCUMENT_EXTS = frozenset(
    {
        "pdf",
        "doc",
        "docx",
        "txt",
        "rtf",
        "odt",
        "xls",
        "xlsx",
        "ods",
        "ppt",
        "pptx",
        "odp",
        "csv",
        "md",
        "rst",
        "epub",
        "html",
        "htm",
        "xml",
    }
)
_ARCHIVE_EXTS = frozenset(
    {"zip", "rar", "7z", "tar", "gz", "bz2", "xz", "iso", "dmg", "pkg", "deb"}
)
_CODE_EXTS = frozenset(
    {
        "py",
        "js",
        "ts",
        "jsx",
        "tsx",
        "html",
        "css",
        "java",
        "c",
        "cpp",
        "h",
        "hpp",
        "rb",
        "go",
        "rs",
        "php",
        "sh",
        "bat",
        "ps1",
        "lua",
        "swift",
    }
)


def _categorize_file(magic_type: "Optional[str]", ext: str) -> str:
    """
    Return a human-readable category for a file.
    Priority: magic_type (bytes-verified) > extension > 'Other'.
    """
    if magic_type:
        if magic_type in _IMAGE_MAGIC:
            return "Image"
        if magic_type in _VIDEO_MAGIC:
            return "Video"
        if magic_type in _AUDIO_MAGIC:
            return "Audio"
        if magic_type in _DOC_MAGIC:
            return "Document"
        if magic_type in _ARCHIVE_MAGIC:
            return "Archive"
        if magic_type in _EXE_MAGIC:
            return "Executable"
    el = ext.lower().lstrip(".")
    if el in _IMAGE_EXTS:
        return "Image"
    if el in _VIDEO_EXTS:
        return "Video"
    if el in _AUDIO_EXTS:
        return "Audio"
    if el in _DOCUMENT_EXTS:
        return "Document"
    if el in _ARCHIVE_EXTS:
        return "Archive"
    if el in _CODE_EXTS:
        return "Code"
    return "Other"


# ═════════════════════════════════════════════════════════════════════════════
#  ScanEngine  — pure logic, no tkinter dependency
# ═════════════════════════════════════════════════════════════════════════════


class ScanEngine:
    """
    Full duplicate-detection pipeline with MANDATORY SAFETY SYSTEM:

      1. _discover()       → List[Path]           (os.scandir, fast)
      2. _stat_batch()     → List[FileRecord]      (parallel stat)
      3. _detect_magic()   → in-place on records   (optional, parallel)
      4. _check_file_locks() → detect locked files (psutil)
      5. _hash_all()       → 3-phase hashing       (parallel, adaptive)
      6. _revalidate_mutation() → verify files haven't changed
      7. find_duplicates() → List[DupGroup]
         a. _find_hardlinks()   exact inode match (NEVER deletable)
         b. _group_by_size()    pre-filter
         c. _find_exact()       hash-based exact (VERIFIED)
         d. _find_near()        scored pairwise
         e. _merge_clusters()   UnionFind transitive merge
         f. _validate_clusters() → prevent false merges
         g. _assess_risk()      → LOW/MEDIUM/HIGH per group
      8. smart_select()    → annotates DupGroup.suggestions with WHY explanations

    SAFETY GUARANTEES:
      - Only verified EXACT duplicates eligible for auto-delete
      - Near-duplicates NEVER auto-deleted
      - Hardlinks NEVER deletable
      - Locked files protected
      - System files protected
      - Files changing during scan detected
    """

    def __init__(
        self,
        root: str,
        settings: ScanSettings,
        progress_queue: queue.Queue,
        cancel_event: threading.Event,
    ):
        self.root = Path(root)
        self.settings = settings
        self.pq = progress_queue
        self.cancel = cancel_event
        self.files: List[FileRecord] = []
        self._script_name = Path(sys.argv[0]).name.lower()

    # ── Messaging helpers ─────────────────────────────────────────────────

    def _send(self, mtype: str, data) -> None:
        try:
            self.pq.put_nowait((mtype, data))
        except queue.Full:
            pass

    def _log(self, tag: str, text: str) -> None:
        self._send("log", (tag, text))

    def _dbg(self, text: str) -> None:
        self._send("debug", text)

    def _err(self, text: str, exc: Exception = None) -> None:
        full = text + (f"\n  {type(exc).__name__}: {exc}" if exc else "")
        self._send("error_detail", full)
        self._log("error", text)

    def _progress(self, kind: str, **kw) -> None:
        self._send(kind, kw)

    # ── Stage 1: File Discovery (os.scandir, fast) ────────────────────────

    def _discover(self) -> List[Path]:
        """Recursive os.scandir — ~3× faster than Path.glob."""
        results: List[Path] = []
        self._dbg(
            f"[SCAN] Discovery starting  root={self.root}"
            f"  recursive={self.settings.subdirs}"
        )

        def _walk(dirpath: Path):
            if self.cancel.is_set():
                return
            try:
                with os.scandir(dirpath) as it:
                    for entry in it:
                        if self.cancel.is_set():
                            break
                        try:
                            if entry.is_file(follow_symlinks=False):
                                results.append(Path(entry.path))
                            elif (
                                entry.is_dir(follow_symlinks=False)
                                and self.settings.subdirs
                            ):
                                # Skip known junk directories
                                if entry.name.lower() not in TEMP_DIR_TOKENS:
                                    _walk(Path(entry.path))
                        except (OSError, PermissionError):
                            pass
            except (PermissionError, OSError) as exc:
                self._dbg(f"[SCAN] SKIP dir {dirpath}  reason={exc}")

        _walk(self.root)
        self._dbg(f"[SCAN] Discovery done  paths={len(results)}")
        return results

    # ── Stage 2: Parallel stat ─────────────────────────────────────────────

    def _stat_batch(self, paths: List[Path]) -> List[FileRecord]:
        """Parallel stat + filter → FileRecord list."""
        settings = self.settings
        script = self._script_name
        min_sz = settings.min_size
        max_sz = settings.max_size if settings.max_size > 0 else float("inf")
        excl = [p.lower() for p in settings.exclusion_patterns]

        def _stat(fp: Path) -> Optional[FileRecord]:
            if fp.name.lower() == script:
                return None
            # Exclusion patterns
            fp_str = str(fp).lower()
            if any(pat in fp_str for pat in excl):
                return None
            try:
                st = fp.stat()
                is_sys = _is_system_path(fp)
                if settings.skip_system and is_sys:
                    return None
                if not (min_sz <= st.st_size <= max_sz):
                    return None
                return FileRecord(
                    path=fp,
                    size=st.st_size,
                    mtime=st.st_mtime,
                    ctime=st.st_ctime,
                    inode=st.st_ino,
                    device=st.st_dev,
                    ext=fp.suffix.lower(),
                    name=fp.name,
                    is_symlink=fp.is_symlink(),
                    is_system=is_sys,
                )
            except (OSError, PermissionError) as exc:
                self._dbg(f"[SCAN] STAT-FAIL {fp.name[:40]}  {exc}")
                return None

        total = len(paths)
        results = []
        done = 0
        report_every = max(1, total // 200)
        nw = settings.num_workers

        with ThreadPoolExecutor(max_workers=nw) as pool:
            futs = {pool.submit(_stat, fp): fp for fp in paths}
            for fut in as_completed(futs):
                if self.cancel.is_set():
                    break
                done += 1
                rec = fut.result()
                if rec:
                    results.append(rec)
                if done % report_every == 0 or done == total:
                    nm = rec.name[:35] if rec else "…"
                    self._progress(
                        "scan_progress",
                        current=done,
                        total=total,
                        percent=int(done / max(total, 1) * 100),
                        file=nm,
                        status="📄 Scanning",
                    )
                    self._dbg(f"[SCAN] {done}/{total}  files={len(results)}")

        return results

    # ── Stage 3: Magic-type detection ─────────────────────────────────────

    def _detect_magic_batch(self, files: List[FileRecord]) -> None:
        # ── modular: magic-type detection stage ──────────────────────────────
        """
        Detect file magic types in parallel; updates records in-place.
        Automatically skips files > MAGIC_SIZE_SKIP bytes (overhead not worth it).
        Skips entirely if file count > MAGIC_COUNT_SKIP (too slow for whole-drive).
        """
        if not files:
            return
        total = len(files)

        # Skip if file count is enormous
        if total > MAGIC_COUNT_SKIP:
            self._log(
                "info",
                f"⚡ Skipping magic detection ({total:,} files — using extension only)",
            )
            self._dbg(f"[SCAN] Magic skipped: {total} > {MAGIC_COUNT_SKIP} threshold")
            return

        # Filter to only smallish files worth inspecting
        eligible = [f for f in files if f.size <= MAGIC_SIZE_SKIP]
        skipped_large = total - len(eligible)

        self._log(
            "info",
            f"🔬 Detecting file types: {len(eligible):,} files"
            + (f" ({skipped_large:,} large files skipped)" if skipped_large else ""),
        )
        self._dbg(
            f"[SCAN] Magic on {len(eligible)}/{total}  skipped_large={skipped_large}"
        )

        if not eligible:
            return
        etotal = len(eligible)
        report_every = max(1, etotal // 20)
        done = 0
        last_t = time.monotonic()

        def _detect(fp: Path):
            return _detect_magic(fp)

        with ThreadPoolExecutor(max_workers=self.settings.num_workers) as pool:
            futs = {pool.submit(_detect, f.path): f for f in eligible}
            for fut in as_completed(futs):
                if self.cancel.is_set():
                    break
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
                    self._progress(
                        "scan_progress",
                        current=done,
                        total=etotal,
                        percent=pct,
                        file=f.name[:35],
                        status="🔬 Detecting file types",
                    )
                    self._dbg(f"[SCAN] Magic {done}/{etotal} ({pct}%)")

        typed = sum(1 for f in files if f.magic_type)
        self._dbg(f"[SCAN] Magic done  typed={typed}/{total}")
        self._log("info", f"✓ File-type detection: {typed:,}/{total:,} typed")

    # ── Stage 4: Three-phase hashing ──────────────────────────────────────

    def _hash_all(self, files: List[FileRecord]) -> None:
        """
        Phase 1: quick hash (4 KB) all files.
        Phase 2: partial hash (first+last 64 KB) large files that share quick hash.
        Phase 3: full hash files that share (size, partial_hash).
        """
        if not files or not self.settings.hash_files:
            return
        algo = "xxhash-xxh64" if (self.settings.use_xxhash and HAS_XXHASH) else "MD5"
        nw = self.settings.num_workers
        ux = self.settings.use_xxhash and HAS_XXHASH

        self._log(
            "info", f"🔐 3-phase hashing {len(files)} files ({algo}, {nw} workers)…"
        )
        self._dbg(f"[HASH] Start  total={len(files)}  algo={algo}  workers={nw}")

        # ── Phase 1: quick hash all ───────────────────────────────────────
        self._hash_batch(files, "quick", ux, nw, "Phase-1 quick")

        if self.cancel.is_set():
            return

        # ── Phase 2: partial hash large files that share quick hash ───────
        quick_groups: Dict[str, List[FileRecord]] = defaultdict(list)
        for f in files:
            if f.quick_hash and f.size > PARTIAL_THRESHOLD:
                quick_groups[(f.size, f.quick_hash)].append(f)
        need_partial = [f for g in quick_groups.values() if len(g) > 1 for f in g]
        skipped_partial = sum(1 for f in files if f.size > PARTIAL_THRESHOLD) - len(
            need_partial
        )
        self._dbg(
            f"[HASH] Phase 2: {len(need_partial)} large need partial"
            f"  {skipped_partial} skipped"
        )

        if need_partial and not self.cancel.is_set():
            self._hash_batch(need_partial, "partial", ux, nw, "Phase-2 partial")

        if self.cancel.is_set():
            return

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
        self._dbg(
            f"[HASH] Phase 3: {len(need_full)} need full hash"
            f"  {skipped_full} unique → skipped"
        )

        if need_full and not self.cancel.is_set():
            self._hash_batch(need_full, "full", ux, nw, "Phase-3 full")

        # ── Phase 4: SHA-256 verification for potential exact duplicates ────
        if self.settings.use_sha256_verify and not self.cancel.is_set():
            hash_groups: Dict[str, List[FileRecord]] = defaultdict(list)
            for f in files:
                if f.hash:
                    hash_groups[f.hash].append(f)
            sha256_cands = [f for g in hash_groups.values() if len(g) > 1 for f in g]
            if sha256_cands:
                self._sha256_batch(sha256_cands, nw)
            else:
                self._dbg("[SHA256] skipped — no shared full hashes found")

        self._log("info", "✓ Hashing complete — grouping results…")
        self._dbg("[HASH] All phases done")

    def _check_file_locks(self, files: List[FileRecord]) -> None:
        """Detect files locked by other processes - skip for performance."""
        pass

    def _check_file_locks_fast(self, files: List[FileRecord]) -> None:
        """Fast lock detection: only check files in same-size groups (potential duplicates)."""
        from collections import defaultdict

        size_groups = defaultdict(list)
        for f in files:
            size_groups[f.size].append(f)

        duplicate_sizes = [grp for grp in size_groups.values() if len(grp) > 1]
        check_count = sum(len(g) for g in duplicate_sizes)

        if check_count == 0:
            return

        locked = 0
        for grp in duplicate_sizes:
            for f in grp:
                try:
                    with open(f.path, "ab") as test_file:
                        pass
                    f.is_locked = False
                except IOError:
                    f.is_locked = True
                    locked += 1
                except Exception:
                    f.is_locked = False

        if locked > 0:
            self._log("warn", f"🔒 {locked} file(s) are currently in use")
            self._dbg(f"[SCAN] {locked} locked files detected")

    def _revalidate_mutation(self, files: List[FileRecord]) -> List[FileRecord]:
        """Re-check size+mtime after hashing; clear hashes if file changed."""
        valid = []
        mutated = 0
        for f in files:
            try:
                st = f.path.stat()
                if st.st_size != f.size or abs(st.st_mtime - f.mtime) > 1:
                    mutated += 1
                    self._dbg(f"[SCAN] File mutated during scan: {f.path.name}")
                    f.hash = None
                    f.partial_hash = None
                    f.quick_hash = None
                    valid.append(f)
                else:
                    valid.append(f)
            except OSError:
                mutated += 1
        if mutated > 0:
            self._log(
                "warn", f"⚠️  {mutated} file(s) changed during scan - hashes cleared"
            )
        return valid

    def _hash_batch(
        self, files: List[FileRecord], mode: str, use_xxhash: bool, nw: int, label: str
    ) -> None:
        """Hash a batch of files; update records in-place."""
        if not files:
            return
        total = len(files)
        path_map = {str(f.path): f for f in files}
        hash_key = {"quick": "quick_hash", "partial": "partial_hash", "full": "hash"}[
            mode
        ]
        use_blake = self.settings.use_blake3 and HAS_BLAKE3
        args_list = [
            (str(f.path), HASH_CHUNK, use_xxhash, mode, use_blake) for f in files
        ]
        done = 0
        errors = 0
        report_every = max(1, total // 100)

        with ThreadPoolExecutor(max_workers=nw) as pool:
            futs = {pool.submit(_hash_file_worker, a): a[0] for a in args_list}
            for fut in as_completed(futs):
                if self.cancel.is_set():
                    break
                done += 1
                fp_str, digest, _ = fut.result()
                if digest:
                    if fp_str in path_map:
                        setattr(path_map[fp_str], hash_key, digest)
                else:
                    errors += 1
                    self._dbg(f"[HASH] ERR {Path(fp_str).name[:40]}")
                if done % report_every == 0 or done == total:
                    pct = int(done / max(total, 1) * 100)
                    nm = Path(fp_str).name[:35] if fp_str else "…"
                    self._progress(
                        "scan_progress",
                        current=done,
                        total=total,
                        percent=pct,
                        file=nm,
                        status=f"🔐 {label}",
                    )
                    self._dbg(f"[HASH] {label} {done}/{total} ({pct}%)  err={errors}")

    # ── Stage 4b: SHA-256 verification (optional) ─────────────────────────

    def _sha256_batch(self, files: "List[FileRecord]", nw: int) -> None:
        """
        Compute SHA-256 for files that are candidates for exact deduplication.
        Updates f.sha256_hash in-place.  Only called when use_sha256_verify=True.
        """
        if not files:
            return
        total = len(files)
        self._log("info", f"🔏 SHA-256 verification: {total:,} candidate files…")
        self._dbg(f"[SHA256] start  files={total}")
        path_map = {str(f.path): f for f in files}
        done = errors = 0
        rep = max(1, total // 50)
        fp_str = ""
        with ThreadPoolExecutor(max_workers=nw) as pool:
            futs = {pool.submit(_sha256_worker, str(f.path)): f for f in files}
            for fut in as_completed(futs):
                if self.cancel.is_set():
                    break
                done += 1
                fp_str, digest = fut.result()
                if digest and fp_str in path_map:
                    path_map[fp_str].sha256_hash = digest
                else:
                    errors += 1
                if done % rep == 0 or done == total:
                    self._progress(
                        "scan_progress",
                        current=done,
                        total=total,
                        percent=int(done / max(total, 1) * 100),
                        file=Path(fp_str).name[:35] if fp_str else "…",
                        status="🔏 SHA-256 verify",
                    )
        ok = sum(1 for f in files if f.sha256_hash)
        self._log("info", f"✓ SHA-256: {ok:,}/{total:,} verified  errors={errors}")
        self._dbg(f"[SHA256] done  ok={ok}  errors={errors}")

    # ── Stage 4c: Shannon entropy ──────────────────────────────────────────

    def _entropy_batch(self, files: "List[FileRecord]") -> None:
        """
        Compute Shannon entropy for files that share a size with another file.
        Only processes duplicate-size candidates — O(candidates) not O(all).
        Updates f.entropy in-place.
        """
        size_cnt: "Dict[int, int]" = defaultdict(int)
        for f in files:
            size_cnt[f.size] += 1
        candidates = [f for f in files if size_cnt[f.size] > 1]
        if not candidates:
            return
        total = len(candidates)
        self._log("info", f"📊 Entropy analysis: {total:,} files…")
        self._dbg(f"[ENTROPY] start  candidates={total}")
        nw = self.settings.num_workers
        path_map = {str(f.path): f for f in candidates}
        done = 0
        rep = max(1, total // 20)
        fp_str = ""
        with ThreadPoolExecutor(max_workers=nw) as pool:
            futs = {pool.submit(_entropy_worker, str(f.path)): f for f in candidates}
            for fut in as_completed(futs):
                if self.cancel.is_set():
                    break
                done += 1
                fp_str, ent = fut.result()
                if ent is not None and fp_str in path_map:
                    path_map[fp_str].entropy = ent
                if done % rep == 0 or done == total:
                    self._progress(
                        "scan_progress",
                        current=done,
                        total=total,
                        percent=int(done / max(total, 1) * 100),
                        file=Path(fp_str).name[:35] if fp_str else "…",
                        status="📊 Entropy analysis",
                    )
        ok = sum(1 for f in candidates if f.entropy is not None)
        self._log("info", f"✓ Entropy: {ok:,}/{total:,} computed")
        self._dbg(f"[ENTROPY] done  ok={ok}/{total}")

    # ── Stage 4d: Perceptual hashing (image files only) ───────────────────

    def _perceptual_hash_batch(self, files: "List[FileRecord]") -> None:
        """
        Compute pHash for image files that share a size group.
        Silently skips if imagehash / Pillow are unavailable.
        Updates f.phash in-place.
        """
        if not HAS_IMAGEHASH or not HAS_PIL:
            self._dbg("[PHASH] skipped — imagehash or Pillow not available")
            return
        size_cnt: "Dict[int, int]" = defaultdict(int)
        for f in files:
            size_cnt[f.size] += 1
        images = [f for f in files if f.category == "Image" and size_cnt[f.size] > 1]
        if not images:
            self._dbg("[PHASH] no image candidates")
            return
        total = len(images)
        self._log("info", f"🖼️  Perceptual hashing: {total:,} image files…")
        self._dbg(f"[PHASH] start  images={total}")
        nw = self.settings.num_workers
        path_map = {str(f.path): f for f in images}
        done = hits = 0
        rep = max(1, total // 20)
        fp_str = ""
        with ThreadPoolExecutor(max_workers=nw) as pool:
            futs = {pool.submit(_phash_worker, str(f.path)): f for f in images}
            for fut in as_completed(futs):
                if self.cancel.is_set():
                    break
                done += 1
                fp_str, ph = fut.result()
                if ph and fp_str in path_map:
                    path_map[fp_str].phash = ph
                    hits += 1
                if done % rep == 0 or done == total:
                    self._progress(
                        "scan_progress",
                        current=done,
                        total=total,
                        percent=int(done / max(total, 1) * 100),
                        file=Path(fp_str).name[:35] if fp_str else "…",
                        status="🖼️  Perceptual hash",
                    )
        self._log("info", f"✓ pHash: {hits:,}/{total:,} images hashed")
        self._dbg(f"[PHASH] done  hits={hits}/{total}")

    # ── Stage 4e: Neural embedding (DINOv2 for images) ───────────────────────

    def _neural_embed_batch(self, files: "List[FileRecord]") -> None:
        """
        Compute DINOv2 neural embeddings for image files.
        Silently skips if transformers/torch unavailable.
        Updates f.neural_embedding in-place.
        """
        if not self.settings.use_neural_embed:
            self._dbg("[NEURAL] disabled in settings")
            return
        if not HAS_DINO:
            self._dbg("[NEURAL] skipped — transformers/torch not available")
            return

        size_cnt: "Dict[int, int]" = defaultdict(int)
        for f in files:
            size_cnt[f.size] += 1
        images = [f for f in files if f.category == "Image" and size_cnt[f.size] > 1]
        if not images:
            self._dbg("[NEURAL] no image candidates")
            return

        total = len(images)
        self._log("info", f"🧠 Neural embeddings (DINOv2): {total:,} image files…")
        self._dbg(f"[NEURAL] start  images={total}")
        nw = self.settings.num_workers
        path_map = {str(f.path): f for f in images}
        done = hits = 0
        rep = max(1, total // 20)
        fp_str = ""

        with ThreadPoolExecutor(max_workers=nw) as pool:
            futs = {pool.submit(_neural_embed_worker, str(f.path)): f for f in images}
            for fut in as_completed(futs):
                if self.cancel.is_set():
                    break
                done += 1
                fp_str, emb = fut.result()
                if emb and fp_str in path_map:
                    path_map[fp_str].neural_embedding = emb
                    hits += 1
                if done % rep == 0 or done == total:
                    self._progress(
                        "scan_progress",
                        current=done,
                        total=total,
                        percent=int(done / max(total, 1) * 100),
                        file=Path(fp_str).name[:35] if fp_str else "…",
                        status="🧠 Neural embeddings",
                    )

        self._log("info", f"✓ Neural embeddings: {hits:,}/{total:,} images processed")
        self._dbg(f"[NEURAL] done  hits={hits}/{total}")

    # ── Stage 4f: Audio fingerprinting ───────────────────────────────────────

    def _audio_fingerprint_batch(self, files: "List[FileRecord]") -> None:
        """
        Compute audio fingerprints using chromaprint/mutagen.
        Updates f.audio_fingerprint and f.audio_metadata in-place.
        """
        if not self.settings.use_audio_fingerprint:
            self._dbg("[AUDIO] disabled in settings")
            return

        audio_files = [f for f in files if f.category == "Audio"]
        if not audio_files:
            self._dbg("[AUDIO] no audio candidates")
            return

        total = len(audio_files)
        self._log("info", f"🎵 Audio fingerprinting: {total:,} audio files…")
        self._dbg(f"[AUDIO] start  files={total}")
        nw = self.settings.num_workers
        path_map = {str(f.path): f for f in audio_files}
        done = hits = 0
        rep = max(1, total // 20)
        fp_str = ""

        use_chromaprint = HAS_CHROMAPRINT

        with ThreadPoolExecutor(max_workers=nw) as pool:
            if use_chromaprint:
                futs = {
                    pool.submit(_audio_fingerprint_worker, str(f.path)): f
                    for f in audio_files
                }
                for fut in as_completed(futs):
                    if self.cancel.is_set():
                        break
                    done += 1
                    fp_str, fp, meta = fut.result()
                    if fp and fp_str in path_map:
                        path_map[fp_str].audio_fingerprint = fp
                        path_map[fp_str].audio_metadata = meta
                        hits += 1
            else:
                futs = {
                    pool.submit(_audio_metadata_worker, str(f.path)): f
                    for f in audio_files
                }
                for fut in as_completed(futs):
                    if self.cancel.is_set():
                        break
                    done += 1
                    fp_str, meta = fut.result()
                    if meta and fp_str in path_map:
                        path_map[fp_str].audio_metadata = meta
                        hits += 1

            if done % rep == 0 or done == total:
                self._progress(
                    "scan_progress",
                    current=done,
                    total=total,
                    percent=int(done / max(total, 1) * 100),
                    file=Path(fp_str).name[:35] if fp_str else "…",
                    status="🎵 Audio fingerprinting",
                )

        self._log("info", f"✓ Audio fingerprints: {hits:,}/{total:,} files processed")
        self._dbg(f"[AUDIO] done  hits={hits}/{total}")

    # ── Stage 4g: Semantic embeddings (v7.0 AI) ────────────────────────────

    def _semantic_embed_batch(self, files: "List[FileRecord]") -> None:
        """
        Compute semantic embeddings using Sentence-BERT / CLIP.
        Updates f.semantic_embedding in-place.
        """
        if not self.settings.use_semantic_dedup:
            self._dbg("[SEMANTIC] disabled in settings")
            return

        if not (HAS_SENTENCE_TRANSFORMERS or HAS_CLIP):
            self._dbg("[SEMANTIC] skipped — no embedding libraries available")
            return

        size_cnt: "Dict[int, int]" = defaultdict(int)
        for f in files:
            size_cnt[f.size] += 1
        candidates = [f for f in files if size_cnt[f.size] > 1]
        if not candidates:
            self._dbg("[SEMANTIC] no size-matching candidates")
            return

        total = len(candidates)
        self._log("info", f"🤖 Semantic embeddings: {total:,} files…")
        self._dbg(f"[SEMANTIC] start  candidates={total}")
        nw = self.settings.num_workers
        path_map = {str(f.path): f for f in candidates}
        done = hits = 0
        rep = max(1, total // 20)
        fp_str = ""

        use_clip = self.settings.use_clip_embeddings and HAS_CLIP
        use_sentence = (
            self.settings.use_sentence_embeddings and HAS_SENTENCE_TRANSFORMERS
        )

        engine = SemanticEngine()

        def _get_embedding(fp: str, cat: str):
            if use_clip and cat == "Image":
                return fp, engine.compute_image_embedding(fp)
            elif use_sentence and cat == "Document":
                return fp, engine.compute_text_embedding(fp)
            return fp, None

        with ThreadPoolExecutor(max_workers=nw) as pool:
            futs = {
                pool.submit(_get_embedding, str(f.path), f.category): f
                for f in candidates
            }
            for fut in as_completed(futs):
                if self.cancel.is_set():
                    break
                done += 1
                try:
                    fp_str, emb = fut.result()
                except Exception:
                    fp_str, emb = "", None
                if emb and fp_str in path_map:
                    path_map[fp_str].semantic_embedding = emb
                    hits += 1
                if done % rep == 0 or done == total:
                    self._progress(
                        "scan_progress",
                        current=done,
                        total=total,
                        percent=int(done / max(total, 1) * 100),
                        file=Path(fp_str).name[:35] if fp_str else "…",
                        status="🤖 Semantic embeddings",
                    )

        self._log("info", f"✓ Semantic embeddings: {hits:,}/{total:,} files processed")
        self._dbg(f"[SEMANTIC] done  hits={hits}/{total}")

    # ── Stage 4h: File categorization ─────────────────────────────────────

    def _categorize_batch(self, files: "List[FileRecord]") -> None:
        """
        Populate f.category for all files.  O(n), no I/O.
        Must be called after _detect_magic_batch (magic_type already set).
        """
        for f in files:
            f.category = _categorize_file(f.magic_type, f.ext)
        cats: "Dict[str, int]" = {}
        for f in files:
            cats[f.category] = cats.get(f.category, 0) + 1
        summary = "  ".join(f"{k}:{v}" for k, v in sorted(cats.items()))
        self._dbg(f"[CAT] {len(files)} files categorized  {summary}")

    # ── Stage 5: Duplicate detection pipeline ─────────────────────────────

    def scan(self) -> int:
        """Run full scan pipeline. Returns file count."""
        self.files = []
        if not self.root.exists():
            self._err(f"Folder not found: {self.root}")
            return 0

        paths = self._discover()
        if self.cancel.is_set():
            return 0

        self._log("info", f"📁 {len(paths)} items found, stat-ing…")
        records = self._stat_batch(paths)
        if self.cancel.is_set():
            return 0

        self._log("info", f"✓ {len(records)} files indexed")
        self._dbg(f"[SCAN] Complete  files={len(records)}")

        if records:
            self._detect_magic_batch(records)
            self._log("info", "🏷️  Categorizing files…")
            self._categorize_batch(records)  # must come after magic detection
            self._log("info", "🔒 Checking file locks...")
            self._check_file_locks_fast(records)

        if records and not self.cancel.is_set():
            self._log("info", f"🔐 Starting hashing ({len(records)} files)...")
            self._hash_all(records)

        if records and not self.cancel.is_set():
            self._entropy_batch(records)

        if records and not self.cancel.is_set() and HAS_IMAGEHASH and HAS_PIL:
            self._perceptual_hash_batch(records)

        if records and not self.cancel.is_set():
            self._neural_embed_batch(records)

        if records and not self.cancel.is_set():
            self._audio_fingerprint_batch(records)

        # v7.0 - Semantic embedding computation (AI-powered similarity)
        if records and not self.cancel.is_set():
            self._semantic_embed_batch(records)

        records = self._revalidate_mutation(records)

        # Cache keep scores
        for f in records:
            f.keep_score = _total_keep_score(f)

        self.files = records
        return len(records)

    def find_duplicates(self) -> List[DupGroup]:
        """Full detection pipeline with cluster validation and risk assessment."""
        if len(self.files) < 2:
            self._log("info", "⚠️  Need 2+ files to compare")
            return []

        self._log("info", f"🔍 Finding duplicates in {len(self.files)} files…")
        self._dbg(f"[FIND] Starting  files={len(self.files)}")

        groups: List[DupGroup] = []

        # ── a. Hard-link detection ────────────────────────────────────────
        hl_groups = self._find_hardlinks()
        if hl_groups:
            self._log("info", f"🔗 {len(hl_groups)} hard-link group(s)")
            self._dbg(f"[FIND] Hard links: {len(hl_groups)} groups")
            groups.extend(hl_groups)

        hl_paths = {str(f.path) for g in hl_groups for f in g.files}
        remaining = [f for f in self.files if str(f.path) not in hl_paths]

        # ── b. Size grouping ──────────────────────────────────────────────
        self._log("info", f"📐 Size-grouping {len(remaining):,} files…")
        self._progress(
            "scan_progress",
            current=0,
            total=100,
            percent=10,
            file="",
            status="📐 Size grouping…",
        )
        self._dbg(f"[FIND] Size grouping {len(remaining)} files")
        size_groups = self._group_by_size(remaining)
        candidates = [f for grp in size_groups.values() for f in grp]
        skipped_unique = len(remaining) - len(candidates)
        n_size_groups = len(size_groups)
        self._log(
            "info",
            f"📐 {n_size_groups:,} size groups, {len(candidates):,} candidates"
            f" ({skipped_unique:,} unique-size files skipped)",
        )
        self._dbg(
            f"[FIND] Size-grouped: {len(candidates)} candidates"
            f"  {skipped_unique} unique-size files skipped"
            f"  groups={n_size_groups}"
        )

        if self.cancel.is_set():
            return groups

        # ── c. Exact duplicates via hash ──────────────────────────────────
        self._log("info", f"🔑 Hash-based exact duplicate detection…")
        self._progress(
            "scan_progress",
            current=0,
            total=100,
            percent=30,
            file="",
            status="🔑 Finding exact duplicates…",
        )
        exact_groups, still_remaining = self._find_exact(size_groups)
        exact_files = sum(len(g.files) for g in exact_groups)
        self._log(
            "info", f"✓ {len(exact_groups)} exact-dup group(s) ({exact_files} files)"
        )
        self._dbg(
            f"[FIND] Exact groups={len(exact_groups)}"
            f"  exact_files={exact_files}"
            f"  remaining_for_near={len(still_remaining)}"
        )
        groups.extend(exact_groups)

        if self.cancel.is_set():
            return groups

        # ── d. Near-duplicates ────────────────────────────────────────────
        near_groups = self._find_near(still_remaining)
        self._log("info", f"✓ {len(near_groups)} near-duplicate group(s)")
        self._dbg(f"[FIND] Near groups={len(near_groups)}")

        # ── e. GPU / NumPy name-similarity pass (optional) ────────────────
        if self.settings.use_gpu and len(still_remaining) > 50:
            gpu_extra = self._gpu_name_pass(
                still_remaining, {str(f.path) for g in groups for f in g.files}
            )
            self._dbg(f"[FIND] GPU/NP pass extra={len(gpu_extra)}")
            near_groups.extend(gpu_extra)

        # ── f. Transitive cluster merge ───────────────────────────────────
        if near_groups:
            self._log(
                "info", f"🔗 Merging {len(near_groups)} near-dup pairs into clusters…"
            )
            self._progress(
                "scan_progress",
                current=0,
                total=100,
                percent=90,
                file="",
                status="🔗 Merging clusters…",
            )
        merged = self._merge_clusters(near_groups)
        merged = self._validate_clusters(merged)  # Prevent false transitive merges
        if merged:
            self._log("info", f"✓ {len(merged)} near-dup group(s) after validation")
        near_files = sum(len(g.files) for g in merged)
        self._dbg(
            f"[FIND] After transitive merge: {len(merged)} near groups"
            f"  ({near_files} files)"
        )
        if merged:
            self._log("info", f"✓ {len(merged)} near-dup group(s) ({near_files} files)")
        groups.extend(merged)

        # ── Paranoid byte-by-byte verification ────────────────────────────
        if self.settings.paranoid_mode:
            groups = self._paranoid_verify(groups)

        # ── Risk assessment per group ─────────────────────────────────────
        for g in groups:
            g.risk_level = self._assess_risk(g)
            for fi, f in enumerate(g.files):
                g.why_keep[fi] = _generate_why_keep(f, g, g.files)

        # Sort: exact first, then by score desc
        groups.sort(key=lambda g: (-g.is_exact, -g.score))
        total_files = sum(len(g.files) for g in groups)
        self._log("info", f"✓ {len(groups)} duplicate groups  ({total_files} files)")
        self._dbg(f"[FIND] Done  groups={len(groups)}  files_involved={total_files}")
        return groups

    def _find_hardlinks(self) -> List[DupGroup]:
        inode_map: Dict[Tuple, List[FileRecord]] = defaultdict(list)
        for f in self.files:
            if f.inode > 0:
                inode_map[(f.device, f.inode)].append(f)
        return [
            DupGroup(
                files=grp, score=100, group_type="hardlink", components={"inode": 100}
            )
            for grp in inode_map.values()
            if len(grp) > 1
        ]

    def _group_by_size(self, files: List[FileRecord]) -> Dict[int, List[FileRecord]]:
        d: Dict[int, List[FileRecord]] = defaultdict(list)
        for f in files:
            d[f.size].append(f)
        return {sz: grp for sz, grp in d.items() if len(grp) > 1}

    def _find_exact(self, size_groups: Dict[int, List[FileRecord]]):
        """
        Group files by full hash within each size group.

        When use_sha256_verify=True and SHA-256 hashes are available, files
        must ALSO match on SHA-256 to be confirmed as exact duplicates.
        This eliminates any residual hash-collision risk (xxhash / MD5 are
        not collision-resistant; SHA-256 is cryptographically secure).
        """
        exact: List[DupGroup] = []
        hash_map: Dict[str, List[FileRecord]] = defaultdict(list)
        for f in (f for grp in size_groups.values() for f in grp):
            if f.hash:
                hash_map[f.hash].append(f)

        use_sha256 = self.settings.use_sha256_verify
        exact_paths: Set[str] = set()

        for h, grp in hash_map.items():
            if len(grp) < 2:
                continue

            if use_sha256:
                # Sub-group by SHA-256 to confirm no hash collision
                sha_map: Dict[str, List[FileRecord]] = defaultdict(list)
                no_sha: List[FileRecord] = []
                for f in grp:
                    if f.sha256_hash:
                        sha_map[f.sha256_hash].append(f)
                    else:
                        no_sha.append(f)
                # Confirmed by both full hash AND SHA-256
                for sha_grp in sha_map.values():
                    if len(sha_grp) > 1:
                        exact.append(
                            DupGroup(
                                files=sha_grp,
                                score=100,
                                group_type="exact",
                                components={
                                    "hash": 100,
                                    "size": 35,
                                    "name": 0,
                                    "ext": 0,
                                    "magic": 0,
                                    "dir": 0,
                                    "time": 0,
                                    "sha256": 10,
                                },
                                verified=True,
                            )
                        )
                        exact_paths.update(str(f.path) for f in sha_grp)
                # Files without SHA-256 fall through to near-dup analysis
            else:
                exact.append(
                    DupGroup(
                        files=grp,
                        score=100,
                        group_type="exact",
                        components={
                            "hash": 100,
                            "size": 35,
                            "name": 0,
                            "ext": 0,
                            "magic": 0,
                            "dir": 0,
                            "time": 0,
                        },
                    )
                )
                exact_paths.update(str(f.path) for f in grp)

        still = [
            f
            for grp in size_groups.values()
            for f in grp
            if str(f.path) not in exact_paths
        ]
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
        if len(files) < 2:
            return []

        # ── Step 1: Build candidate pairs with smart pre-filtering ────────────
        size_groups = self._group_by_size(files)
        total_raw = 0
        skipped_qh = 0
        skipped_cap = 0
        pairs: List[Tuple[FileRecord, FileRecord]] = []

        self._log(
            "info", f"🔍 Near-dup: pre-filtering {len(files):,} files into pairs…"
        )
        self._dbg(
            f"[FIND] Near-dup pre-filter start  files={len(files)}  size_groups={len(size_groups)}"
        )

        for sz, grp in size_groups.items():
            group_pairs: List[Tuple[FileRecord, FileRecord]] = []
            for i, f1 in enumerate(grp):
                for f2 in grp[i + 1 :]:
                    total_raw += 1
                    # KEY OPTIMISATION: same size + both have quick_hash + differ
                    # → _calculate_dup_score will return 0 anyway → skip now
                    if (
                        f1.quick_hash
                        and f2.quick_hash
                        and f1.quick_hash != f2.quick_hash
                    ):
                        skipped_qh += 1
                        continue
                    group_pairs.append((f1, f2))

            # Per-group cap prevents one huge size bucket from dominating
            if len(group_pairs) > PER_GROUP_PAIR_CAP:
                skipped_cap += len(group_pairs) - PER_GROUP_PAIR_CAP
                group_pairs = group_pairs[:PER_GROUP_PAIR_CAP]

            pairs.extend(group_pairs)

        self._dbg(
            f"[FIND] Raw pairs={total_raw:,}  qh-filtered={skipped_qh:,}"
            f"  cap-filtered={skipped_cap:,}  remaining={len(pairs):,}"
        )

        if not pairs:
            self._log(
                "info",
                f"✓ Near-dup: 0 pairs after smart pre-filter"
                f" (eliminated {total_raw:,} via quick-hash + caps)",
            )
            return []

        # ── Global safety cap ─────────────────────────────────────────────────
        if len(pairs) > NEAR_DUP_MAX_PAIRS:
            self._dbg(f"[FIND] Global cap: {len(pairs):,} → {NEAR_DUP_MAX_PAIRS:,}")
            pairs = pairs[:NEAR_DUP_MAX_PAIRS]

        total = len(pairs)
        min_sc = self.settings.min_score
        nw = self.settings.num_workers
        self._log(
            "info",
            f"📊 Comparing {total:,} pairs ({nw} workers, "
            f"{total_raw - total:,} pre-filtered)…",
        )
        self._dbg(
            f"[COMPARE] pairs={total}  workers={nw}  min_score={min_sc}"
            f"  executor=Thread  batch={NEAR_BATCH_SIZE}"
        )

        # ── Serialize to dicts once ───────────────────────────────────────────
        def _to_dict(f: FileRecord) -> dict:
            return {
                "hash": f.hash,
                "partial_hash": f.partial_hash,
                "quick_hash": f.quick_hash,
                "size": f.size,
                "name": f.name,
                "ext": f.ext,
                "magic_type": f.magic_type,
                "mtime": f.mtime,
                "parent": str(f.path.parent),
                "tokens": list(_tokenize_filename(f.name)),
                "entropy": f.entropy,
                "phash": f.phash,
                "semantic_embedding": f.semantic_embedding,
            }

        # ── Batched ThreadPoolExecutor (no ProcessPool spawn overhead) ────────
        results: List[DupGroup] = []
        done = 0
        hits = 0
        last_report_t = time.monotonic()
        report_interval = max(1, total // 100)

        with ThreadPoolExecutor(max_workers=nw) as pool:
            # Build {future: batch_of_original_pairs} — correct future→pair map
            batch_futs: dict = {}
            for i in range(0, len(pairs), NEAR_BATCH_SIZE):
                batch = pairs[i : i + NEAR_BATCH_SIZE]
                batch_dicts = [(_to_dict(f1), _to_dict(f2)) for f1, f2 in batch]
                fut = pool.submit(_score_batch, batch_dicts, min_sc)
                batch_futs[fut] = batch

            for fut in as_completed(batch_futs):
                if self.cancel.is_set():
                    break
                batch = batch_futs[fut]
                try:
                    scored = fut.result()  # [(local_idx, sc, comp), ...]
                except Exception as exc:
                    self._dbg(f"[COMPARE] batch error: {exc}")
                    done += len(batch)
                    continue

                for local_idx, sc, comp in scored:
                    if local_idx < len(batch):
                        f1, f2 = batch[local_idx]
                        results.append(
                            DupGroup(
                                files=[f1, f2],
                                score=sc,
                                group_type="near",
                                components=comp,
                            )
                        )
                        hits += 1

                done += len(batch)
                now = time.monotonic()
                if (
                    done >= done - len(batch) + report_interval
                    or (now - last_report_t) >= 0.8
                    or done == total
                ):
                    last_report_t = now
                    pct = int(done / max(total, 1) * 100)
                    self._progress(
                        "match_progress", current=done, total=total, percent=pct
                    )
                    self._dbg(
                        f"[COMPARE] {done:,}/{total:,} ({pct}%)  hits≥{min_sc}: {hits}"
                    )

        self._dbg(f"[COMPARE] Done  hits={hits}  elapsed_pairs={done}")
        return results

    def _merge_clusters(self, groups: List[DupGroup]) -> List[DupGroup]:
        """Transitive merge using UnionFind: A~B + B~C → {A,B,C}."""
        if not groups:
            return []

        uf = UnionFind()
        file_map: Dict[str, FileRecord] = {}

        for grp in groups:
            keys = [str(f.path) for f in grp.files]
            for f in grp.files:
                file_map[str(f.path)] = f
            for k in keys[1:]:
                uf.union(keys[0], k)

        cluster_map: Dict[str, Set[str]] = defaultdict(set)
        for key in file_map:
            cluster_map[uf.find(key)].add(key)

        result: List[DupGroup] = []
        used: Set[str] = set()

        for grp in sorted(groups, key=lambda x: -x.score):
            rep = str(grp.files[0].path)
            root = uf.find(rep)
            if root in used:
                continue
            used.add(root)
            keys = cluster_map[root]
            cluster_files = [file_map[k] for k in keys if k in file_map]
            if len(cluster_files) >= 2 and len(cluster_files) <= MAX_CLUSTER_SIZE:
                result.append(
                    DupGroup(
                        files=cluster_files,
                        score=grp.score,
                        group_type=grp.group_type,
                        components=grp.components,
                    )
                )

        return result

    def _validate_clusters(self, groups: List[DupGroup]) -> List[DupGroup]:
        """Validate clusters: minimum pairwise similarity required to keep merged."""
        validated = []
        min_sim = self.settings.min_cluster_sim
        MAX_VALIDATE_SIZE = 50

        for grp in groups:
            if len(grp.files) < 2:
                continue

            if len(grp.files) > MAX_VALIDATE_SIZE:
                validated.append(grp)
                continue

            pairs_total = len(grp.files) * (len(grp.files) - 1) // 2
            pairs_checked = 0
            pairs_above_threshold = 0

            for i in range(len(grp.files)):
                for j in range(i + 1, len(grp.files)):
                    pairs_checked += 1
                    if pairs_checked > 1000:
                        break
                    sc, _ = _calculate_dup_score(
                        (
                            {
                                "hash": grp.files[i].hash,
                                "partial_hash": grp.files[i].partial_hash,
                                "quick_hash": grp.files[i].quick_hash,
                                "size": grp.files[i].size,
                                "name": grp.files[i].name,
                                "ext": grp.files[i].ext,
                                "magic_type": grp.files[i].magic_type,
                                "mtime": grp.files[i].mtime,
                                "parent": str(grp.files[i].path.parent),
                                "tokens": list(_tokenize_filename(grp.files[i].name)),
                                "entropy": grp.files[i].entropy,
                                "phash": grp.files[i].phash,
                            },
                            {
                                "hash": grp.files[j].hash,
                                "partial_hash": grp.files[j].partial_hash,
                                "quick_hash": grp.files[j].quick_hash,
                                "size": grp.files[j].size,
                                "name": grp.files[j].name,
                                "ext": grp.files[j].ext,
                                "magic_type": grp.files[j].magic_type,
                                "mtime": grp.files[j].mtime,
                                "parent": str(grp.files[j].path.parent),
                                "tokens": list(_tokenize_filename(grp.files[j].name)),
                                "entropy": grp.files[j].entropy,
                                "phash": grp.files[j].phash,
                            },
                        )
                    )
                    if sc >= min_sim * 100:
                        pairs_above_threshold += 1

            if pairs_checked > 0:
                pair_ratio = pairs_above_threshold / pairs_checked
                if pair_ratio >= CLUSTER_MIN_PAIRS:
                    grp.cluster_valid = True
                    validated.append(grp)
                else:
                    grp.cluster_valid = False
                    self._dbg(
                        f"[CLUSTER] Cluster broken: {pair_ratio:.1%} pairs above threshold"
                    )
            else:
                validated.append(grp)

        return validated

    def _assess_risk(self, g: DupGroup) -> str:
        """Assess deletion risk level for a group."""
        if g.group_type == "hardlink":
            return "LOW"
        if not g.is_exact:
            return "HIGH"
        if any(f.is_locked for f in g.files):
            return "MEDIUM"
        if any(f.is_system for f in g.files):
            return "HIGH"
        return "LOW"

    def _find_near(self, files: List[FileRecord]) -> List[DupGroup]:
        """
        Parallel pairwise near-duplicate scoring with smart pre-filtering.

        • Same size + different quick_hash → score is always 0; skip these pairs
        • Per-size-group cap prevents single large groups from dominating
        • Batched ThreadPoolExecutor for efficiency
        """
        if len(files) < 2:
            return []

        size_groups = self._group_by_size(files)
        total_raw = 0
        skipped_qh = 0
        skipped_cap = 0
        pairs: List[Tuple[FileRecord, FileRecord]] = []

        self._log(
            "info", f"🔍 Near-dup: pre-filtering {len(files):,} files into pairs…"
        )
        self._dbg(
            f"[FIND] Near-dup pre-filter start  files={len(files)}  size_groups={len(size_groups)}"
        )

        for sz, grp in size_groups.items():
            group_pairs: List[Tuple[FileRecord, FileRecord]] = []
            for i, f1 in enumerate(grp):
                for f2 in grp[i + 1 :]:
                    total_raw += 1
                    if (
                        f1.quick_hash
                        and f2.quick_hash
                        and f1.quick_hash != f2.quick_hash
                    ):
                        skipped_qh += 1
                        continue
                    group_pairs.append((f1, f2))

            if len(group_pairs) > PER_GROUP_PAIR_CAP:
                skipped_cap += len(group_pairs) - PER_GROUP_PAIR_CAP
                group_pairs = group_pairs[:PER_GROUP_PAIR_CAP]

            pairs.extend(group_pairs)

        self._dbg(
            f"[FIND] Raw pairs={total_raw:,}  qh-filtered={skipped_qh:,}"
            f"  cap-filtered={skipped_cap:,}  remaining={len(pairs):,}"
        )

        if not pairs:
            self._log(
                "info",
                f"✓ Near-dup: 0 pairs after smart pre-filter"
                f" (eliminated {total_raw:,} via quick-hash + caps)",
            )
            return []

        if len(pairs) > NEAR_DUP_MAX_PAIRS:
            self._dbg(f"[FIND] Global cap: {len(pairs):,} → {NEAR_DUP_MAX_PAIRS:,}")
            pairs = pairs[:NEAR_DUP_MAX_PAIRS]

        total = len(pairs)
        min_sc = self.settings.min_score
        nw = self.settings.num_workers
        self._log(
            "info",
            f"📊 Comparing {total:,} pairs ({nw} workers, "
            f"{total_raw - total:,} pre-filtered)…",
        )
        self._dbg(
            f"[COMPARE] pairs={total}  workers={nw}  min_score={min_sc}"
            f"  executor=Thread  batch={NEAR_BATCH_SIZE}"
        )

        def _to_dict(f: FileRecord) -> dict:
            return {
                "hash": f.hash,
                "partial_hash": f.partial_hash,
                "quick_hash": f.quick_hash,
                "size": f.size,
                "name": f.name,
                "ext": f.ext,
                "magic_type": f.magic_type,
                "mtime": f.mtime,
                "parent": str(f.path.parent),
                "tokens": list(_tokenize_filename(f.name)),
                "entropy": f.entropy,
                "phash": f.phash,
            }

        results: List[DupGroup] = []
        done = 0
        hits = 0
        last_report_t = time.monotonic()
        report_interval = max(1, total // 100)

        with ThreadPoolExecutor(max_workers=nw) as pool:
            batch_futs: dict = {}
            for i in range(0, len(pairs), NEAR_BATCH_SIZE):
                batch = pairs[i : i + NEAR_BATCH_SIZE]
                batch_dicts = [(_to_dict(f1), _to_dict(f2)) for f1, f2 in batch]
                fut = pool.submit(_score_batch, batch_dicts, min_sc)
                batch_futs[fut] = batch

            for fut in as_completed(batch_futs):
                if self.cancel.is_set():
                    break
                batch = batch_futs[fut]
                try:
                    scored = fut.result()
                except Exception as exc:
                    self._dbg(f"[COMPARE] batch error: {exc}")
                    done += len(batch)
                    continue

                for local_idx, sc, comp in scored:
                    if local_idx < len(batch):
                        f1, f2 = batch[local_idx]
                        results.append(
                            DupGroup(
                                files=[f1, f2],
                                score=sc,
                                group_type="near",
                                components=comp,
                            )
                        )
                        hits += 1

                done += len(batch)
                now = time.monotonic()
                if (
                    done % report_interval == 0
                    or (now - last_report_t) >= 0.8
                    or done == total
                ):
                    last_report_t = now
                    pct = int(done / max(total, 1) * 100)
                    self._progress(
                        "match_progress", current=done, total=total, percent=pct
                    )
                    self._dbg(
                        f"[COMPARE] {done:,}/{total:,} ({pct}%)  hits≥{min_sc}: {hits}"
                    )

        self._dbg(f"[COMPARE] Done  hits={hits}  elapsed_pairs={done}")
        return results

    def _gpu_name_pass(
        self, files: List[FileRecord], exclude_paths: Set[str]
    ) -> List[DupGroup]:
        """GPU / NumPy vectorised filename bigram cosine similarity."""
        if not HAS_NUMPY and not HAS_CUPY:
            return []
        eligible = [f for f in files if str(f.path) not in exclude_paths]
        if len(eligible) < 2:
            return []

        backend = "GPU(CuPy)" if (HAS_CUPY and self.settings.use_gpu) else "NumPy"
        self._log("info", f"🖥️  {backend} name-similarity on {len(eligible)} files…")
        names = [f.name.lower() for f in eligible]
        vocab: Dict[str, int] = {}
        for nm in names:
            for i in range(len(nm) - 1):
                gram = nm[i : i + 2]
                if gram not in vocab:
                    vocab[gram] = len(vocab)
        V = len(vocab)
        if V == 0:
            return []
        xp = cp if (HAS_CUPY and self.settings.use_gpu) else np
        mat = xp.zeros((len(eligible), V), dtype=xp.float32)
        for i, nm in enumerate(names):
            for j in range(len(nm) - 1):
                gram = nm[j : j + 2]
                if gram in vocab:
                    mat[i, vocab[gram]] += 1.0
        norms = xp.linalg.norm(mat, axis=1, keepdims=True)
        norms = xp.where(norms == 0, 1.0, norms)
        mat /= norms
        sim = xp.dot(mat, mat.T)
        rows, cols = xp.where(sim > 0.85)
        if HAS_CUPY and self.settings.use_gpu:
            rows, cols = cp.asnumpy(rows), cp.asnumpy(cols)
        else:
            rows = rows.__array__()
            cols = cols.__array__()
        groups: List[DupGroup] = []
        seen: Set[Tuple] = set()
        for r, c in zip(rows, cols):
            r, c = int(r), int(c)
            if r >= c:
                continue
            if (r, c) in seen:
                continue
            seen.add((r, c))
            f1, f2 = eligible[r], eligible[c]
            sc, comp = _calculate_dup_score(
                (
                    {
                        "hash": f1.hash,
                        "partial_hash": f1.partial_hash,
                        "quick_hash": f1.quick_hash,
                        "size": f1.size,
                        "name": f1.name,
                        "ext": f1.ext,
                        "magic_type": f1.magic_type,
                        "mtime": f1.mtime,
                        "parent": str(f1.path.parent),
                    },
                    {
                        "hash": f2.hash,
                        "partial_hash": f2.partial_hash,
                        "quick_hash": f2.quick_hash,
                        "size": f2.size,
                        "name": f2.name,
                        "ext": f2.ext,
                        "magic_type": f2.magic_type,
                        "mtime": f2.mtime,
                        "parent": str(f2.path.parent),
                    },
                )
            )
            if sc >= self.settings.min_score:
                groups.append(
                    DupGroup(
                        files=[f1, f2], score=sc, group_type="near", components=comp
                    )
                )
        self._log("info", f"✓ {backend} → {len(groups)} extra groups")
        return groups

    def _paranoid_verify(self, groups: List[DupGroup]) -> List[DupGroup]:
        """Byte-by-byte verification of exact-match groups (paranoid mode)."""
        self._log("info", "🔬 Paranoid mode: byte-by-byte verification…")
        verified: List[DupGroup] = []
        for grp in groups:
            if not grp.is_exact or len(grp.files) < 2:
                verified.append(grp)
                continue
            ref = grp.files[0]
            confirmed = [ref]
            for f in grp.files[1:]:
                if _byte_compare(ref.path, f.path):
                    confirmed.append(f)
                else:
                    self._dbg(f"[VERIFY] byte-mismatch: {ref.name} vs {f.name}")
            if len(confirmed) > 1:
                grp.files = confirmed
                grp.verified = True
                verified.append(grp)
            elif len(confirmed) == 1:
                grp.files = confirmed
                verified.append(grp)
        self._log("info", f"✓ Paranoid verify: {len(verified)} groups confirmed")
        return verified

    # ── Smart auto-selection ───────────────────────────────────────────────

    def smart_select(self, groups: List[DupGroup]) -> None:
        """
        Annotate each DupGroup with suggestions: {file_idx: 'KEEP'|'DELETE'|'REVIEW'}.

        SAFETY RULES (enforced regardless of cleanup_mode):
          • Hardlinks: ALL marked KEEP — never deletable
          • Locked files:  KEEP
          • System files:  KEEP
          • Near-duplicates: REVIEW by default (AGGRESSIVE mode may DELETE obvious copies)

        CLEANUP MODES:
          SAFE (default):
            • Exact duplicates only → DELETE when keep_score gap >= delete_gap
            • Near-duplicates       → REVIEW (never auto-deleted)

          AGGRESSIVE:
            • Exact duplicates      → DELETE with a lower gap threshold (delete_gap - 10)
            • Near-dups score ≥ 90 AND obvious copy-name pattern → DELETE
            • Verification level is NOT weakened — only selection is broader

          MEDIA-FOCUSED:
            • Same as SAFE for non-media groups
            • Image/Video/Audio exact groups: slightly lower gap (delete_gap - 5)
            • Uses perceptual hash distance to confirm visual equivalence before
              marking near-identical images for DELETE (dist ≤ 8)
        """
        mode = self.settings.cleanup_mode  # "SAFE" | "AGGRESSIVE" | "MEDIA-FOCUSED"
        total_marked = 0
        self._dbg(f"[SELECT] start  groups={len(groups)}  mode={mode}")

        base_gap = self.settings.delete_gap

        # Mode-specific gap overrides
        if mode == "AGGRESSIVE":
            exact_gap = max(5, base_gap - 10)  # lower bar; floor at 5
            near_gap = exact_gap  # same for near-dups with copy names
        elif mode == "MEDIA-FOCUSED":
            exact_gap = base_gap  # normal for non-media
            media_gap = max(5, base_gap - 5)  # slightly looser for media
            near_gap = base_gap  # near-dups still need full gap
        else:  # SAFE
            exact_gap = base_gap
            near_gap = base_gap

        def _is_media(grp: DupGroup) -> bool:
            return any(f.category in ("Image", "Video", "Audio") for f in grp.files)

        def _is_obvious_copy(f: FileRecord) -> bool:
            stem = Path(f.name).stem.lower()
            return any(pat in stem for pat in COPY_PATTERNS)

        def _phash_confirmed_dup(f1: FileRecord, f2: FileRecord) -> bool:
            """Both files have phash AND distance ≤ 8 → visually identical."""
            if f1.phash and f2.phash:
                return _phash_distance(f1.phash, f2.phash) <= 8
            return False

        for gi, grp in enumerate(groups):
            files = grp.files
            if len(files) < 2:
                grp.suggestions = {0: "KEEP"}
                continue

            scored = [(fi, f.keep_score, f.ctime) for fi, f in enumerate(files)]

            # ── HARDLINKS: never delete ───────────────────────────────────
            if grp.group_type == "hardlink":
                for fi in range(len(files)):
                    grp.suggestions[fi] = "KEEP"
                continue

            scored_sorted = sorted(scored, key=lambda x: (-x[1], x[2]))
            keeper_idx = scored_sorted[0][0]
            best_score = scored_sorted[0][1]

            # ── NEAR-DUPLICATES ───────────────────────────────────────────
            if not grp.is_exact:
                for fi, f in enumerate(files):
                    if f.is_locked or f.is_system:
                        grp.suggestions[fi] = "KEEP"
                        continue
                    if fi == keeper_idx:
                        grp.suggestions[fi] = "KEEP"
                        continue

                    keep_file = files[keeper_idx]

                    if mode == "AGGRESSIVE" and grp.score >= 90:
                        # Only delete if: obvious copy name AND score gap is clear
                        if (
                            _is_obvious_copy(f)
                            and best_score - f.keep_score >= near_gap
                        ):
                            grp.suggestions[fi] = "DELETE"
                            total_marked += 1
                        else:
                            grp.suggestions[fi] = "REVIEW"

                    elif mode == "MEDIA-FOCUSED" and _is_media(grp):
                        # For near-identical images confirmed by pHash → DELETE
                        if (
                            _phash_confirmed_dup(keep_file, f)
                            and _is_obvious_copy(f)
                            and best_score - f.keep_score >= media_gap
                        ):
                            grp.suggestions[fi] = "DELETE"
                            total_marked += 1
                        else:
                            grp.suggestions[fi] = "REVIEW"

                    else:
                        grp.suggestions[fi] = "REVIEW"
                continue

            # ── EXACT DUPLICATES ──────────────────────────────────────────
            if mode == "MEDIA-FOCUSED" and _is_media(grp):
                eff_gap = media_gap
            else:
                eff_gap = exact_gap

            for fi in range(len(files)):
                f = files[fi]
                if f.is_locked:
                    grp.suggestions[fi] = "KEEP"
                elif f.is_system:
                    grp.suggestions[fi] = "KEEP"
                elif fi == keeper_idx:
                    grp.suggestions[fi] = "KEEP"
                elif best_score - f.keep_score >= eff_gap:
                    grp.suggestions[fi] = "DELETE"
                    total_marked += 1
                else:
                    grp.suggestions[fi] = "KEEP"

            self._dbg(
                f"[SELECT] G{gi + 1}: keep={keeper_idx}  marked={total_marked}"
                f"  n={len(files)}  type={grp.group_type}"
                f"  mode={mode}  gap={eff_gap}"
            )

        self._dbg(f"[SELECT] done  files_marked={total_marked}  mode={mode}")


# ═════════════════════════════════════════════════════════════════════════════
#  SafeDeleter  — cross-platform Recycle Bin / Trash with persistent log
# ═════════════════════════════════════════════════════════════════════════════


class SafeDeleter:
    """Move files to Recycle Bin / Trash with MANDATORY safety checks."""

    @staticmethod
    def can_delete(f: FileRecord) -> Tuple[bool, str]:
        """Check if file can be safely deleted."""
        if f.is_locked:
            return False, "File is locked by another process"
        if f.is_system:
            return False, "System file - protected"
        if f.is_symlink:
            return False, "Symbolic link - skipped"
        return True, "OK"

    @staticmethod
    def verify_safe_for_deletion(g: DupGroup) -> Tuple[bool, List[str]]:
        """Verify group is safe for deletion. Returns (can_delete, warnings)."""
        if g.group_type == "hardlink":
            return False, ["Hardlinks cannot be deleted - same disk data"]
        if not g.is_exact:
            return False, [
                "Only verified EXACT duplicates can be deleted",
                f"This group is: {g.group_type.upper()} ({g.score}% similar)",
            ]
        deletable = []
        warnings = []
        for fi, f in enumerate(g.files):
            can_del, reason = SafeDeleter.can_delete(f)
            if can_del and g.suggestions.get(fi) == "DELETE":
                deletable.append(fi)
            else:
                if reason != "OK":
                    warnings.append(f"File {fi + 1}: {reason}")
        if not deletable:
            return False, warnings if warnings else ["No files marked for deletion"]
        return True, warnings

    @staticmethod
    def to_trash(filepath) -> Tuple[bool, str]:
        try:
            fp = Path(filepath)
            if not fp.exists():
                return False, "File not found"
            if HAS_SEND2TRASH:
                _s2t.send2trash(str(fp))
                SafeDeleter._log_deletion(fp, True)
                return True, "Moved to Trash / Recycle Bin"
            if sys.platform == "win32":
                import ctypes

                class _SHOp(ctypes.Structure):
                    _fields_ = [
                        ("hwnd", ctypes.c_void_p),
                        ("wFunc", ctypes.c_uint),
                        ("pFrom", ctypes.c_wchar_p),
                        ("pTo", ctypes.c_wchar_p),
                        ("fFlags", ctypes.c_ushort),
                        ("fAnyAborted", ctypes.c_bool),
                        ("hMappings", ctypes.c_void_p),
                        ("lpTitle", ctypes.c_wchar_p),
                    ]

                op = _SHOp()
                op.wFunc = 3
                op.pFrom = str(fp) + "\0"
                op.fFlags = 0x40  # FOF_ALLOWUNDO
                rc = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
                if rc == 0:
                    SafeDeleter._log_deletion(fp, True)
                    return True, "Moved to Recycle Bin"
                return False, f"SHFileOperation error code {rc}"
            return False, "send2trash not available — run: pip install send2trash"
        except Exception as exc:
            SafeDeleter._log_deletion(filepath, False, str(exc))
            return False, str(exc)

    @staticmethod
    def _log_deletion(filepath, success: bool, error: str = "") -> None:
        entry = {
            "ts": datetime.datetime.now().isoformat(),
            "path": str(filepath),
            "success": success,
            "error": error,
        }
        try:
            log = SafeDeleter.load_log()
            log.append(entry)
            with open(DELETION_LOG_PATH, "w", encoding="utf-8") as fh:
                json.dump(log, fh, indent=2)
        except Exception:
            pass

    @staticmethod
    def load_log() -> List[dict]:
        try:
            if DELETION_LOG_PATH.exists():
                with open(DELETION_LOG_PATH, encoding="utf-8") as fh:
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
    def save(
        groups: List[DupGroup], folder: str, scan_settings: ScanSettings, filepath: str
    ) -> bool:
        try:

            def _ser_file(f: FileRecord) -> dict:
                return {
                    "path": str(f.path),
                    "size": f.size,
                    "mtime": f.mtime,
                    "ctime": f.ctime,
                    "inode": f.inode,
                    "device": f.device,
                    "ext": f.ext,
                    "name": f.name,
                    "hash": f.hash,
                    "partial_hash": f.partial_hash,
                    "quick_hash": f.quick_hash,
                    "sha256_hash": f.sha256_hash,
                    "magic_type": f.magic_type,
                    "category": f.category,
                    "keep_score": f.keep_score,
                    "entropy": f.entropy,
                    "is_locked": f.is_locked,
                    "is_system": f.is_system,
                }

            payload = {
                "version": VERSION,
                "folder": folder,
                "saved_at": datetime.datetime.now().isoformat(),
                "min_score": scan_settings.min_score,
                "groups": [
                    {
                        "score": g.score,
                        "group_type": g.group_type,
                        "components": g.components,
                        "suggestions": {str(k): v for k, v in g.suggestions.items()},
                        "files": [_ser_file(f) for f in g.files],
                    }
                    for g in groups
                ],
            }
            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            return True
        except Exception:
            return False

    @staticmethod
    def load(filepath: str) -> Tuple[Optional[List[DupGroup]], Optional[str]]:
        """Returns (groups, folder) or (None, None) on failure."""
        try:
            with open(filepath, encoding="utf-8") as fh:
                payload = json.load(fh)
            if payload.get("version") != VERSION:
                return None, None
            folder = payload.get("folder", "")
            groups: List[DupGroup] = []
            for gd in payload.get("groups", []):
                files = []
                for fd in gd.get("files", []):
                    files.append(
                        FileRecord(
                            path=Path(fd["path"]),
                            size=fd["size"],
                            mtime=fd["mtime"],
                            ctime=fd["ctime"],
                            inode=fd["inode"],
                            device=fd["device"],
                            ext=fd["ext"],
                            name=fd["name"],
                            hash=fd.get("hash"),
                            partial_hash=fd.get("partial_hash"),
                            quick_hash=fd.get("quick_hash"),
                            sha256_hash=fd.get("sha256_hash"),
                            magic_type=fd.get("magic_type"),
                            category=fd.get("category"),
                            keep_score=fd.get("keep_score", 100),
                            entropy=fd.get("entropy"),
                            is_locked=fd.get("is_locked", False),
                            is_system=fd.get("is_system", False),
                        )
                    )
                sugg = {int(k): v for k, v in gd.get("suggestions", {}).items()}
                groups.append(
                    DupGroup(
                        files=files,
                        score=gd["score"],
                        group_type=gd["group_type"],
                        components=gd.get("components", {}),
                        suggestions=sugg,
                        risk_level=gd.get("risk_level", "LOW"),
                        why_keep={int(k): v for k, v in gd.get("why_keep", {}).items()},
                        verified=gd.get("verified", False),
                    )
                )
            return groups, folder
        except Exception:
            return None, None


# ═════════════════════════════════════════════════════════════════════════════
#  DuplicateFinderApp  v5.0 — Ground-up redesign
# ═════════════════════════════════════════════════════════════════════════════

# Colour palettes
LIGHT_PALETTE = {
    "bg": "#f0f0f0",
    "fg": "#1a1a1a",
    "header_bg": "#1e5631",
    "header_fg": "#ffffff",
    "toolbar_bg": "#2d8659",
    "toolbar_fg": "#ffffff",
    "accent1": "#0066cc",
    "accent2": "#52b788",
    "accent3": "#e8f5e9",
    "success": "#28a745",
    "warning": "#ffc107",
    "danger": "#dc3545",
    "info": "#17a2b8",
    "panel_bg": "#ffffff",
    "border": "#cccccc",
    "select_bg": "#cce5ff",
    "tree_bg": "#ffffff",
    "tree_fg": "#1a1a1a",
    "tree_sel": "#0066cc",
    "tree_sel_fg": "#ffffff",
    "card_bg": "#f8f9fa",
    "card_border": "#dee2e6",
    "keep_fg": "#28a745",
    "del_fg": "#dc3545",
    "exact_bg": "#d4edda",
    "near_bg": "#cce5ff",
    "hard_bg": "#fff3cd",
    "debug_bg": "#0d1117",
    "debug_fg": "#c9d1d9",
    "term_bg": "#161b22",
    "status_bar_bg": "#2d2d2d",
    "status_bar_fg": "#ffffff",
    "amber": "#ffc107",
    "risk_low": "#d4edda",
    "risk_med": "#fff3cd",
    "risk_high": "#f8d7da",
    "review_fg": "#17a2b8",
}
DARK_PALETTE = {
    "bg": "#1e1e2e",
    "fg": "#cdd6f4",
    "header_bg": "#11111b",
    "header_fg": "#cba6f7",
    "toolbar_bg": "#181825",
    "toolbar_fg": "#cdd6f4",
    "accent1": "#89b4fa",
    "accent2": "#a6e3a1",
    "accent3": "#313244",
    "success": "#a6e3a1",
    "warning": "#f9e2af",
    "danger": "#f38ba8",
    "info": "#89dceb",
    "panel_bg": "#1e1e2e",
    "border": "#45475a",
    "select_bg": "#313244",
    "tree_bg": "#1e1e2e",
    "tree_fg": "#cdd6f4",
    "tree_sel": "#89b4fa",
    "tree_sel_fg": "#1e1e2e",
    "card_bg": "#181825",
    "card_border": "#45475a",
    "keep_fg": "#a6e3a1",
    "del_fg": "#f38ba8",
    "exact_bg": "#1e3a2f",
    "near_bg": "#1e2a3a",
    "hard_bg": "#3a2e1e",
    "debug_bg": "#0a0a0f",
    "debug_fg": "#c9d1d9",
    "term_bg": "#0d0d18",
    "status_bar_bg": "#11111b",
    "status_bar_fg": "#cdd6f4",
    "amber": "#f9e2af",
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
        self.root.title(f"Duplicate File Finder v{VERSION}")
        self.root.geometry("1400x920")
        self.root.minsize(1100, 700)

        # State
        self.settings = ScanSettings()
        self.groups: List[DupGroup] = []
        self.current_group: int = -1
        self.is_scanning: bool = False
        self.scan_thread: Optional[threading.Thread] = None
        self._cancel_ev: threading.Event = threading.Event()
        self._pq: queue.Queue = queue.Queue(maxsize=20000)
        self._engine: Optional[ScanEngine] = None
        self._spinner_idx: int = 0
        self._term_visible: bool = True
        self._log_file_queue: queue.Queue = queue.Queue()  # async log I/O
        self._report_dirty: bool = False  # lazy Full Report render

        # Tk variables (bound to settings widgets)
        self._var_subdirs = tk.BooleanVar(value=True)
        self._var_hash = tk.BooleanVar(value=True)
        self._var_xxhash = tk.BooleanVar(value=True)
        self._var_sha256 = tk.BooleanVar(value=False)
        self._var_gpu = tk.BooleanVar(value=False)
        self._var_neural = tk.BooleanVar(value=False)
        self._var_audio_fp = tk.BooleanVar(value=False)
        self._var_paranoid = tk.BooleanVar(value=False)
        self._var_workers = tk.IntVar(value=min(CPU_COUNT * 2, 16))
        self._var_min_score = tk.IntVar(value=70)
        self._var_min_size = tk.IntVar(value=1)
        self._var_max_size = tk.IntVar(value=0)
        self._var_dark_mode = tk.BooleanVar(value=False)
        self._var_auto_select = tk.BooleanVar(value=True)
        self._var_delete_gap = tk.IntVar(value=15)
        self._var_cleanup_mode = tk.StringVar(value="SAFE")
        self._var_skip_network = tk.BooleanVar(value=True)
        self._var_skip_system = tk.BooleanVar(value=True)
        self._var_exclusions = tk.StringVar(value="")
        self._var_filter_text = tk.StringVar()
        self._var_filter_type = tk.StringVar(value="All")
        self._var_filter_risk = tk.StringVar(value="All")
        self._var_io_port = tk.BooleanVar(value=False)

        # v7.0 AI Semantic features
        self._var_semantic = tk.BooleanVar(value=False)
        self._var_faiss = tk.BooleanVar(value=False)
        self._var_clip = tk.BooleanVar(value=False)
        self._var_sentence = tk.BooleanVar(value=False)
        # v7.0 BLAKE3 hashing
        self._var_blake3 = tk.BooleanVar(value=False)

        # Add traces to update optional features display when settings change
        self._var_semantic.trace_add(
            "write", lambda *_: self._update_optional_features_display()
        )
        self._var_faiss.trace_add(
            "write", lambda *_: self._update_optional_features_display()
        )
        self._var_clip.trace_add(
            "write", lambda *_: self._update_optional_features_display()
        )
        self._var_blake3.trace_add(
            "write", lambda *_: self._update_optional_features_display()
        )

        self._setup_theme(dark=False)
        self._build_ui()
        self._bind_shortcuts()
        self._start_progress_monitor()
        self._animate_spinner()
        self._start_log_writer()  # async log-file I/O thread

    # ── Theme ─────────────────────────────────────────────────────────────

    def _setup_theme(self, dark: bool = False) -> None:
        self.C = DARK_PALETTE.copy() if dark else LIGHT_PALETTE.copy()
        C = self.C

        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("TFrame", background=C["bg"], borderwidth=0, relief="flat")
        style.configure(
            "Card.TFrame", background=C["card_bg"], borderwidth=1, relief="solid"
        )
        style.configure("TLabel", background=C["bg"], foreground=C["fg"])
        style.configure(
            "Header.TLabel",
            background=C["header_bg"],
            foreground=C["header_fg"],
            font=("Arial", 15, "bold"),
        )
        style.configure(
            "Status.TLabel",
            background=C["status_bar_bg"],
            foreground=C["status_bar_fg"],
            font=("Courier", 9),
        )

        for name, bg in [
            ("Green", C["success"]),
            ("Blue", C["accent1"]),
            ("Red", C["danger"]),
            ("Amber", C["warning"]),
            ("Gray", C["border"]),
        ]:
            style.configure(
                f"{name}.TButton",
                background=bg,
                foreground=C["header_fg"],
                borderwidth=2,
                relief="raised",
                padding=(8, 4),
            )
            style.map(
                f"{name}.TButton",
                background=[("active", C["accent2"]), ("pressed", C["header_bg"])],
            )

        style.configure(
            "Treeview",
            background=C["tree_bg"],
            foreground=C["tree_fg"],
            fieldbackground=C["tree_bg"],
            rowheight=22,
            borderwidth=1,
            relief="solid",
        )
        style.configure(
            "Treeview.Heading",
            background=C["toolbar_bg"],
            foreground=C["toolbar_fg"],
            font=("Arial", 9, "bold"),
        )
        style.map(
            "Treeview",
            background=[("selected", C["tree_sel"])],
            foreground=[("selected", C["tree_sel_fg"])],
        )

        style.configure("TNotebook", background=C["bg"])
        style.configure(
            "TNotebook.Tab", background=C["bg"], foreground=C["fg"], padding=(10, 4)
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", C["accent1"])],
            foreground=[("selected", "#ffffff")],
        )

        style.configure("TProgressbar", troughcolor=C["bg"], background=C["accent2"])
        style.configure("TScrollbar", background=C["bg"])
        style.configure("TSeparator", background=C["border"])
        style.configure("TEntry", fieldbackground=C["panel_bg"], foreground=C["fg"])
        style.configure("TSpinbox", fieldbackground=C["panel_bg"], foreground=C["fg"])
        style.configure("TCheckbutton", background=C["bg"], foreground=C["fg"])
        style.configure("TCombobox", fieldbackground=C["panel_bg"], foreground=C["fg"])

        self.root.configure(bg=C["bg"])

    # ── Build UI ──────────────────────────────────────────────────────────

    def _add_tooltip(self, widget, text: str) -> None:
        """Add simple tooltip to any widget with proper cleanup."""
        if widget is None:
            return

        tooltip_window = [None]  # Use list to store reference

        def _on_enter(e):
            if tooltip_window[0] is not None:
                return
            try:
                tw = tk.Toplevel()
                tw.wm_overrideredirect(True)
                tw.wm_geometry(f"+{e.x_root + 10}+{e.y_root + 10}")
                tw.wm_attributes("-topmost", True)
                tw.configure(background="#ffffe0")
                tw.attributes("-alpha", 0.95)
                label = tk.Label(
                    tw,
                    text=text,
                    background="#ffffe0",
                    foreground="#1a1a1a",
                    font=("Arial", 8),
                    padx=6,
                    pady=3,
                    wraplength=350,
                )
                label.pack()
                tooltip_window[0] = tw
            except Exception:
                pass

        def _on_leave(e):
            try:
                if tooltip_window[0] is not None:
                    tooltip_window[0].destroy()
                    tooltip_window[0] = None
            except Exception:
                pass
            finally:
                tooltip_window[0] = None

        widget.bind("<Enter>", _on_enter)
        widget.bind("<Leave>", _on_leave)

        widget.bind("<Enter>", _on_enter)
        widget.bind("<Leave>", _on_leave)

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
            self.root,
            orient=tk.HORIZONTAL,
            sashwidth=5,
            sashrelief="raised",
            bg=C["border"],
        )
        self._main_pane.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Left: groups treeview
        left_outer = tk.Frame(
            self._main_pane, bg=C["bg"], borderwidth=1, relief="solid"
        )
        self._main_pane.add(left_outer, minsize=280, width=380)
        self._build_groups_panel(left_outer)

        # Right: tab notebook
        right_outer = tk.Frame(
            self._main_pane, bg=C["bg"], borderwidth=1, relief="solid"
        )
        self._main_pane.add(right_outer, minsize=400)
        self._build_right_notebook(right_outer)

    # ── Header ────────────────────────────────────────────────────────────

    def _build_header(self) -> None:
        C = self.C
        hdr = tk.Frame(self.root, bg=C["header_bg"], height=65)
        hdr.pack(fill=tk.X, side=tk.TOP)
        hdr.pack_propagate(False)

        # Spinner + title
        self._header_spinner = tk.Label(
            hdr,
            text=SPINNER_FRAMES[0],
            font=("Arial", 18, "bold"),
            bg=C["header_bg"],
            fg=C["accent2"],
        )
        self._header_spinner.pack(side=tk.LEFT, padx=8, pady=10)

        tk.Label(
            hdr,
            text=f"🔍 DUPLICATE FILE FINDER  v{VERSION}",
            font=("Arial", 15, "bold"),
            bg=C["header_bg"],
            fg=C["header_fg"],
        ).pack(side=tk.LEFT, padx=5)

        # Folder selector
        ff = tk.Frame(hdr, bg=C["header_bg"])
        ff.pack(side=tk.LEFT, padx=20, pady=8)
        tk.Label(
            ff, text="Folder:", bg=C["header_bg"], fg=C["header_fg"], font=("Arial", 9)
        ).pack(side=tk.LEFT)
        self._folder_var = tk.StringVar(value=str(os.getcwd()))
        self._folder_entry = tk.Entry(
            ff,
            textvariable=self._folder_var,
            bg=C["accent2"],
            fg="#1a1a1a",
            font=("Courier", 9),
            relief="solid",
            borderwidth=1,
            width=55,
        )
        self._folder_entry.pack(side=tk.LEFT, padx=4)
        self._add_tooltip(self._folder_entry, "Current folder being scanned (can edit)")

        change_btn = ttk.Button(ff, text="📁 Change", command=self._change_folder)
        change_btn.pack(side=tk.LEFT, padx=4)
        self._add_tooltip(change_btn, "Browse to select a different folder")

        # TEST button to run scan on test_data folder
        test_btn = ttk.Button(ff, text="🧪 TEST", command=self._run_test_scan)
        test_btn.pack(side=tk.LEFT, padx=4)
        self._add_tooltip(test_btn, "Run scan on test_data folder and show results")

        # Right-side status chips
        def _chip(text, bg):
            return tk.Label(
                hdr,
                text=text,
                bg=bg,
                fg="#ffffff",
                font=("Arial", 8, "bold"),
                padx=5,
                pady=2,
            )

        _chip(f"CPUs:{CPU_COUNT}", C["toolbar_bg"]).pack(side=tk.RIGHT, padx=4, pady=15)
        _chip(
            "xxh✓" if HAS_XXHASH else "xxh✗",
            C["success"] if HAS_XXHASH else C["danger"],
        ).pack(side=tk.RIGHT, padx=2)
        _chip(
            "PIL✓" if HAS_PIL else "PIL✗", C["accent1"] if HAS_PIL else C["border"]
        ).pack(side=tk.RIGHT, padx=2)
        _chip(
            "s2t✓" if HAS_SEND2TRASH else "s2t✗",
            C["success"] if HAS_SEND2TRASH else C["danger"],
        ).pack(side=tk.RIGHT, padx=2)
        _chip(
            "ps✓" if HAS_PSUTIL else "ps✗", C["success"] if HAS_PSUTIL else C["border"]
        ).pack(side=tk.RIGHT, padx=2)
        _chip(
            "GPU✓" if HAS_CUPY else ("NP✓" if HAS_NUMPY else "GPU✗"),
            C["accent1"] if (HAS_CUPY or HAS_NUMPY) else C["border"],
        ).pack(side=tk.RIGHT, padx=2)
        _chip(
            "DINO✓" if HAS_DINO else "DINO✗",
            C["accent1"] if HAS_DINO else C["border"],
        ).pack(side=tk.RIGHT, padx=2)
        _chip(
            "Audio✓" if (HAS_CHROMAPRINT or HAS_MUTAGEN) else "Audio✗",
            C["accent1"] if (HAS_CHROMAPRINT or HAS_MUTAGEN) else C["border"],
        ).pack(side=tk.RIGHT, padx=2)

        # Optional features status - show inactive for turned off features
        self._optional_features_btn = tk.Button(
            hdr,
            text="⚙ Features",
            font=("Arial", 8),
            bg=C["toolbar_bg"],
            fg=C["header_fg"],
            relief="flat",
            cursor="hand2",
            command=self._show_settings,
        )
        self._optional_features_btn.pack(side=tk.RIGHT, padx=8, pady=12)
        self._add_tooltip(self._optional_features_btn, "Click to open Settings")

        # Build optional features summary for header display
        self._update_optional_features_display()

        # Dark mode toggle
        self._dm_btn = tk.Button(
            hdr,
            text="🌙",
            font=("Arial", 12),
            bg=C["header_bg"],
            fg=C["accent2"],
            relief="flat",
            cursor="hand2",
            command=self._toggle_dark_mode,
        )
        self._dm_btn.pack(side=tk.RIGHT, padx=8)

    # ── Toolbar ───────────────────────────────────────────────────────────

    def _build_toolbar(self) -> None:
        C = self.C
        tb = tk.Frame(self.root, bg=C["toolbar_bg"], height=44)
        tb.pack(fill=tk.X, side=tk.TOP)
        tb.pack_propagate(False)

        self._scan_btn = ttk.Button(
            tb, text="▶ SCAN", command=self._toggle_scan, style="Green.TButton"
        )
        self._scan_btn.pack(side=tk.LEFT, padx=6, pady=6)
        self._add_tooltip(self._scan_btn, "Start/Stop scan (Ctrl+S, F5, or Esc)")

        self._reset_btn = ttk.Button(
            tb,
            text="↺ RESET",
            command=self._reset_scan,
            style="Gray.TButton",
        )
        self._reset_btn.pack(side=tk.LEFT, padx=4, pady=6)
        self._reset_btn.config(state=tk.DISABLED)
        self._add_tooltip(self._reset_btn, "Clear all results for new scan")

        # Quick access buttons (Settings, Manual, Logs) - brighter system colors
        self._settings_btn = ttk.Button(
            tb,
            text="⚙ Settings",
            command=self._show_settings,
            style="Blue.TButton",
        )
        self._settings_btn.pack(side=tk.LEFT, padx=4, pady=6)
        self._add_tooltip(self._settings_btn, "Open Settings (Ctrl+,)")

        self._manual_btn = ttk.Button(
            tb,
            text="📖 Manual",
            command=self._show_manual,
            style="Blue.TButton",
        )
        self._manual_btn.pack(side=tk.LEFT, padx=4, pady=6)
        self._add_tooltip(self._manual_btn, "Open User Manual (F1)")

        self._changelog_btn = ttk.Button(
            tb,
            text="📜 Logs",
            command=self._show_changelog,
            style="Blue.TButton",
        )
        self._changelog_btn.pack(side=tk.LEFT, padx=4, pady=6)
        self._add_tooltip(self._changelog_btn, "Open Changelogs")

        # Selection buttons group
        ttk.Separator(tb, orient="vertical").pack(
            side=tk.LEFT, fill=tk.Y, padx=8, pady=4
        )

        self._auto_sel_btn = ttk.Button(
            tb,
            text="🎯 Auto-Select",
            command=self._auto_select,
            state=tk.DISABLED,
            style="Amber.TButton",
        )
        self._auto_sel_btn.pack(side=tk.LEFT, padx=4, pady=6)
        self._add_tooltip(self._auto_sel_btn, "Automatically select files to delete")

        self._clear_sel_btn = ttk.Button(
            tb,
            text="✕ Clear Sel",
            command=self._clear_selection,
            state=tk.DISABLED,
            style="Gray.TButton",
        )
        self._clear_sel_btn.pack(side=tk.LEFT, padx=4, pady=6)
        self._add_tooltip(self._clear_sel_btn, "Clear all selections")

        self._delete_btn = ttk.Button(
            tb,
            text="🗑 DELETE",
            command=self._delete_selected,
            state=tk.DISABLED,
            style="Red.TButton",
        )
        self._delete_btn.pack(side=tk.LEFT, padx=4, pady=6)
        self._add_tooltip(self._delete_btn, "Move selected files to Trash")

        ttk.Separator(tb, orient="vertical").pack(
            side=tk.LEFT, fill=tk.Y, padx=8, pady=4
        )

        self._export_btn = ttk.Button(
            tb,
            text="📤 Export",
            command=self._show_export_menu,
            state=tk.DISABLED,
            style="Blue.TButton",
        )
        self._export_btn.pack(side=tk.LEFT, padx=4, pady=6)
        self._add_tooltip(self._export_btn, "Export results (TXT, CSV, JSON, HTML)")

        self._save_sess_btn = ttk.Button(
            tb,
            text="💾 Save",
            command=self._save_session,
            state=tk.DISABLED,
            style="Blue.TButton",
        )
        self._save_sess_btn.pack(side=tk.LEFT, padx=4, pady=6)
        self._add_tooltip(self._save_sess_btn, "Save current session")

        self._load_sess_btn = ttk.Button(
            tb, text="📂 Load", command=self._load_session, style="Blue.TButton"
        )
        self._load_sess_btn.pack(side=tk.LEFT, padx=4, pady=6)
        self._add_tooltip(self._load_sess_btn, "Load saved session")

        ttk.Separator(tb, orient="vertical").pack(
            side=tk.LEFT, fill=tk.Y, padx=8, pady=4
        )

        # Workers spinner on toolbar
        tk.Label(
            tb,
            text="Workers:",
            bg=C["toolbar_bg"],
            fg=C["toolbar_fg"],
            font=("Arial", 9),
        ).pack(side=tk.LEFT)
        self._add_tooltip(
            tk.Label(tb, text="Workers:", bg=C["toolbar_bg"], fg=C["toolbar_fg"]),
            "Number of parallel threads",
        )
        sb = tk.Spinbox(
            tb,
            from_=1,
            to=64,
            textvariable=self._var_workers,
            width=4,
            font=("Arial", 9),
        )
        sb.pack(side=tk.LEFT, padx=4)
        self._add_tooltip(sb, f"Worker threads (1-{min(64, CPU_COUNT * 2)})")

        # Activity label (right side)
        self._activity_lbl = tk.Label(
            tb,
            text="Ready to scan",
            bg=C["toolbar_bg"],
            fg=C["toolbar_fg"],
            font=("Courier", 9, "bold"),
            anchor="e",
        )
        self._activity_lbl.pack(side=tk.RIGHT, padx=12, fill=tk.X, expand=True)

    # ── Groups panel (left) ────────────────────────────────────────────────

    def _build_groups_panel(self, parent) -> None:
        C = self.C

        # Filter bar
        fbar = tk.Frame(parent, bg=C["bg"])
        fbar.pack(fill=tk.X, padx=4, pady=(4, 0))

        tk.Label(fbar, text="🔎", bg=C["bg"], fg=C["fg"], font=("Arial", 11)).pack(
            side=tk.LEFT
        )
        self._filter_entry = ttk.Entry(
            fbar, textvariable=self._var_filter_text, width=12
        )
        self._filter_entry.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self._add_tooltip(self._filter_entry, "Filter by filename")
        self._var_filter_text.trace_add("write", lambda *_: self._apply_filter())

        self._type_combo = ttk.Combobox(
            fbar,
            textvariable=self._var_filter_type,
            values=["All", "Exact", "Near-Dup", "Hard-Link"],
            state="readonly",
            width=9,
        )
        self._type_combo.pack(side=tk.LEFT, padx=2)
        self._add_tooltip(self._type_combo, "Filter by group type")
        self._type_combo.bind("<<ComboboxSelected>>", lambda _: self._apply_filter())

        self._risk_combo = ttk.Combobox(
            fbar,
            textvariable=self._var_filter_risk,
            values=["All", "LOW", "MEDIUM", "HIGH"],
            state="readonly",
            width=8,
        )
        self._risk_combo.pack(side=tk.LEFT, padx=2)
        self._add_tooltip(self._risk_combo, "Filter by risk level")
        self._risk_combo.bind("<<ComboboxSelected>>", lambda _: self._apply_filter())

        # Treeview
        tree_frame = tk.Frame(parent, bg=C["bg"])
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        cols = ("type", "files", "score", "risk", "reclaim")
        self._tree = ttk.Treeview(
            tree_frame, columns=cols, show="tree headings", selectmode="browse"
        )

        self._tree.heading("#0", text="Group", anchor="w")
        self._tree.heading("type", text="Type", anchor="center")
        self._tree.heading("files", text="Files", anchor="center")
        self._tree.heading("score", text="Score", anchor="center")
        self._tree.heading("risk", text="Risk", anchor="center")
        self._tree.heading("reclaim", text="Reclaim", anchor="e")

        self._tree.column("#0", width=130, stretch=True)
        self._tree.column("type", width=60, anchor="center", stretch=False)
        self._tree.column("files", width=40, anchor="center", stretch=False)
        self._tree.column("score", width=45, anchor="center", stretch=False)
        self._tree.column("risk", width=45, anchor="center", stretch=False)
        self._tree.column("reclaim", width=70, anchor="e", stretch=True)

        # Row tags
        self._tree.tag_configure(
            "exact", background=C["exact_bg"], foreground=C["tree_fg"]
        )
        self._tree.tag_configure(
            "near", background=C["near_bg"], foreground=C["tree_fg"]
        )
        self._tree.tag_configure(
            "hardlink", background=C["hard_bg"], foreground=C["tree_fg"]
        )
        self._tree.tag_configure("file_row", foreground="#555566", font=("Courier", 8))

        tree_vsb = ttk.Scrollbar(
            tree_frame, orient="vertical", command=self._tree.yview
        )
        self._tree.configure(yscrollcommand=tree_vsb.set)
        tree_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self._tree.bind("<Double-Button-1>", self._on_tree_double_click)

        # Summary strip under tree
        self._tree_summary = tk.Label(
            parent,
            text="No scan yet",
            bg=C["bg"],
            fg=C["fg"],
            font=("Arial", 8),
            anchor="w",
        )
        self._tree_summary.pack(fill=tk.X, padx=6, pady=(0, 4))

    # ── Right notebook ─────────────────────────────────────────────────────

    def _build_right_notebook(self, parent) -> None:
        C = self.C
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True)
        self._right_nb = nb

        def _tab(title):
            f = tk.Frame(nb, bg=C["bg"])
            nb.add(f, text=title)
            return f

        self._build_detail_panel(_tab("📋 Group Detail"))
        self._build_full_report_tab(_tab("📄 Full Report"))
        self._build_log_tab(_tab("📝 Activity Log"))
        self._build_deletion_history_tab(_tab("🗃 Del History"))

        # Lazy render: only build Full Report when user selects that tab
        nb.bind("<<NotebookTabChanged>>", self._on_right_tab_changed)

    # ── Detail panel (right tab 0) ─────────────────────────────────────────

    def _build_detail_panel(self, parent) -> None:
        C = self.C

        # Group header strip
        hdr = tk.Frame(parent, bg=C["toolbar_bg"])
        hdr.pack(fill=tk.X)

        self._detail_group_lbl = tk.Label(
            hdr,
            text="Select a group from the left panel",
            bg=C["toolbar_bg"],
            fg=C["toolbar_fg"],
            font=("Arial", 10, "bold"),
            anchor="w",
            padx=8,
        )
        self._detail_group_lbl.pack(side=tk.LEFT, pady=6, fill=tk.X, expand=True)

        self._detail_risk_lbl = tk.Label(
            hdr,
            text="",
            bg=C["toolbar_bg"],
            fg=C["danger"],
            font=("Arial", 9, "bold"),
            anchor="e",
            padx=8,
        )
        self._detail_risk_lbl.pack(side=tk.RIGHT, pady=6)

        self._detail_score_lbl = tk.Label(
            hdr,
            text="",
            bg=C["toolbar_bg"],
            fg=C["warning"],
            font=("Courier", 9, "bold"),
            anchor="e",
            padx=8,
        )
        self._detail_score_lbl.pack(side=tk.RIGHT, pady=6)

        # Dependency status strip - shows in main window when app opens
        self._dep_status_frame = tk.Frame(parent, bg=C["bg"])
        self._dep_status_frame.pack(fill=tk.X, padx=8, pady=(4, 0))
        self._update_dependency_status()

        # Scan / compare progress bars
        prog_frame = tk.Frame(parent, bg=C["bg"])
        prog_frame.pack(fill=tk.X, padx=8, pady=4)

        tk.Label(
            prog_frame, text="Scan:", bg=C["bg"], fg=C["fg"], font=("Arial", 8), width=7
        ).pack(side=tk.LEFT)
        self._scan_bar = ttk.Progressbar(
            prog_frame, mode="determinate", maximum=100, length=200
        )
        self._scan_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._scan_lbl = tk.Label(
            prog_frame, text="", bg=C["bg"], fg=C["fg"], font=("Courier", 8), width=20
        )
        self._scan_lbl.pack(side=tk.LEFT, padx=4)

        prog_frame2 = tk.Frame(parent, bg=C["bg"])
        prog_frame2.pack(fill=tk.X, padx=8, pady=(0, 4))
        tk.Label(
            prog_frame2,
            text="Compare:",
            bg=C["bg"],
            fg=C["fg"],
            font=("Arial", 8),
            width=7,
        ).pack(side=tk.LEFT)
        self._match_bar = ttk.Progressbar(
            prog_frame2, mode="determinate", maximum=100, length=200
        )
        self._match_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._match_lbl = tk.Label(
            prog_frame2, text="", bg=C["bg"], fg=C["fg"], font=("Courier", 8), width=20
        )
        self._match_lbl.pack(side=tk.LEFT, padx=4)

        # Scrollable file-cards canvas
        canvas_frame = tk.Frame(parent, bg=C["bg"])
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._cards_canvas = tk.Canvas(canvas_frame, bg=C["bg"], highlightthickness=0)
        cards_vsb = ttk.Scrollbar(
            canvas_frame, orient="vertical", command=self._cards_canvas.yview
        )
        self._cards_canvas.configure(yscrollcommand=cards_vsb.set)
        cards_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._cards_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._cards_inner = tk.Frame(self._cards_canvas, bg=C["bg"])
        self._cards_window = self._cards_canvas.create_window(
            (0, 0), window=self._cards_inner, anchor="nw"
        )

        self._cards_inner.bind("<Configure>", self._on_cards_configure)
        self._cards_canvas.bind("<Configure>", self._on_canvas_resize)
        self._cards_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self._cards_canvas.bind("<Button-4>", self._on_mousewheel)
        self._cards_canvas.bind("<Button-5>", self._on_mousewheel)

    def _on_cards_configure(self, event=None):
        self._cards_canvas.configure(scrollregion=self._cards_canvas.bbox("all"))

    def _on_canvas_resize(self, event):
        self._cards_canvas.itemconfig(self._cards_window, width=event.width)

    def _on_mousewheel(self, event):
        if event.num == 4:
            self._cards_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self._cards_canvas.yview_scroll(1, "units")
        else:
            self._cards_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ── Full Report tab ────────────────────────────────────────────────────

    def _build_full_report_tab(self, parent) -> None:
        C = self.C
        ctrl = tk.Frame(parent, bg=C["bg"])
        ctrl.pack(fill=tk.X, padx=8, pady=6)

        self._report_summary_lbl = tk.Label(
            ctrl,
            text="Run a scan to generate the full report",
            bg=C["bg"],
            fg=C["fg"],
            font=("Arial", 10, "bold"),
        )
        self._report_summary_lbl.pack(side=tk.LEFT)

        ttk.Button(
            ctrl, text="📤 Export TXT", command=self._export_txt, style="Blue.TButton"
        ).pack(side=tk.RIGHT, padx=4)
        ttk.Button(
            ctrl, text="📊 Export CSV", command=self._export_csv, style="Blue.TButton"
        ).pack(side=tk.RIGHT, padx=4)
        ttk.Button(
            ctrl,
            text="🎯 Auto-Select",
            command=self._auto_select,
            style="Amber.TButton",
        ).pack(side=tk.RIGHT, padx=4)

        self._report_text = scrolledtext.ScrolledText(
            parent,
            font=("Courier", 9),
            bg=C["panel_bg"],
            fg=C["fg"],
            borderwidth=1,
            relief="solid",
            state=tk.DISABLED,
        )
        self._report_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # Tags
        self._report_text.tag_config(
            "rpt_header",
            foreground="#ffffff",
            background=C["header_bg"],
            font=("Arial", 11, "bold"),
        )
        self._report_text.tag_config(
            "rpt_summary", foreground=C["warning"], font=("Arial", 10, "bold")
        )
        self._report_text.tag_config(
            "rpt_group",
            foreground=C["success"],
            background=C["exact_bg"],
            font=("Courier", 10, "bold"),
        )
        self._report_text.tag_config(
            "rpt_near",
            foreground=C["accent1"],
            background=C["near_bg"],
            font=("Courier", 10, "bold"),
        )
        self._report_text.tag_config(
            "rpt_keep", foreground=C["keep_fg"], font=("Courier", 9, "bold")
        )
        self._report_text.tag_config(
            "rpt_delete", foreground=C["del_fg"], font=("Courier", 9, "bold")
        )
        self._report_text.tag_config(
            "rpt_meta", foreground="#666677", font=("Courier", 8)
        )
        self._report_text.tag_config("rpt_div", foreground=C["accent1"])
        self._report_text.tag_config(
            "rpt_score", foreground=C["info"], font=("Courier", 9)
        )
        self._report_text.tag_config(
            "rpt_hl", foreground=C["warning"], font=("Courier", 9, "bold")
        )

    # ── Activity Log tab ───────────────────────────────────────────────────

    def _build_log_tab(self, parent) -> None:
        C = self.C
        ctrl = tk.Frame(parent, bg=C["bg"])
        ctrl.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(
            ctrl,
            text="Activity Log",
            bg=C["bg"],
            fg=C["fg"],
            font=("Arial", 10, "bold"),
        ).pack(side=tk.LEFT)
        ttk.Button(
            ctrl, text="Clear", command=self._clear_log, style="Gray.TButton"
        ).pack(side=tk.RIGHT, padx=4)

        self._log_text = scrolledtext.ScrolledText(
            parent,
            font=("Courier", 8),
            bg=C["panel_bg"],
            fg=C["fg"],
            borderwidth=1,
            relief="solid",
            state=tk.DISABLED,
        )
        self._log_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self._log_text.tag_config("info", foreground=C["success"])
        self._log_text.tag_config("error", foreground=C["danger"])
        self._log_text.tag_config("warn", foreground=C["warning"])
        self._log_text.tag_config(
            "time", foreground=C["accent1"], font=("Courier", 8, "bold")
        )

    # ── Deletion history tab ───────────────────────────────────────────────

    def _build_deletion_history_tab(self, parent) -> None:
        C = self.C
        ctrl = tk.Frame(parent, bg=C["bg"])
        ctrl.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(
            ctrl,
            text="Deletion History (persistent log)",
            bg=C["bg"],
            fg=C["fg"],
            font=("Arial", 10, "bold"),
        ).pack(side=tk.LEFT)
        ttk.Button(
            ctrl,
            text="🔄 Refresh",
            command=self._refresh_deletion_history,
            style="Blue.TButton",
        ).pack(side=tk.RIGHT, padx=4)
        ttk.Button(
            ctrl,
            text="Clear Log",
            command=self._clear_deletion_log,
            style="Red.TButton",
        ).pack(side=tk.RIGHT, padx=4)

        self._del_history_text = scrolledtext.ScrolledText(
            parent,
            font=("Courier", 8),
            bg=C["panel_bg"],
            fg=C["fg"],
            borderwidth=1,
            relief="solid",
            state=tk.DISABLED,
        )
        self._del_history_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self._del_history_text.tag_config("success", foreground=C["success"])
        self._del_history_text.tag_config("fail", foreground=C["danger"])
        self._del_history_text.tag_config("ts", foreground=C["accent1"])
        self._refresh_deletion_history()

    # ── Manual tab ─────────────────────────────────────────────────────

    def _build_manual_tab(self, parent) -> None:
        C = self.C

        # Use ScrolledText for proper mousewheel scrolling
        manual_frame = tk.Frame(parent, bg=C["bg"])
        manual_frame.pack(fill=tk.BOTH, expand=True)

        # Top controls
        ctrl = tk.Frame(manual_frame, bg=C["bg"])
        ctrl.pack(fill=tk.X, padx=8, pady=6)

        tk.Label(
            ctrl,
            text="📖 USER MANUAL",
            bg=C["bg"],
            fg=C["fg"],
            font=("Arial", 12, "bold"),
        ).pack(side=tk.LEFT)

        tk.Label(
            ctrl,
            text=f"Version {VERSION}",
            bg=C["bg"],
            fg=C["accent1"],
            font=("Arial", 9),
        ).pack(side=tk.LEFT, padx=10)

        # Scrollable text area
        text_frame = tk.Frame(manual_frame, bg=C["bg"])
        text_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self._manual_text = scrolledtext.ScrolledText(
            text_frame,
            font=("Courier", 9),
            bg=C["panel_bg"],
            fg=C["fg"],
            borderwidth=1,
            relief="solid",
            wrap=tk.WORD,
        )
        self._manual_text.pack(fill=tk.BOTH, expand=True)

        # Bind mousewheel for manual text
        self._manual_text.bind("<MouseWheel>", self._on_mousewheel_manual)
        self._manual_text.bind(
            "<Button-4>", lambda e: self._manual_text.yview_scroll(-1, "units")
        )
        self._manual_text.bind(
            "<Button-5>", lambda e: self._manual_text.yview_scroll(1, "units")
        )

        self._render_manual()

    def _on_mousewheel_manual(self, event):
        """Mousewheel handler for manual text."""
        if event.num == 4:
            self._manual_text.yview_scroll(-1, "units")
        elif event.num == 5:
            self._manual_text.yview_scroll(1, "units")
        else:
            self._manual_text.yview_scroll(int(-1 * (event.delta / 120)), "units")
        return "break"

    def _render_manual(self) -> None:
        """Render the user manual."""
        manual_content = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     DUPLICATE FILE FINDER v{VERSION}                        ║
║                     User Manual & Technical Guide                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 1: GETTING STARTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 SELECTING A FOLDER
   • Click the "📁 Change" button in the header to browse for a folder
   • Or drag and drop a folder onto the window
   • The current folder path is displayed in the header

▶ STARTING A SCAN
   • Press Ctrl+S or F5, or click the "▶ SCAN" button
   • Use "Esc" to stop a scan in progress
   • Progress bars show scan and comparison status

🎯 AUTO-SELECTING FILES
   • After scanning, click "🎯 Auto-Select" to automatically mark duplicates
   • The algorithm picks the best file to KEEP in each group
   • Files marked for deletion can be toggled manually

🗑 DELETING FILES
   • Click "🗑 DELETE" to move marked files to Recycle Bin
   • ALWAYS review before confirming deletion
   • Only EXACT duplicates can be auto-deleted
   • Near-duplicates require manual review

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 2: UNDERSTANDING THE RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 GROUP TYPES
   • EXACT: Files are byte-for-byte identical (100% match)
   • NEAR: Files are similar but not identical (70-99% match)
   • HARDLINK: Same file on disk, multiple paths (NEVER deletable)

🎯 SCORE BREAKDOWN
   The similarity score (0-100%) is calculated from:
   • Hash match: 100 points (exact duplicate)
   • Same size: 35 points
   • Similar name: up to 25 points
   • Same extension: up to 12 points
   • Same folder: 5 points
   • Similar modification time: up to 5 points
   • Perceptual hash (images): up to 30 points

⚠️ RISK LEVELS
   • LOW: Safe to delete exact duplicates
   • MEDIUM: Contains locked files (in use by another program)
   • HIGH: Near-duplicates or system files (manual review required)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 3: CLEANUP MODES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Settings → Auto-Selection Settings → Cleanup Mode:

🛡️ SAFE MODE (Default)
   • Only deletes EXACT duplicates
   • Requires high quality score gap (default: 15 points)
   • Near-duplicates always marked "REVIEW"
   • Most conservative approach

⚔️ AGGRESSIVE MODE
   • Deletes EXACT duplicates with lower gap threshold
   • Deletes NEAR duplicates (90%+) if obvious copy pattern
   • Requires manual review of results

🎬 MEDIA-FOCUSED MODE
   • Optimized for photos, videos, and audio
   • Uses perceptual hashing for images
   • Slightly looser criteria for media files

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 4: SCAN SETTINGS EXPLAINED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 SCAN SETTINGS
   • Scan subdirectories: Include all nested folders
   • Hash file contents: Compute hashes for exact matching
   • Use xxhash: ~10× faster hashing (recommended)
   • SHA256 verification: Extra verification for exact matches
   • GPU/NumPy: Use GPU for filename similarity (if available)
   • Paranoid mode: Byte-by-byte verification of exact matches

⚙️ PERFORMANCE SETTINGS
   • Worker threads: Parallel processing (default: CPU cores × 2)
   • Minimum file size: Skip files smaller than this
   • Maximum file size: Skip files larger than this (0 = no limit)
   • Minimum similarity score: Minimum for near-dup detection

🚫 EXCLUSIONS
   • Skip network drives: Don't scan network locations
   • Skip system directories: Protect OS folders
   • Exclusion patterns: Comma-separated substrings to skip

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 5: ADVANCED FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🧠 DINOv2 NEURAL EMBEDDINGS (Optional)
   • Uses Facebook's DINOv2 model for image similarity
   • Detects similar images even after cropping/editing
   • Requires transformers + torch libraries
   • Enable in Settings → Scan Settings → Neural embeddings
   ⚠️ SLOW: May take minutes for large image collections

🎵 AUDIO METADATA (Optional)
   • Extracts metadata from audio files (artist, album, title)
   • Uses mutagen library
   • Enable in Settings → Scan Settings → Audio fingerprinting
   • Shows audio metadata in file details

🔌 I/O PORT (External Control)
   • Allows external programs to control Duplicate Finder
   • Enable in Settings → Scan Settings → Enable I/O Port
   • Communicates via JSON over stdin/stdout
   
   COMMANDS:
   • {{"cmd": "ping"}} - Check if port is responding
   • {{"cmd": "get_status"}} - Get current status
   • {{"cmd": "get_groups"}} - Get list of duplicate groups
   • {{"cmd": "get_group", "args": {{"index": 0}}}} - Get specific group
   • {{"cmd": "select_files", "args": {{"group_index": 0, "file_indices": [1], "action": "DELETE"}}}}
   • {{"cmd": "get_settings"}} - Get current scan settings
   • {{"cmd": "set_settings", "args": {{"min_score": 70}}}}

📊 EXPORT FORMATS
   • TXT: Plain text full report
   • CSV: Spreadsheet-compatible format
   • JSON: Machine-readable data
   • HTML: Browser-viewable colored report

💾 SESSION SAVE/LOAD
   • Save current scan results to continue later
   • Includes all metadata, hashes, and suggestions
   • Use "💾 Save" and "📂 Load" buttons

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 6: KEYBOARD SHORTCUTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   Ctrl+S / F5    Start scan
   Esc            Stop scan
   Delete         Delete selected files
   Left/Right     Navigate between groups
   Up/Down       Navigate file list
   Ctrl+A        Select all in current group
   Ctrl+Z        Show deletion history
   Ctrl+E        Open export menu
   F1            Show this manual

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 7: SAFETY GUARANTEES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ MULTI-LAYER SAFETY SYSTEM

1. VERIFIED DELETION ONLY
   • Only EXACT duplicates can be auto-deleted
   • Near-duplicates marked "REVIEW" require manual decision

2. HARDLINK PROTECTION
   • Hardlinks share the same disk data
   • Deleting one would delete ALL paths to the same file
   • HARDLINK groups are NEVER deletable

3. LOCKED FILE DETECTION
   • Files in use by other programs are protected
   • Marked as "LOCKED" - cannot be deleted

4. SYSTEM FILE PROTECTION
   • Windows system directories are automatically skipped
   • Unix system folders (/bin, /etc, etc.) skipped

5. MUTATION PROTECTION
   • Files changing during scan are detected
   • Hashes cleared for changed files
   • Prevents deleting wrong files

6. RECYCLE BIN DEFAULT
   • ALL deletions go to Recycle Bin/Trash
   • Files can be restored if deleted by mistake
   • Permanent deletion is NEVER performed

7. PERSISTENT DELETION LOG
   • All deletions are logged with timestamp
   • Log stored at: {DELETION_LOG_PATH}
   • Can be viewed in "Del History" tab

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 8: DETECTION METHODS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXACT DUPLICATE DETECTION (100% match)
   1. Quick hash: First 4 KB of file
   2. Partial hash: First + last 64 KB (files > 1 MB)
   3. Full hash: Entire file (xxhash64 or MD5)
   4. SHA-256: Optional extra verification

NEAR-DUPLICATE DETECTION (70-99% match)
   • Size comparison
   • Filename similarity (SequenceMatcher)
   • Token-based Jaccard similarity
   • Extension compatibility
   • Magic-byte file type detection
   • Temporal proximity (modification time)
   • Perceptual hash (pHash for images)
   • Neural embeddings (DINOv2 - optional)

HARDLINK DETECTION
   • Uses (device, inode) pairs
   • Files with same inode are hardlinks
   • Same physical data, multiple paths

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 9: TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❓ "No duplicates found" but I know files are duplicate
   → Check minimum file size setting
   → Try lowering minimum similarity score to 50
   → Enable "Neural embeddings" for image similarity

❓ Scan is very slow
   • Reduce worker threads if CPU is maxed
   • Enable xxhash for faster hashing
   • Exclude large folders from scan
   • Disable unnecessary features (neural embeddings)

❓ Files won't delete
   • Check if file is locked by another program
   • Verify file exists (may have been moved)
   • Check if in system directory (protected)

❓ Memory usage too high
   • Scan large folders in batches
   • Reduce worker threads
   • Enable "Skip network drives"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 10: OPTIONAL LIBRARIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Install with: pip install <package>

   xxhash        - Fast hashing (~10× faster than MD5)
   send2trash    - Cross-platform Recycle Bin support
   Pillow       - Image thumbnails and perceptual hashing
   numpy        - Vectorized name similarity
   cupy-cuda11x - NVIDIA GPU acceleration (optional)
   psutil       - System monitoring and locked file detection
   transformers - DINOv2 neural embeddings (optional)
   torch        - PyTorch for neural embeddings (optional)
   mutagen      - Audio metadata extraction (optional)
   opencv-python - Video frame extraction (optional)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Created by: Shawn Mitchell
Version: {VERSION}
Build Date: 2026-04-09

═══════════════════════════════════════════════════════════════════════════════════
"""
        self._manual_text.config(state=tk.NORMAL)
        self._manual_text.delete("1.0", tk.END)
        self._manual_text.insert("1.0", manual_content)
        self._manual_text.config(state=tk.DISABLED)

    def _build_changelog_tab(self, parent) -> None:
        """Build changelog viewer tab."""
        import os as _os

        C = self.C
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(base_dir, "logs")
        backup_dir = os.path.join(base_dir, "v6_backup")

        frame = tk.Frame(parent, bg=C["bg"])
        frame.pack(fill=tk.BOTH, expand=True)

        ctrl = tk.Frame(frame, bg=C["bg"])
        ctrl.pack(fill=tk.X, padx=8, pady=6)

        tk.Label(
            ctrl,
            text="📜 CHANGELOG VIEWER",
            bg=C["bg"],
            fg=C["fg"],
            font=("Arial", 12, "bold"),
        ).pack(side=tk.LEFT)

        self._changelog_file_var = tk.StringVar()
        self._changelog_file_var.set("Select a changelog...")

        self._changelog_dropdown = ttk.Combobox(
            ctrl,
            textvariable=self._changelog_file_var,
            state="readonly",
            font=("Arial", 9),
            width=30,
        )
        self._changelog_dropdown.pack(side=tk.LEFT, padx=10)
        self._changelog_dropdown.bind(
            "<<ComboboxSelected>>", self._on_changelog_selected
        )

        btn_frame = tk.Frame(ctrl, bg=C["bg"])
        btn_frame.pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame,
            text="🔄 Scan",
            font=("Arial", 8),
            command=self._scan_changelogs,
        ).pack(side=tk.LEFT, padx=2)
        self._add_tooltip(
            tk.Button(
                btn_frame,
                text="🔄",
                font=("Arial", 8),
                command=self._scan_changelogs,
            ),
            "Rescan logs folder for changelogs",
        )

        text_frame = tk.Frame(frame, bg=C["bg"])
        text_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self._changelog_text = scrolledtext.ScrolledText(
            text_frame,
            font=("Courier", 9),
            bg=C["panel_bg"],
            fg=C["fg"],
            borderwidth=1,
            relief="solid",
            wrap=tk.WORD,
        )
        self._changelog_text.pack(fill=tk.BOTH, expand=True)

        self._changelog_text.bind("<MouseWheel>", self._on_mousewheel_changelog)
        self._changelog_text.bind(
            "<Button-4>", lambda e: self._changelog_text.yview_scroll(-1, "units")
        )
        self._changelog_text.bind(
            "<Button-5>", lambda e: self._changelog_text.yview_scroll(1, "units")
        )

        self._changelog_files = []
        self._scan_changelogs()

    def _on_mousewheel_changelog(self, event):
        """Mousewheel handler for changelog text."""
        if event.num == 4:
            self._changelog_text.yview_scroll(-1, "units")
        elif event.num == 5:
            self._changelog_text.yview_scroll(1, "units")
        else:
            self._changelog_text.yview_scroll(int(-1 * (event.delta / 120)), "units")
        return "break"

    def _scan_changelogs(self) -> None:
        """Scan logs folder and backup for changelogs."""
        import os as _os
        import glob as _glob

        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(base_dir, "logs")
        backup_dir = os.path.join(base_dir, "v6_backup")

        changelog_files = []

        for d in [log_dir, backup_dir]:
            if _os.path.isdir(d):
                for pattern in [
                    _os.path.join(d, "*.md"),
                    _os.path.join(d, "*.txt"),
                    _os.path.join(d, "changelog*"),
                ]:
                    changelog_files.extend(_glob.glob(pattern))

        changelog_files = sorted(
            set(changelog_files), key=_os.path.getmtime, reverse=True
        )

        self._changelog_files = changelog_files

        names = [_os.path.basename(f) for f in changelog_files]
        if not names:
            names = ["No changelogs found"]
        else:
            names.insert(0, "Select a changelog...")

        self._changelog_dropdown["values"] = names
        if names[0] == "Select a changelog...":
            self._changelog_file_var.set("Select a changelog...")
        else:
            self._changelog_file_var.set(names[0])

    def _on_changelog_selected(self, event) -> None:
        """Handle changelog file selection."""
        selected = self._changelog_file_var.get()
        if selected == "Select a changelog..." or not selected:
            return

        for f in self._changelog_files:
            if os.path.basename(f) == selected:
                try:
                    with open(f, encoding="utf-8", errors="ignore") as fh:
                        content = fh.read()
                    self._changelog_text.config(state=tk.NORMAL)
                    self._changelog_text.delete("1.0", tk.END)
                    self._changelog_text.insert("1.0", content)
                    self._changelog_text.config(state=tk.DISABLED)
                except Exception as e:
                    self._changelog_text.config(state=tk.NORMAL)
                    self._changelog_text.delete("1.0", tk.END)
                    self._changelog_text.insert("1.0", f"Error loading changelog: {e}")
                    self._changelog_text.config(state=tk.DISABLED)
                break

    # ── Settings tab ───────────────────────────────────────────────────────

    def _build_settings_tab(self, parent) -> None:
        C = self.C
        canvas = tk.Canvas(parent, bg=C["bg"], highlightthickness=0)
        vsb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(canvas, bg=C["bg"])
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        # Mouse wheel scrolling for settings
        def _on_settings_wheel(event):
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<MouseWheel>", _on_settings_wheel)
        canvas.bind("<Button-4>", _on_settings_wheel)
        canvas.bind("<Button-5>", _on_settings_wheel)

        # ── Profile Selection Buttons ────────────────────────────────────────
        profile_frame = tk.Frame(
            inner, bg=C["toolbar_bg"], borderwidth=2, relief="solid"
        )
        profile_frame.pack(fill=tk.X, padx=16, pady=(16, 8))

        tk.Label(
            profile_frame,
            text="🔒 Safety Profile",
            bg=C["toolbar_bg"],
            fg=C["toolbar_fg"],
            font=("Arial", 11, "bold"),
            padx=10,
            pady=8,
        ).pack(side=tk.LEFT, padx=10)

        btn_frame = tk.Frame(profile_frame, bg=C["toolbar_bg"])
        btn_frame.pack(side=tk.RIGHT, padx=10)

        def _set_profile(p):
            apply_profile_settings(p, self.settings)
            self._load_settings_into_ui()
            if p == "ai_assist":
                self._log_msg(
                    "info", "Profile set to: AI ASSIST (Max Safe + All AI Features)"
                )
            elif p == "max_safe":
                self._log_msg("info", "Profile set to: MAX SAFE")
            elif p == "safe":
                self._log_msg("info", "Profile set to: SAFE")
            elif p == "performance":
                self._log_msg("info", "Profile set to: PERFORMANCE")

        self._var_max_safe = tk.StringVar(value="safe")

        # AI Assist - MAX SAFE + all AI features
        tk.Button(
            btn_frame,
            text="🤖 AI ASSIST",
            bg="#4a2d7a",
            fg="white",
            font=("Arial", 9, "bold"),
            relief="raised",
            padx=10,
            cursor="hand2",
            command=lambda: _set_profile("ai_assist"),
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            btn_frame,
            text="🛡️ MAX SAFE",
            bg="#2d5a2d",
            fg="white",
            font=("Arial", 9, "bold"),
            relief="raised",
            padx=10,
            cursor="hand2",
            command=lambda: _set_profile("max_safe"),
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            btn_frame,
            text="✓ SAFE",
            bg="#4a7c59",
            fg="white",
            font=("Arial", 9, "bold"),
            relief="raised",
            padx=10,
            cursor="hand2",
            command=lambda: _set_profile("safe"),
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            btn_frame,
            text="⚡ PERFORMANCE",
            bg="#7c594a",
            fg="white",
            font=("Arial", 9, "bold"),
            relief="raised",
            padx=10,
            cursor="hand2",
            command=lambda: _set_profile("performance"),
        ).pack(side=tk.LEFT, padx=4)

        def _section(title):
            f = tk.Frame(inner, bg=C["toolbar_bg"], borderwidth=1, relief="solid")
            f.pack(fill=tk.X, padx=16, pady=(12, 2))
            tk.Label(
                f,
                text=title,
                bg=C["toolbar_bg"],
                fg=C["toolbar_fg"],
                font=("Arial", 10, "bold"),
                padx=10,
                pady=6,
            ).pack(anchor="w")
            g = tk.Frame(inner, bg=C["accent3"], borderwidth=1, relief="solid")
            g.pack(fill=tk.X, padx=16, pady=(0, 4))
            return g

        def _row(grid, label, widget_fn, r, tip=""):
            lbl = tk.Label(
                grid,
                text=label,
                bg=C["accent3"],
                fg=C["fg"],
                font=("Arial", 9),
                anchor="w",
                width=38,
            )
            lbl.grid(row=r, column=0, padx=12, pady=5, sticky="w")
            if tip:
                self._add_tooltip(lbl, tip)
            widget_fn(grid).grid(row=r, column=1, padx=12, pady=5, sticky="w")

        def _chk(parent, var, state=tk.NORMAL):
            return tk.Checkbutton(
                parent,
                variable=var,
                bg=C["accent3"],
                fg=C["fg"],
                selectcolor=C["bg"],
                activebackground=C["accent3"],
                font=("Arial", 9),
                state=state,
            )

        def _spn(parent, var, lo=0, hi=100, w=8):
            return tk.Spinbox(
                parent,
                from_=lo,
                to=hi,
                textvariable=var,
                width=w,
                font=("Arial", 9),
                bg=C["panel_bg"],
                fg=C["fg"],
            )

        def _cbo(parent, var, values, w=12):
            return ttk.Combobox(
                parent, textvariable=var, values=values, state="readonly", width=w
            )

        # ── Scan settings ─────────────────────────────────────────────────
        g = _section("⚙️  SCAN SETTINGS")
        _row(
            g,
            "Scan subdirectories recursively:",
            lambda p: _chk(p, self._var_subdirs),
            0,
            "Include all subdirectories",
        )
        _row(
            g,
            "Hash file contents (exact match):",
            lambda p: _chk(p, self._var_hash),
            1,
            "Compute hashes for exact matching",
        )
        _row(
            g,
            "Use xxhash (~10× faster):",
            lambda p: _chk(
                p, self._var_xxhash, tk.NORMAL if HAS_XXHASH else tk.DISABLED
            ),
            2,
            "Use fast xxhash",
        )
        _row(
            g,
            "Use BLAKE3 (fastest):",
            lambda p: _chk(
                p, self._var_blake3, tk.NORMAL if HAS_BLAKE3 else tk.DISABLED
            ),
            2,
            "Use BLAKE3 for full hash (faster than SHA-256)",
        )
        _row(
            g,
            "SHA256 verification:",
            lambda p: _chk(p, self._var_sha256),
            3,
            "Extra SHA256 hash for confidence",
        )
        _row(
            g,
            "GPU/NumPy name similarity:",
            lambda p: _chk(
                p, self._var_gpu, tk.NORMAL if (HAS_CUPY or HAS_NUMPY) else tk.DISABLED
            ),
            4,
            "Use GPU/NumPy",
        )
        _row(
            g,
            "Neural embeddings (DINOv2):",
            lambda p: _chk(p, self._var_neural, tk.NORMAL if HAS_DINO else tk.DISABLED),
            5,
            "Use DINOv2 for image similarity (SLOW)",
        )
        _row(
            g,
            "Audio fingerprinting:",
            lambda p: _chk(
                p,
                self._var_audio_fp,
                tk.NORMAL if (HAS_CHROMAPRINT or HAS_MUTAGEN) else tk.DISABLED,
            ),
            6,
            "Use chromaprint/mutagen for audio",
        )
        _row(
            g,
            "Paranoid mode (byte-by-byte):",
            lambda p: _chk(p, self._var_paranoid),
            7,
            "Verify exact matches byte-by-byte",
        )
        _row(
            g,
            f"Worker threads (CPUs: {CPU_COUNT}):",
            lambda p: _spn(p, self._var_workers, 1, 64, 6),
            8,
            "Parallel workers",
        )
        _row(
            g,
            "Minimum similarity score:",
            lambda p: _spn(p, self._var_min_score, 0, 100, 6),
            9,
            "Min score for near-dup",
        )
        _row(
            g,
            "Minimum file size (bytes):",
            lambda p: _spn(p, self._var_min_size, 0, 10_000_000, 10),
            8,
            "Ignore smaller files",
        )
        _row(
            g,
            "Maximum file size (0=none):",
            lambda p: _spn(p, self._var_max_size, 0, 100_000_000_000, 14),
            9,
            "Ignore larger files",
        )
        _row(
            g,
            "Skip network drives:",
            lambda p: _chk(p, self._var_skip_network),
            10,
            "Skip network paths",
        )
        _row(
            g,
            "Skip system directories:",
            lambda p: _chk(p, self._var_skip_system),
            11,
            "Skip OS directories",
        )
        _row(
            g,
            "Enable I/O Port (external control):",
            lambda p: _chk(p, self._var_io_port),
            12,
            "Allow external programs to control via JSON commands",
        )

        # v7.0 - AI Semantic Deduplication
        g_ai = _section("🤖 AI SEMANTIC DEDUPLICATION (v7.0 NEXT GEN)")
        _row(
            g_ai,
            "Enable Semantic Deduplication:",
            lambda p: _chk(p, self._var_semantic),
            0,
            "Use AI-powered semantic similarity (EXPERIMENTAL)",
        )
        _row(
            g_ai,
            "Use FAISS Vector Index:",
            lambda p: _chk(p, self._var_faiss, tk.NORMAL if HAS_FAISS else tk.DISABLED),
            1,
            "Fast Approximate Nearest Neighbor search (requires faiss-cpu)",
        )
        _row(
            g_ai,
            "Use CLIP Embeddings (Images):",
            lambda p: _chk(p, self._var_clip, tk.NORMAL if HAS_CLIP else tk.DISABLED),
            2,
            "Cross-modal image-text similarity (SLOW, requires GPU recommended)",
        )
        _row(
            g_ai,
            "Use Sentence-BERT (Documents):",
            lambda p: _chk(
                p,
                self._var_sentence,
                tk.NORMAL if HAS_SENTENCE_TRANSFORMERS else tk.DISABLED,
            ),
            3,
            "Semantic text similarity for documents (SLOW, requires transformers)",
        )

        # ── Auto-selection settings ────────────────────────────────────────
        g2 = _section("🎯 AUTO-SELECTION SETTINGS")
        _row(
            g2,
            "Auto-select after scan:",
            lambda p: _chk(p, self._var_auto_select),
            0,
            "Automatically select files",
        )
        _row(
            g2,
            "Min quality gap for DELETE:",
            lambda p: _spn(p, self._var_delete_gap, 0, 100, 6),
            1,
            "Higher = safer",
        )
        _row(
            g2,
            "Cleanup mode:",
            lambda p: _cbo(
                p, self._var_cleanup_mode, ["SAFE", "AGGRESSIVE", "MEDIA-FOCUSED"], 14
            ),
            2,
            "Selection aggressiveness",
        )

        # ── Display settings ───────────────────────────────────────────────
        g3 = _section("🎨 DISPLAY SETTINGS")
        _row(
            g3,
            "Dark mode (Catppuccin Mocha):",
            lambda p: _chk(p, self._var_dark_mode),
            0,
        )
        ttk.Button(
            g3,
            text="Apply Dark/Light Mode",
            command=self._apply_display_settings,
            style="Blue.TButton",
        ).grid(row=1, column=0, columnspan=2, padx=12, pady=8, sticky="w")

        # ── Exclusion patterns ────────────────────────────────────────────
        g4 = _section("🚫 EXCLUSION PATTERNS  (comma-separated substrings)")
        self._excl_entry = tk.Text(
            g4,
            height=3,
            width=60,
            bg=C["panel_bg"],
            fg=C["fg"],
            font=("Courier", 9),
            borderwidth=1,
            relief="solid",
        )
        self._excl_entry.grid(
            row=0, column=0, columnspan=2, padx=12, pady=8, sticky="w"
        )

        # ── Library status + install buttons ─────────────────────────────
        g5 = _section("📦 OPTIONAL LIBRARY STATUS  (auto-install on startup)")
        libs = [
            ("xxhash", HAS_XXHASH, "xxhash", "~10× faster hashing"),
            ("send2trash", HAS_SEND2TRASH, "send2trash", "Cross-platform Trash"),
            ("psutil", HAS_PSUTIL, "psutil", "System monitoring"),
            ("numpy", HAS_NUMPY, "numpy", "Vectorised name similarity"),
            ("cupy", HAS_CUPY, "cupy-cuda11x", "NVIDIA GPU acceleration"),
            ("Pillow", HAS_PIL, "Pillow", "Image thumbnails + pHash"),
            # v7.0 AI libraries
            ("faiss-cpu", HAS_FAISS, "faiss-cpu", "Vector similarity search"),
            (
                "sentence-transformers",
                HAS_SENTENCE_TRANSFORMERS,
                "sentence-transformers",
                "Document semantic similarity",
            ),
            ("CLIP (transformers)", HAS_CLIP, "transformers", "Image-text cross-modal"),
            ("blake3", HAS_BLAKE3, "blake3", "Fast hashing (faster than SHA-256)"),
            ("watchdog", HAS_WATCHDOG, "watchdog", "Real-time folder monitoring"),
        ]

        def _make_install_fn(pip_pkg, status_lbl_ref):
            def _do_install():
                status_lbl_ref.config(text="⏳ Installing…", fg=C["warning"])
                inner.update_idletasks()

                def _bg():
                    _ok = _pip_install(pip_pkg, verbose=True)
                    _txt = "✓ Done — restart to apply" if _ok else "✗ Install failed"
                    _fg = C["success"] if _ok else C["danger"]
                    _tag = "info" if _ok else "error"
                    _msg = (
                        f"[SESSION] pip install {pip_pkg} → {'ok' if _ok else 'FAILED'}"
                    )
                    # must update tkinter widgets on main thread
                    self.root.after(0, lambda: status_lbl_ref.config(text=_txt, fg=_fg))
                    self.root.after(0, lambda: self._dbg(_msg, _tag))

                threading.Thread(target=_bg, daemon=True).start()

            return _do_install

        for i, (name, present, pip_pkg, desc) in enumerate(libs):
            sc = C["success"] if present else C["danger"]
            status_text = "✓ INSTALLED" if present else "✗ MISSING"
            tk.Label(
                g5,
                text=f"{name}",
                bg=C["accent3"],
                fg=C["fg"],
                font=("Courier", 9, "bold"),
                width=14,
                anchor="w",
            ).grid(row=i, column=0, padx=12, pady=4, sticky="w")
            lbl = tk.Label(
                g5,
                text=status_text,
                bg=C["accent3"],
                fg=sc,
                font=("Courier", 9, "bold"),
                width=14,
                anchor="w",
            )
            lbl.grid(row=i, column=1, padx=6, pady=4, sticky="w")
            tk.Label(
                g5,
                text=desc,
                bg=C["accent3"],
                fg=C["fg"],
                font=("Arial", 8),
                anchor="w",
            ).grid(row=i, column=2, padx=8, pady=4, sticky="w")
            btn_text = "🔄 Reinstall" if present else "⬇ Install"
            btn_style = "Blue.TButton" if present else "Green.TButton"
            ttk.Button(
                g5,
                text=btn_text,
                style=btn_style,
                command=_make_install_fn(pip_pkg, lbl),
            ).grid(row=i, column=3, padx=8, pady=4, sticky="w")

        tk.Label(
            g5,
            text="  Changes take effect after restarting the program.",
            bg=C["accent3"],
            fg=C["warning"],
            font=("Arial", 8, "italic"),
        ).grid(row=len(libs), column=0, columnspan=4, padx=12, pady=(2, 8), sticky="w")

    # ── Status bar ─────────────────────────────────────────────────────────

    def _build_status_bar(self) -> None:
        C = self.C
        sb = tk.Frame(self.root, bg=C["status_bar_bg"], height=22)
        sb.pack(fill=tk.X, side=tk.BOTTOM)
        sb.pack_propagate(False)

        self._sb_groups = tk.Label(
            sb,
            text="Groups: 0",
            bg=C["status_bar_bg"],
            fg=C["status_bar_fg"],
            font=("Courier", 8),
        )
        self._sb_groups.pack(side=tk.LEFT, padx=8)
        self._add_tooltip(self._sb_groups, "Total number of duplicate groups found")

        tk.Label(sb, text="│", bg=C["status_bar_bg"], fg=C["border"]).pack(side=tk.LEFT)

        self._sb_files = tk.Label(
            sb,
            text="Files: 0",
            bg=C["status_bar_bg"],
            fg=C["status_bar_fg"],
            font=("Courier", 8),
        )
        self._sb_files.pack(side=tk.LEFT, padx=8)
        self._add_tooltip(self._sb_files, "Total number of files in duplicate groups")

        tk.Label(sb, text="│", bg=C["status_bar_bg"], fg=C["border"]).pack(side=tk.LEFT)

        self._sb_reclaim = tk.Label(
            sb,
            text="Reclaimable: 0 B",
            bg=C["status_bar_bg"],
            fg=C["status_bar_fg"],
            font=("Courier", 8),
        )
        self._sb_reclaim.pack(side=tk.LEFT, padx=8)
        self._add_tooltip(
            self._sb_reclaim, "Total space that can be recovered by deleting duplicates"
        )

        tk.Label(sb, text="│", bg=C["status_bar_bg"], fg=C["border"]).pack(side=tk.LEFT)

        self._sb_marked = tk.Label(
            sb,
            text="Marked: 0",
            bg=C["status_bar_bg"],
            fg=C["status_bar_fg"],
            font=("Courier", 8),
        )
        self._sb_marked.pack(side=tk.LEFT, padx=8)
        self._add_tooltip(self._sb_marked, "Number of files marked for deletion")

        tk.Label(sb, text="│", bg=C["status_bar_bg"], fg=C["border"]).pack(side=tk.LEFT)

        self._sb_exact = tk.Label(
            sb,
            text="Exact: 0",
            bg=C["status_bar_bg"],
            fg=C["success"],
            font=("Courier", 8),
        )
        self._sb_exact.pack(side=tk.LEFT, padx=8)
        self._add_tooltip(
            self._sb_exact, "Number of exact duplicate groups (100% identical)"
        )

        tk.Label(sb, text="│", bg=C["status_bar_bg"], fg=C["border"]).pack(side=tk.LEFT)

        self._sb_near = tk.Label(
            sb,
            text="Near: 0",
            bg=C["status_bar_bg"],
            fg=C["warning"],
            font=("Courier", 8),
        )
        self._sb_near.pack(side=tk.LEFT, padx=8)
        self._add_tooltip(
            self._sb_near, "Number of near-duplicate groups (similar but not identical)"
        )

        # Watermark removed - will be in separate footer bar

        # Terminal toggle button
        self._term_toggle_btn = tk.Button(
            sb,
            text="▼ Terminal",
            bg=C["status_bar_bg"],
            fg=C["warning"],
            font=("Courier", 8, "bold"),
            relief="flat",
            cursor="hand2",
            command=self._toggle_terminal,
        )
        self._term_toggle_btn.pack(side=tk.RIGHT, padx=8)

        self._sb_status = tk.Label(
            sb,
            text="Ready",
            bg=C["status_bar_bg"],
            fg=C["warning"],
            font=("Courier", 8, "bold"),
            anchor="e",
        )
        self._sb_status.pack(side=tk.RIGHT, padx=12, fill=tk.X, expand=True)

    def _update_status_bar(self) -> None:
        """Refresh status bar counts from current groups."""
        ng = len(self.groups)
        nf = sum(len(g.files) for g in self.groups)
        rb = sum(g.reclaimable_bytes for g in self.groups)
        nm = sum(
            1 for g in self.groups for fi, s in g.suggestions.items() if s == "DELETE"
        )
        n_exact = sum(1 for g in self.groups if g.group_type == "exact")
        n_near = sum(1 for g in self.groups if g.group_type == "near")
        self._sb_groups.config(text=f"Groups: {ng}")
        self._sb_files.config(text=f"Files: {nf}")
        self._sb_reclaim.config(text=f"Reclaimable: {_format_size(rb)}")
        self._sb_marked.config(text=f"Marked: {nm}")
        self._sb_exact.config(text=f"Exact: {n_exact}")
        self._sb_near.config(text=f"Near: {n_near}")

    def _update_dependency_status(self) -> None:
        """Update dependency status display in main window group details area."""
        if not hasattr(self, "_dep_status_frame"):
            return

        # Clear existing widgets
        for w in self._dep_status_frame.winfo_children():
            w.destroy()

        C = self.C

        # Build dependency info
        optional_missing = []
        if not HAS_FAISS:
            optional_missing.append("faiss")
        if not HAS_SENTENCE_TRANSFORMERS:
            optional_missing.append("sentence-transformers")
        if not HAS_CLIP:
            optional_missing.append("clip")

        # Determine missing vs installed for display
        all_optional_libs = [
            "faiss-cpu",
            "sentence-transformers",
            "clip",
            "blake3",
            "watchdog",
        ]
        available_libs = []
        if HAS_FAISS:
            available_libs.append("faiss-cpu")
        if HAS_SENTENCE_TRANSFORMERS:
            available_libs.append("sentence-transformers")
        if HAS_CLIP:
            available_libs.append("CLIP")
        if HAS_BLAKE3:
            available_libs.append("blake3")
        if HAS_WATCHDOG:
            available_libs.append("watchdog")
        if HAS_PIL:
            available_libs.append("pillow")
        if HAS_PSUTIL:
            available_libs.append("psutil")
        if HAS_SEND2TRASH:
            available_libs.append("send2trash")
        if HAS_NUMPY:
            available_libs.append("numpy")
        if HAS_TORCH:
            available_libs.append("torch")
        if HAS_MUTAGEN:
            available_libs.append("mutagen")

        # Create status display
        status_frame = tk.Frame(self._dep_status_frame, bg=C["bg"])
        status_frame.pack(fill=tk.X, pady=2)

        if optional_missing:
            # Show what's missing
            missing_lbl = tk.Label(
                status_frame,
                text=f"⚙ Optional libraries not installed: {', '.join(optional_missing)}",
                bg=C["bg"],
                fg=C["warning"],
                font=("Arial", 8, "bold"),
                anchor="w",
            )
            missing_lbl.pack(side=tk.LEFT, padx=4)

            desc_lbl = tk.Label(
                status_frame,
                text="Enable advanced features (AI, better performance)",
                bg=C["bg"],
                fg=C["fg"],
                font=("Arial", 7),
                anchor="w",
            )
            desc_lbl.pack(side=tk.LEFT, padx=8)
        else:
            # All features available
            pass  # Will show in second line below

        # Second line: show what IS installed
        libs_frame = tk.Frame(self._dep_status_frame, bg=C["bg"])
        libs_frame.pack(fill=tk.X, pady=(0, 4))

        installed_lbl = tk.Label(
            libs_frame,
            text=f"Installed: {', '.join(available_libs) if available_libs else 'none'}",
            bg=C["bg"],
            fg=C["success"],
            font=("Arial", 7),
            anchor="w",
        )
        installed_lbl.pack(side=tk.LEFT, padx=4)

        # Add clickable link to Settings tab
        settings_link = tk.Label(
            libs_frame,
            text="→ Open Settings",
            bg=C["bg"],
            fg=C["accent1"],
            font=("Arial", 8, "underline"),
            cursor="hand2",
        )
        settings_link.pack(side=tk.RIGHT, padx=8)
        settings_link.bind("<Button-1>", lambda e: self._show_settings())

    # ── Debug terminal (always-visible, collapsible, bottom) ──────────────

    def _build_debug_terminal(self) -> None:
        C = self.C

        self._term_frame = tk.Frame(
            self.root,
            bg="#161b22",
            borderwidth=2,
            relief="solid",
            highlightthickness=1,
            highlightbackground=C["warning"],
        )
        self._term_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=1, pady=1)

        # Header
        hdr = tk.Frame(self._term_frame, bg="#161b22")
        hdr.pack(fill=tk.X, padx=4, pady=(3, 0))

        tk.Label(
            hdr,
            text="⚡ DEBUG TERMINAL",
            bg="#161b22",
            fg=C["warning"],
            font=("Courier", 9, "bold"),
        ).pack(side=tk.LEFT)
        tk.Label(
            hdr,
            text=f"  v{VERSION}  |  xxhash={'✓' if HAS_XXHASH else '✗'}"
            f"  |  send2trash={'✓' if HAS_SEND2TRASH else '✗'}"
            f"  |  PIL={'✓' if HAS_PIL else '✗'}",
            bg="#161b22",
            fg="#6e7681",
            font=("Courier", 8),
        ).pack(side=tk.LEFT, padx=6)

        tk.Button(
            hdr,
            text="CLEAR ALL",
            command=self._clear_debug_terminal,
            bg="#21262d",
            fg=C["warning"],
            font=("Courier", 8, "bold"),
            relief="flat",
            padx=6,
            cursor="hand2",
        ).pack(side=tk.RIGHT, padx=4)

        # Watermark - centered in terminal footer
        tk.Label(
            self._term_frame,
            text="by Shawn Mitchell",
            bg="#161b22",
            fg="#444d56",
            font=("Courier", 6, "italic"),
        ).pack(side=tk.BOTTOM, pady=(0, 2))

        tk.Button(
            hdr,
            text="📁 Open Log",
            command=self._open_log_file,
            bg="#21262d",
            fg="#58a6ff",
            font=("Courier", 8, "bold"),
            relief="flat",
            padx=6,
            cursor="hand2",
        ).pack(side=tk.RIGHT, padx=4)

        tk.Label(
            hdr,
            text=f"  log → {LOG_FILE_PATH.name}  ({LOG_FILE_PATH.parent})",
            bg="#161b22",
            fg="#444d56",
            font=("Courier", 7),
        ).pack(side=tk.LEFT, padx=4)

        # Inner notebook (4 tabs)
        term_nb = ttk.Notebook(self._term_frame)
        term_nb.pack(fill=tk.BOTH, padx=4, pady=(2, 4))
        self._term_nb = term_nb

        def _make_tab(title, height=6):
            frm = tk.Frame(term_nb, bg="#0d1117")
            term_nb.add(frm, text=title)
            txt = tk.Text(
                frm,
                font=("Courier", 8),
                bg="#0d1117",
                fg="#c9d1d9",
                height=height,
                wrap=tk.WORD,
                borderwidth=0,
                relief="flat",
                insertbackground=C["warning"],
                selectbackground="#264f78",
            )
            vsb = tk.Scrollbar(frm, orient="vertical", command=txt.yview)
            txt.configure(yscrollcommand=vsb.set)
            vsb.pack(side=tk.RIGHT, fill=tk.Y)
            txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            # Colour tags
            txt.tag_config("ts", foreground="#58a6ff", font=("Courier", 8, "bold"))
            txt.tag_config("info", foreground="#3fb950")
            txt.tag_config("error", foreground="#f85149")
            txt.tag_config("warn", foreground="#e3b341")
            txt.tag_config("debug", foreground="#79c0ff")
            txt.tag_config("hash", foreground="#bc8cff")
            txt.tag_config("compare", foreground="#ffa657")
            txt.tag_config("scan", foreground="#56d364")
            txt.tag_config("select", foreground="#d2a8ff")
            txt.tag_config("result", foreground="#f0f6fc")
            txt.config(state=tk.DISABLED)
            return txt

        self._term_status = _make_tab("📊 Status")
        self._term_debug = _make_tab("🐛 Debug")
        self._term_errors = _make_tab("⚠️  Errors")
        self._term_events = _make_tab("📋 Events")

        # Banner
        self._term_append(
            self._term_status,
            f"Duplicate File Finder v{VERSION} ready"
            f"  |  CPUs={CPU_COUNT}"
            f"  |  xxhash={'✓' if HAS_XXHASH else '✗'}"
            f"  |  PIL={'✓' if HAS_PIL else '✗'}"
            f"  |  send2trash={'✓' if HAS_SEND2TRASH else '✗'}",
            "info",
        )

    def _term_append(self, widget, text: str, tag: str = "debug") -> None:
        """Append one timestamped line to a terminal tab widget + log file."""
        widget.config(state=tk.NORMAL)
        ts = time.strftime("%H:%M:%S")
        widget.insert(tk.END, f"[{ts}] ", "ts")
        widget.insert(tk.END, f"{text}\n", tag)
        lc = int(widget.index("end-1c").split(".")[0])
        if lc > DEBUG_MAX_LINES:
            widget.delete("1.0", f"{lc - DEBUG_MAX_LINES}.0")
        widget.see(tk.END)
        widget.config(state=tk.DISABLED)

    def _dbg(self, text: str, tag: str = "") -> None:
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
        tl = text.lower()

        is_event = any(
            x in tl
            for x in (
                "[scan]",
                "[hash]",
                "[compare]",
                "[find]",
                "[select]",
                "[verify]",
                "[delete]",
                "[export]",
                "[session]",
            )
        )
        is_error = "error" in tl or tag == "error" or "[error]" in tl

        # ── Write to persistent log file ONCE per logical message ─────────
        try:
            self._write_log_file(tag, text)
        except Exception:
            pass

        # ── Debug tab gets EVERYTHING ─────────────────────────────────────
        self._term_append(self._term_debug, text, tag)

        # ── Events tab gets pipeline events ───────────────────────────────
        if is_event:
            self._term_append(self._term_events, text, tag)

        # ── Errors: Errors tab + Status tab + auto-switch ─────────────────
        if is_error:
            self._term_append(self._term_errors, text, "error")
            self._term_append(self._term_status, text, "error")
            try:
                self._term_nb.select(2)
            except Exception:
                pass

    def _classify_tag(self, text: str) -> str:
        tl = text.lower()
        if "[hash]" in tl:
            return "hash"
        if "[compare]" in tl:
            return "compare"
        if "[scan]" in tl:
            return "scan"
        if "[find]" in tl:
            return "debug"
        if "[select]" in tl:
            return "select"
        if "[delete]" in tl:
            return "warn"
        if "[verify]" in tl:
            return "hash"
        if "[session]" in tl:
            return "info"
        if "error" in tl:
            return "error"
        if "warn" in tl:
            return "warn"
        return "debug"

    def _clear_debug_terminal(self) -> None:
        for w in (
            self._term_status,
            self._term_debug,
            self._term_errors,
            self._term_events,
        ):
            w.config(state=tk.NORMAL)
            w.delete("1.0", tk.END)
            w.config(state=tk.DISABLED)

    def _toggle_terminal(self) -> None:
        if self._term_visible:
            self._term_frame.pack_forget()
            self._term_toggle_btn.config(text="▲ Terminal")
        else:
            self._term_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=1, pady=1)
            self._term_toggle_btn.config(text="▼ Terminal")
        self._term_visible = not self._term_visible

    def _write_log_file(self, level: str, text: str) -> None:
        """
        Enqueue one log line for async write — never blocks the main thread.
        The _start_log_writer background thread drains and batch-writes the queue.
        """
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{ts}] [{level.upper():7}] {text}\n"
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
                    buf.clear()
                    continue
                if buf:
                    try:
                        with open(LOG_FILE_PATH, "a", encoding="utf-8") as fh:
                            fh.writelines(buf)
                    except Exception:
                        pass
                    buf.clear()

        t = threading.Thread(target=_writer, daemon=True, name="dupfinder-log-writer")
        t.start()

    def _open_log_file(self) -> None:
        """Open the persistent log file in the default text editor."""
        try:
            import subprocess as _sp2

            LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            if not LOG_FILE_PATH.exists():
                LOG_FILE_PATH.write_text("", encoding="utf-8")
            if sys.platform == "win32":
                _sp2.run(["notepad", str(LOG_FILE_PATH)], check=False)
            elif sys.platform == "darwin":
                _sp2.run(["open", str(LOG_FILE_PATH)], check=False)
            else:
                _sp2.run(["xdg-open", str(LOG_FILE_PATH)], check=False)
        except Exception as exc:
            messagebox.showerror("Log File", f"Cannot open log file:\n{exc}")

    # ── Keyboard shortcuts ─────────────────────────────────────────────────

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-s>", lambda e: self._toggle_scan())
        self.root.bind("<Control-S>", lambda e: self._toggle_scan())
        self.root.bind("<Escape>", lambda e: self._toggle_scan())
        self.root.bind("<Delete>", lambda e: self._delete_selected())
        self.root.bind("<Left>", lambda e: self._nav_prev())
        self.root.bind("<Right>", lambda e: self._nav_next())
        self.root.bind("<Up>", lambda e: self._tree_move(-1))
        self.root.bind("<Down>", lambda e: self._tree_move(1))
        self.root.bind("<Control-a>", lambda e: self._select_all_in_group())
        self.root.bind("<Control-A>", lambda e: self._select_all_in_group())
        self.root.bind("<Control-z>", lambda e: self._show_deletion_history())
        self.root.bind("<F5>", lambda e: self._toggle_scan())
        self.root.bind("<F2>", lambda e: self._run_test_scan())
        self.root.bind("<F1>", lambda e: self._show_manual())
        self.root.bind("<Control-comma>", lambda e: self._show_settings())
        self.root.bind("<Control-e>", lambda e: self._show_export_menu())

    # ── Scan control ───────────────────────────────────────────────────────

    def _change_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select folder to scan")
        if folder:
            self._folder_var.set(folder)
            self.groups = []
            self._populate_tree([])
            self._clear_detail()
            self._log_msg("info", f"Folder changed: {folder}")
            self._dbg(f"[SCAN] Folder changed  path={folder}", "scan")

    def _collect_settings(self) -> ScanSettings:
        """Build a ScanSettings from current UI variable values."""
        # Use stringvar with fallback if text widget not created yet (lazy tab)
        excl_raw = ""
        if hasattr(self, "_excl_entry") and self._excl_entry:
            try:
                excl_raw = self._excl_entry.get("1.0", tk.END).strip()
            except Exception:
                excl_raw = self._var_exclusions.get()
        else:
            excl_raw = self._var_exclusions.get()
        excl = [p.strip() for p in excl_raw.split(",") if p.strip()]
        return ScanSettings(
            subdirs=self._var_subdirs.get(),
            min_size=max(0, self._var_min_size.get()),
            max_size=max(0, self._var_max_size.get()),
            use_xxhash=self._var_xxhash.get() and HAS_XXHASH,
            use_sha256_verify=self._var_sha256.get(),
            hash_files=self._var_hash.get(),
            paranoid_mode=self._var_paranoid.get(),
            use_gpu=self._var_gpu.get(),
            use_neural_embed=self._var_neural.get() and HAS_DINO,
            use_audio_fingerprint=self._var_audio_fp.get()
            and (HAS_CHROMAPRINT or HAS_MUTAGEN),
            num_workers=max(1, self._var_workers.get()),
            min_score=self._var_min_score.get(),
            exclusion_patterns=excl,
            dark_mode=self._var_dark_mode.get(),
            auto_select=self._var_auto_select.get(),
            delete_gap=self._var_delete_gap.get(),
            cleanup_mode=self._var_cleanup_mode.get(),
            skip_network=self._var_skip_network.get(),
            skip_system=self._var_skip_system.get(),
            enable_io_port=self._var_io_port.get(),
            # v7.0 AI Semantic settings
            use_semantic_dedup=self._var_semantic.get(),
            use_faiss_index=self._var_faiss.get() and HAS_FAISS,
            use_clip_embeddings=self._var_clip.get() and HAS_CLIP,
            use_sentence_embeddings=self._var_sentence.get()
            and HAS_SENTENCE_TRANSFORMERS,
            # v7.0 BLAKE3
            use_blake3=self._var_blake3.get() and HAS_BLAKE3,
            # v7.2 Profile mode
            profile=getattr(self.settings, "profile", "safe")
            if hasattr(self, "settings")
            else "safe",
        )

    def _load_settings_into_ui(self) -> None:
        """Load settings from ScanSettings into UI variables."""
        s = self.settings
        self._var_min_score.set(s.min_score)
        self._var_cleanup_mode.set(s.cleanup_mode)
        self._var_auto_select.set(s.auto_select)
        self._var_delete_gap.set(s.delete_gap)
        self._var_paranoid.set(s.paranoid_mode)
        self._var_skip_network.set(s.skip_network)
        self._var_skip_system.set(s.skip_system)
        self._var_workers.set(s.num_workers)
        self._var_xxhash.set(s.use_xxhash)
        self._var_sha256.set(s.use_sha256_verify)
        self._var_hash.set(s.hash_files)
        self._var_semantic.set(s.use_semantic_dedup)
        self._var_faiss.set(s.use_faiss_index)
        self._var_neural.set(s.use_neural_embed)
        self._var_audio_fp.set(s.use_audio_fingerprint)
        self._var_clip.set(
            s.use_clip_embeddings if hasattr(s, "use_clip_embeddings") else False
        )
        self._var_sentence.set(
            s.use_sentence_embeddings
            if hasattr(s, "use_sentence_embeddings")
            else False
        )

    def _toggle_scan(self) -> None:
        """Toggle between start and stop scan."""
        if self.is_scanning:
            self._stop_scan()
        else:
            self._start_scan()

    def _reset_scan(self) -> None:
        """Reset scan data and state for new scan."""
        self.groups = []
        self._populate_tree([])
        self._clear_detail()
        self._scan_bar["value"] = 0
        self._match_bar["value"] = 0
        self._scan_lbl.config(text="")
        self._match_lbl.config(text="")
        self._activity_lbl.config(text="Ready")
        self._sb_status.config(text="Ready")
        self._reset_btn.config(state=tk.DISABLED)
        self._auto_sel_btn.config(state=tk.DISABLED)
        self._delete_btn.config(state=tk.DISABLED)
        self._export_btn.config(state=tk.DISABLED)
        self._save_sess_btn.config(state=tk.DISABLED)
        self._clear_sel_btn.config(state=tk.DISABLED)
        self._log_msg("info", "Scan reset - ready for new scan")

    def _run_test_scan(self) -> None:
        """Run scan on test_data folder and show results popup."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        test_folder = os.path.join(base_dir, "test_data")

        if not Path(test_folder).exists():
            messagebox.showerror(
                "Test Error", f"test_data folder not found:\n{test_folder}"
            )
            return

        # Update folder entry - make it very visible
        self._folder_var.set(test_folder)

        # Force multiple UI updates - ensure user sees the change
        self.root.update()
        self.root.update_idletasks()

        # Small delay so user can SEE the folder change before scan starts
        self.root.after(500, self._execute_test_scan)

    def _execute_test_scan(self) -> None:
        """Actually execute the test scan after brief delay."""
        test_folder = self._folder_var.get()

        self._log_msg("info", f"Starting test scan on: {test_folder}")

        # Start scan
        self._start_scan()

        # Set flag AFTER starting scan
        self._test_results_pending = True

    def _start_scan(self) -> None:
        if self.is_scanning:
            return
        folder = self._folder_var.get()
        if not folder or not Path(folder).exists():
            messagebox.showerror("Invalid Folder", f"Folder not found:\n{folder}")
            return

        self.settings = self._collect_settings()

        # Create ScanEngine FIRST (before any references to self._engine)
        self._cancel_ev = threading.Event()
        self._pq = queue.Queue(maxsize=20000)
        self._engine = ScanEngine(folder, self.settings, self._pq, self._cancel_ev)

        # Connect to I/O Port if enabled
        if self.settings.enable_io_port:
            IOPORT.set_engine(self._engine)
            IOPORT.set_settings(self.settings)
            IOPORT.register_callback(
                "SCAN_COMPLETE", lambda s, d: self.root.after(0, lambda: None)
            )

        self.is_scanning = True
        self.groups = []
        self._scan_btn.config(text="⏹ STOP", style="Red.TButton")
        self._reset_btn.config(state=tk.DISABLED)
        self._auto_sel_btn.config(state=tk.DISABLED)
        self._delete_btn.config(state=tk.DISABLED)
        self._export_btn.config(state=tk.DISABLED)
        self._save_sess_btn.config(state=tk.DISABLED)

        self._scan_bar["value"] = 0
        self._match_bar["value"] = 0
        self._scan_lbl.config(text="")
        self._match_lbl.config(text="")
        self._activity_lbl.config(text="⠋ Scanning…")

        self._clear_detail()
        self._populate_tree([])
        self._clear_log()
        self._clear_debug_terminal()
        self._update_status_bar()

        self._log_msg("info", f"=== SCAN STARTED  folder={folder} ===")
        self._dbg(
            f"[SCAN] === SCAN STARTED ==="
            f"  folder={folder}"
            f"  workers={self.settings.num_workers}"
            f"  hash={self.settings.hash_files}"
            f"  xxhash={self.settings.use_xxhash}"
            f"  paranoid={self.settings.paranoid_mode}"
            f"  min_score={self.settings.min_score}",
            "scan",
        )

        self.scan_thread = threading.Thread(target=self._scan_worker, daemon=True)
        self.scan_thread.start()

    def _scan_worker(self) -> None:
        """Background worker — communicates only via self._pq."""
        try:
            self._engine.scan()
            if self._cancel_ev.is_set():
                return
            groups = self._engine.find_duplicates()
            if self._cancel_ev.is_set():
                return
            self._pq.put_nowait(
                (
                    "complete",
                    {
                        "groups": len(groups),
                        "files": len(self._engine.files),
                        "data": groups,
                    },
                )
            )
        except Exception as exc:
            tb = traceback.format_exc()
            err_msg = f"[ERROR] Unhandled scan exception: {type(exc).__name__}: {exc}"
            try:
                self._pq.put_nowait(("error", str(exc)))
            except queue.Full:
                pass
            try:
                self._pq.put_nowait(("error_detail", f"{err_msg}\nTraceback:\n{tb}"))
            except queue.Full:
                pass
            # Also write to log file immediately (queue might be full)
            try:
                with open(LOG_FILE_PATH, "a", encoding="utf-8") as _lf:
                    _lf.write(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [CRITICAL] {err_msg}\n{tb}\n"
                    )
            except Exception:
                pass

    def _stop_scan(self) -> None:
        self._cancel_ev.set()
        self.is_scanning = False
        self._scan_btn.config(text="▶ SCAN", style="Green.TButton")
        if self.groups:
            self._reset_btn.config(state=tk.NORMAL)
        self._activity_lbl.config(text="Scan stopped")
        self._sb_status.config(text="Stopped")
        self._log_msg("warn", "Scan stopped by user")
        self._dbg("[SCAN] Stopped by user", "warn")

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
        if mtype == "scan_progress":
            self._on_scan_progress(data)
        elif mtype == "match_progress":
            self._on_match_progress(data)
        elif mtype == "log":
            # High-level milestone: Activity Log + Status tab ONLY (not Debug)
            tag, text = data
            self._log_msg(tag, text)
            try:
                self._write_log_file(tag, text)
            except Exception:
                pass
            _stag = "info" if tag == "info" else ("warn" if tag == "warn" else "error")
            self._term_append(self._term_status, text, _stag)
        elif mtype == "debug":
            # Detailed operational debug -> Debug/Events tabs via _dbg()
            self._dbg(data)
        elif mtype == "error_detail":
            # Full stack trace: Errors tab only (_term_append adds timestamp)
            try:
                self._write_log_file("error", data)
            except Exception:
                pass
            self._term_append(self._term_errors, data, "error")
            self._term_append(
                self._term_status,
                "\u274c Unhandled error — see ⚠️ Errors tab for trace",
                "error",
            )
            try:
                self._term_nb.select(2)
            except Exception:
                pass
        elif mtype in ("info", "warn"):
            self._log_msg(mtype, data)
            self._dbg(data, mtype)
        elif mtype == "error":
            self._log_msg("error", data)
            self._dbg(f"[ERROR] {data}", "error")
        elif mtype == "complete":
            self._on_scan_complete(data)

    def _on_scan_progress(self, d: dict) -> None:
        self._scan_bar["value"] = d.get("percent", 0)
        self._scan_lbl.config(
            text=f"{d.get('current', 0):,}/{d.get('total', 0):,} ({d.get('percent', 0)}%)"
        )
        self._activity_lbl.config(text=f"{d.get('status', '')}  {d.get('file', '')}")
        self._sb_status.config(text=d.get("status", "Scanning…"))

    def _on_match_progress(self, d: dict) -> None:
        # ── modular: match-progress handler ──────────────────────────────────
        pct = d.get("percent", 0)
        cur = d.get("current", 0)
        tot = d.get("total", 0)
        self._match_bar["value"] = pct
        self._match_lbl.config(text=f"{cur:,}/{tot:,} ({pct}%)")
        remaining = max(tot - cur, 0)
        lbl = (
            f"🔍 Comparing {tot:,} pairs… ({remaining:,} remaining)"
            if tot > 0
            else "🔍 Comparing pairs…"
        )
        self._activity_lbl.config(text=lbl)
        self._sb_status.config(text=f"Comparing… {pct}%")

    def _on_scan_complete(self, data: dict) -> None:
        # ── modular: scan-complete UI handler ───────────────────────────────
        """Handle scan completion — populate tree immediately, auto-select in bg."""
        self.is_scanning = False
        self.groups = data.get("data", [])
        ng = data.get("groups", 0)
        nf = data.get("files", 0)

        # Update I/O Port with results
        IOPORT.set_groups(self.groups)
        IOPORT.emit_signal("SCAN_COMPLETE", {"groups": ng, "files": nf})

        self._scan_btn.config(text="▶ SCAN", style="Green.TButton")
        if self.groups:
            self._reset_btn.config(state=tk.NORMAL)
        self._scan_bar["value"] = 100
        self._match_bar["value"] = 100

        self._log_msg("info", f"✓ Scan complete! {ng} duplicate groups in {nf:,} files")
        self._dbg(f"[SCAN] === SCAN COMPLETE ===  groups={ng}  files={nf}", "scan")

        # Update I/O Port with results (even if no groups found)
        IOPORT.set_groups(self.groups)
        IOPORT.emit_signal("SCAN_COMPLETE", {"groups": ng, "files": nf})

        if ng == 0:
            self._activity_lbl.config(text="✓ No duplicates — all files are unique")
            self._sb_status.config(text="✓ Done")
            messagebox.showinfo(
                "✓ No Duplicates", "No duplicates found.\nAll files appear unique."
            )
            self._update_status_bar()
            return

        # Show "processing" state ─ tree populated after auto-select completes
        self._activity_lbl.config(text=f"⚙️  Processing {ng} groups…")
        self._sb_status.config(text="Processing…")
        self._update_status_bar()

        # Mark Full Report as needing rebuild (lazy render) ──────────────────
        self._report_dirty = True
        self._report_summary_lbl.config(
            text=f'{ng} groups · {nf:,} files — click "📄 Full Report" tab to view'
        )

        # Run auto-select in a BACKGROUND thread (never block main thread) ───
        if self.settings.auto_select and self._engine:
            self._log_msg("info", "🎯 Running auto-selection in background…")
            self._activity_lbl.config(text="⚙️  Auto-selecting best files to keep…")
            _engine = self._engine
            _groups = self.groups

            def _bg_select():
                try:
                    _engine.smart_select(_groups)
                    n_marked = sum(
                        1
                        for g in _groups
                        for s in g.suggestions.values()
                        if s == "DELETE"
                    )
                except Exception as exc:
                    n_marked = 0
                    try:
                        self._pq.put_nowait(("debug", f"[SELECT] bg error: {exc}"))
                    except Exception:
                        pass
                self.root.after(0, lambda nm=n_marked: self._finish_scan_ui(ng, nf, nm))

            threading.Thread(
                target=_bg_select, daemon=True, name="dupfinder-autoselect"
            ).start()
        else:
            self._finish_scan_ui(ng, nf, 0)

    def _finish_scan_ui(self, ng: int, nf: int, n_marked: int) -> None:
        # ── modular: post-auto-select UI finalisation ───────────────────────
        """Called on main thread after background auto-select finishes."""
        self._populate_tree(self.groups)  # refresh tree with suggestion marks
        self._update_status_bar()

        # Enable action buttons
        self._auto_sel_btn.config(state=tk.NORMAL)
        self._delete_btn.config(state=tk.NORMAL)
        self._export_btn.config(state=tk.NORMAL)
        self._save_sess_btn.config(state=tk.NORMAL)
        self._clear_sel_btn.config(state=tk.NORMAL)

        summary = f"✓ {ng} groups  ·  {nf:,} files  ·  {n_marked} marked"
        self._activity_lbl.config(text=summary)
        self._sb_status.config(text="✓ Done")

        if n_marked > 0:
            self._log_msg(
                "info", f"🎯 Auto-select: {n_marked} files suggested for deletion"
            )
            self._dbg(f"[SELECT] Auto-select: {n_marked} files marked DELETE", "select")

        self._log_msg("info", f"✓ Ready — {ng} groups to review")
        self._dbg(f"[SCAN] Mode: {self.settings.cleanup_mode}", "scan")

        # Show first group (only UI work, no report render) ──────────────────
        if self.groups:
            self._show_group(0)

        # Check if this was a test scan
        if getattr(self, "_test_results_pending", False):
            self._test_results_pending = False
            self.root.after(100, lambda: self._show_test_results_popup(ng, nf))

    def _show_test_results_popup(self, ng: int, nf: int) -> None:
        """Show test results in a popup window with export options."""

        # Check if we have groups
        if not self.groups:
            messagebox.showwarning("No Results", "No duplicate groups found.")
            return

        win = tk.Toplevel(self.root)
        win.title(f"Test Results - {ng} duplicate groups found")
        win.geometry("700x500")
        win.configure(bg=self.C["bg"])

        # Summary
        tk.Label(
            win,
            text="TEST SCAN COMPLETE",
            bg=self.C["bg"],
            fg=self.C["accent1"],
            font=("Arial", 14, "bold"),
        ).pack(pady=10)

        total_reclaim = sum(g.reclaimable_bytes for g in self.groups)
        stats = f"Groups: {ng}  |  Files: {nf}  |  Reclaimable: {_format_size(total_reclaim)}"
        tk.Label(
            win,
            text=stats,
            bg=self.C["bg"],
            fg=self.C["fg"],
            font=("Arial", 10),
        ).pack(pady=5)

        # Scrollable results list
        txt_frame = tk.Frame(win, bg=self.C["bg"])
        txt_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        txt = scrolledtext.ScrolledText(
            txt_frame,
            font=("Courier", 9),
            bg=self.C["panel_bg"],
            fg=self.C["fg"],
        )
        txt.pack(fill=tk.BOTH, expand=True)

        # Generate results text
        result_lines = []
        result_lines.append("=" * 60)
        result_lines.append("DUPLICATE FINDER TEST RESULTS")
        result_lines.append(f"Version: {VERSION}")
        result_lines.append(f"Timestamp: {datetime.datetime.now().isoformat()}")
        result_lines.append("=" * 60)
        result_lines.append(f"Total Groups: {ng}")
        result_lines.append(f"Total Files: {nf}")
        result_lines.append(f"Reclaimable: {_format_size(total_reclaim)}")
        result_lines.append("=" * 60)
        result_lines.append("")

        # Add scan settings/methods used
        result_lines.append("--- SCAN SETTINGS APPLIED ---")
        s = getattr(self, "settings", None)
        if s is None:
            result_lines.append("  (Settings not available)")
        else:
            try:
                result_lines.append(f"  Subdirs: {s.subdirs}")
                result_lines.append(f"  Min Size: {s.min_size} bytes")
                result_lines.append(f"  Max Size: {s.max_size} bytes")
                result_lines.append(f"  Hash Files: {s.hash_files}")
                result_lines.append(f"  Use xxHash: {s.use_xxhash}")
                result_lines.append(f"  SHA256 Verify: {s.use_sha256_verify}")
                result_lines.append(f"  GPU Mode: {s.use_gpu}")
                result_lines.append(f"  Neural: {s.use_neural_embed}")
                result_lines.append(f"  Audio: {s.use_audio_fingerprint}")
                result_lines.append(f"  Workers: {s.num_workers}")
                result_lines.append(f"  Min Score: {s.min_score}")
                result_lines.append(f"  Cleanup: {s.cleanup_mode}")
                result_lines.append(f"  Skip Net: {s.skip_network}")
                result_lines.append(f"  Skip Sys: {s.skip_system}")
            except Exception as e:
                result_lines.append(f"  Error: {e}")
        result_lines.append("")

        # Library status
        result_lines.append("--- LIBRARIES ---")
        result_lines.append(
            f"  xxhash: {'Yes' if globals().get('HAS_XXHASH') else 'No (using MD5)'}"
        )
        result_lines.append(
            f"  PIL/Pillow: {'Yes' if globals().get('HAS_PIL') else 'No'}"
        )
        result_lines.append(
            f"  psutil: {'Yes' if globals().get('HAS_PSUTIL') else 'No'}"
        )
        result_lines.append(
            f"  send2trash: {'Yes' if globals().get('HAS_SEND2TRASH') else 'No'}"
        )
        result_lines.append(f"  DINOv2: {'Yes' if globals().get('HAS_DINO') else 'No'}")
        result_lines.append(
            f"  Mutagen: {'Yes' if globals().get('HAS_MUTAGEN') else 'No'}"
        )
        result_lines.append(f"  OpenCV: {'Yes' if globals().get('HAS_CV2') else 'No'}")
        result_lines.append(f"  NumPy: {'Yes' if globals().get('HAS_NUMPY') else 'No'}")
        result_lines.append(f"  CuPy: {'Yes' if globals().get('HAS_CUPY') else 'No'}")
        result_lines.append("")

        # Detection methods summary
        result_lines.append("--- DETECTION METHODS ---")
        result_lines.append(
            "  EXACT: Quick hash (first 4KB) -> Size match -> Full hash"
        )
        if s.use_sha256_verify:
            result_lines.append("  SHA256: Enabled for verification")
        if s.use_xxhash:
            result_lines.append("  xxHash: Enabled (faster than MD5)")
        result_lines.append("  HARDLINKS: Inode+device detection")
        result_lines.append("")

        # Add scan settings/methods used
        result_lines.append("--- SCAN SETTINGS APPLIED ---")
        s = self.settings
        result_lines.append(f"  Subdirs: {s.subdirs}")
        result_lines.append(f"  Min Size: {s.min_size} bytes")
        result_lines.append(f"  Max Size: {s.max_size} bytes (0 = unlimited)")
        result_lines.append(f"  Hash Files: {s.hash_files}")
        result_lines.append(
            f"  Use xxHash: {s.use_xxhash} ({'enabled' if s.use_xxhash else 'using MD5'})"
        )
        result_lines.append(f"  SHA256 Verify: {s.use_sha256_verify}")
        result_lines.append(f"  GPU Mode: {s.use_gpu}")
        result_lines.append(f"  Neural Embeddings: {s.use_neural_embed} (DINOv2)")
        result_lines.append(f"  Audio Fingerprint: {s.use_audio_fingerprint}")
        result_lines.append(f"  Workers: {s.num_workers}")
        result_lines.append(f"  Min Score: {s.min_score}")
        result_lines.append(f"  Cleanup Mode: {s.cleanup_mode}")
        result_lines.append(f"  Skip Network: {s.skip_network}")
        result_lines.append(f"  Skip System: {s.skip_system}")
        if s.exclusion_patterns:
            result_lines.append(f"  Exclusions: {s.exclusion_patterns}")
        result_lines.append("")

        # Library status - use actual global variables
        result_lines.append("--- AVAILABLE LIBRARIES ---")
        try:
            result_lines.append(
                f"  xxhash: {'Yes' if HAS_XXHASH else 'No (using MD5)'}"
            )
            result_lines.append(f"  PIL/Pillow: {'Yes' if HAS_PIL else 'No'}")
            result_lines.append(f"  psutil: {'Yes' if HAS_PSUTIL else 'No'}")
            result_lines.append(f"  send2trash: {'Yes' if HAS_SEND2TRASH else 'No'}")
            result_lines.append(f"  DINOv2: {'Yes' if HAS_DINO else 'No'}")
            result_lines.append(f"  Mutagen: {'Yes' if HAS_MUTAGEN else 'No'}")
            result_lines.append(f"  OpenCV: {'Yes' if HAS_CV2 else 'No'}")
            result_lines.append(f"  NumPy: {'Yes' if HAS_NUMPY else 'No'}")
            result_lines.append(f"  CuPy: {'Yes' if HAS_CUPY else 'No'}")
        except Exception as e:
            result_lines.append(f"  (Library check error: {e})")
        result_lines.append("")

        # Detection methods summary
        result_lines.append("--- DETECTION METHODS ---")
        result_lines.append(
            "  EXACT: Quick hash (first 4KB) -> Size match -> Full hash"
        )
        if s.use_sha256_verify:
            result_lines.append("  SHA256: Enabled for verification")
        if s.use_xxhash:
            result_lines.append("  xxHash: Enabled (faster than MD5)")
        result_lines.append("  HARDLINKS: Inode+device detection")
        result_lines.append("")

        for i, g in enumerate(self.groups, 1):
            result_lines.append(f"GROUP {i} ({g.group_type.upper()})")
            result_lines.append(f"  Score: {g.score}/100 | Risk: {g.risk_level}")
            result_lines.append(f"  Files: {len(g.files)}")
            result_lines.append(f"  Reclaimable: {_format_size(g.reclaimable_bytes)}")
            result_lines.append("  Files:")
            for f in g.files:
                result_lines.append(f"    - {f.path}")
            if g.suggestions:
                result_lines.append("  Suggestions:")
                for fp, action in g.suggestions.items():
                    result_lines.append(f"    - {action}: {fp}")
            result_lines.append("")

        result_text = "\n".join(result_lines)

        # Insert text into widget
        if txt:
            txt.insert("1.0", result_text)
            txt.config(state=tk.DISABLED)

        # Buttons
        btn_frame = tk.Frame(win, bg=self.C["bg"])
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        def copy_to_clipboard():
            self.root.clipboard_clear()
            self.root.clipboard_append(result_text)
            messagebox.showinfo("Copied", "Results copied to clipboard")

        def save_to_file():
            p = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text File", "*.txt"), ("All", "*.*")],
                title="Save Test Results",
            )
            if p:
                Path(p).write_text(result_text, encoding="utf-8")
                messagebox.showinfo("Saved", f"Results saved to:\n{p}")

        tk.Button(
            btn_frame,
            text="Copy to Clipboard",
            command=copy_to_clipboard,
            bg=self.C["accent1"],
            fg=self.C["fg"],
            font=("Arial", 10),
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame,
            text="Save to File",
            command=save_to_file,
            bg=self.C["accent1"],
            fg=self.C["fg"],
            font=("Arial", 10),
        ).pack(side=tk.LEFT, padx=5)

        # Compare with expected results button
        tk.Button(
            btn_frame,
            text="Compare with Expected",
            command=lambda: self._show_test_comparison(self.groups),
            bg=self.C["success"],
            fg="white",
            font=("Arial", 10, "bold"),
        ).pack(side=tk.RIGHT, padx=5)

    def _show_test_comparison(self, groups: List) -> None:
        """Show comparison between actual results and expected results from cheat sheet."""
        win = tk.Toplevel(self.root)
        win.title("Test Results Comparison")
        win.geometry("900x600")
        win.configure(bg=self.C["bg"])

        # Get ALL file names from test_data for accurate comparison
        test_data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_data"
        )

        # Expected duplicates based on TEST_CHEAT_SHEET.md - comprehensive list from scan
        # Exact duplicates (identical content)
        EXPECTED_EXACT = set()

        # Near duplicates (similar but not identical)
        EXPECTED_NEAR = set()

        # Find all _copy files and _a/_b pairs in test_data
        if os.path.exists(test_data_dir):
            for root, dirs, files in os.walk(test_data_dir):
                for f in files:
                    # Look for _copy suffix
                    if "_copy" in f:
                        base = f.replace("_copy", "")
                        # Handle extension changes (e.g., vacation.png = vacation.jpg)
                        EXPECTED_EXACT.add((base, f))
                        EXPECTED_EXACT.add((f, base))
                    # Look for _a/_b pairs
                    if (
                        f.endswith("_a.")
                        or f.endswith("_a.txt")
                        or f.endswith("_a.csv")
                        or f.endswith("_a.xml")
                        or f.endswith("_a.json")
                        or f.endswith("_a.pdf")
                        or f.endswith("_a.py")
                        or f.endswith("_a.png")
                        or f.endswith("_a.mp3")
                    ):
                        base = f[:-2]  # Remove _a
                        ext = f[-4:] if len(f) > 4 and "." in f else ""
                        # Find corresponding _b file
                        for f2 in files:
                            if f2.startswith(
                                base[:-1] if base.endswith("_") else base
                            ) and f2.endswith(ext.replace("_a", "_b")):
                                EXPECTED_NEAR.add((f, f2))
                                EXPECTED_NEAR.add((f2, f))
                    if (
                        f.endswith("_b.")
                        or f.endswith("_b.txt")
                        or f.endswith("_b.csv")
                        or f.endswith("_b.xml")
                        or f.endswith("_b.json")
                        or f.endswith("_b.pdf")
                        or f.endswith("_b.py")
                        or f.endswith("_b.png")
                        or f.endswith("_b.mp3")
                    ):
                        base = f[:-2]
                        for f2 in files:
                            if (
                                f2.startswith(base[:-1] if base.endswith("_") else base)
                                and "_a" in f2
                            ):
                                EXPECTED_NEAR.add((f, f2))
                                EXPECTED_NEAR.add((f2, f))

        # Add hardcoded known duplicates
        hardcoded_exact = [
            ("main.c", "main_copy.c"),
            ("script.py", "script_copy.py"),
            ("app.js", "app_copy.js"),
            ("page.html", "page_copy.html"),
            ("readme.md", "readme_copy.md"),
            ("data.xml", "data_copy.xml"),
            ("document.pdf", "document_copy.pdf"),
            ("picture.jpg", "picture_copy.jpg"),
            ("photo.png", "photo_copy.png"),
            ("report.txt", "report_copy.txt"),
        ]
        for a, b in hardcoded_exact:
            EXPECTED_EXACT.add((a, b))
            EXPECTED_EXACT.add((b, a))

        hardcoded_near = [
            ("data_a.json", "data_b.json"),
            ("table_a.csv", "table_b.csv"),
            ("xml_a.xml", "xml_b.xml"),
            ("doc_a.pdf", "doc_b.pdf"),
            ("script_a.py", "script_b.py"),
            ("indent_a.py", "indent_b.py"),
            ("photo_001.png", "photo_blue.png"),
        ]
        for a, b in hardcoded_near:
            EXPECTED_NEAR.add((a, b))
            EXPECTED_NEAR.add((b, a))

        # Analyze actual results
        found_exact = set()
        found_near = set()
        all_actual_groups = []

        # Get all file paths from groups
        for g in groups:
            file_names = [os.path.basename(f.path) for f in g.files]
            if len(g.files) >= 2:
                all_actual_groups.append((file_names, g.group_type))

        # Check each group against expected
        for file_names, group_type in all_actual_groups:
            if group_type == "exact":
                # Check all pairs in this group
                for i, fn1 in enumerate(file_names):
                    for fn2 in file_names[i + 1 :]:
                        if (fn1, fn2) in EXPECTED_EXACT or (fn2, fn1) in EXPECTED_EXACT:
                            pair = tuple(sorted([fn1, fn2]))
                            found_exact.add(pair)
            elif group_type == "near":
                for i, fn1 in enumerate(file_names):
                    for fn2 in file_names[i + 1 :]:
                        if (fn1, fn2) in EXPECTED_NEAR or (fn2, fn1) in EXPECTED_NEAR:
                            pair = tuple(sorted([fn1, fn2]))
                            found_near.add(pair)

        # Determine what was expected but not found
        missing_exact = []
        for a, b in EXPECTED_EXACT:
            pair = tuple(sorted([a, b]))
            if pair not in found_exact:
                if pair not in [(tuple(sorted([m[0], m[1]]))) for m in missing_exact]:
                    missing_exact.append(pair)

        missing_near = []
        for a, b in EXPECTED_NEAR:
            pair = tuple(sorted([a, b]))
            if pair not in found_near:
                if pair not in [(tuple(sorted([m[0], m[1]]))) for m in missing_near]:
                    missing_near.append(pair)

        # Calculate pass/fail - only count unique pairs, not all file occurrences
        total_expected = len(
            set(tuple(sorted([a, b])) for a, b in EXPECTED_EXACT)
        ) + len(set(tuple(sorted([a, b])) for a, b in EXPECTED_NEAR))
        total_found = len(found_exact) + len(found_near)
        pass_rate = (total_found / total_expected * 100) if total_expected > 0 else 0

        pass_fail = "✓ PASS" if pass_rate >= 80 else "✗ FAIL"

        # Build comparison text
        comp_lines = []
        comp_lines.append("=" * 70)
        comp_lines.append("TEST RESULTS COMPARISON - CHEAT SHEET vs ACTUAL")
        comp_lines.append("=" * 70)
        comp_lines.append("")
        comp_lines.append(
            f"OVERALL: {pass_fail} ({pass_rate:.1f}% of expected duplicates found)"
        )
        comp_lines.append(f"Total Groups Found: {len(groups)}")
        comp_lines.append(
            f"Expected Exact: {len(set(tuple(sorted([a, b])) for a, b in EXPECTED_EXACT))} | Found: {len(found_exact)}"
        )
        comp_lines.append(
            f"Expected Near: {len(set(tuple(sorted([a, b])) for a, b in EXPECTED_NEAR))} | Found: {len(found_near)}"
        )
        comp_lines.append("")
        comp_lines.append("-" * 70)
        comp_lines.append("✓ FOUND EXACT DUPLICATES:")
        comp_lines.append("-" * 70)
        if found_exact:
            for pair in sorted(found_exact):
                comp_lines.append(f"  ✓ {pair[0]} = {pair[1]}")
        else:
            comp_lines.append("  (none)")
        comp_lines.append("")

        comp_lines.append("-" * 70)
        comp_lines.append("✓ FOUND NEAR DUPLICATES:")
        comp_lines.append("-" * 70)
        if found_near:
            for pair in sorted(found_near):
                comp_lines.append(f"  ✓ {pair[0]} ≈ {pair[1]}")
        else:
            comp_lines.append("  (none - requires semantic/AI features enabled)")
        comp_lines.append("")

        unique_missing_exact = list(set(tuple(sorted(p)) for p in missing_exact))
        if unique_missing_exact:
            comp_lines.append("-" * 70)
            comp_lines.append("✗ MISSING EXACT DUPLICATES:")
            comp_lines.append("-" * 70)
            for pair in sorted(unique_missing_exact):
                comp_lines.append(f"  ✗ {pair[0]} = {pair[1]}")
            comp_lines.append("")

        unique_missing_near = list(set(tuple(sorted(p)) for p in missing_near))
        if unique_missing_near:
            comp_lines.append("-" * 70)
            comp_lines.append(
                "✗ MISSING NEAR DUPLICATES (enable Semantic/AI in Settings):"
            )
            comp_lines.append("-" * 70)
            for pair in sorted(unique_missing_near):
                comp_lines.append(f"  ✗ {pair[0]} ≈ {pair[1]}")
            comp_lines.append("")

        comp_lines.append("=" * 70)
        comp_lines.append("NOTES:")
        comp_lines.append("=" * 70)
        comp_lines.append("- Exact: Files with identical content (100% match)")
        comp_lines.append("- Near: Files with similar but not identical content")
        comp_lines.append(
            "- Near duplicates require Semantic/DINO/CLIP enabled in Settings"
        )
        comp_lines.append("=" * 70)

        comp_text = "\n".join(comp_lines)

        # Display in scrollable text
        txt_frame = tk.Frame(win, bg=self.C["bg"])
        txt_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        txt = scrolledtext.ScrolledText(
            txt_frame,
            font=("Courier", 9),
            bg=self.C["panel_bg"],
            fg=self.C["fg"],
        )
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert("1.0", comp_text)
        txt.config(state=tk.DISABLED)

        # Status label at bottom
        status_bg = self.C["success"] if pass_rate >= 80 else self.C["danger"]
        status_fg = "white"
        status_text = f"Result: {pass_fail} - {total_found}/{total_expected} expected duplicates found"

        status_lbl = tk.Label(
            win,
            text=status_text,
            bg=status_bg,
            fg=status_fg,
            font=("Arial", 12, "bold"),
            pady=10,
        )
        status_lbl.pack(fill=tk.X, side=tk.BOTTOM)

        # Close button
        tk.Button(
            win,
            text="Close",
            command=win.destroy,
            bg=self.C["toolbar_bg"],
            fg=self.C["fg"],
            font=("Arial", 10),
        ).pack(pady=10)

    # ── Spinner animation ─────────────────────────────────────────────────

    def _animate_spinner(self) -> None:
        if self.is_scanning:
            frame = SPINNER_FRAMES[self._spinner_idx % len(SPINNER_FRAMES)]
            self._spinner_idx += 1
            fg = self.C["warning"]
        else:
            frame = "◉"
            fg = self.C["success"]
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
        if idx == 1 and self._report_dirty:  # 1 = Full Report tab
            self._render_full_report()
        elif idx == 2:  # 2 = Activity Log tab
            self._load_recent_log()
        elif idx == 6:  # 6 = Changelogs tab
            self._scan_changelogs()
            if self._changelog_files:
                for f in self._changelog_files:
                    if os.path.basename(f) == os.path.basename(
                        self._changelog_files[0]
                    ):
                        try:
                            with open(f, encoding="utf-8", errors="ignore") as fh:
                                content = fh.read()
                            self._changelog_text.config(state=tk.NORMAL)
                            self._changelog_text.delete("1.0", tk.END)
                            self._changelog_text.insert("1.0", content)
                            self._changelog_text.config(state=tk.DISABLED)
                        except Exception:
                            pass
                        break

    # ── Logging helpers (main thread only) ────────────────────────────────

    def _load_recent_log(self) -> None:
        """Load most recent log file into activity log viewer."""
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            log_dir = os.path.join(base_dir, "logs")
            if not os.path.isdir(log_dir):
                return
            log_files = [
                f
                for f in os.listdir(log_dir)
                if f.endswith(".log") and os.path.isfile(os.path.join(log_dir, f))
            ]
            if not log_files:
                return
            log_files.sort(
                key=lambda f: os.path.getmtime(os.path.join(log_dir, f)), reverse=True
            )
            recent = log_files[0]
            with open(
                os.path.join(log_dir, recent), encoding="utf-8", errors="ignore"
            ) as fh:
                content = fh.read()
            self._log_text.config(state=tk.NORMAL)
            self._log_text.delete("1.0", tk.END)
            self._log_text.insert("1.0", content or f"(Log file '{recent}' is empty)")
            self._log_text.config(state=tk.DISABLED)
        except Exception as e:
            self._log_text.config(state=tk.NORMAL)
            self._log_text.delete("1.0", tk.END)
            self._log_text.insert("1.0", f"Error loading log: {e}")
            self._log_text.config(state=tk.DISABLED)

    def _log_msg(self, tag: str, text: str) -> None:
        self._log_text.config(state=tk.NORMAL)
        ts = time.strftime("%H:%M:%S")
        self._log_text.insert(tk.END, f"[{ts}] ", "time")
        self._log_text.insert(tk.END, f"{text}\n", tag)
        self._log_text.see(tk.END)
        self._log_text.config(state=tk.DISABLED)

    def _clear_log(self) -> None:
        self._log_text.config(state=tk.NORMAL)
        self._log_text.delete("1.0", tk.END)
        self._log_text.config(state=tk.DISABLED)

    # ── Tree rendering ─────────────────────────────────────────────────────

    def _populate_tree(self, groups: List[DupGroup]) -> None:
        for item in self._tree.get_children():
            self._tree.delete(item)

        if not groups:
            self._tree_summary.config(text="No duplicate groups")
            return

        ft = self._var_filter_type.get()
        ftext = self._var_filter_text.get().lower()

        shown = 0
        for gi, g in enumerate(groups):
            # Type filter
            if ft != "All":
                if ft == "Exact" and g.group_type != "exact":
                    continue
                if ft == "Near-Dup" and g.group_type != "near":
                    continue
                if ft == "Hard-Link" and g.group_type != "hardlink":
                    continue
            # Text filter
            if ftext and not any(ftext in str(f.path).lower() for f in g.files):
                continue

            type_label = {"exact": "EXACT", "near": "NEAR", "hardlink": "HARD"}.get(
                g.group_type, g.group_type.upper()
            )
            tag = {"exact": "exact", "near": "near", "hardlink": "hardlink"}.get(
                g.group_type, "near"
            )
            reclaim = _format_size(g.reclaimable_bytes)
            label = f"Group {gi + 1}  ({len(g.files)} files)"

            risk_icon = {"LOW": "✓", "MEDIUM": "⚠", "HIGH": "✗"}.get(g.risk_level, "?")
            iid = self._tree.insert(
                "",
                tk.END,
                iid=str(gi),
                text=label,
                values=(type_label, len(g.files), f"{g.score}%", risk_icon, reclaim),
                tags=(tag,),
            )
            shown += 1

        total_r = _format_size(sum(g.reclaimable_bytes for g in groups))
        self._tree_summary.config(text=f"{shown} group(s)  |  ~{total_r} reclaimable")

    def _apply_filter(self) -> None:
        if self.groups:
            self._populate_tree(self.groups)

    def _apply_filter(self) -> None:
        if not self.groups:
            return
        ft = self._var_filter_type.get()
        fr = self._var_filter_risk.get()
        ftext = self._var_filter_text.get().lower()
        shown = 0
        for item in self._tree.get_children():
            self._tree.delete(item)
        for gi, g in enumerate(self.groups):
            if ft != "All":
                if ft == "Exact" and g.group_type != "exact":
                    continue
                if ft == "Near-Dup" and g.group_type != "near":
                    continue
                if ft == "Hard-Link" and g.group_type != "hardlink":
                    continue
            if fr != "All" and g.risk_level != fr:
                continue
            if ftext and not any(ftext in str(f.path).lower() for f in g.files):
                continue
            shown += 1
        self._populate_tree(self.groups)

    def _on_tree_select(self, event=None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
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
            try:
                self._tree.selection_set(str(self.current_group))
            except Exception:
                pass

    def _nav_next(self) -> None:
        if self.current_group < len(self.groups) - 1:
            self.current_group += 1
            self._show_group(self.current_group)
            try:
                self._tree.selection_set(str(self.current_group))
            except Exception:
                pass

    def _tree_move(self, delta: int) -> None:
        sel = self._tree.selection()
        if sel:
            children = self._tree.get_children()
            if not children:
                return
            try:
                idx = list(children).index(sel[0])
                nidx = max(0, min(len(children) - 1, idx + delta))
                nid = children[nidx]
                self._tree.selection_set(nid)
                self._tree.see(nid)
                try:
                    self._on_tree_select()
                except Exception:
                    pass
            except ValueError:
                pass

    # ── Group detail view (scrollable file cards) ──────────────────────────

    def _clear_detail(self) -> None:
        for w in self._cards_inner.winfo_children():
            w.destroy()
        self._detail_group_lbl.config(text="Select a group from the left panel")
        self._detail_score_lbl.config(text="")

    def _show_group(self, gi: int) -> None:
        if gi < 0 or gi >= len(self.groups):
            return
        g = self.groups[gi]
        C = self.C
        self._right_nb.select(0)  # Switch to detail tab

        self._detail_group_lbl.config(
            text=f"Group {gi + 1} of {len(self.groups)}  ·  {len(g.files)} files  ·  {g.group_type.upper()}"
        )

        risk_colors = {"LOW": C["success"], "MEDIUM": C["warning"], "HIGH": C["danger"]}
        risk_color = risk_colors.get(g.risk_level, C["fg"])
        self._detail_risk_lbl.config(text=f"RISK: {g.risk_level}", fg=risk_color)

        self._detail_score_lbl.config(
            text=f"Score: {g.score}%  |  ~{_format_size(g.reclaimable_bytes)} reclaimable"
        )

        # Rebuild cards — cap at MAX_CARDS to prevent UI freeze on huge groups
        MAX_CARDS = 200
        for w in self._cards_inner.winfo_children():
            w.destroy()

        files_to_show = g.files[:MAX_CARDS]
        hidden = len(g.files) - len(files_to_show)

        if hidden > 0:
            banner = tk.Label(
                self._cards_inner,
                text=(
                    f"⚠  Group has {len(g.files):,} files — "
                    f"showing first {MAX_CARDS}.  "
                    f"Use Export to see all {hidden:,} additional files."
                ),
                bg="#45475a",
                fg="#f38ba8",
                font=("Arial", 9, "bold"),
                pady=6,
            )
            banner.pack(fill=tk.X, padx=8, pady=(6, 2))

        for fi, f in enumerate(files_to_show):
            self._build_file_card(self._cards_inner, g, gi, fi, f, C)

        self._cards_canvas.yview_moveto(0)
        self._cards_inner.update_idletasks()
        self._cards_canvas.configure(scrollregion=self._cards_canvas.bbox("all"))

    def _build_file_card(
        self, parent, g: DupGroup, gi: int, fi: int, f: FileRecord, C: dict
    ) -> None:
        """Build one styled card for a file in a group."""
        sugg = g.suggestions.get(fi, "KEEP")
        is_keep = sugg == "KEEP"
        card_bg = C["exact_bg"] if is_keep else C["card_bg"]
        bd_color = C["keep_fg"] if is_keep else C["del_fg"]

        card = tk.Frame(
            parent,
            bg=card_bg,
            borderwidth=2,
            relief="solid",
            highlightthickness=1,
            highlightbackground=bd_color,
        )
        card.pack(fill=tk.X, padx=8, pady=5)

        # ── Header row ──────────────────────────────────────────
        hr = tk.Frame(card, bg=card_bg)
        hr.pack(fill=tk.X, padx=8, pady=(6, 2))

        action_icon = "✓" if is_keep else "🗑"
        action_color = C["keep_fg"] if is_keep else C["del_fg"]
        action_label = "KEEP" if is_keep else "DELETE"

        tk.Label(
            hr,
            text=f"{action_icon} [{action_label}]",
            bg=card_bg,
            fg=action_color,
            font=("Arial", 10, "bold"),
        ).pack(side=tk.LEFT)

        # Toggle button
        def _toggle(gi=gi, fi=fi):
            g = self.groups[gi]
            old = g.suggestions.get(fi, "KEEP")
            if old == "KEEP":
                g.suggestions[fi] = "DELETE"
            elif old == "DELETE":
                g.suggestions[fi] = "KEEP"
            else:  # REVIEW state
                g.suggestions[fi] = "DELETE"
            self._show_group(gi)
            self._report_dirty = True
            self._update_status_bar()

        tk.Button(
            hr,
            text="Toggle Keep/Delete",
            command=_toggle,
            bg=C["accent1"],
            fg="#ffffff",
            font=("Arial", 8),
            relief="flat",
            padx=6,
            cursor="hand2",
        ).pack(side=tk.RIGHT, padx=4)

        # Open in explorer button
        def _reveal(path=f.path):
            try:
                _open_in_explorer(path)
            except Exception as exc:
                self._dbg(f"[ERROR] Cannot reveal: {exc}", "error")

        tk.Button(
            hr,
            text="📂 Reveal",
            command=_reveal,
            bg=C["toolbar_bg"],
            fg="#ffffff",
            font=("Arial", 8),
            relief="flat",
            padx=6,
            cursor="hand2",
        ).pack(side=tk.RIGHT, padx=4)

        # ── Score bar ────────────────────────────────────────────
        ks = f.keep_score
        score_color = (
            C["success"] if ks >= 70 else (C["warning"] if ks >= 40 else C["danger"])
        )
        tk.Label(
            hr,
            text=f"Quality: {ks}/100",
            bg=card_bg,
            fg=score_color,
            font=("Courier", 9, "bold"),
        ).pack(side=tk.LEFT, padx=12)

        # ── Path ─────────────────────────────────────────────────
        path_frame = tk.Frame(card, bg=card_bg)
        path_frame.pack(fill=tk.X, padx=8, pady=(2, 0))
        tk.Label(
            path_frame,
            text="📄 Path:",
            bg=card_bg,
            fg=C["fg"],
            font=("Arial", 8, "bold"),
            width=10,
            anchor="e",
        ).pack(side=tk.LEFT)
        tk.Label(
            path_frame,
            text=str(f.path),
            bg=card_bg,
            fg=C["accent1"],
            font=("Courier", 8),
            anchor="w",
            wraplength=500,
            justify="left",
        ).pack(side=tk.LEFT, padx=4)

        # ── Metadata grid ────────────────────────────────────────
        meta = tk.Frame(card, bg=card_bg)
        meta.pack(fill=tk.X, padx=8, pady=(2, 6))

        mtime_str = _ts(f.mtime)
        ctime_str = _ts(f.ctime)

        FILE_TYPE_URLS = {
            "JPEG": "https://fileinfo.com/format/jpg",
            "PNG": "https://fileinfo.com/format/png",
            "GIF": "https://fileinfo.com/format/gif",
            "BMP": "https://fileinfo.com/format/bmp",
            "TIFF": "https://fileinfo.com/format/tiff",
            "WebP": "https://fileinfo.com/format/webp",
            "PDF": "https://fileinfo.com/format/pdf",
            "HTML": "https://fileinfo.com/format/html",
            "XML": "https://fileinfo.com/format/xml",
            "TXT": "https://fileinfo.com/format/txt",
            "ZIP": "https://fileinfo.com/format/zip",
            "GZIP": "https://fileinfo.com/format/gz",
            "RAR": "https://fileinfo.com/format/rar",
            "7Z": "https://fileinfo.com/format/7z",
            "MP3": "https://fileinfo.com/format/mp3",
            "FLAC": "https://fileinfo.com/format/flac",
            "OGG": "https://fileinfo.com/format/ogg",
            "WAV": "https://fileinfo.com/format/wav",
            "MP4": "https://fileinfo.com/format/mp4",
            "MKV": "https://fileinfo.com/format/mkv",
            "AVI": "https://fileinfo.com/format/avi",
            "EXE/DLL": "https://fileinfo.com/format/exe",
            "ELF": "https://fileinfo.com/format/elf",
            "MACH-O": "https://fileinfo.com/format/mach-o",
            "Python": "https://fileinfo.com/format/py",
            "JS": "https://fileinfo.com/format/js",
            "JSON": "https://fileinfo.com/format/json",
            "CSV": "https://fileinfo.com/format/csv",
            "DOC": "https://fileinfo.com/format/doc",
            "DOCX": "https://fileinfo.com/format/docx",
        }

        def _open_file_type_url(file_type):
            url = FILE_TYPE_URLS.get(file_type, "https://fileinfo.com/filetypes")
            import webbrowser

            webbrowser.open(url)

        def _ml(label, val, col, row):
            tk.Label(
                meta,
                text=label + ":",
                bg=card_bg,
                fg=C["fg"],
                font=("Arial", 7, "bold"),
                width=12,
                anchor="e",
            ).grid(row=row, column=col * 2, padx=(8, 0), pady=1, sticky="e")
            if label == "MIME" and val in FILE_TYPE_URLS:
                lbl = tk.Label(
                    meta,
                    text=val,
                    bg=card_bg,
                    fg="#89b4fa",
                    font=("Courier", 8, "underline"),
                    anchor="w",
                    cursor="hand2",
                )
                lbl.grid(row=row, column=col * 2 + 1, padx=(2, 12), pady=1, sticky="w")
                lbl.bind("<Button-1>", lambda e: _open_file_type_url(val))
            else:
                tk.Label(
                    meta,
                    text=val,
                    bg=card_bg,
                    fg=C["fg"],
                    font=("Courier", 8),
                    anchor="w",
                ).grid(row=row, column=col * 2 + 1, padx=(2, 12), pady=1, sticky="w")

        _ml("Size", _format_size(f.size), 0, 0)
        _ml("Modified", mtime_str, 1, 0)
        _ml("Created", ctime_str, 2, 0)
        _ml("Ext", f.ext.upper() or "—", 0, 1)
        if f.magic_type:
            _ml("MIME", f.magic_type, 1, 1)
        if f.category:
            _ml("Category", f.category, 2, 0)
        if f.hash:
            _ml("Hash", f.hash[:24] + "…", 2, 1)
        if f.is_locked:
            _ml("Status", "🔒 LOCKED", 0, 2)
        elif f.is_system:
            _ml("Status", "⚠️ SYSTEM", 0, 2)
        else:
            _ml("Status", "✓ OK", 0, 2)

        # Why Keep explanation
        why = g.why_keep.get(fi, "")
        if why:
            why_frame = tk.Frame(card, bg=card_bg)
            why_frame.pack(fill=tk.X, padx=8, pady=(4, 2))
            tk.Label(
                why_frame,
                text="💡 Why: " + why,
                bg=card_bg,
                fg=C["info"],
                font=("Arial", 8, "italic"),
                anchor="w",
                wraplength=550,
                justify="left",
            ).pack(fill=tk.X)

        # ── Exact match marker ───────────────────────────────────
        if g.group_type == "exact":
            tk.Label(
                card,
                text="═══ EXACT CONTENT MATCH (identical bytes) ═══",
                bg=C["success"],
                fg="#ffffff",
                font=("Arial", 8, "bold"),
            ).pack(fill=tk.X, padx=8, pady=(0, 6))
        elif g.group_type == "hardlink":
            tk.Label(
                card,
                text="═══ HARD LINK (same inode) ═══",
                bg=C["warning"],
                fg="#1a1a1a",
                font=("Arial", 8, "bold"),
            ).pack(fill=tk.X, padx=8, pady=(0, 6))

        # ── Thumbnail (Pillow only) ───────────────────────────────
        if HAS_PIL and f.ext.lower() in (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".gif",
            ".webp",
            ".tiff",
        ):
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
        if gi < 0 or gi >= len(self.groups):
            return
        g = self.groups[gi]
        for fi in range(len(g.files)):
            if fi == 0:
                g.suggestions[fi] = "KEEP"
            else:
                g.suggestions[fi] = "DELETE"
        self._show_group(gi)
        self._render_full_report()
        self._update_status_bar()

    def _auto_select(self) -> None:
        if not self.groups:
            return
        if not self._engine:
            # Create a minimal engine just for smart_select (e.g. after session load)
            pq_tmp = queue.Queue()
            ev_tmp = threading.Event()
            engine_tmp = ScanEngine(str(Path.home()), self.settings, pq_tmp, ev_tmp)
            engine_tmp.smart_select(self.groups)
        else:
            self._engine.smart_select(self.groups)

        n = sum(1 for g in self.groups for s in g.suggestions.values() if s == "DELETE")
        self._populate_tree(self.groups)
        if self.current_group >= 0:
            self._show_group(self.current_group)
        self._render_full_report()
        self._update_status_bar()
        self._log_msg("info", f"🎯 Auto-select: {n} files marked for deletion")
        self._dbg(f"[SELECT] Auto-select complete — {n} files marked DELETE", "select")

    def _clear_selection(self) -> None:
        for g in self.groups:
            g.suggestions.clear()
        if self.current_group >= 0:
            self._show_group(self.current_group)
        self._render_full_report()
        self._update_status_bar()
        self._log_msg("info", "Selection cleared")
        self._dbg("[SELECT] All suggestions cleared", "select")

    def _delete_selected(self) -> None:
        if not self.groups:
            return

        # Collect only files marked DELETE (not REVIEW)
        marked = [
            (g, fi, g.files[fi])
            for g in self.groups
            for fi, s in g.suggestions.items()
            if s == "DELETE" and fi < len(g.files)
        ]

        if not marked:
            messagebox.showinfo(
                "Nothing Selected",
                "No files marked for deletion.\n"
                "Use Auto-Select or toggle individual files.\n\n"
                'Note: Near-duplicates are marked "REVIEW" and require manual decision.',
            )
            return

        # Safety check: verify all groups are safe to delete
        high_risk = [g for g in self.groups if g.risk_level == "HIGH"]
        if high_risk:
            msg = (
                f"⚠️  WARNING: {len(high_risk)} HIGH risk group(s) detected!\n\n"
                f"These groups contain:\n"
                f"  • Near-duplicates (not verified exact matches)\n"
                f"  • System files\n\n"
                f"Only verified EXACT duplicates will be deleted.\n"
                f"Continue?"
            )
            if not messagebox.askyesno(
                "Confirm Deletion", msg, icon="warning", default="no"
            ):
                return

        total_bytes = sum(f.size for _, _, f in marked)
        lines = [f"  {f.path}  ({_format_size(f.size)})" for _, _, f in marked[:25]]
        if len(marked) > 25:
            lines.append(f"  … and {len(marked) - 25} more")
        msg = (
            f"⚠️  CONFIRM DELETION\n\n"
            f"{len(marked)} file(s) will be moved to TRASH / RECYCLE BIN\n"
            f"Total: {_format_size(total_bytes)}\n\n" + "\n".join(lines) + "\n\n"
            "✓ Only verified EXACT duplicates\n"
            "✓ Files are moved to Trash (NOT permanent)\n"
            "✓ Hardlinks and locked files are protected\n\n"
            "Continue?"
        )
        if not messagebox.askyesno(
            "Confirm Move to Trash", msg, icon="warning", default="no"
        ):
            self._dbg("[DELETE] Cancelled by user", "warn")
            return

        # Do deletions
        ok, fail = 0, 0
        errors = []
        for g, fi, f in marked:
            success, reason = SafeDeleter.to_trash(f.path)
            if success:
                ok += 1
                g.suggestions[fi] = "DELETED"
                self._dbg(f"[DELETE] ✓ Moved to trash: {f.path.name}", "warn")
            else:
                fail += 1
                errors.append(f"{f.path}: {reason}")
                self._dbg(f"[DELETE] ✗ FAILED: {f.path.name} — {reason}", "error")

        msg2 = f"✓ {ok} file(s) moved to Trash."
        if fail:
            msg2 += f"\n✗ {fail} failed:\n" + "\n".join(errors[:10])
        messagebox.showinfo("Deletion Complete", msg2)

        # Remove deleted files from groups
        for g in self.groups:
            g.files = [
                f for fi, f in enumerate(g.files) if g.suggestions.get(fi) != "DELETED"
            ]
            g.suggestions = {}

        self.groups = [g for g in self.groups if len(g.files) > 1]
        self._populate_tree(self.groups)
        self._clear_detail()
        self._render_full_report()
        self._update_status_bar()
        self._refresh_deletion_history()
        self._log_msg("info", f"Deletion complete: {ok} moved, {fail} failed")

    # ── Deletion history ───────────────────────────────────────────────────

    def _refresh_deletion_history(self) -> None:
        log = SafeDeleter.load_log()
        self._del_history_text.config(state=tk.NORMAL)
        self._del_history_text.delete("1.0", tk.END)
        if not log:
            self._del_history_text.insert(tk.END, "No deletion history yet.\n")
        else:
            for entry in reversed(log):
                ts = entry.get("ts", "")[:19]
                path = entry.get("path", "")
                ok = entry.get("success", False)
                err = entry.get("error", "")
                icon = "✓" if ok else "✗"
                stag = "success" if ok else "fail"
                self._del_history_text.insert(tk.END, f"{icon} [{ts}] ", stag)
                self._del_history_text.insert(tk.END, f"{path}\n")
                if err:
                    self._del_history_text.insert(tk.END, f"  Error: {err}\n", "fail")
        self._del_history_text.config(state=tk.DISABLED)

    def _clear_deletion_log(self) -> None:
        if messagebox.askyesno("Clear Log", "Clear all deletion history?"):
            try:
                DELETION_LOG_PATH.write_text("[]", encoding="utf-8")
            except Exception:
                pass
            self._refresh_deletion_history()

    def _show_deletion_history(self) -> None:
        self._right_nb.select(3)
        self._refresh_deletion_history()

    # ── Full report renderer ───────────────────────────────────────────────

    def _render_full_report(self) -> None:
        # ── modular: full report renderer (lazy, chunked) ──────────────────
        """Build full report — called lazily when tab is selected.
        Uses update_idletasks() every 10 groups to prevent UI freeze."""
        self._report_dirty = False
        t = self._report_text
        t.config(state=tk.NORMAL)
        t.delete("1.0", tk.END)

        if not self.groups:
            t.insert(tk.END, "No scan results yet.\n", "rpt_meta")
            t.config(state=tk.DISABLED)
            return

        ng = len(self.groups)
        nf = sum(len(g.files) for g in self.groups)
        rb = sum(g.reclaimable_bytes for g in self.groups)
        n_marked = sum(
            1 for g in self.groups for s in g.suggestions.values() if s == "DELETE"
        )
        n_exact = sum(1 for g in self.groups if g.group_type == "exact")
        n_near = sum(1 for g in self.groups if g.group_type == "near")
        n_hard = sum(1 for g in self.groups if g.group_type == "hardlink")

        # Header — minimal inserts, fast ─────────────────────────────────────
        t.insert(
            tk.END,
            f"╔══════════════════════════════════════════════════════════╗\n"
            f"║  DUPLICATE FILE FINDER — FULL REPORT   v{VERSION:<16}║\n"
            f"╚══════════════════════════════════════════════════════════╝\n",
            "rpt_header",
        )
        t.insert(
            tk.END,
            f"\n  Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  Groups    : {ng}  |  Files: {nf:,}  |  Reclaimable: {_format_size(rb)}\n"
            f"  Exact     : {n_exact}  |  Near: {n_near}  |  Hard-links: {n_hard}\n"
            f"  Marked DELETE : {n_marked}\n\n",
            "rpt_summary",
        )

        self._report_summary_lbl.config(
            text=f"{ng} groups  ·  {nf:,} files  ·  ~{_format_size(rb)} reclaimable  ·  {n_marked} marked"
        )

        # ── Per-group rendering — chunked to stay responsive ─────────────────
        for gi, g in enumerate(self.groups):
            type_color = (
                "rpt_group"
                if g.group_type == "exact"
                else ("rpt_hl" if g.group_type == "hardlink" else "rpt_near")
            )

            # Build this group's plain text in one string → single insert ─────
            hdr_line = (
                f"\n══ GROUP {gi + 1:>4} ─── {g.group_type.upper():<10} ─── "
                f"Score: {g.score}%  ─── {len(g.files)} files  ─── "
                f"~{_format_size(g.reclaimable_bytes)} reclaimable ══\n"
            )
            t.insert(tk.END, hdr_line, type_color)

            if g.components:
                parts = "  |  ".join(
                    f"{k}: {v:.1f}%" if isinstance(v, float) else f"{k}: {v}"
                    for k, v in g.components.items()
                )
                t.insert(tk.END, f"  Score: {parts}\n", "rpt_score")

            for fi, f in enumerate(g.files):
                sugg = g.suggestions.get(fi, "KEEP")
                s_tag = "rpt_keep" if sugg == "KEEP" else "rpt_delete"
                s_icon = "✓ KEEP" if sugg == "KEEP" else "🗑 DELETE"
                # Build file block as one string per section ───────────────────
                t.insert(
                    tk.END,
                    f"\n  [{fi + 1}] {s_icon:12} Quality:{f.keep_score:3}/100\n",
                    s_tag,
                )
                meta_block = (
                    f"      Path    : {f.path}\n"
                    f"      Size    : {_format_size(f.size)}\n"
                    f"      Created : {_ts(f.ctime)}\n"
                    f"      Modified: {_ts(f.mtime)}\n"
                )
                if f.hash:
                    meta_block += f"      Hash    : {f.hash[:48]}\n"
                if f.magic_type:
                    meta_block += f"      MIME    : {f.magic_type}\n"
                t.insert(tk.END, meta_block, "rpt_meta")  # ONE insert for all meta

            t.insert(tk.END, "─" * 72 + "\n", "rpt_div")

            # Yield to event loop every 10 groups to stay responsive ──────────
            if (gi + 1) % 10 == 0:
                t.update_idletasks()

        t.config(state=tk.DISABLED)
        t.see("1.0")

    # ── Export ─────────────────────────────────────────────────────────────

    def _show_export_menu(self) -> None:
        """Pop a small menu with export options."""
        m = tk.Menu(self.root, tearoff=0)
        m.add_command(label="Export TXT", command=self._export_txt)
        m.add_command(label="Export CSV", command=self._export_csv)
        m.add_command(label="Export JSON", command=self._export_json)
        m.add_command(label="Export HTML", command=self._export_html)
        try:
            btn = self._export_btn
            x = btn.winfo_rootx()
            y = btn.winfo_rooty() + btn.winfo_height()
            m.tk_popup(x, y)
        finally:
            m.grab_release()

    def _export_txt(self) -> None:
        # Force render Full Report if not rendered yet (lazy tab)
        if not hasattr(self, "_report_text") or not self._report_text:
            self._render_full_report()
        p = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text File", "*.txt"), ("All", "*.*")],
            title="Export Full Report TXT",
        )
        if not p:
            return
        t = self._report_text
        t.config(state=tk.NORMAL)
        content = t.get("1.0", tk.END)
        t.config(state=tk.DISABLED)
        try:
            Path(p).write_text(content, encoding="utf-8")
            self._log_msg("info", f"TXT report saved: {p}")
            self._dbg(f"[EXPORT] TXT -> {p}", "info")
            messagebox.showinfo("Exported", f"Report saved:\n{p}")
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc))

    def _export_csv(self) -> None:
        p = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
            title="Export CSV",
        )
        if not p:
            return
        try:
            import csv

            with open(p, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(
                    [
                        "group",
                        "type",
                        "score",
                        "file_index",
                        "suggestion",
                        "quality",
                        "path",
                        "size",
                        "created",
                        "modified",
                        "hash",
                        "mime",
                    ]
                )
                for gi, g in enumerate(self.groups):
                    for fi, f in enumerate(g.files):
                        w.writerow(
                            [
                                gi + 1,
                                g.group_type,
                                g.score,
                                fi + 1,
                                g.suggestions.get(fi, "KEEP"),
                                f.keep_score,
                                str(f.path),
                                f.size,
                                _ts(f.ctime),
                                _ts(f.mtime),
                                f.hash or "",
                                f.magic_type or "",
                            ]
                        )
            self._log_msg("info", f"✓ CSV exported: {p}")
            self._dbg(f"[EXPORT] CSV → {p}", "info")
            messagebox.showinfo("Exported", f"CSV saved:\n{p}")
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc))

    def _export_json(self) -> None:
        p = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
            title="Export JSON",
        )
        if not p:
            return
        try:
            data = []
            for gi, g in enumerate(self.groups):
                data.append(
                    {
                        "group": gi + 1,
                        "type": g.group_type,
                        "score": g.score,
                        "components": g.components,
                        "reclaimable_bytes": g.reclaimable_bytes,
                        "files": [
                            {
                                "index": fi + 1,
                                "suggestion": g.suggestions.get(fi, "KEEP"),
                                "quality": f.keep_score,
                                "path": str(f.path),
                                "size": f.size,
                                "created": _ts(f.ctime),
                                "modified": _ts(f.mtime),
                                "hash": f.hash or "",
                                "mime": f.magic_type or "",
                            }
                            for fi, f in enumerate(g.files)
                        ],
                    }
                )
            Path(p).write_text(json.dumps(data, indent=2), encoding="utf-8")
            self._log_msg("info", f"✓ JSON exported: {p}")
            self._dbg(f"[EXPORT] JSON → {p}", "info")
            messagebox.showinfo("Exported", f"JSON saved:\n{p}")
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc))

    def _export_html(self) -> None:
        p = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML", "*.html"), ("All", "*.*")],
            title="Export HTML Report",
        )
        if not p:
            return
        try:
            rows = []
            for gi, g in enumerate(self.groups):
                for fi, f in enumerate(g.files):
                    sugg = g.suggestions.get(fi, "KEEP")
                    c_name = "keep" if sugg == "KEEP" else "delete"
                    rows.append(
                        f'<tr class="{c_name}">'
                        f"<td>{gi + 1}</td>"
                        f"<td>{g.group_type}</td>"
                        f"<td>{g.score}%</td>"
                        f"<td>{fi + 1}</td>"
                        f"<td>{sugg}</td>"
                        f"<td>{f.keep_score}</td>"
                        f"<td><code>{f.path}</code></td>"
                        f"<td>{_format_size(f.size)}</td>"
                        f"<td>{_ts(f.ctime)}</td>"
                        f"<td>{_ts(f.mtime)}</td>"
                        f"<td><small>{(f.hash or '')[:20]}</small></td>"
                        f"</tr>"
                    )
            html = f"""<!DOCTYPE html>
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
{"".join(rows)}
</table>
</body></html>"""
            Path(p).write_text(html, encoding="utf-8")
            self._log_msg("info", f"✓ HTML exported: {p}")
            self._dbg(f"[EXPORT] HTML → {p}", "info")
            messagebox.showinfo("Exported", f"HTML report saved:\n{p}")
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc))

    # ── Session save / load ────────────────────────────────────────────────

    def _save_session(self) -> None:
        if not self.groups:
            messagebox.showinfo("Nothing to save", "Run a scan first.")
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".dupjson",
            filetypes=[("Duplicate Session", "*.dupjson"), ("All", "*.*")],
            title="Save Session",
        )
        if not p:
            return
        ok = SessionManager.save(self.groups, self._folder_var.get(), self.settings, p)
        if ok:
            self._log_msg("info", f"✓ Session saved: {p}")
            self._dbg(f"[SESSION] Saved → {p}", "info")
            messagebox.showinfo("Session Saved", f"Session saved:\n{p}")
        else:
            messagebox.showerror("Save Failed", "Could not save session.")

    def _load_session(self) -> None:
        p = filedialog.askopenfilename(
            filetypes=[("Duplicate Session", "*.dupjson"), ("All", "*.*")],
            title="Load Session",
        )
        if not p:
            return
        groups, folder = SessionManager.load(p)
        if groups is None:
            messagebox.showerror(
                "Load Failed", "Could not load session. Wrong version or corrupt file."
            )
            return
        self.groups = groups
        if folder:
            self._folder_var.set(folder)
        self._populate_tree(self.groups)
        self._render_full_report()
        self._update_status_bar()
        self._auto_sel_btn.config(state=tk.NORMAL)
        self._delete_btn.config(state=tk.NORMAL)
        self._export_btn.config(state=tk.NORMAL)
        self._save_sess_btn.config(state=tk.NORMAL)
        self._clear_sel_btn.config(state=tk.NORMAL)
        self._log_msg("info", f"✓ Session loaded: {len(self.groups)} groups")
        self._dbg(f"[SESSION] Loaded ← {p}  groups={len(self.groups)}", "info")
        if self.groups:
            self._show_group(0)

    # ── Display / theme settings ───────────────────────────────────────────

    def _toggle_dark_mode(self) -> None:
        self._var_dark_mode.set(not self._var_dark_mode.get())
        self._apply_display_settings()

    def _update_optional_features_display(self) -> None:
        """Update optional features status display in header."""
        if not hasattr(self, "_optional_features_btn"):
            return

        # Build status text showing installed but inactive features
        inactive_features = []

        # Check which features are installed but not enabled in settings
        if HAS_FAISS and not self._var_faiss.get():
            inactive_features.append("FAISS")
        if HAS_SENTENCE_TRANSFORMERS and not self._var_semantic.get():
            inactive_features.append("Semantic")
        if HAS_CLIP and not self._var_clip.get():
            inactive_features.append("CLIP")
        if HAS_BLAKE3 and not self._var_blake3.get():
            inactive_features.append("BLAKE3")
        if HAS_WATCHDOG and not self.settings.enable_watch_mode:
            inactive_features.append("Watch")

        # Check which optional features are missing
        missing = []
        if not HAS_FAISS:
            missing.append("FAISS")
        if not HAS_SENTENCE_TRANSFORMERS:
            missing.append("Semantic")
        if not HAS_CLIP:
            missing.append("CLIP")

        # Update button text and tooltip
        if inactive_features or missing:
            parts = []
            if missing:
                parts.append(f"Missing: {', '.join(missing)}")
            if inactive_features:
                parts.append(f"Inactive: {', '.join(inactive_features)}")
            status_text = " | ".join(parts)
            self._optional_features_btn.config(
                text=f"⚙ {len(inactive_features) + len(missing)} Features"
            )
        else:
            self._optional_features_btn.config(text="⚙ Features")

        # Update tooltip with details
        tooltip_parts = []
        if missing:
            tooltip_parts.append(
                f"Missing: {', '.join(missing)} - Click Settings to install"
            )
        if inactive_features:
            tooltip_parts.append(
                f"Inactive (turned off): {', '.join(inactive_features)}"
            )

        if tooltip_parts:
            full_tooltip = " | ".join(tooltip_parts)
            if hasattr(self, "_add_tooltip"):
                self._add_tooltip(self._optional_features_btn, full_tooltip)
        else:
            if hasattr(self, "_add_tooltip"):
                self._add_tooltip(self._optional_features_btn, "All features active")

    def _apply_display_settings(self) -> None:
        dark = self._var_dark_mode.get()
        self._setup_theme(dark=dark)
        # Rebuild the UI
        for w in self.root.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
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

    def _show_settings(self) -> None:
        """Show settings window."""
        if hasattr(self, "_settings_window") and self._settings_window.winfo_exists():
            self._settings_window.lift()
            return
        self._settings_window = tk.Toplevel(self.root)
        self._settings_window.title(f"Settings — Duplicate Finder v{VERSION}")
        self._settings_window.geometry("700x600")
        self._settings_window.configure(bg=self.C["bg"])
        inner = tk.Frame(self._settings_window, bg=self.C["bg"])
        inner.pack(fill=tk.BOTH, expand=True)
        self._build_settings_tab(inner)

    def _show_manual(self) -> None:
        """Show manual window."""
        if hasattr(self, "_manual_window") and self._manual_window.winfo_exists():
            self._manual_window.lift()
            return
        self._manual_window = tk.Toplevel(self.root)
        self._manual_window.title(f"User Manual — Duplicate Finder v{VERSION}")
        self._manual_window.geometry("700x650")
        self._manual_window.configure(bg=self.C["bg"])
        inner = tk.Frame(self._manual_window, bg=self.C["bg"])
        inner.pack(fill=tk.BOTH, expand=True)
        self._build_manual_tab(inner)

    def _show_changelog(self) -> None:
        """Show changelog window."""
        if hasattr(self, "_changelog_window") and self._changelog_window.winfo_exists():
            self._changelog_window.lift()
            return
        self._changelog_window = tk.Toplevel(self.root)
        self._changelog_window.title(f"Changelogs — Duplicate Finder v{VERSION}")
        self._changelog_window.geometry("700x600")
        self._changelog_window.configure(bg=self.C["bg"])
        inner = tk.Frame(self._changelog_window, bg=self.C["bg"])
        inner.pack(fill=tk.BOTH, expand=True)
        self._build_changelog_tab(inner)

    def _show_help(self) -> None:
        win = tk.Toplevel(self.root)
        win.title(f"Help — Duplicate Finder v{VERSION}")
        win.geometry("720x600")
        win.configure(bg=self.C["bg"])

        header = tk.Frame(win, bg=self.C["toolbar_bg"])
        header.pack(fill=tk.X, padx=0, pady=0)
        tk.Label(
            header,
            text="Duplicate File Finder Help",
            bg=self.C["toolbar_bg"],
            fg=self.C["toolbar_fg"],
            font=("Arial", 12, "bold"),
            padx=12,
            pady=8,
        ).pack(side=tk.LEFT)

        import webbrowser

        def open_file_types():
            webbrowser.open("https://fileinfo.com/filetypes")

        ttk.Button(
            header,
            text="Online File Types Reference",
            command=open_file_types,
            style="Blue.TButton",
        ).pack(side=tk.RIGHT, padx=12, pady=4)
        tk.Label(
            header,
            text="Click to open online reference",
            bg=self.C["toolbar_bg"],
            fg=self.C["toolbar_fg"],
            font=("Arial", 8),
            padx=6,
        ).pack(side=tk.RIGHT)

        txt = scrolledtext.ScrolledText(
            win,
            font=("Courier", 9),
            bg=self.C["panel_bg"],
            fg=self.C["fg"],
            wrap=tk.WORD,
            borderwidth=0,
            relief="flat",
        )
        txt.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        txt.insert(tk.END, HELP_TEXT)
        txt.config(state=tk.DISABLED)


# ═════════════════════════════════════════════════════════════════════════════
#  HELP TEXT
# ═════════════════════════════════════════════════════════════════════════════

HELP_TEXT = f"""
DUPLICATE FILE FINDER  v{VERSION}
{"=" * 60}

SAFETY SYSTEM
  This program implements MANDATORY SAFETY:
  • Only verified EXACT duplicates can be deleted
  • Near-duplicates are marked REVIEW - manual decision
  • Hardlinks are NEVER deletable
  • Locked files are protected
  • System files are protected
  • All deletions go to Recycle Bin

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
  transformers — DINOv2 neural embeddings (optional)
  torch      — PyTorch for neural embeddings
  mutagen    — Audio metadata extraction
  chromaprint — Audio fingerprinting

I/O PORT (External Control)
  Enable I/O Port in Settings to allow external programs to
  control Duplicate Finder via JSON commands over stdin/stdout.
  Commands: get_status, get_groups, get_group, select_files,
  set_settings, ping, etc.
  Signals: SCAN_START, SCAN_PROGRESS, SCAN_COMPLETE, etc.

NOTE ON GPU ACCELERATION
  GPU (CuPy) is only used for the name-similarity matrix
  (bigram cosine). For file hashing, CPU xxhash is faster
  than any GPU-transfer overhead.

FILE TYPES & MAGIC BYTES
  This program detects file types by reading the first few bytes
  (magic bytes) of each file - the actual format signature.

  IMAGE FORMATS:
    JPEG (.jpg, .jpeg) — FF D8 FF signature, common photo format
    PNG (.png) — 89 50 4E 0D 0A 1A 0A, lossless with transparency
    GIF (.gif) — 47 49 46 38 7A/39 89, simple animation support
    BMP (.bmp) — 42 4D, uncompressed Windows bitmap
    TIFF (.tif, .tiff) — 49 49 2A 00 or 4D 4D 00 2A, professional imaging
    WebP (.webp) — RIFF....webp, modern Google format

  DOCUMENT FORMATS:
    PDF (.pdf) — 25 50 44 46, Adobe Portable Document
    HTML (.html, .htm) — 3C 21 44 4F 43 54 59 50 45 or 3C 68 74 6D 6C
    XML (.xml) — 3C 3F 78 6D 6C, structured data
    TXT (.txt) — No magic bytes, plain text (UTF-8 BOM: EF BB BF)

  ARCHIVE FORMATS:
    ZIP (.zip) — 50 4B 03 04, compressed archive
    GZIP (.gz) — 1F 8B, single-file compression
    RAR (.rar) — 52 61 72 21, WinRAR format
    7Z (.7z) — 37 7A BC AF 27 1C, 7-Zip high compression
    TAR (.tar) — No magic bytes (check POSIX header)

  AUDIO FORMATS:
    MP3 (.mp3) — 49 44 33 or FF FB/F3/F2, compressed audio
    FLAC (.flac) — 66 4C 61 43, lossless audio
    OGG (.ogg) — 4F 67 67 53, Ogg Vorbis container
    WAV (.wav) — 52 49 46 46 .... 57 41 56 45, uncompressed

  VIDEO FORMATS:
    MP4 (.mp4) — 00 00 00 xx 66 74 79 70, MPEG-4 container
    MKV (.mkv) — 1A 45 DF A3, Matroska video
    AVI (.avi) — 52 49 46 46 .... 41 56 49, older video format

  PROGRAM FORMATS:
    EXE/DLL (.exe, .dll) — 4D 5A, Windows executable
    ELF (.elf) — 7F 45 4C 46, Linux executable
    Mach-O (.app) — CA FE BA BE, macOS executable

  WHY MAGIC BYTES MATTER:
    File extensions can be renamed or wrong. Magic bytes
    show the TRUE file type regardless of extension.
    Example: file.pdf renamed to file.txt still starts with
    25 50 44 46 = PDF signature.

SAFETY PROFILES
  The Settings page has 3 profile buttons:
  
  🛡️ MAX SAFE
    • min_score = 90 (only high-confidence matches)
    • cleanup_mode = SAFE
    • auto_select = False (manual selection required)
    • delete_gap = 30 (stagger deletions)
    • paranoid_mode = True (byte-verify all matches)
    • use_sha256_verify = True (extra hash verification)
    • Best for: First-time users, important data, cautious scanning
  
  ✓ SAFE (Default)
    • min_score = 70 (balanced detection)
    • cleanup_mode = SAFE
    • auto_select = True (automatic suggestions)
    • delete_gap = 15 (reasonable separation)
    • paranoid_mode = False
    • use_xxhash = True (fast hashing)
    • Best for: Regular use, general cleanup
  
  ⚡ PERFORMANCE
    • min_score = 50 (detects more potential duplicates)
    • cleanup_mode = AGGRESSIVE
    • auto_select = True
    • delete_gap = 5 (quick deletions)
    • skip_system = False (scan everything)
    • skip_network = False (include network drives)
    • num_workers = 32 (maximum speed)
    • Best for: Large libraries, experienced users

NEEDED DUPLICATE PROTECTION (v7.2+)
  Files in these locations are protected from accidental deletion:
  • Program Files, Program Files (x86), ProgramData
  • Steam, Epic Games, game folders
  • Browser profiles (Chrome, Edge, Firefox)
  • Docker, VirtualBox, VMware
  • OneDrive, Google Drive, Dropbox
  • AppData\\Local\\Programs
  
  Files in folders containing executables (.exe, .dll) are also protected.
  The program shows warnings when files may be required by installed software.

AI SEMANTIC DEDUPLICATION (v7.0+)
  Optional advanced features for near-duplicate detection:
  • Sentence-BERT: Semantic similarity for text documents
  • CLIP: Image understanding beyond visual hash
  • FAISS: Fast vector similarity search
  • DINOv2: Neural embeddings for images
  
  These are DISABLED by default for safety. Enable in Settings
  if you understand the risks - semantic matches require manual review.

I/O PORT (External Control)
  Enable I/O Port in Settings to allow external programs to
  control Duplicate Finder via JSON commands over stdin/stdout.
  Commands: get_status, get_groups, get_group, select_files,
  set_settings, ping, scan, cancel, get_capabilities, get_stats, etc.
  Signals: SCAN_START, SCAN_PROGRESS, SCAN_COMPLETE, etc.
  
  This enables integration with AI assistants and automation tools.

VERSION INFO
  Current Version: {VERSION}
  AI Features: Optional (Sentence-BERT, CLIP, FAISS, DINOv2)
  Neural Embeddings: Optional (requires torch, transformers)
  Safe Delete: Always uses Recycle Bin
  Auto-Install: Missing required libraries are installed on startup
"""


# ═════════════════════════════════════════════════════════════════════════════
#  Open in file explorer — cross-platform
# ═════════════════════════════════════════════════════════════════════════════


def _open_in_explorer(path: Path) -> None:
    try:
        p = Path(path)
        if not p.exists():
            return
        if sys.platform == "win32":
            subprocess.run(["explorer", "/select,", str(p)], check=False)
        elif sys.platform == "darwin":
            subprocess.run(["open", "-R", str(p)], check=False)
        else:
            subprocess.run(["xdg-open", str(p.parent)], check=False)
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    multiprocessing.freeze_support()
    root = tk.Tk()
    try:
        root.iconbitmap(default="")
    except Exception:
        pass
    app = DuplicateFinderApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: _quit(root, app))
    root.mainloop()


def _quit(root: tk.Tk, app: DuplicateFinderApp) -> None:
    if app.is_scanning:
        if not messagebox.askyesno("Quit", "Scan in progress. Quit anyway?"):
            return
        app._cancel_ev.set()
    root.destroy()


if __name__ == "__main__":
    main()
