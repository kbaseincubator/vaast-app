"""
Concurrent utilities for synchronization between processes, including file-based locking.
"""

import fcntl
from pathlib import Path
from typing import TextIO


class OperationLock:
    """
    Lock operations using a file lock to synchronize between processes acting within the same container
    """

    def __init__(self, name: str | None = None, wdir: Path | None = None):
        """
        Create Lock

        :param name: Name of lock file prefix
        :param wdir: Directory in which to create file (default is current working directory)
        """
        if wdir is None:
            wdir = Path(".")
        self._lock_file: Path = wdir.joinpath(f"{f'{name}-' if name else ''}lockfile.LOCK")
        self._lock_fd: TextIO | None = None

    def __enter__(self) -> TextIO:
        self._lock_fd = open(self._lock_file, "w+")
        fcntl.lockf(self._lock_fd, fcntl.LOCK_EX)
        return self._lock_fd

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock_fd.close()
        self._lock_fd = None
