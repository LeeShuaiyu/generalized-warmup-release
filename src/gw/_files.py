"""Install verified files on local and removable filesystems."""
import errno
import os
from pathlib import Path
import shutil


_COPY_ERRORS = {errno.EPERM, errno.EXDEV, errno.ENOSYS, errno.ENOTSUP,
                errno.EOPNOTSUPP}


def _remove_created(destination, created):
    try:
        current = destination.lstat()
    except FileNotFoundError:
        return
    if os.path.samestat(created, current):
        destination.unlink()


def install_verified_file(source, destination, *, reserve_bytes=0):
    """Create a destination exclusively, using a link or a streamed copy.

    Callers verify the completed source file before installation. Existing
    destinations raise FileExistsError. Failed copies remove their own output.
    """
    source, destination = Path(source), Path(destination)
    try:
        os.link(source, destination)
        return
    except OSError as error:
        if error.errno not in _COPY_ERRORS:
            raise
    size = source.stat().st_size
    if shutil.disk_usage(destination.parent).free < size + reserve_bytes:
        raise RuntimeError('Insufficient space for file installation')
    with source.open('rb') as stream:
        output = destination.open('xb')
        created = os.fstat(output.fileno())
        try:
            with output:
                try:
                    shutil.copyfileobj(stream, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                    if output.tell() != size:
                        raise OSError('Installed file size differs from source')
                except BaseException:
                    try:
                        output.flush()
                    except OSError:
                        pass
                    # exFAT assigns its file identity during allocation.
                    created = os.fstat(output.fileno())
                    try:
                        _remove_created(destination, created)
                    except PermissionError:
                        # Windows releases an open file during context exit.
                        pass
                    raise
        except BaseException:
            _remove_created(destination, created)
            raise
