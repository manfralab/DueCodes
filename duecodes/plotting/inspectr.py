import sys
import time
import subprocess
import hashlib
import argparse
from pathlib import Path, PurePath
from pyqtgraph.Qt import QtCore, QtGui

import logging
from plottr import log as plottrlog
from plottr.apps import inspectr


def get_qc_database():
    import qcodes
    qc_config = qcodes.config.current_config
    return qc_config['core']['db_location']


def filepath_hash(dbPath):
    md5_hash = hashlib.md5(dbPath.encode('utf-8'))
    return md5_hash.hexdigest()


def get_lockfile_path(dbPath):
    path_hash = filepath_hash(str(dbPath))
    # lock_file_path = Path(dbPath).parent / f'plottr_{path_hash}.lock'
    lock_file_path = Path(QtCore.QDir.tempPath()).parent / f'plottr_{path_hash}.lock'
    return lock_file_path.expanduser().absolute()


def is_locked(dbPath):
    lockfile_path = get_lockfile_path(dbPath)
    lockfile = QtCore.QLockFile(str(lockfile_path))
    if lockfile.tryLock(100):
        return False

    return True


def run_with_lockfile(dbPath: str):
    from PyQt5 import QtCore, QtWidgets

    dbPath = Path(dbPath).expanduser().absolute()
    lockfile_path = get_lockfile_path(dbPath)
    print(lockfile_path)
    lockfile = QtCore.QLockFile(str(lockfile_path))

    if lockfile.tryLock(100):
        app = QtGui.QApplication([])

        # what is the right way to create a logger for this?
        # log_win = plottrlog.setupLogging(level=logging.DEBUG)
        plottrlog.enableStreamHandler(False) # no log handler at this point

        win = inspectr.inspectr(dbPath=dbPath)
        win.show()

        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            app.exec()
        return

    raise RuntimeWarning("plottr instance already open for this database!") # inspectr already running


def start_inspectr(dbPath=None, timeout=10.0):

    if not dbPath:
        dbPath = get_qc_database()

    python_path = str(Path(sys.executable))

    import duecodes
    duecodes_path_parts = Path(duecodes.__file__).parts[:-1]
    duecodes_path_full = str(Path(*duecodes_path_parts, "plotting\inspectr.py"))

    if is_locked(dbPath):
        print(f"plottr instance with qcodes database at {dbPath} is already running.")
        return

    print(f"starting plottr with qcodes database at {dbPath} ...")
    out = subprocess.Popen(
        [python_path, duecodes_path_full, f'--dbpath={dbPath}'],
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    time.sleep(0.1)

    start = time.time()
    while True:
        if is_locked(dbPath):
            print("plottr successfully started.")
            return
        if time.time() - start > timeout:
            print("plottr failed to start!")
            return
        time.sleep(0.1)

    print("plottr failed to start!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inspectr -- sifting through qcodes data.')
    parser.add_argument(
        '--dbpath',
        help='path to qcodes .db file',
        default=get_qc_database(),
    )

    args = parser.parse_args()

    print(f"starting plottr with qcodes database at {args.dbpath} ...")
    run_with_lockfile(args.dbpath)
