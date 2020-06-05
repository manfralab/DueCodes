import sys
import subprocess
import hashlib
import argparse
from pathlib import Path, PurePath
from pyqtgraph.Qt import QtCore, QtGui

from plottr import log as plottrlog
from plottr.apps import inspectr


def get_qc_database():
    import qcodes
    qc_config = qcodes.config.current_config
    return qc_config['core']['db_location']


def filepath_hash(filepath):
    md5_hash = hashlib.md5(filepath.encode('utf-8'))
    return md5_hash.hexdigest()


def run_with_lockfile(dbPath: str):
    from PyQt5 import QtCore, QtWidgets

    dbPath = Path(dbPath).expanduser().absolute()
    path_hash = filepath_hash(str(dbPath))
    lock_file_name = f'plottr_{path_hash}.lock'
    lockfile = QtCore.QLockFile(QtCore.QDir.tempPath() + lock_file_name)

    if lockfile.tryLock(100):
        app = QtGui.QApplication([])
        plottrlog.enableStreamHandler(True)

        win = inspectr.inspectr(dbPath=dbPath)
        win.show()

        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            app.exec()
        return

    raise RuntimeWarning("plottr instance already open for this database!") # inspectr already running


def start_inspectr(db_path=None):

    if not db_path:
        db_path = get_qc_database()

    python_path = str(Path(sys.executable))

    import duecodes
    duecodes_path_parts = Path(duecodes.__file__).parts[:-1]
    duecodes_path_full = str(Path(*duecodes_path_parts, "plotting\inspectr.py"))

    print(f"starting plottr with qcodes database at {db_path} ...")
    out = subprocess.Popen(
        [python_path, duecodes_path_full, f'--dbpath={db_path}'],
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


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
