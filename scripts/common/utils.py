import contextlib
import datetime
import os
import subprocess
import time


def check_cmd(cmd):
    job = subprocess.run(cmd, shell=True, capture_output=True)
    if job.returncode != 0:
        print(job)
    job.check_returncode()


def cmd_with_tmpdir(cmd, tmpdir):
    os.makedirs(tmpdir, exist_ok=True)
    check_cmd(cmd)
    with contextlib.suppress(FileNotFoundError):
        os.rmdir(tmpdir)


class CopyStream(object):
    def __init__(self, stream, filename):
        self.stream = stream
        self.log = open(filename, "a")

    def write(self, msg):
        msg = str(msg)
        self.stream.write(msg)
        if "\n" in msg:
            self.stream.flush()
            token = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "> "
            msg = msg.replace("\n", "\n" + token)
        try:
            self.log.write(msg)
            self.log.flush()
        except Exception as e:
            print(f"Log write failed: {e}")

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


class LockFile(object):
    def __init__(self, filename, timeout=60*30):
        self.filename = filename
        self.f = None
        self.timeout = timeout

    def exists(self):
        return os.path.exists(self.filename)

    def stale(self):
        return self.exists() and time.time() - os.path.getmtime(self.filename) > self.timeout

    def check(self, strict=False):
        if self.exists():
            if self.stale():
                if strict:
                    print("Lockfile is stale, probably can delete")
                else:
                    print("Lockfile is stale, deleting and continuing")
                    self.delete()
                    return False
            return True
        return False

    def touch(self):
        with open(self.filename, "w") as f:
            f.write(f"{os.getpid()} {time.time()}")

    def delete(self):
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.filename)

    def __del__(self):
        self.delete()


def memory_usage():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 ** 3
    except ImportError:
        return 0
