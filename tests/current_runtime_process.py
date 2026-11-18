"""Run one focused suite, retaining both streams and nested-process cleanup."""
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time


def process_row(pid):
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(state=fields[0], ppid=int(fields[1]), process_group=int(fields[2]),
                session=int(fields[3]), start_ticks=int(fields[19]))


def process_table():
    result = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            result[int(path.name)] = process_row(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return result


class ProcessTree:
    """Follow only the freshly spawned runner and its exact PID incarnations."""
    def __init__(self, pid):
        self.root = pid
        self.known = {}
        self.root_initialized = False
        self.cleanup_errors = []
        self.terminated_groups = set()
        self.track()

    def track(self):
        table = process_table()
        root = table.get(self.root)
        if root is not None and not self.root_initialized:
            self.known[self.root] = root
        self.root_initialized = True
        alive = {pid for pid, row in self.known.items()
                 if pid in table and table[pid]['start_ticks'] == row['start_ticks']}
        while True:
            # A numeric historical SID grants no authority. There must be a
            # currently alive, exact-starttime owned member of that session.
            sessions = {table[pid]['session'] for pid in alive
                        if table[pid]['session'] != os.getsid(0)}
            discovered = {pid for pid, row in table.items()
                          if row['ppid'] in alive or row['session'] in sessions}
            discovered -= alive
            if not discovered:
                break
            added = False
            for pid in discovered:
                row = table[pid]
                if pid in self.known and self.known[pid]['start_ticks'] != row['start_ticks']:
                    continue
                self.known[pid] = row
                alive.add(pid)
                added = True
            if not added:
                break
        return {pid: table[pid] for pid in alive if pid in table}

    def running(self):
        return {pid: row for pid, row in self.track().items() if row['state'] not in ('Z', 'X')}

    def signal_exact(self, pid, row, wanted):
        if pid == os.getpid() or row['session'] == os.getsid(0):
            self.cleanup_errors.append(dict(operation='refused-own-session-termination', pid=pid))
            return False
        descriptor = None
        try:
            descriptor = os.pidfd_open(pid)
            # Bind after pidfd_open as well: reuse before opening must fail the
            # start-time check; reuse after it cannot redirect this pidfd signal.
            now = process_row(pid)
            if now['start_ticks'] != row['start_ticks']:
                return False
            signal.pidfd_send_signal(descriptor, wanted)
            return True
        except (FileNotFoundError, ProcessLookupError):
            return False
        except OSError as error:
            self.cleanup_errors.append(dict(operation=signal.Signals(wanted).name, pid=pid, error=repr(error)))
            return False
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def stop(self):
        stopped = set()
        # Freeze exact owned incarnations, including newly discovered nested
        # sessions, before killing them. No numeric process-group signal is used.
        for _ in range(4):
            rows = self.running()
            new = [(pid, row) for pid, row in rows.items()
                   if (pid, row['start_ticks']) not in stopped]
            for pid, row in new:
                if self.signal_exact(pid, row, signal.SIGSTOP):
                    stopped.add((pid, row['start_ticks']))
            if not new:
                break
        for pid, row in self.running().items():
            if self.signal_exact(pid, row, signal.SIGKILL):
                self.terminated_groups.add(row['process_group'])


def execute(command, directory, timeout_seconds, *, cwd, mirror_stderr=True):
    directory = Path(directory)
    started = time.monotonic()
    receipt = dict(argv=[str(value) for value in command], timeout_seconds=timeout_seconds,
                   started=False, stdout_EOF=False, stderr_EOF=False, forced_cleanup=False,
                   stderr_mirrored=mirror_stderr)
    child = tree = None
    selector = selectors.DefaultSelector()
    hashes = {name: hashlib.sha256() for name in ('stdout', 'stderr')}
    sizes = {name: 0 for name in hashes}

    def stop():
        receipt['forced_cleanup'] = True
        tree.stop()

    try:
        if type(mirror_stderr) is not bool:
            raise TypeError('mirror_stderr must be an explicit bool')
        if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
            raise RuntimeError('focused native supervision requires Linux pidfd support')
        with (directory / 'runner.stdout').open('xb') as stdout, (directory / 'runner.stderr').open('xb') as stderr:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, cwd=cwd, start_new_session=True,
                env={'PATH': os.defpath, 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8'})
            receipt.update(started=True, pid=child.pid)
            tree = ProcessTree(child.pid)
            for name, pipe, destination, console in (('stdout', child.stdout, stdout, sys.stdout),
                                                     ('stderr', child.stderr, stderr, sys.stderr if mirror_stderr else None)):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, (name, destination, console))
            deadline = started + timeout_seconds
            while selector.get_map():
                tree.track()
                if time.monotonic() >= deadline:
                    if receipt.get('timed_out'):
                        raise TimeoutError('runner streams did not reach EOF after full process-tree termination')
                    receipt['timed_out'] = True
                    stop()
                    deadline = time.monotonic() + 10
                for key, _ in selector.select(timeout=max(0, min(1, deadline - time.monotonic()))):
                    name, destination, console = key.data
                    try:
                        raw = os.read(key.fd, 65536)
                    except BlockingIOError:
                        continue
                    if not raw:
                        receipt[name + '_EOF'] = True
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    destination.write(raw); destination.flush()
                    hashes[name].update(raw); sizes[name] += len(raw)
                    if console is None:
                        continue
                    if hasattr(console, 'buffer'):
                        console.buffer.write(raw); console.buffer.flush()
                    else:
                        console.write(raw.decode('utf-8', errors='replace')); console.flush()
            try:
                child.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                receipt['timed_out'] = True
                stop()
                child.wait(timeout=10)
            receipt['exit_code'] = child.returncode
    except BaseException as error:
        receipt['execution_error'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        if tree is not None and tree.running():
            stop()
        if child is not None:
            if child.poll() is None:
                child.wait(timeout=10)
            receipt['exit_code'] = child.returncode
            for pipe in (child.stdout, child.stderr):
                if pipe is not None:
                    pipe.close()
        if tree is not None:
            cleanup_deadline = time.monotonic() + 1
            remaining = tree.running()
            while remaining and time.monotonic() < cleanup_deadline:
                time.sleep(0.02)
                remaining = tree.running()
            receipt.update(tracked_processes=[dict(pid=pid, **row) for pid, row in sorted(tree.known.items())],
                           termination_method='exact-PID-starttime-and-pidfd-signals',
                           terminated_process_groups=sorted(tree.terminated_groups),
                           cleanup_errors=tree.cleanup_errors,
                           running_descendants_after_cleanup=sorted(remaining))
            if remaining:
                receipt['cleanup_errors'].append(dict(operation='remaining-running-processes', pids=sorted(remaining)))
        selector.close()
        receipt.update(elapsed_seconds=time.monotonic() - started,
                       stdout_bytes=sizes['stdout'], stdout_sha256=hashes['stdout'].hexdigest(),
                       stderr_bytes=sizes['stderr'], stderr_sha256=hashes['stderr'].hexdigest())
        (directory / 'runner.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    return receipt
