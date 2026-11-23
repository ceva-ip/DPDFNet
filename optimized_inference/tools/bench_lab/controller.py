#!/usr/bin/env python3
"""Reversible Linux benchmark laboratory. Root only controls the environment.

State and rollback code are root-owned in /run. Workers always drop privileges.
No application build or arbitrary command is ever executed with root privileges.
"""
import argparse
import ctypes
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid

RUN = Path('/run/dpdfnet-bench-lab')
CG = Path('/sys/fs/cgroup/dpdfnetbenchlab.slice')
LEAF = CG / 'workers'
HOLDER = 'dpdfnet-bench-lab-holder.service'
TIMER = 'dpdfnet-bench-lab-restore'
CPU = Path('/sys/devices/system/cpu')


def read(path):
    return Path(path).read_text().strip()


def write(path, value):
    Path(path).write_text(str(value) + '\n')


def cpus(text):
    result = set()
    for item in text.split(','):
        if not item:
            continue
        a = item.split('-')
        if len(a) > 2 or not all(x.isdecimal() for x in a):
            raise ValueError('Invalid CPU list')
        result.update(range(int(a[0]), int(a[-1]) + 1))
    return result


def cpulist(values):
    return ','.join(map(str, sorted(values)))


def cpumask(values):
    value = sum(1 << cpu for cpu in values)
    words = []
    while value:
        words.append(f'{value & 0xffffffff:08x}')
        value >>= 32
    return ','.join(reversed(words)) or '0'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def command(*args):
    return subprocess.check_output(list(map(str, args)), text=True, stderr=subprocess.STDOUT).strip()


def save(state):
    temporary = RUN / 'state.tmp'
    with temporary.open('w') as f:
        json.dump(state, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(RUN / 'state.json')


def load():
    require((RUN / 'state.json').exists(), 'No benchmark environment has been prepared')
    return json.loads((RUN / 'state.json').read_text())


def change(state, path, value, optional=False):
    """Write-ahead rollback journal, including partially completed writes."""
    path = str(path)
    before = read(path)
    if before == str(value):
        return
    row = dict(path=path, before=before, requested=str(value))
    state['journal'].append(row)
    save(state)
    try:
        write(path, value)
        row['applied'] = read(path)
    except OSError as error:
        row['error'] = str(error)
        if not optional:
            save(state)
            raise
    save(state)


def identity(pid):
    text = read(f'/proc/{pid}/stat')
    fields = text[text.rindex(')') + 2:].split()
    return dict(start=fields[19], state=fields[0], flags=int(fields[6]), comm=text[text.index('(') + 1:text.rindex(')')])


def power_snapshot():
    names = ['scaling_governor', 'energy_performance_preference', 'scaling_min_freq', 'scaling_max_freq']
    return {str(p): {n: read(p / n) for n in names if (p / n).exists()}
            for p in CPU.glob('cpufreq/policy*')}


def set_limits(policy, low, high):
    """Avoid transient min > max both when lowering and when raising bounds."""
    require(0 < low <= high, 'Invalid frequency limits')
    if low > int(read(policy / 'scaling_max_freq')):
        write(policy / 'scaling_max_freq', high)
    elif high < int(read(policy / 'scaling_min_freq')):
        write(policy / 'scaling_min_freq', low)
    write(policy / 'scaling_min_freq', low)
    write(policy / 'scaling_max_freq', high)


def topology(cpu):
    online = cpus(read(CPU / 'online'))
    require(cpu in online, 'Worker CPU is offline')
    cache = []
    for entry in (CPU / f'cpu{cpu}/cache').glob('index*'):
        if read(entry / 'level') == '3' and read(entry / 'type') == 'Unified':
            cache.append(cpus(read(entry / 'shared_cpu_list')))
    require(len(cache) == 1, 'Exactly one unified L3 domain is required')
    reserved = cache[0] & online
    siblings = cpus(read(CPU / f'cpu{cpu}/topology/thread_siblings_list')) & online
    house = online - reserved
    require(siblings <= reserved and house, 'No separate housekeeping CPU domain')
    return online, reserved, siblings, house


def foreign_threads(state):
    """Validate effective per-thread affinity; allowed masks alone are insufficient."""
    reserved = set(state['reserved'])
    foreign = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            ident = identity(int(proc.name))
            if ident['state'] == 'Z' or ident['flags'] & 0x200000:
                continue
            if str(CG.name) + '/workers' in read(proc / 'cgroup'):
                continue
            for task in (proc / 'task').iterdir():
                try:
                    if os.sched_getaffinity(int(task.name)) & reserved:
                        foreign.append(dict(pid=int(proc.name), tid=int(task.name)))
                except ProcessLookupError:
                    pass
        except (FileNotFoundError, ProcessLookupError):
            pass
    return foreign


def noise_snapshot(cpu):
    result = dict(time_ns=time.time_ns())
    for name in ['interrupts', 'softirqs']:
        lines = Path('/proc/' + name).read_text().splitlines()
        column = lines[0].split().index('CPU' + str(cpu)) + 1
        values = {}
        for line in lines[1:]:
            fields = line.split()
            try:
                values[fields[0].rstrip(':')] = int(fields[column])
            except (IndexError, ValueError):
                pass
        result[name] = values
    return result


def verify(state):
    require(state['phase'] == 'ready', 'Environment is not ready')
    require(hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest() == state['controller_sha256'], 'Controller changed during the session')
    require(hashlib.sha256(Path(state['probe']).read_bytes()).hexdigest() == state['probe_sha256'], 'Measurement worker changed during the session')
    require(time.monotonic() < state['expires_monotonic'], 'Restoration deadline has expired')
    require(command('systemctl', 'is-active', state['timer'] + '.timer') == 'active', 'Independent restoration timer is not active')
    require(read(CG / 'cpuset.cpus.partition') == 'isolated', 'CPU partition is not isolated')
    require(cpus(read(CG / 'cpuset.cpus.effective')) == set(state['reserved']) - set(state['parked']), 'L3 reservation changed')
    require(cpus(read(LEAF / 'cpuset.cpus.effective')) == {state['cpu']}, 'Worker partition changed')
    require(read(CPU / 'cpufreq/boost') == state['boost_during'], 'Boost policy changed')
    require(read(CPU / 'online') == state['online_during'], 'CPU online state changed')
    p = CPU / f"cpu{state['cpu']}/cpufreq"
    for name, expected in state['worker_power'].items():
        require(read(p / name) == expected, 'Frequency policy changed: ' + name)
    for entry in state['journal']:
        if 'applied' in entry:
            require(read(entry['path']) == entry['applied'], 'Setting changed: ' + entry['path'])
    foreign = foreign_threads(state)
    require(not foreign, 'Foreign threads may run in reserved L3 domain: ' + str(foreign[:8]))
    return dict(structurally_ready=True, session_id=state['id'], cpu=state['cpu'], reserved_l3_cpus=state['reserved'],
                housekeeping_cpus=state['house'], parked_siblings=state['parked'],
                frequency_khz=state['frequency_khz'], power_profile=state['power_profile'], layout_profile=state.get('layout_profile','randomized'), foreign_userspace_threads=foreign,
                restoration_timer=state['timer'] + '.timer', controller_sha256=state['controller_sha256'], worker_sha256=state['probe_sha256'],
                worker_power=state['worker_power'], expires_at=state['expires_at'],
                residual_kernel_work_possible=True, calibration=state.get('calibration'), noise=noise_snapshot(state['cpu']))


def launch(state, argv):
    verify(state)
    require(argv and Path(argv[0]).is_absolute() and Path(argv[0]).is_file(), 'Use an absolute executable path')
    require(state['uid'] > 0, 'Refusing a root benchmark worker')
    write(LEAF / 'cgroup.procs', os.getpid())
    os.sched_setaffinity(0, {state['cpu']})
    libc = ctypes.CDLL(None, use_errno=True)
    for cap in range(int(read('/proc/sys/kernel/cap_last_cap')) + 1):
        require(libc.prctl(24, cap, 0, 0, 0) == 0, 'Cannot clear capability bounding set')
    user = pwd.getpwuid(state['uid'])
    os.initgroups(user.pw_name, state['gid'])
    os.setgid(state['gid'])
    os.setuid(state['uid'])
    require(libc.prctl(38, 1, 0, 0, 0) == 0, 'Cannot set no_new_privs')
    status = read('/proc/self/status')
    for field in ['CapEff', 'CapPrm', 'CapAmb', 'CapBnd']:
        require(int(next(x.split()[1] for x in status.splitlines() if x.startswith(field + ':')), 16) == 0, 'Worker retains capabilities')
    os.environ.update(HOME=user.pw_dir, USER=user.pw_name, LOGNAME=user.pw_name, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='0')
    libc.personality.argtypes=[ctypes.c_ulong];libc.personality.restype=ctypes.c_int
    old=libc.personality(0xffffffff)
    desired=(old|0x40000) if state.get('layout_profile','randomized')=='fixed' else (old&~0x40000)
    require(old>=0 and libc.personality(desired)>=0, 'Cannot apply the requested process-only ASLR policy')
    # The controller lock is CLOEXEC. No root file descriptor is passed to exec.
    os.execv(argv[0], argv)


def frequency_qualified(rows, state):
    import statistics
    values = [r['effective_khz'] for r in rows]
    if state['power_profile'] == 'steady':
        mean = statistics.mean(values)
        return abs(mean / state['frequency_khz'] - 1) <= .01 and statistics.stdev(values) / mean <= .003
    # Boost is deliberately variable. Validate limits without claiming stability.
    low = int(state['worker_power']['scaling_min_freq'])
    high = int(state['worker_power']['scaling_max_freq'])
    return all(low * .97 <= value <= high * 1.03 for value in values)


def calibrate(state, repeats=3):
    verify(state)
    probe = Path(state['probe'])
    require(hashlib.sha256(probe.read_bytes()).hexdigest() == state['probe_sha256'], 'Frequency probe changed')
    reference_khz = state['reference_khz']
    rows = []
    for _ in range(repeats):
        argv = [str(probe), 'probe', str(state['cpu']), '600']
        # This controller is single-threaded. The child drops privileges before
        # exec; the parent retains the short control lock during calibration.
        data = json.loads(subprocess.check_output(argv, preexec_fn=lambda: launch(state, argv),
                                                  text=True, timeout=15))
        data['effective_khz'] = reference_khz * data['delivered_delta'] / data['reference_delta']
        rows.append(data)
    import statistics
    mean = statistics.mean(r['effective_khz'] for r in rows)
    sd = statistics.stdev(r['effective_khz'] for r in rows)
    passed = frequency_qualified(rows, state)
    result = dict(samples=rows, mean_khz=mean, sd_khz=sd, target_khz=state['frequency_khz'],
                  passed=passed, fixed_frequency=state['power_profile']=='steady',
                  method='CPPC delivered/reference hardware counters; nominal_perf == reference_perf',
                  tolerance_mean_percent=1 if state['power_profile']=='steady' else None,
                  tolerance_sd_percent=.3 if state['power_profile']=='steady' else None, dynamic_range_tolerance_percent=3 if state['power_profile']=='boost' else None, time_ns=time.time_ns())
    state['calibration'] = result
    save(state)
    require(result['passed'], 'Actual frequency does not satisfy the selected power profile: ' + json.dumps(result))
    return result


def restore(state):
    if state['phase'] == 'restored':
        return state['restoration']
    state['phase'] = 'restoring'
    save(state)
    errors = []
    def attempt(label, fn):
        try:
            fn()
        except Exception as error:
            errors.append(dict(action=label, error=str(error)))
    if (LEAF / 'cgroup.kill').exists():
        attempt('kill benchmark workers', lambda: write(LEAF / 'cgroup.kill', 1))
    for cpu in state.get('parked', []):
        attempt('online sibling ' + str(cpu), lambda cpu=cpu: write(CPU / f'cpu{cpu}/online', 1) if read(CPU / f'cpu{cpu}/online') != '1' else None)
    if (CG / 'cpuset.cpus.partition').exists():
        attempt('release CPU partition', lambda: write(CG / 'cpuset.cpus.partition', 'member'))
    # Restore boost before original frequency ranges (which may require boost).
    attempt('restore boost', lambda: write(CPU / 'cpufreq/boost', state['boost_before']))
    for path, values in state['power_before'].items():
        def original_power(path=path, values=values):
            p = Path(path)
            set_limits(p, int(values['scaling_min_freq']), int(values['scaling_max_freq']))
            write(p / 'scaling_governor', values['scaling_governor'])
            if 'energy_performance_preference' in values:
                write(p / 'energy_performance_preference', values['energy_performance_preference'])
        attempt('power ' + path, original_power)
    for entry in reversed(state['journal']):
        attempt(entry['path'], lambda entry=entry: write(entry['path'], entry['before']) if read(entry['path']) != entry['before'] else None)
    for row in reversed(state['kernel_moves']):
        def affinity(row=row):
            try:
                if identity(row['pid'])['start'] == row['start']:
                    os.sched_setaffinity(row['pid'], row['before'])
            except (FileNotFoundError, ProcessLookupError):
                pass
        attempt('kernel affinity ' + str(row['pid']), affinity)
    if LEAF.exists():
        def remove_leaf():
            for _ in range(100):
                try:
                    LEAF.rmdir()
                    return
                except OSError:
                    time.sleep(.02)
            raise RuntimeError('Worker cgroup did not drain')
        attempt('remove worker cgroup', remove_leaf)
    unit_loaded = command('systemctl', 'show', '--property=LoadState', '--value', CG.name) != 'not-found'
    cleanup = [('stop', HOLDER), ('set-property', '--runtime', CG.name, 'AllowedCPUs='),
                 ('stop', CG.name), ('revert', CG.name)] if unit_loaded else []
    cleanup.append(('stop', state['timer'] + '.timer'))
    for args in cleanup:
        if args[0] == 'stop' and command('systemctl', 'show', '--property=LoadState', '--value', args[1]) == 'not-found':
            continue
        attempt('systemctl ' + ' '.join(args), lambda args=args: command('systemctl', *args))
    if state['irqbalance_before'] == 'active':
        attempt('restart irqbalance', lambda: command('systemctl', 'start', 'irqbalance'))
    power = power_snapshot()
    mismatches = [entry['path'] for entry in state['journal'] if read(entry['path']) != entry['before']]
    result = dict(restored=not errors and not mismatches and power == state['power_before']
                  and read(CPU / 'cpufreq/boost') == state['boost_before']
                  and read(CPU / 'online') == state['online_before'] and not CG.exists()
                  and read('/sys/fs/cgroup/cgroup.subtree_control') == state['root_controllers_before']
                  and subprocess.run(['systemctl', 'is-active', 'irqbalance'], capture_output=True, text=True).stdout.strip() == state['irqbalance_before'],
                  errors=errors, mismatched_files=mismatches, power_matches=power == state['power_before'],
                  online=read(CPU / 'online'), timestamp_ns=time.time_ns())
    state['restoration'] = result
    state['phase'] = 'restored' if result['restored'] else 'restore_failed'
    save(state)
    require(result['restored'], 'Restoration incomplete: ' + json.dumps(result))
    return result


def prepare(args):
    if (RUN / 'state.json').exists():
        previous = load()
        require(previous['phase'] == 'restored', 'An environment already exists; run bench-ready or bench-restore')
        shutil.copyfile(RUN / 'state.json', RUN / ('state-' + previous['id'] + '.json'))
    require(read('/sys/devices/system/clocksource/clocksource0/current_clocksource') == 'tsc', 'This validated backend requires the stable-TSC clocksource')
    require(not CG.exists(), 'Benchmark cgroup already exists')
    for unit in [CG.name, HOLDER]:
        for prop in ['FragmentPath', 'DropInPaths']:
            require(not command('systemctl', 'show', '--property=' + prop, '--value', unit), 'Existing unit configuration must not be overwritten: ' + unit)
    online, reserved, siblings, house = topology(args.cpu)
    user = int(os.environ.get('SUDO_UID', '-1'))
    require(user > 0, 'Prepare via sudo from an ordinary user account')
    probe = Path(args.probe).resolve()
    require(probe.is_file(), 'Compile the worker before preparing isolation')
    cppc = CPU / f'cpu{args.cpu}/acpi_cppc'
    nominal = int(read(cppc / 'nominal_freq')) * 1000
    require(nominal > 0 and read(cppc / 'nominal_perf') == read(cppc / 'reference_perf'), 'Unsupported CPPC reference frequency; refusing unverified frequency claims')
    frequency = nominal if args.frequency_khz == 'auto' else int(args.frequency_khz)
    require(args.power_profile == 'steady' or args.frequency_khz == 'auto', 'Boost profile requires automatic dynamic bounds')
    require(int(read(CPU / f'cpu{args.cpu}/cpufreq/cpuinfo_min_freq')) <= frequency <= nominal, 'Choose a frequency no higher than the nominal non-boost frequency')
    require(60 <= args.ttl <= 7200, 'Restoration timeout must be 60..7200 seconds')
    state = dict(id=uuid.uuid4().hex, phase='preparing', controller_sha256=hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(), cpu=args.cpu, uid=user,
                 gid=int(os.environ['SUDO_GID']), reserved=sorted(reserved), siblings=sorted(siblings),
                 house=sorted(house), online_before=read(CPU / 'online'), parked=[],
                 power_before=power_snapshot(), boost_before=read(CPU / 'cpufreq/boost'),
                 root_controllers_before=read('/sys/fs/cgroup/cgroup.subtree_control'),
                 irqbalance_before=subprocess.run(['systemctl', 'is-active', 'irqbalance'],capture_output=True,text=True).stdout.strip(),
                 reference_khz=nominal, frequency_khz=frequency if args.power_profile=='steady' else None,
                 power_profile=args.power_profile, layout_profile=args.layout, boost_during='0' if args.power_profile=='steady' else '1', probe=str(probe),
                 probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(), journal=[], kernel_moves=[],
                 unmovable_kernel=[], expires_at=time.time()+args.ttl, expires_monotonic=time.monotonic()+args.ttl, created_ns=time.time_ns())
    state['timer'] = TIMER + '-' + state['id'][:12]
    save(state)
    # Immutable rollback code survives edits/deletion of the workspace script.
    shutil.copyfile(Path(__file__).resolve(), RUN / 'rollback.py')
    os.chmod(RUN / 'rollback.py', 0o700)
    try:
        command('systemd-run', '--collect', '--unit='+state['timer'], '--on-active='+str(args.ttl)+'s', '--timer-property=AccuracySec=1s',
                '/usr/bin/python3', RUN / 'rollback.py', 'restore', '--session', state['id'])
        if state['irqbalance_before'] == 'active':
            command('systemctl', 'stop', 'irqbalance')
        command('systemd-run', '--unit='+HOLDER, '--slice='+CG.name, '--remain-after-exit', '--collect',
                '--property=AllowedCPUs='+cpulist(reserved), '/usr/bin/true')
        command('systemctl', 'set-property', '--runtime', CG.name, 'AllowedCPUs='+cpulist(reserved))
        write(CG / 'cpuset.cpus.partition', 'isolated')
        write(CG / 'cgroup.subtree_control', '+cpuset')
        LEAF.mkdir()
        write(LEAF / 'cpuset.cpus', args.cpu)
        write(LEAF / 'cpuset.mems', read(CG / 'cpuset.mems.effective'))
        files = [Path('/proc/irq/default_smp_affinity'), Path('/sys/devices/virtual/workqueue/cpumask'), Path('/proc/sys/kernel/watchdog_cpumask')]
        files += sorted(Path('/proc/irq').glob('[0-9]*/smp_affinity_list'))
        for p in files:
            current = read(p)
            if p.name == 'smp_affinity_list':
                if not cpus(current) & reserved:
                    continue
                target = cpulist((cpus(current)-reserved) or house)
            elif p.name == 'watchdog_cpumask':
                target = cpulist(house)
            else:
                target = cpumask(house)
            change(state, p, target, optional=p.name == 'smp_affinity_list')
        for p in Path('/proc').glob('[0-9]*'):
            pid = int(p.name)
            try:
                ident = identity(pid)
                if pid == 2 or not ident['flags'] & 0x200000:
                    continue
                affinity = os.sched_getaffinity(pid)
                if not affinity & reserved:
                    continue
                row = dict(pid=pid, start=ident['start'], before=sorted(affinity))
                state['kernel_moves'].append(row)
                save(state)
                try:
                    os.sched_setaffinity(pid, (affinity-reserved) or house)
                except OSError as error:
                    state['kernel_moves'].pop()
                    state['unmovable_kernel'].append(dict(pid=pid, comm=ident['comm'], error=str(error)))
                save(state)
            except (FileNotFoundError, ProcessLookupError):
                pass
        write(CPU / 'cpufreq/boost', state['boost_during'])
        p = CPU / f'cpu{args.cpu}/cpufreq'
        write(p / 'scaling_governor', 'performance')
        if (p / 'energy_performance_preference').exists():
            write(p / 'energy_performance_preference', 'performance')
        if args.power_profile == 'steady':
            set_limits(p, frequency, frequency)
        else:
            original = state['power_before'][str(p.resolve())]
            set_limits(p, int(original['scaling_min_freq']), int(read(p / 'cpuinfo_max_freq')))
        state['worker_power'] = {n: read(p / n) for n in ['scaling_governor','energy_performance_preference','scaling_min_freq','scaling_max_freq'] if (p / n).exists()}
        # Restrict deep idle on the worker only; this is a laboratory power policy.
        for entry in (CPU / f'cpu{args.cpu}/cpuidle').glob('state*'):
            if args.power_profile == 'steady' and int(read(entry / 'latency')) > 1:
                change(state, entry / 'disable', 1)
        for cpu in sorted(siblings-{args.cpu}):
            state['parked'].append(cpu)
            save(state)
            write(CPU / f'cpu{cpu}/online', 0)
        state['online_during'] = read(CPU / 'online')
        state['phase'] = 'ready'
        save(state)
        verify(state)
        # Let the package settle before the three hardware-counter probes.
        time.sleep(2)
        calibrate(state)
        return verify(state)
    except BaseException:
        restore(state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--cpu', type=int, default=3)
    p.add_argument('--frequency-khz', default='auto')
    p.add_argument('--power-profile', choices=['steady','boost'], default='steady')
    p.add_argument('--layout', choices=['randomized','fixed'], default='randomized')
    p.add_argument('--ttl', type=int, default=1800)
    p.add_argument('--probe', required=True)
    sub.add_parser('ready')
    sub.add_parser('status')
    p = sub.add_parser('restore')
    p.add_argument('--session')
    p = sub.add_parser('exec')
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    require(os.getuid() == 0, 'Run the controller via sudo')
    RUN.mkdir(mode=0o700, exist_ok=True)
    st = RUN.lstat()
    require(stat.S_ISDIR(st.st_mode) and st.st_uid == 0 and not st.st_mode & 0o077, 'Unsafe state directory')
    # No caller-selected state or lock paths; root must not follow user symlinks.
    with (RUN / 'lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        if args.mode == 'prepare':
            result = prepare(args)
        elif args.mode == 'restore':
            if args.session:
                require(load()['id'] == args.session, 'Session changed; refusing to restore another session')
            result = restore(load()) if (RUN / 'state.json').exists() else dict(restored=True, already_absent=True)
        elif args.mode == 'exec':
            # calibrate() starts an exec controller; release the parent lock first.
            state = load()
            argv = args.command[1:] if args.command[:1] == ['--'] else args.command
            launch(state, argv)
            return
        else:
            state = load()
            if args.mode == 'ready':
                try:
                    calibrate(state)
                except BaseException:
                    restore(state)
                    raise
            result = verify(state) if state['phase'] == 'ready' else state.get('restoration', dict(phase=state['phase']))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt(signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        failure = dict(error=str(error), type=type(error).__name__)
        if (RUN / 'state.json').exists():
            state = load()
            failure.update(phase=state['phase'], restoration=state.get('restoration'))
        print(json.dumps(failure), flush=True)
        sys.exit(1)
