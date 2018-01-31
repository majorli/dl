class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def rc_input(pmt):
    print(pmt, end="", flush=True)
    return input().strip()

def rc_warn(s):
    print(bcolors.WARNING + s + bcolors.ENDC)
    return

def rc_warn_in(pmt):
    print(bcolors.WARNING + pmt + bcolors.ENDC, end="", flush=True)
    return input().strip()

def rc_fail(s):
    print(bcolors.FAIL + s + bcolors.ENDC)
    return

def rc_header(s):
    print(bcolors.HEADER + s + bcolors.ENDC)
    return

def rc_state(s):
    print(bcolors.OKBLUE + s + bcolors.ENDC)
    return

def rc_result(s):
    print(bcolors.OKGREEN + s + bcolors.ENDC)
    return

def rc_highlight(s):
    print(bcolors.BOLD + s + bcolors.ENDC)
    return

def rc_highlight_in(pmt):
    print(bcolors.BOLD + pmt + bcolors.ENDC, end="", flush=True)
    return input().strip()
