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

def _rc_type_in(pmt, t):
    if t == "warn":
        y = rc_warn_in(pmt)
    elif t == "highlight":
        y = rc_highlight_in(pmt)
    else:
        y = rc_input(pmt)

    return y

def rc_ensure(pmt, t="", default=True):
    y = None
    while y not in ["", "Y", "N"]:
        y = _rc_type_in(pmt, t).upper()

    return y=="Y" if y != "" else default

def rc_select(pmt, range_=range(2), t=""):
    y = None
    while y not in range_:
        y = _rc_type_in(pmt, t)
        try:
            y = int(y)
        except ValueError:
            y = None
    return y

def rc_choose(pmt, range_=["Yes", "No"], t="", case_sensitive=False):
    if not case_sensitive:
        r = [i.upper() for i in range_]
    else:
        r = range_
    y = None
    while y not in r:
        y = _rc_type_in(pmt, t)
        if not case_sensitive:
            y = y.upper()
    return y

def rc_getstr(pmt, t="", keepblank=False):
    if keepblank:
        return _rc_type_in(pmt,t)
    else:
        y = ""
        while y == "":
            y = _rc_type_in(pmt, t)
        return y

def rc_getnum(pmt, t="", blankas=0.0):
    while True:
        y = _rc_type_in(pmt, t)
        try:
            y = float(y)
            break
        except ValueError:
            if y == "":
                y = blankas
                break
    return y

def rc_getint(pmt, t="", blankas=0):
    while True:
        y = _rc_type_in(pmt, t)
        try:
            y = int(y)
            break
        except ValueError:
            if y == "":
                y = blankas
                break
    return y

