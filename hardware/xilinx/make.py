import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path

ROOT_DIR = Path.cwd()
VTA_HW_DIR = (ROOT_DIR / "../../").resolve() 
# Long Windows paths break Vivado, so the build directory can be moved to a
# shorter location (e.g. D:/vta-build) with VTA_BUILD_DIR.
BUILD_DIR = Path(os.environ.get("VTA_BUILD_DIR", VTA_HW_DIR / "build")).resolve() / "hardware" / "xilinx"
SCRIPT_DIR = ROOT_DIR / "scripts"
SRC_DIR = ROOT_DIR / "src"

VIVADO_HLS = "vivado_hls"
VIVADO = "vivado"
PYTHON_EXEC = sys.executable

VTA_CONFIG = VTA_HW_DIR / "config" / "vta_config.py"

def get_conf_string():
    try:
        result = subprocess.check_output(
            [PYTHON_EXEC, str(VTA_CONFIG), "--cfg-str"],
            text=True
        )
        return result.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting config string: {e}")
        raise SystemExit(1)

CONF = get_conf_string()

IP_BUILD_PATH = BUILD_DIR / "hls" / CONF
HW_BUILD_PATH = BUILD_DIR / "vivado" / CONF

INCLUDE_DIR = BUILD_DIR / "include"
CONFIG_TCL = INCLUDE_DIR / "vta_config.tcl"

IP_PATH = IP_BUILD_PATH / "vta_compute" / "soln" / "impl" / "ip" / "xilinx_com_hls_compute_1_0.zip"
BIT_PATH = HW_BUILD_PATH / "export" / f"{CONF}.bit"

# Vivado fails to create its per-user Tcl Store in %APPDATA% and emits a
# CRITICAL WARNING (Common 17-739) every time, so give it one in the build dir.
os.environ.setdefault("XILINX_TCLSTORE_USERAREA", str(BUILD_DIR / "tclstore"))

def run_command(cmd, cwd=None):
    cmd_str = " ".join(str(x) for x in cmd)
    print(f"[{cwd or 'Current Dir'}] Executing: {cmd_str}")
    
    try:
        subprocess.run(cmd, cwd=cwd, check=True, shell=True)
    except subprocess.CalledProcessError:
        print(f"Command failed: {cmd_str}")
        raise SystemExit(1)

def ensure_directory(path):
    """mkdir -p equivalent."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

def task_config_tcl():
    ensure_directory(INCLUDE_DIR)
    
    if CONFIG_TCL.exists():
        print(f"Skipping config gen: {CONFIG_TCL} exists.")
        return

    run_command([PYTHON_EXEC, str(VTA_CONFIG), "--export-tcl", str(CONFIG_TCL)])

def task_ip():
    """Builds the IP (HLS)."""

    task_config_tcl()
    
    if IP_PATH.exists():
        print(f"Skipping IP build: {IP_PATH} exists.")
        return

    print("Building IP...")
    ensure_directory(IP_BUILD_PATH)
    
    cmd = [
        VIVADO_HLS,
        "-f", str((SCRIPT_DIR / "hls.tcl").as_posix()),
        "-tclargs",
        str((VTA_HW_DIR).as_posix()),
        str((CONFIG_TCL).as_posix())
    ]
    run_command(cmd, cwd=IP_BUILD_PATH)

def task_bit():
    """Builds the Bitstream (Vivado)."""

    task_ip()

    if BIT_PATH.exists():
        print(f"Skipping Bitstream build: {BIT_PATH} exists.")
        return

    print("Building Bitstream...")
    ensure_directory(HW_BUILD_PATH)

    cmd = [
        VIVADO,
        "-mode", "tcl",
        "-source", str(SCRIPT_DIR / "vivado.tcl"),
        "-tclargs",
        str(BUILD_DIR / "hls" / CONF),
        str(CONFIG_TCL)
    ]
    run_command(cmd, cwd=HW_BUILD_PATH)

def task_clean():
    """Removes .out and .log files."""
    print("Cleaning logs and temporary files...")
    patterns = ["*.out", "*.log"]
    for pattern in patterns:
        for f in ROOT_DIR.glob(pattern):
            try:
                os.remove(f)
                print(f"Removed: {f}")
            except OSError as e:
                print(f"Error removing {f}: {e}")

def task_cleanall():
    """Removes build directory and logs."""
    task_clean()
    if BUILD_DIR.exists():
        print(f"Removing build directory: {BUILD_DIR}")
        shutil.rmtree(BUILD_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python builder for VTA Hardware")
    parser.add_argument(
        "target", 
        nargs="?", 
        default="bit", 
        choices=["all", "bit", "ip", "clean", "cleanall"],
        help="Build target (default: bit)"
    )
    
    args = parser.parse_args()

    if args.target == "all":
        task_bit() # 'all' maps to 'bit' in the Makefile
    elif args.target == "bit":
        task_bit()
    elif args.target == "ip":
        task_ip()
    elif args.target == "clean":
        task_clean()
    elif args.target == "cleanall":
        task_cleanall()
