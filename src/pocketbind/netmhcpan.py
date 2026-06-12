from __future__ import annotations

import os
import subprocess
import tarfile
from pathlib import Path

import pandas as pd

from pocketbind.data import load_predictions_tsv


def configure_netmhcpan_script(tool_dir: Path, symlink: Path) -> None:
    script = tool_dir / "netMHCpan"
    text = script.read_text()
    lines = []
    for line in text.splitlines():
        if line.startswith("setenv\tNMHOME") or line.startswith("setenv NMHOME"):
            lines.append(f"setenv\tNMHOME\t{symlink}")
        elif line.strip().startswith("setenv  TMPDIR"):
            lines.append("\tsetenv  TMPDIR  $NMHOME/tmp")
        elif line.startswith("setenv NETMHCpan "):
            lines.append('setenv NETMHCpan "$NMHOME/$PLATFORM"')
        elif line.strip() == "if ( -x $NETMHCpan/bin/netMHCpan-4.2 ) then":
            lines.append('if ( -x "$NETMHCpan/bin/netMHCpan-4.2" ) then')
        elif line.strip() == "$NETMHCpan/bin/netMHCpan-4.2 $*":
            lines.append('   "$NETMHCpan/bin/netMHCpan-4.2" $*')
        else:
            lines.append(line)
    script.write_text("\n".join(lines) + "\n")


def ensure_netmhcpan_install(
    tool_dir: Path,
    *,
    archive_path: Path | None = None,
    symlink: Path = Path("/tmp/PocketBind_netMHCpan-4.2"),
) -> None:
    if not tool_dir.exists():
        if archive_path is None:
            archive_path = tool_dir.parent / "netMHCpan-4.2c.Darwin_arm64.tar.gz"
        if not archive_path.exists():
            raise FileNotFoundError(f"Missing NetMHCpan directory and archive: {archive_path}")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(tool_dir.parent)

    (tool_dir / "tmp").mkdir(parents=True, exist_ok=True)
    ensure_netmhcpan_symlink(tool_dir, symlink=symlink)
    configure_netmhcpan_script(tool_dir, symlink)


def ensure_netmhcpan_symlink(tool_dir: Path, symlink: Path = Path("/tmp/PocketBind_netMHCpan-4.2")) -> None:
    tool_dir = tool_dir.resolve()
    if symlink.exists() or symlink.is_symlink():
        if symlink.resolve() == tool_dir:
            return
        symlink.unlink()
    symlink.symlink_to(tool_dir, target_is_directory=True)


def run_netmhcpan(
    *,
    netmhcpan_script: Path,
    input_path: Path,
    output_path: Path,
    include_context: bool,
    include_ba: bool,
    pathogen: bool = False,
    neo: bool = False,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_path.with_suffix(output_path.suffix + ".log")

    cmd = [str(netmhcpan_script), "-pmhc", str(input_path), "-xls", "-xlsfile", str(output_path)]
    if include_context:
        cmd.append("-context")
    if include_ba:
        cmd.append("-BA")
    if pathogen:
        cmd.append("-pathogen")
    if neo:
        cmd.append("-neo")

    env = os.environ.copy()
    env["PATH"] = f"/opt/homebrew/bin:{env.get('PATH', '')}"
    with log_path.open("w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)

    return load_predictions_tsv(output_path)
