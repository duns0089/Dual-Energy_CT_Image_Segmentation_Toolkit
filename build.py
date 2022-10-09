from argparse import ArgumentParser
from pathlib import Path
from platform import system
from shutil import rmtree
from subprocess import CalledProcessError, check_call
from sys import exit

CC = "clang"
CXX = "clang++"
CC_LD = "lld"
CXX_LD = "lld"

SRC_ROOT = Path("src")
BUILD_ROOT = Path("build")


def build_linux(commands: list[str], target: str, config: str) -> None:
    build_path = BUILD_ROOT / target / config
    meson_config = "--buildtype " + config
    meson_reinit = "--reconfigure"

    for command in commands:
        if command == "init" or command == "reinit":
            check_call(
                " ".join(
                    [
                        f"{CC=}",
                        f"{CXX=}",
                        f"{CC_LD=}",
                        f"{CXX_LD=}",
                        "meson",
                        meson_config,
                        meson_reinit if command == "reinit" else "",
                        str(SRC_ROOT),
                        str(build_path),
                    ]
                ),
                shell=True,
            )
        elif command == "build":
            check_call(["ninja"], cwd=build_path)
        elif command == "clean":
            rmtree(build_path, ignore_errors=True)
        elif command == "cleanall":
            rmtree(BUILD_ROOT, ignore_errors=True)


def build_windows(commands: list[str], target: str, config: str) -> None:
    sln_path = r"src\LTRI.sln"

    # TODO: Determine MSBuild.exe path procedurally.
    msbuild = r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe"
    msbuild_target = "/p:Platform=" + target
    msbuild_config = "/p:Configuration=" + config.capitalize()
    msbuild_clean = "/t:clean"

    for command in commands:
        if command == "build":
            check_call([msbuild, sln_path, msbuild_target, msbuild_config])
        elif command == "clean":
            check_call(
                [msbuild, sln_path, msbuild_clean, msbuild_target, msbuild_config]
            )
        elif command == "cleanall":
            check_call([msbuild, sln_path, msbuild_clean])


if __name__ == "__main__":
    filename = Path(__file__).name
    example = f"example: {filename} clean init build -t x64 -c debug"

    parser = ArgumentParser(description=example)
    parser.add_argument(
        "commands",
        nargs="+",
        choices=["init", "reinit", "build", "clean", "cleanall"],
        help="A sequence of build steps to perform.",
    )
    parser.add_argument(
        "-t",
        "--target",
        choices=["x64"],
        default="x64",
        help="The desired build target.",
    )
    parser.add_argument(
        "-c",
        "--config",
        choices=["debug", "release"],
        default="debug",
        help="The desired build configuration.",
    )

    try:
        if system() == "Linux":
            build_linux(**vars(parser.parse_args()))
        elif system() == "Windows":
            build_windows(**vars(parser.parse_args()))
        else:
            raise NotImplementedError(f"Unexpected system: {system()}")
    except CalledProcessError as error:
        exit(error.returncode)
