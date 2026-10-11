"""Cache the original Isaac training assets in this project using HTTPS.

Run in isaaclab_61, then set ISAACSIM_ASSET_ROOT to assets/isaac_cache
before starting training. USD references and textures retain their layout.
"""

from pathlib import Path
import subprocess
import sys
from urllib.parse import quote, unquote, urljoin, urlparse

sys.dont_write_bytecode = True

from pxr import Sdf, UsdUtils

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / "assets/isaac_cache"
REMOTE_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/"
ASSETS = (
    "Isaac/Robots_Multiphysics/RobotStudio/so101_new_calib_SysID/so101_new_calib.usda",
    "Isaac/IsaacLab/Environments/Grid/default_ground_plane_checker_v1/default_ground_plane.usda",
    "Isaac/Props/UIElements/arrow_x.usd",
)


def main():
    pending = [urljoin(REMOTE_ROOT, asset) for asset in ASSETS]
    visited = set()
    while pending:
        url = pending.pop()
        if url in visited:
            continue
        if not url.startswith(REMOTE_ROOT):
            raise ValueError(f"Dependency outside the Isaac asset root: {url}")
        visited.add(url)
        relative = unquote(url[len(REMOTE_ROOT):])
        destination = (CACHE_ROOT / relative).resolve()
        if not destination.is_relative_to(CACHE_ROOT.resolve()):
            raise ValueError(f"Dependency escapes the project cache: {relative}")
        if not destination.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name + ".partial")
            parsed = urlparse(url)
            encoded = parsed._replace(path=quote(unquote(parsed.path), safe="/")).geturl()
            print(f"Downloading {relative}", flush=True)
            subprocess.run(
                ["curl", "-fsSL", "--retry", "2", "--connect-timeout", "10",
                 "--max-time", "90", encoded, "-o", str(partial)],
                check=True,
            )
            partial.replace(destination)
        if destination.suffix.lower() in (".usd", ".usda", ".usdc"):
            layer = Sdf.Layer.FindOrOpen(str(destination))
            if layer is None:
                raise RuntimeError(f"Invalid USD layer: {destination}")
            dependencies = set()

            def collect(path):
                if path:
                    dependencies.add(path)
                return path

            UsdUtils.ModifyAssetPaths(layer, collect)
            for dependency in sorted(dependencies):
                # OmniPBR is supplied by the installed renderer, not the asset tree.
                if dependency == "OmniPBR.mdl":
                    continue
                pending.append(urljoin(url, dependency))
    print(f"Verified {len(visited)} cached files: {CACHE_ROOT}", flush=True)


if __name__ == "__main__":
    main()
