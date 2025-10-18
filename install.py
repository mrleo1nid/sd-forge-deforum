# Copyright (C) 2023 Deforum LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the authors: https://deforum.github.io/

import launch
import os

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

print("Deforum: Installing dependencies for Wan 2.2 support...")

# Force upgrade critical dependencies for Wan 2.2 TI2V support
critical_upgrades = {
    'peft': '0.17.1',
    'accelerate': '1.10.1',
}

for package, version in critical_upgrades.items():
    try:
        import importlib.metadata
        current_version = importlib.metadata.version(package)
        if current_version != version:
            print(f"Deforum: Upgrading {package} {current_version} → {version} for Wan 2.2...")
            launch.run_pip(f"install {package}=={version}", f"Deforum Wan 2.2 requirement: {package}=={version}")
    except:
        print(f"Deforum: Installing {package}=={version}...")
        launch.run_pip(f"install {package}=={version}", f"Deforum Wan 2.2 requirement: {package}=={version}")

with open(req_file) as file:
    for lib in file:
        lib = lib.strip()
        if not lib or lib.startswith('#'):
            continue

        # Force install git diffusers for Wan 2.2 support
        if lib.startswith('git+'):
            print(f"Deforum: Installing diffusers from git for Wan 2.2 support...")
            launch.run_pip(f"install --upgrade {lib}", f"Deforum Wan 2.2 requirement: diffusers (git main)")
            continue

        # Skip version-range packages already handled above
        if any(lib.startswith(pkg) for pkg in ['peft', 'accelerate']):
            continue

        # Install other packages normally
        if not launch.is_installed(lib.split('>=')[0].split('==')[0].split('<')[0]):
            launch.run_pip(f"install {lib}", f"Deforum requirement: {lib}")
