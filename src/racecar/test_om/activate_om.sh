#!/usr/bin/env bash
# Source this file in each terminal used for yellow-line testing.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Use: source /home/davinci-mini/racecar/src/racecar/test_om/activate_om.sh'
    exit 1
fi
if [[ -n "${CONDA_PREFIX:-}" ]] && declare -F conda >/dev/null; then
    conda deactivate
fi
source /opt/ros/humble/setup.bash
source /home/davinci-mini/racecar/install/local_setup.bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/davinci-mini/.venvs/racecar_om/bin/activate
echo "OM + ROS Python: $(command -v python3)"
