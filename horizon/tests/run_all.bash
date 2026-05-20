set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

python -m unittest ilqr_tests.py
python test_horizon.py
python test_solvers.py
python -m unittest paramter_support_test.py
python -m unittest solver_lq_test.py
python -m unittest test_get_set.py

echo "All tests passed!"