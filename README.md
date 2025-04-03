# Causal Bandits

## Repository Structure

```plaintext
.
├── algorithms/
│   ├── __init__.py               
│   ├── confounded_budgeted.py
│   ├── pomis.py
│   ├── raps.py
│   ├── raps_ucb.py
├── causal_models/ # Submodule
├── experiments/                
├── integrator/                   
│   ├── __init__.py
│   ├── causal_bandit.py  
├── mab/ # Submodule
├── scripts/
├── tests/  
├── Makefile
├── README.md  
└── setup.sh             

```
## 🔧 Project Setup & Submodule Management

This project integrates multiple submodules (e.g., `causal_models`, `mab`) that live in separate repositories. To manage them efficiently, use the provided script:

### 📄 `scripts/setup.sh`

| Command                 | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `bash setup.sh init`    | Initialize all submodules (`git submodule update --init --recursive`)     |
| `bash setup.sh update`  | Pull the latest changes in each submodule                                  |
| `bash setup.sh status`  | Show current submodule status                                               |
| `bash setup.sh install` | Install submodules as editable packages (via `pip install -e`)             |
| `bash setup.sh venv`    | Create a Python virtual environment (`.venv/`)                              |
| `bash setup.sh clean`   | Remove all `__pycache__` folders and `.pyc` files                           |
| `bash setup.sh help`    | Show available commands                                                     |

---

