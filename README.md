# Causal Bandits


## 🔧 Project Setup & Submodule Management

This project integrates multiple submodules (e.g., `causal_models`, `mab`) that live in separate repositories. To manage them efficiently, use the provided script:

### 📄 `scripts/setup.sh`

| Command                         | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `bash scripts/setup.sh init`    | Initialize all submodules (`git submodule update --init --recursive`)     |
| `bash scripts/setup.sh update`  | Pull the latest changes in each submodule                                  |
| `bash scripts/setup.sh status`  | Show current submodule status                                               |
| `bash scripts/setup.sh install` | Install submodules as editable packages (via `pip install -e`)             |
| `bash scripts/setup.sh venv`    | Create a Python virtual environment (`.venv/`)                              |
| `bash scripts/setup.sh clean`   | Remove all `__pycache__` folders and `.pyc` files                           |
| `bash scripts/setup.sh help`    | Show available commands                                                     |

---

### Example Usage

```bash
# Clone the main project with submodules
git clone --recurse-submodules <main-repo-url>

# Enter the project directory
cd my-integrated-project

# Initialize submodules
bash scripts/setup.sh init

# Optionally update submodules to the latest main branch
bash scripts/setup.sh update

# Create a virtual environment
bash scripts/setup.sh venv

# Activate it (Linux/macOS)
source .venv/bin/activate

# Or on Windows:
.venv\Scripts\activate

# Install dependencies in editable mode
bash scripts/setup.sh install

