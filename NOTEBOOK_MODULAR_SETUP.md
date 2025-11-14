# Modular Notebook with Self-Cloning Repository

## Overview

The notebook has been restructured to maintain clean modular architecture while being 100% standalone. Instead of embedding all code in the notebook, it now clones its own GitHub repository to obtain supporting files.

## How It Works

### 1. Repository Structure

Your repository at `https://github.com/MattKevan/VQ` should contain:

```
VQ/
├── vqgan_clip_modern.ipynb    # Main notebook
├── vqgan_clip_utils.py        # Utilities module
├── requirements.txt           # Python dependencies
├── config.yaml                # Optional configuration
└── README.md                  # Documentation
```

### 2. Notebook Workflow

**Cell 1 (Markdown):** Introduction and instructions

**Cell 2 (Setup):** Automatic repository cloning and dependency installation
- Clones `https://github.com/MattKevan/VQ` to `/mnt/store/vqgan-clip/`
- Clones `taming-transformers` to `/mnt/store/taming-transformers/`
- Uses `/mnt/store` so files **persist across kernel restarts**
- Only clones if directories don't exist (skips if already present)
- Verifies required files are present
- Installs dependencies from `requirements.txt`
- Patches `taming-transformers` for PyTorch 2.x compatibility

**Cell 3 (Imports):** Load modules from cloned repositories
- Adds `/mnt/store/vqgan-clip/` to Python path
- Adds `/mnt/store/taming-transformers/` to Python path
- Imports `vqgan_clip_utils` from cloned directory
- Shows helpful error messages if setup wasn't run

**Cells 4+:** Rest of notebook (configuration, model loading, generation, etc.)

### 3. User Experience

For users, the workflow is simple:

1. Download just `vqgan_clip_modern.ipynb` from GitHub
2. Place VQGAN models in `./models/` directory
3. Run Cell 2 (takes 5-10 minutes first time)
4. Restart kernel
5. Run remaining cells

All supporting files are automatically fetched from the repository!

## Benefits

✅ **Clean modular architecture** - Files stay separate and maintainable
✅ **Version control friendly** - Easy to update via git
✅ **Standalone notebook** - Users only need the .ipynb file
✅ **Always up-to-date** - Fresh clone ensures latest code
✅ **Easy debugging** - Can edit `vqgan_clip_utils.py` directly in `./vqgan-clip/`

## Configuration

The repository URL and directories are set at the top of Cell 2:

```python
REPO_URL = "https://github.com/MattKevan/VQ"
REPO_DIR = Path("/mnt/store/vqgan-clip")
TAMING_DIR = Path("/mnt/store/taming-transformers")
```

**Why `/mnt/store`?**
- Files persist across kernel restarts
- No need to re-clone on every restart
- Faster subsequent runs (only installs dependencies)

Users don't need to change this - it just works!

## Files in Repository

### vqgan_clip_utils.py (existing)
No changes needed - already modular and ready to use.

### requirements.txt (existing)
No changes needed - contains all dependencies.

### vqgan_clip_modern.ipynb (updated)
- Cell 1: Updated instructions
- Cell 2: New setup with git clone + dependency installation
- Cell 3: Updated imports to use `./vqgan-clip/` path
- Removed old Cell 3 that embedded utils code

### config.yaml (optional)
Can be used if you add config file loading, but currently notebook uses ipywidgets for interactive configuration.

## Persistent Storage with `/mnt/store`

The notebook uses `/mnt/store` as a persistent storage location:

**Benefits:**
- Files survive kernel restarts
- No need to re-clone repositories
- Faster startup on subsequent runs
- Can edit files directly and changes persist

**File Locations:**
- `/mnt/store/vqgan-clip/` - Main repository
- `/mnt/store/taming-transformers/` - VQGAN dependencies

**To force update:**
```bash
# Delete directories to trigger fresh clone
rm -rf /mnt/store/vqgan-clip
rm -rf /mnt/store/taming-transformers
```

Then re-run Cell 2.

## Troubleshooting

**"Repository not found" error:**
- Check that `REPO_URL` is correct
- Ensure repository is public or user has access
- Verify git is installed

**"Module not found" error (taming-transformers):**
- Make sure Cell 2 completed successfully
- Restart kernel after running Cell 2
- Check that both `/mnt/store/vqgan-clip/` and `/mnt/store/taming-transformers/` exist
- Cell 3 adds both directories to Python path

**"Already exists" message:**
- Normal - Cell 2 skips cloning if directories already exist
- To force fresh clone, delete directories (see above)

## Development Workflow

### Making Changes

1. Edit files in main repository (vqgan_clip_utils.py, requirements.txt, etc.)
2. Commit and push changes to GitHub
3. Users re-run Cell 2 to get latest version

### Testing Locally

You can test the notebook locally even without pushing to GitHub:

1. Create a local git repository in a different directory
2. Clone it to `./vqgan-clip/` manually
3. Or temporarily edit Cell 2 to use a local path

## Next Steps

- Commit the updated notebook to your repository
- Users can download it and start generating immediately
- Consider adding example prompts or preset configurations
- Could add option to use git pull instead of fresh clone (for faster updates)

## Comparison: Before vs After

**Before (Embedded Approach):**
- Cell 3 contained ~11KB of embedded Python code
- Hard to maintain - needed to edit within notebook
- Version control showed huge diffs
- But truly standalone

**After (Modular with Self-Cloning):**
- Cell 2 clones repository with all files
- Easy to maintain - edit separate .py files
- Clean git history
- Still standalone - just needs internet for first run

Best of both worlds! 🎉
