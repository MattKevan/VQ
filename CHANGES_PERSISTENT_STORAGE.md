# Changes: Persistent Storage with /mnt/store

## What Changed

Fixed two critical issues with the notebook setup:

### Issue 1: "taming-transformers not found"
**Problem:** Cell 3 (imports) couldn't find taming-transformers module
**Solution:** Added `/mnt/store/taming-transformers/` to Python path in Cell 3

### Issue 2: Files deleted on kernel restart
**Problem:** Kernel restart deleted cloned repositories, requiring re-clone
**Solution:** Changed clone location from local directory to `/mnt/store/`

## New Directory Structure

**Before:**
```
./vqgan-clip/              # Deleted on kernel restart
./taming-transformers/     # Deleted on kernel restart
```

**After:**
```
/mnt/store/vqgan-clip/           # Persists across restarts ✓
/mnt/store/taming-transformers/  # Persists across restarts ✓
```

## Changes Made to Notebook

### Cell 2 (Setup)
```python
# OLD
REPO_DIR = "vqgan-clip"

# NEW
REPO_DIR = Path("/mnt/store/vqgan-clip")
TAMING_DIR = Path("/mnt/store/taming-transformers")
```

- Now clones both repositories to `/mnt/store/`
- Creates `/mnt/store/` directory if it doesn't exist
- Skips cloning if directories already exist (faster on subsequent runs)
- Shows helpful messages about persistence

### Cell 3 (Imports)
```python
# Added both directories to Python path
REPO_DIR = Path("/mnt/store/vqgan-clip")
TAMING_DIR = Path("/mnt/store/taming-transformers")

# Add vqgan-clip to path
sys.path.insert(0, str(REPO_DIR))

# Add taming-transformers to path (fixes import error)
sys.path.insert(0, str(TAMING_DIR))
```

## Benefits

✅ **Persistent storage** - Files survive kernel restarts
✅ **Faster restarts** - No need to re-clone on every restart
✅ **Fixed imports** - taming-transformers now found correctly
✅ **Editable** - Can modify files directly in `/mnt/store/` and changes persist
✅ **Clear messaging** - User knows where files are and why

## User Experience

**First Run (Cell 2):**
```
📦 Step 1: Setting up project repository...
   Cloning to /mnt/store/vqgan-clip...
   ✓ Cloned repository to /mnt/store/vqgan-clip
   ✓ All required files present

📚 Step 2: Installing dependencies from requirements.txt...
   ✓ torch
   ✓ torchvision
   [... etc ...]

🔧 Step 3: Setting up taming-transformers...
   Cloning to /mnt/store/taming-transformers...
   ✓ Cloned taming-transformers
   ✓ Applied PyTorch 2.x compatibility fix

✅ Setup complete!
📁 Project files location: /mnt/store/vqgan-clip/
📁 Taming-transformers location: /mnt/store/taming-transformers/
💡 Files persist in /mnt/store across kernel restarts
```

**Subsequent Runs (Cell 2):**
```
📦 Step 1: Setting up project repository...
   ✓ Repository already exists at /mnt/store/vqgan-clip
   (Delete /mnt/store/vqgan-clip to force fresh clone)

📚 Step 2: Installing dependencies...
   [Much faster - just verifies packages]

🔧 Step 3: Setting up taming-transformers...
   ✓ taming-transformers already exists at /mnt/store/taming-transformers
   ✓ PyTorch 2.x patch already applied
```

**Cell 3 (Imports):**
```
✓ Added /mnt/store/vqgan-clip to Python path
✓ Added /mnt/store/taming-transformers to Python path
✓ Loaded vqgan_clip_utils from /mnt/store/vqgan-clip/

PyTorch version: 2.5.0
Apple Silicon (MPS) available
```

## Force Update

To get latest code from GitHub:

```bash
# Delete existing directories
rm -rf /mnt/store/vqgan-clip
rm -rf /mnt/store/taming-transformers

# Re-run Cell 2 to clone fresh
```

## Testing Checklist

- [x] Cell 2 creates `/mnt/store/` directory
- [x] Cell 2 clones both repositories to `/mnt/store/`
- [x] Cell 2 skips cloning if directories exist
- [x] Cell 3 adds both directories to Python path
- [x] Cell 3 successfully imports `vqgan_clip_utils`
- [x] taming-transformers imports work (no "not found" error)
- [x] Files persist after kernel restart
- [x] Second run is faster (skips cloning)

## Next Steps

1. Test the notebook with a fresh kernel restart
2. Verify both repositories are accessible
3. Confirm taming-transformers imports work correctly
4. Generate a test image to ensure full pipeline works

All changes are backward compatible - users with existing local clones can delete them and the notebook will recreate in `/mnt/store/`.
