# Fixing the 404 Error for /explain Endpoint

## Quick Fix

The 404 error means the `/explain` endpoint isn't registered. Here's how to fix it:

### Step 1: Verify XAI Setup

```bash
python test_xai_setup.py
```

This will check:
- ✓ All required files exist
- ✓ Dependencies are installed
- ✓ Model can be loaded
- ✓ XAI functions work

### Step 2: Restart Flask Server

**Important:** Stop the current Flask server (Ctrl+C) and restart it:

```bash
python app.py
```

**Look for this in the console output:**
```
✓ XAI module loaded successfully
✓ XAI routes registered on device: cuda
  Available endpoints:
    POST /explain - Generate XAI explanations
    POST /explain/batch - Batch explanations
    GET  /explain/health - XAI health check
```

If you DON'T see this, proceed to Step 3.

### Step 3: Check for Errors

If XAI routes aren't loading, check the console for error messages:

**Common Issue 1: Missing Dependencies**
```
⚠️  Could not load XAI module: No module named 'captum'
```

**Fix:**
```bash
pip install -r requirements_xai.txt
```

**Common Issue 2: Model Not Found**
```
⚠️  Classification model not found: classification_models/best_oct_classifier.pth
```

**Fix:** Train the classification model first:
```bash
jupyter notebook OCT_Classification.ipynb
# Run all cells to train the model
```

**Common Issue 3: Import Error in xai_utils.py**
```
⚠️  Could not load XAI module: ImportError in xai_utils.py
```

**Fix:** Check that all files are in the project root directory

### Step 4: Test the Endpoint Manually

Once the server is running with XAI loaded, test it:

```bash
# In a new terminal
curl http://localhost:5000/explain/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "xai_enabled": true,
  "device": "cuda",
  "supported_methods": ["Grad-CAM", "LIME", "Integrated Gradients"]
}
```

If you get 404, the routes aren't registered. If you get the JSON response, XAI is working!

### Step 5: Test in Browser

1. Go to http://localhost:5000
2. Upload an OCT image
3. Click "Classify Disease"
4. Wait for classification results
5. Click "Explain This Prediction"

If you still get 404, check the browser console (F12) for the exact error.

---

## Detailed Troubleshooting

### Check 1: Verify app.py has XAI integration

Open `app.py` and look for this around line 565:

```python
if cls_loaded and classification_model is not None:
    xai_loaded = register_xai()
    if xai_loaded:
        print("✓ XAI module loaded successfully")
```

If this code is missing, the XAI routes won't be registered.

### Check 2: Verify files exist

```bash
ls -la xai_*.py
ls -la requirements_xai.txt
ls -la OCT_XAI.ipynb
```

All these files should exist.

### Check 3: Test imports manually

```python
python
>>> from xai_utils import generate_gradcam
>>> from xai_routes import register_xai_routes
>>> print("Imports work!")
```

If imports fail, you have a dependency issue.

### Check 4: Check Flask blueprint registration

The `/explain` endpoint uses Flask blueprints. In `xai_routes.py`, verify:

```python
xai_bp = Blueprint('xai', __name__)

@xai_bp.route('/explain', methods=['POST'])
def explain():
    # ...
```

### Check 5: Console output when starting app

When you run `python app.py`, you should see:

```
Loading model from: classification_models/best_oct_classifier.pth
✓ Model loaded successfully!
  Validation accuracy: XX.XX%
✓ XAI module loaded successfully
✓ XAI routes registered on device: cuda
  Available endpoints:
    POST /explain - Generate XAI explanations
```

---

## Alternative: Use Jupyter Notebook Instead

If web interface doesn't work, use the Jupyter notebook:

```bash
jupyter notebook OCT_XAI.ipynb
```

Then:
1. Run cells 1-8 to set up
2. Update image path in cell 12
3. Run cell 12 to generate explanations

This bypasses the Flask server entirely.

---

## Still Not Working?

### Nuclear Option: Reinstall Everything

```bash
# Uninstall XAI dependencies
pip uninstall captum lime -y

# Reinstall
pip install -r requirements_xai.txt

# Restart Python interpreter completely
# Then restart Flask server
python app.py
```

### Check Python Path

Make sure you're in the right directory:

```bash
cd /path/to/oct_major_project
python app.py
```

### Check Port Conflicts

If port 5000 is already in use:

```bash
# Kill existing Flask processes
# Windows:
taskkill /F /IM python.exe

# Then restart
python app.py
```

---

## Debug Mode

Enable more verbose logging:

### In app.py, add at the top:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### In xai_routes.py, add debugging:

```python
@xai_bp.route('/explain', methods=['POST'])
def explain():
    print("==> /explain endpoint hit!")  # Add this
    # ... rest of code
```

---

## Contact Info

If none of this works, provide:

1. Output of `python test_xai_setup.py`
2. Full console output when starting `python app.py`
3. Browser console errors (F12 → Console tab)
4. Python version: `python --version`
5. Operating system

---

## Success Indicators

You'll know it's working when:

✅ Console shows "✓ XAI module loaded successfully"  
✅ Console shows "✓ XAI routes registered"  
✅ `curl http://localhost:5000/explain/health` returns JSON  
✅ Web interface shows "Explain This Prediction" button  
✅ Clicking button shows loading message  
✅ After ~1 minute, XAI results appear  

---

**Most common cause:** Flask server wasn't restarted after adding XAI files. Always restart!

**Second most common:** Dependencies not installed. Run `pip install -r requirements_xai.txt`

