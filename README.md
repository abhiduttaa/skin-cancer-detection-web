# 🔬 DermatoAI - Skin Cancer Detection & Explainable AI Web Application
### Standalone Web Application with Grad-CAM & Render Deployment
MY website Link : https://skin-cancer-detection-web-igpo.onrender.com/

---

## 📂 Project Structure
```
cancer-detection-web/
├── app.py                      # Flask backend API & PyTorch / Grad-CAM engine
├── templates/
│   └── index.html              # Modern Tailwind CSS clinical triage dashboard
├── static/
│   └── samples/                # Bundled test cases (Melanoma, Benign Mole, BCC)
│       ├── melanoma_sample.jpg
│       ├── benign_nevus_sample.jpg
│       └── basal_cell_sample.jpg
├── models/                     # Checkpoints directory
├── requirements.txt            # CPU-optimized PyTorch dependencies for Render
├── Procfile                    # Production Gunicorn worker definition
├── render.yaml                 # Infrastructure-as-code for Render cloud
└── .gitignore
```

---

## 🚀 How to Deploy on Render (Step-by-Step)

### Step 1: Push This Project to Your GitHub
1. Go to [github.com](https://github.com) and click **New Repository**.
2. Name it `skin-cancer-detection-web`, set it to **Public**, and click **Create repository** (do not check initialize with README).
3. Open PowerShell or Terminal on your PC and run:
```bash
cd "C:\Users\ABHIJIT DUTTA\Desktop\cancer-detection-web"
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/skin-cancer-detection-web.git
git branch -M main
git push -u origin main
```

---

### Step 2: Deploy on Render (Free Cloud Tier)
1. Go to [render.com](https://render.com) and Sign In (or Sign Up with GitHub).
2. Click the blue **New +** button $\rightarrow$ select **Web Service**.
3. Choose **Build and deploy from a Git repository** $\rightarrow$ Click **Next**.
4. Select your `skin-cancer-detection-web` repository from the list.
5. Fill in the deployment settings:
   - **Name:** `dermato-ai-detector` *(or any unique name you like)*
   - **Region:** Any region near you (e.g., Singapore, Frankfurt, Oregon)
   - **Branch:** `main`
   - **Runtime:** `Python 3`
   - **Build Command:**
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command:**
     ```bash
     gunicorn app:app --workers 1 --threads 4 --timeout 120
     ```
   - **Instance Type:** Select **Free** (512 MB RAM / 0.1 CPU).
6. Click **Deploy Web Service** at the bottom!

---

### Step 3: Access Your Live Website
Render will take ~2 minutes to install dependencies and boot your application. Once done, you will get your permanent live URL:
```text
https://dermato-ai-detector.onrender.com
```
Anyone in the world—including your college professors and external examiners—can visit this link, upload skin images, and see the live prediction, malignancy risk banner, and Grad-CAM visual attention heatmap!
