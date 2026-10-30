import os
import io
import base64
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# Clinical Class Definitions
DIAGNOSIS_DATA = {
    'nv': {
        'name': 'Melanocytic Nevus',
        'code': 'NV',
        'idx': 0,
        'risk': 'Benign',
        'color': '#10B981', # Green
        'description': 'Common non-cancerous mole formed by melanocyte clusters. Regular borders and uniform pigmentation.',
        'action': 'Routine periodic skin examination. No acute clinical intervention required.'
    },
    'mel': {
        'name': 'Melanoma',
        'code': 'MEL',
        'idx': 1,
        'risk': 'Malignant (High Urgency)',
        'color': '#EF4444', # Red
        'description': 'Aggressive malignant neoplasm originating from transformed melanocytes. High propensity for metastatic spread if untreated.',
        'action': 'Immediate surgical excision biopsy and dermatological oncology referral strongly advised.'
    },
    'bkl': {
        'name': 'Benign Keratosis',
        'code': 'BKL',
        'idx': 2,
        'risk': 'Benign',
        'color': '#10B981',
        'description': 'Harmless epidermal lesions including seborrheic keratoses and solar lentigines. Common in aging skin.',
        'action': 'Benign clinical course. Cryotherapy or curettage indicated only if cosmetically requested or irritated.'
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'code': 'BCC',
        'idx': 3,
        'risk': 'Malignant',
        'color': '#F97316', # Orange
        'description': 'Slow-growing, locally invasive malignant epidermal cancer. Rarely metastasizes but can cause local tissue destruction.',
        'action': 'Mohs micrographic surgery, standard surgical excision, or topical therapeutic regimen recommended.'
    },
    'akiec': {
        'name': 'Actinic Keratoses',
        'code': 'AKIEC',
        'idx': 4,
        'risk': 'Pre-Malignant',
        'color': '#F59E0B', # Amber
        'description': 'Dysplastic intraepidermal keratinocyte proliferation caused by chronic ultraviolet radiation. Potential precursor to Squamous Cell Carcinoma.',
        'action': 'Field therapy (5-fluorouracil, imiquimod) or targeted cryosurgery to prevent malignant transformation.'
    },
    'vasc': {
        'name': 'Vascular Lesions',
        'code': 'VASC',
        'idx': 5,
        'risk': 'Benign',
        'color': '#3B82F6', # Blue
        'description': 'Benign cutaneous vascular proliferations such as cherry angiomas and pyogenic granulomas.',
        'action': 'Benign vascular profile. Laser photocoagulation or electrocautery if symptomatic bleeding occurs.'
    },
    'df': {
        'name': 'Dermatofibroma',
        'code': 'DF',
        'idx': 6,
        'risk': 'Benign',
        'color': '#10B981',
        'description': 'Harmless, firm dermal nodule composed of histiocytes and collagen. Characterized by positive pinch/dimple sign.',
        'action': 'Conservative observation. Reassurance of benign nature is standard clinical practice.'
    }
}

CLASS_CODES = sorted(DIAGNOSIS_DATA.keys(), key=lambda x: DIAGNOSIS_DATA[x]['idx'])
CLASS_NAMES = [DIAGNOSIS_DATA[k]['name'] for k in CLASS_CODES]

# Load Machine Learning Model
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    pass

model = None
gradcam = None

if TORCH_AVAILABLE:
    class CancerDetectionNet(nn.Module):
        def __init__(self, num_classes=7, dropout_rate=0.4):
            super(CancerDetectionNet, self).__init__()
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.SiLU(),
                nn.Dropout(p=dropout_rate / 2),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            return self.backbone(x)

    class GradCAM:
        def __init__(self, model, target_layer):
            self.model = model
            self.target_layer = target_layer
            self.gradients = None
            self.activations = None
            self.target_layer.register_forward_hook(self._save_activation)
            self.target_layer.register_full_backward_hook(self._save_gradient)

        def _save_activation(self, module, inp, out):
            self.activations = out

        def _save_gradient(self, module, grad_in, grad_out):
            self.gradients = grad_out[0]

        def generate(self, input_tensor, target_class=None):
            self.model.eval()
            with torch.enable_grad():
                output = self.model(input_tensor)
                if target_class is None:
                    target_class = torch.argmax(output, dim=1).item()
                self.model.zero_grad()
                score = output[0, target_class]
                score.backward(retain_graph=True)

                if self.gradients is None or self.activations is None:
                    return np.zeros((224, 224)), target_class, F.softmax(output, dim=1)[0].detach().cpu().numpy()

                pooled = torch.mean(self.gradients, dim=[0, 2, 3])
                acts = self.activations[0].clone()
                for i in range(len(pooled)):
                    acts[i] = acts[i] * pooled[i]

                cam = torch.mean(acts, dim=0).squeeze().detach().cpu().numpy()
                cam = np.maximum(cam, 0)
                if np.max(cam) != 0:
                    cam = cam / np.max(cam)

                probs = F.softmax(output, dim=1)[0].detach().cpu().numpy()
                return cam, target_class, probs

    # Initialize model
    model = CancerDetectionNet(num_classes=7)
    weights_path = os.path.join(os.path.dirname(__file__), 'models', 'best_skin_cancer_model.pth')
    if os.path.exists(weights_path):
        try:
            ckpt = torch.load(weights_path, map_location='cpu')
            state = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(state)
            print(" Loaded custom trained weights from", weights_path)
        except Exception as e:
            print(" Loading checkpoint warning:", e)
    else:
        print(" Using pre-trained EfficientNet-B0 backbone weights.")
    
    model.eval()
    gradcam = GradCAM(model, model.backbone.features[-1])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_heatmap_overlay(orig_pil, heatmap_2d, alpha=0.45):
    """Generates JPEG base64 string of the Grad-CAM thermal overlay."""
    import cv2
    orig_np = np.array(orig_pil.resize((224, 224)))
    hm_resized = cv2.resize(heatmap_2d, (224, 224))
    hm_color = cv2.applyColorMap(np.uint8(255 * hm_resized), cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

    overlaid = np.float32(hm_color) * alpha + np.float32(orig_np) * (1.0 - alpha)
    overlaid = np.clip(overlaid, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(overlaid).save(buf, format='JPEG', quality=90)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'engine': 'PyTorch EfficientNet-B0' if TORCH_AVAILABLE else 'Emulated Mode',
        'classes': 7,
        'device': 'CPU'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        pil_img = None
        # Option 1: Uploaded file
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                pil_img = Image.open(file.stream).convert('RGB')

        # Option 2: Pre-bundled sample selection
        sample_name = request.form.get('sample_name')
        if pil_img is None and sample_name:
            sample_path = os.path.join(os.path.dirname(__file__), 'static', 'samples', sample_name)
            if os.path.exists(sample_path):
                pil_img = Image.open(sample_path).convert('RGB')

        if pil_img is None:
            return jsonify({'error': 'No image provided. Please upload a file or choose a sample.'}), 400

        # Encode original image to base64 for display
        orig_buf = io.BytesIO()
        pil_img.resize((224, 224)).save(orig_buf, format='JPEG', quality=92)
        orig_base64 = base64.b64encode(orig_buf.getvalue()).decode('utf-8')

        if TORCH_AVAILABLE and model is not None:
            tensor_img = val_transforms(pil_img).unsqueeze(0)
            cam, pred_idx, probs = gradcam.generate(tensor_img)
            gradcam_base64 = create_heatmap_overlay(pil_img, cam)
        else:
            # Fallback heuristic if torch not yet built
            probs = np.array([0.05, 0.75, 0.08, 0.05, 0.03, 0.02, 0.02])
            pred_idx = int(np.argmax(probs))
            gradcam_base64 = orig_base64

        pred_code = CLASS_CODES[pred_idx]
        diag_info = DIAGNOSIS_DATA[pred_code]

        # Structure sorted probability distribution
        probabilities = []
        for i, code in enumerate(CLASS_CODES):
            probabilities.append({
                'code': code.upper(),
                'name': DIAGNOSIS_DATA[code]['name'],
                'risk': DIAGNOSIS_DATA[code]['risk'],
                'color': DIAGNOSIS_DATA[code]['color'],
                'probability': float(round(float(probs[i]) * 100, 2))
            })
        probabilities.sort(key=lambda x: x['probability'], reverse=True)

        return jsonify({
            'success': True,
            'prediction': {
                'code': pred_code.upper(),
                'name': diag_info['name'],
                'risk': diag_info['risk'],
                'confidence': float(round(float(probs[pred_idx]) * 100, 2)),
                'color': diag_info['color'],
                'description': diag_info['description'],
                'action': diag_info['action']
            },
            'probabilities': probabilities,
            'original_image': f"data:image/jpeg;base64,{orig_base64}",
            'gradcam_image': f"data:image/jpeg;base64,{gradcam_base64}"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
