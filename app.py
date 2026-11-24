import json
from pathlib import Path

import matplotlib
import numpy as np
import requests
import streamlit as st
import torch
from anomalib.models import Patchcore
from PIL import Image
from torchvision.transforms.v2.functional import (
    to_dtype,
    to_image,
)


# ============================================================
# Configuration générale
# ============================================================

st.set_page_config(
    page_title="Inspection visuelle PatchCore",
    page_icon="🔍",
    layout="wide",
)

GITHUB_USERNAME = "Nouhaila76"
GITHUB_REPOSITORY = "quality-control-streamlit"
RELEASE_TAG = "v1.0.0"

APP_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = APP_DIR / "artifacts"

MODEL_CACHE_DIR = (
    Path.home() / ".cache" / "patchcore_models"
)

MODEL_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEVICE = torch.device("cpu")

PRODUCTS = {
    "Bouteille": "bottle",
    "Câble": "cable",
    "Carrelage": "tile",
}

MODEL_URLS = {
    category: (
        f"https://github.com/{GITHUB_USERNAME}/"
        f"{GITHUB_REPOSITORY}/releases/download/"
        f"{RELEASE_TAG}/patchcore_{category}.pt"
    )
    for category in PRODUCTS.values()
}


# ============================================================
# Téléchargement et chargement des modèles
# ============================================================

def download_model(category):
    """Télécharge un modèle depuis GitHub Releases."""

    destination = (
        MODEL_CACHE_DIR
        / f"patchcore_{category}.pt"
    )

    # Un modèle valide déjà téléchargé est réutilisé.
    if (
        destination.exists()
        and destination.stat().st_size > 50_000_000
    ):
        return destination

    temporary_file = destination.with_suffix(
        ".download"
    )

    if temporary_file.exists():
        temporary_file.unlink()

    response = requests.get(
        MODEL_URLS[category],
        stream=True,
        timeout=300,
    )

    response.raise_for_status()

    with open(temporary_file, "wb") as model_file:
        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):
            if chunk:
                model_file.write(chunk)

    if temporary_file.stat().st_size < 50_000_000:
        temporary_file.unlink(
            missing_ok=True
        )
        raise RuntimeError(
            "Le modèle téléchargé semble incomplet. "
            "Vérifiez la GitHub Release."
        )

    temporary_file.replace(destination)

    return destination


@st.cache_resource(
    max_entries=1,
    show_spinner=(
        "Téléchargement et chargement "
        "du modèle PatchCore…"
    ),
)
def load_model(category):
    """Charge le modèle et le seuil d'une catégorie."""

    model_path = download_model(category)

    threshold_path = (
        ARTIFACTS_DIR
        / category
        / "threshold.json"
    )

    if not threshold_path.exists():
        raise FileNotFoundError(
            f"Fichier de seuil absent : {threshold_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    configuration = checkpoint["configuration"]

    state_dict = dict(
        checkpoint["model_state_dict"]
    )

    memory_key = "model.memory_bank"

    if memory_key not in state_dict:
        raise RuntimeError(
            "La mémoire PatchCore est absente "
            "du fichier modèle."
        )

    # La mémoire possède une taille spécifique
    # à chaque catégorie.
    memory_bank = state_dict.pop(memory_key)

    patchcore = Patchcore(
        backbone=configuration["backbone"],
        layers=configuration["layers"],
        pre_trained=False,
        coreset_sampling_ratio=configuration[
            "coreset_sampling_ratio"
        ],
        num_neighbors=configuration[
            "num_neighbors"
        ],
    )

    patchcore.load_state_dict(
        state_dict,
        strict=False,
    )

    patchcore.model.memory_bank = (
        memory_bank.to(DEVICE)
    )

    patchcore.to(DEVICE)
    patchcore.eval()
    patchcore.model.eval()

    for parameter in patchcore.parameters():
        parameter.requires_grad = False

    with open(
        threshold_path,
        "r",
        encoding="utf-8",
    ) as threshold_file:
        threshold_data = json.load(
            threshold_file
        )

    threshold = float(
        threshold_data["threshold"]
    )

    return patchcore, threshold, configuration


# ============================================================
# Prétraitement et inférence
# ============================================================

def preprocess_image(model, image):
    """Applique le prétraitement utilisé par PatchCore."""

    image_tensor = to_dtype(
        to_image(image),
        dtype=torch.float32,
        scale=True,
    )

    image_tensor = (
        model.pre_processor.transform(
            image_tensor
        )
    )

    return image_tensor.unsqueeze(0).to(DEVICE)


def create_visualizations(image, anomaly_map):
    """Crée une carte colorée et une superposition."""

    width, height = image.size

    resized_map = Image.fromarray(
        anomaly_map.astype(np.float32)
    ).resize(
        (width, height),
        resample=Image.Resampling.BILINEAR,
    )

    resized_map = np.asarray(
        resized_map,
        dtype=np.float32,
    )

    normalized_map = (
        resized_map - resized_map.min()
    ) / (
        resized_map.max()
        - resized_map.min()
        + 1e-12
    )

    colormap = matplotlib.colormaps["jet"]

    heatmap_array = (
        colormap(normalized_map)[..., :3] * 255
    ).astype(np.uint8)

    heatmap_image = Image.fromarray(
        heatmap_array
    )

    overlay_image = Image.blend(
        image,
        heatmap_image,
        alpha=0.45,
    )

    return heatmap_image, overlay_image


def inspect_product(image, category):
    """Inspecte une image avec PatchCore."""

    model, threshold, configuration = (
        load_model(category)
    )

    input_tensor = preprocess_image(
        model,
        image,
    )

    with torch.inference_mode():
        outputs = model.model(input_tensor)

    score = float(
        outputs.pred_score
        .detach()
        .flatten()[0]
        .cpu()
    )

    anomaly_map = (
        outputs.anomaly_map
        .detach()
        .squeeze()
        .cpu()
        .numpy()
    )

    heatmap, overlay = create_visualizations(
        image,
        anomaly_map,
    )

    return {
        "score": score,
        "threshold": threshold,
        "is_anomalous": score >= threshold,
        "heatmap": heatmap,
        "overlay": overlay,
        "memory_vectors": configuration[
            "memory_vectors"
        ],
    }


# ============================================================
# Interface Streamlit
# ============================================================

st.title("🔍 Inspection visuelle industrielle")
st.subheader(
    "Détection d’anomalies avec PatchCore"
)

st.markdown(
    """
Cette application inspecte trois catégories de produits :

- bouteilles ;
- câbles ;
- carrelage.

Sélectionnez le produit correspondant, importez une image,
puis lancez l’inspection.
"""
)

with st.sidebar:
    st.header("Configuration")

    selected_product = st.selectbox(
        "Type de produit",
        options=list(PRODUCTS.keys()),
    )

    category = PRODUCTS[selected_product]

    st.warning(
        "Le produit sélectionné doit correspondre "
        "au produit présent dans l’image."
    )

    st.markdown(
        """
### Modèle

- PatchCore
- Wide ResNet-50-2
- Coreset de 1 %
- Résolution de 256 × 256
- Poids du réseau figés
"""
    )

uploaded_file = st.file_uploader(
    "Importer une image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "webp",
    ],
)

if uploaded_file is None:
    st.info(
        "Importez une image pour commencer."
    )

else:
    input_image = Image.open(
        uploaded_file
    ).convert("RGB")

    image_column, action_column = st.columns(2)

    with image_column:
        st.image(
            input_image,
            caption="Image à inspecter",
            use_container_width=True,
        )

    with action_column:
        st.markdown(
            f"### Produit : {selected_product}"
        )

        st.write(
            "Le premier chargement d’un modèle "
            "peut prendre quelques secondes."
        )

        inspect_button = st.button(
            "Lancer l’inspection",
            type="primary",
            use_container_width=True,
        )

    if inspect_button:
        try:
            with st.spinner(
                "Analyse de l’image en cours…"
            ):
                result = inspect_product(
                    input_image,
                    category,
                )

            if result["is_anomalous"]:
                st.error(
                    "🔴 Produit à vérifier"
                )
            else:
                st.success(
                    "🟢 Produit considéré comme conforme"
                )

            metric_1, metric_2, metric_3 = (
                st.columns(3)
            )

            metric_1.metric(
                "Score d’anomalie",
                f"{result['score']:.3f}",
            )

            metric_2.metric(
                "Seuil de décision",
                f"{result['threshold']:.3f}",
            )

            metric_3.metric(
                "Vecteurs de référence",
                f"{result['memory_vectors']:,}",
            )

            heatmap_column, overlay_column = (
                st.columns(2)
            )

            with heatmap_column:
                st.image(
                    result["heatmap"],
                    caption="Carte d’anomalie",
                    use_container_width=True,
                )

            with overlay_column:
                st.image(
                    result["overlay"],
                    caption="Zones suspectes",
                    use_container_width=True,
                )

            st.caption(
                "Le score est une distance par rapport "
                "aux produits conformes. Il ne s’agit "
                "pas d’une probabilité."
            )

        except requests.HTTPError:
            st.error(
                "Impossible de télécharger le modèle. "
                "Vérifiez le nom du dépôt, le tag "
                "v1.0.0 et les fichiers de la Release."
            )

        except Exception as error:
            st.exception(error)

st.divider()

st.markdown(
    """
### Limites du prototype

Cette application a été évaluée sur MVTec AD dans des
conditions de prise de vue contrôlées. Une variation
importante du cadrage, de l’éclairage ou du fond peut
être signalée comme une anomalie.

La carte de chaleur aide à localiser une zone suspecte,
mais ne constitue pas une segmentation industrielle
certifiée.
"""
)
