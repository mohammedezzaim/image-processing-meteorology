import os
import cv2
import numpy as np
from tkinter import messagebox
from datetime import datetime
from fpdf import FPDF
import shutil

def clear_data_folder(app):
    """Supprime tous les fichiers du dossier de données"""
    if os.path.exists(app.data_folder):
        for filename in os.listdir(app.data_folder):
            file_path = os.path.join(app.data_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Échec de la suppression de {file_path}. Raison: {e}")

def generate_report(app):
    """Générer un rapport d'analyse"""
    report = create_analysis_report(app)
    report_path = os.path.join(app.output_folder, f"rapport_analyse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        messagebox.showinfo("Succès", f"Rapport généré avec succès:\n{report_path}")
    except Exception as e:
        messagebox.showerror("Erreur", f"Impossible de générer le rapport: {str(e)}")

def generate_detailed_report(app):
    """Génère un rapport PDF détaillé avec graphiques"""
    if not app.current_images:
        messagebox.showwarning("Attention", "Aucune image chargée")
        return

    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Rapport d'Analyse Météorologique", ln=1, align='C')
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", ln=1)
        pdf.ln(10)

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Image Analysée", ln=1)
        pdf.set_font("Arial", size=12)

        image_path = os.path.join(app.data_folder, app.current_images[app.current_index])
        image = cv2.imread(image_path)
        temp_img_path = os.path.join(app.output_folder, "temp_report_image.jpg")
        cv2.imwrite(temp_img_path, image)
        pdf.image(temp_img_path, x=10, w=180)
        pdf.ln(5)

        if hasattr(app, 'cloud_properties') and app.cloud_properties:
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="Propriétés des Nuages", ln=1)
            pdf.set_font("Arial", size=10)

            pdf.cell(40, 10, "Nuage", border=1)
            pdf.cell(40, 10, "Aire (px)", border=1)
            pdf.cell(40, 10, "Périmètre (px)", border=1)
            pdf.cell(40, 10, "Circularité", border=1)
            pdf.ln()

            for prop in app.cloud_properties:
                pdf.cell(40, 10, str(prop['id']), border=1)
                pdf.cell(40, 10, f"{prop['area']:.1f}", border=1)
                pdf.cell(40, 10, f"{prop['perimeter']:.1f}", border=1)
                pdf.cell(40, 10, f"{prop['circularity']:.2f}", border=1)
                pdf.ln()

            pdf.ln(10)

        if hasattr(app, 'temporal_analysis_results') and app.temporal_analysis_results:
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="Analyse Temporelle", ln=1)
            pdf.set_font("Arial", size=12)

            if 'mean_movement' in app.temporal_analysis_results:
                pdf.cell(200, 10,
                         txt=f"Mouvement moyen des nuages: {app.temporal_analysis_results['mean_movement']:.2f} px/image",
                         ln=1)
                pdf.cell(200, 10,
                         txt=f"Direction prédominante: {app.temporal_analysis_results['dominant_direction']}",
                         ln=1)

            if 'cyclone_tracking' in app.temporal_analysis_results:
                pdf.cell(200, 10,
                         txt=f"Déplacement moyen du cyclone: {app.temporal_analysis_results['cyclone_tracking']['mean_movement']:.2f} px/image",
                         ln=1)
                pdf.cell(200, 10,
                         txt=f"Direction du cyclone: {app.temporal_analysis_results['cyclone_tracking']['direction']}",
                         ln=1)

            pdf.ln(10)

        if hasattr(app, 'histogram_fig'):
            histogram_path = os.path.join(app.output_folder, "temp_histogram.png")
            app.histogram_fig.savefig(histogram_path)
            pdf.image(histogram_path, x=10, w=180)
            pdf.ln(5)

        

        report_path = os.path.join(app.output_folder,
                                  f"rapport_detaille_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        pdf.output(report_path)

        messagebox.showinfo("Succès", f"Rapport PDF généré avec succès:\n{report_path}")

        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        if os.path.exists(histogram_path):
            os.remove(histogram_path)
        

    except Exception as e:
        messagebox.showerror("Erreur", f"Impossible de générer le rapport PDF: {str(e)}")

def create_analysis_report(app):
    """Créer le contenu du rapport d'analyse"""
    report = f"""
RAPPORT D'ANALYSE - TRAITEMENT D'IMAGES MÉTÉOROLOGIQUES
========================================================

Date et heure: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
Système: Traitement d'Images Météorologiques GOES-19
Université Cadi Ayyad - Master IAII

INFORMATIONS GÉNÉRALES
---------------------
Nombre total d'images analysées: {len(app.current_images)}
Dossier source: {app.data_folder}
Dossier de sortie: {app.output_folder}

IMAGES TRAITÉES
--------------
"""
    for i, image_name in enumerate(app.current_images):
        image_path = os.path.join(app.data_folder, image_name)
        if os.path.exists(image_path):
            try:
                image = cv2.imread(image_path)
                height, width = image.shape[:2]
                file_size = os.path.getsize(image_path)

                report += f"""
Image {i+1}: {image_name}
  - Dimensions: {width} × {height} pixels
  - Taille: {file_size/1024:.1f} KB
  - Résolution: {width*height/1000000:.1f} Mpx
"""
            except:
                report += f"\nImage {i+1}: {image_name} (erreur de lecture)\n"

    report += f"""

TECHNIQUES DE TRAITEMENT APPLIQUÉES
----------------------------------
Nombre de techniques utilisées: {len(app.processing_results)}

Techniques disponibles dans le système:
1. Égalisation d'histogramme - Amélioration du contraste
2. Filtrage gaussien - Réduction du bruit
3. Détection de contours (Canny) - Identification des structures
4. Opérations morphologiques - Nettoyage et structuration
5. Extraction de nuages - Identification des masses nuageuses
6. Détection de cyclones - Localisation des formations spiralées
7. Analyse des précipitations - Classification par intensité
8. Flot optique - Analyse du mouvement atmosphérique
9. Segmentation multi-seuils - Classification des régions

RÉSULTATS D'ANALYSE
------------------
"""
    if "cloud_extraction" in app.processing_results:
        report += "✓ Extraction de nuages effectuée\n"
    if "cyclone_detection" in app.processing_results:
        report += "✓ Détection de cyclones effectuée\n"
    if "precipitation" in app.processing_results:
        report += "✓ Analyse des précipitations effectuée\n"
    if "optical_flow" in app.processing_results:
        report += "✓ Calcul du flot optique effectué\n"

    if hasattr(app, 'cloud_properties') and app.cloud_properties:
        report += "\nPROPRIÉTÉS DES NUAGES\n"
        for prop in app.cloud_properties:
            report += (f"Nuage {prop['id']}: Aire={prop['area']:.1f} px, "
                      f"Périmètre={prop['perimeter']:.1f} px, "
                      f"Circularité={prop['circularity']:.2f}\n")

    if hasattr(app, 'temporal_analysis_results') and app.temporal_analysis_results:
        report += "\nANALYSE TEMPORELLE\n"
        if 'mean_movement' in app.temporal_analysis_results:
            report += (f"Mouvement moyen des nuages: {app.temporal_analysis_results['mean_movement']:.2f} px/image\n"
                      f"Direction prédominante: {app.temporal_analysis_results['dominant_direction']}\n")

        if 'cyclone_tracking' in app.temporal_analysis_results:
            report += (f"Déplacement moyen du cyclone: {app.temporal_analysis_results['cyclone_tracking']['mean_movement']:.2f} px/image\n"
                      f"Direction du cyclone: {app.temporal_analysis_results['cyclone_tracking']['direction']}\n")

    report += f"""

RECOMMANDATIONS
--------------
1. Surveillance continue des formations cycloniques détectées
2. Analyse temporelle pour le suivi des déplacements
3. Corrélation avec les données météorologiques terrain
4. Utilisation des résultats pour la modélisation prédictive

MÉTHODOLOGIE TECHNIQUE
---------------------
Les techniques implémentées sont basées sur les méthodes classiques
de traitement d'images en météorologie:

- Filtrage homomorphique pour la correction d'illumination
- Morphologie mathématique pour l'extraction de structures
- Calcul de flot optique pour l'estimation du mouvement
- Segmentation adaptative pour la classification

CONCLUSION
----------
Le système a traité {len(app.current_images)} image(s) avec succès.
{len(app.processing_results)} technique(s) de traitement ont été appliquées.

Ce rapport a été généré automatiquement par le système de traitement
d'images météorologiques développé dans le cadre du projet académique.

========================================================
Fin du rapport
"""
    return report

def save_results(app):
    """Sauvegarder les résultats de traitement"""
    if not app.processing_results:
        messagebox.showwarning("Attention", "Aucun résultat de traitement à sauvegarder")
        return

    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for technique_name, result_image in app.processing_results.items():
            filename = f"{technique_name}_{timestamp}.jpg"
            filepath = os.path.join(app.output_folder, filename)
            cv2.imwrite(filepath, result_image)

        messagebox.showinfo("Succès", f"{len(app.processing_results)} résultat(s) sauvegardé(s) dans:\n{app.output_folder}")
    except Exception as e:
        messagebox.showerror("Erreur", f"Impossible de sauvegarder: {str(e)}")