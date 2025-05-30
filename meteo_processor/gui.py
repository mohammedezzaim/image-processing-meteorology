import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from processing import apply_histogram_equalization, apply_gaussian_filter, apply_canny_edges, \
                      apply_morphological_ops, apply_cloud_extraction, apply_cyclone_detection, \
                      apply_precipitation_analysis, apply_optical_flow, apply_homomorphic_filter, \
                      apply_multi_threshold_segmentation
from utils import clear_data_folder, generate_report, generate_detailed_report, save_results
from config import DATA_FOLDER, OUTPUT_FOLDER

class MeteorologicalImageProcessor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Système Avancé de Traitement d'Images Météorologiques")
        self.root.configure(bg="#2c3e50")
        self.root.geometry("1400x900")

        # Variables globales
        self.data_folder = DATA_FOLDER
        self.output_folder = OUTPUT_FOLDER
        self.current_images = []
        self.current_index = 0
        self.is_playing = False
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.previous_gray = None
        self.processing_results = {}
        self.cloud_properties = []
        self.temporal_analysis_results = {}
        self.myvariabl = ""
        self.fullscreen_state = False

        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.clear_data_folder()
        self.create_directories()
        self.setup_ui()
        self.load_images()

    def create_directories(self):
        """Créer les dossiers nécessaires pour le projet"""
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def setup_ui(self):
        """Configuration de l'interface utilisateur principale"""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#2c3e50', foreground='white')
        style.configure('Subtitle.TLabel', font=('Arial', 12), background='#2c3e50', foreground='white')

        header_frame = tk.Frame(self.root, bg="#2c3e50")
        header_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(header_frame, text="Système Avancé de Traitement d'Images Météorologiques",
                 font=('Arial', 18, 'bold'), bg="#2c3e50", fg="white").pack()
        tk.Label(header_frame, text="Analyse automatique des données satellitaires GOES-19 - Master IAII",
                 font=('Arial', 12), bg="#2c3e50", fg="#ecf0f1").pack()

        main_container = tk.Frame(self.root, bg="#2c3e50")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        control_column = tk.Frame(main_container, bg="#34495e", relief=tk.RAISED, bd=2, width=350)
        control_column.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        control_column.pack_propagate(False)

        visualization_column = tk.Frame(main_container, bg="#34495e", relief=tk.RAISED, bd=2)
        visualization_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.setup_control_panel(control_column)
        self.setup_display_panel(visualization_column)
        self.setup_advanced_features(control_column)
        self.setup_scientific_visualization(visualization_column)

    def setup_control_panel(self, parent):
        """Configuration du panneau de contrôle"""
        tk.Label(parent, text="Panneau de Contrôle", font=('Arial', 14, 'bold'),
                 bg="#34495e", fg="white").pack(pady=10)

        data_frame = tk.LabelFrame(parent, text="Données Satellitaires", bg="#34495e", fg="white",
                                  font=('Arial', 10, 'bold'))
        data_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(data_frame, text="📁 Charger Images", command=self.load_local_images,
                  bg="#3498db", fg="white", font=('Arial', 10)).pack(fill=tk.X, padx=5, pady=2)

        processing_frame = tk.LabelFrame(parent, text="Techniques de Traitement", bg="#34495e", fg="white",
                                        font=('Arial', 10, 'bold'))
        processing_frame.pack(fill=tk.X, padx=10, pady=5)

        techniques = [
            ("📊 Égalisation d'Histogramme", self.apply_histogram_equalization),
            ("🔍 Filtrage Gaussien", self.apply_gaussian_filter),
            ("📐 Détection de Contours (Canny)", self.apply_canny_edges),
            ("🔄 Opérations Morphologiques", self.apply_morphological_ops),
            ("☁️ Extraction de Nuages", self.apply_cloud_extraction),
            ("🌀 Détection de Cyclones", self.apply_cyclone_detection),
            ("💧 Analyse des Précipitations", self.apply_precipitation_analysis),
            ("🎯 Flot Optique", self.apply_optical_flow),
            (" Filtre Homomorphique", self.apply_homomorphic_filter),
            ("📈 Segmentation Multi-seuils", self.apply_multi_threshold_segmentation)
        ]

        for text, command in techniques:
            tk.Button(processing_frame, text=text, command=command,
                      bg="#e74c3c", fg="white", font=('Arial', 9)).pack(fill=tk.X, padx=5, pady=1)

        playback_frame = tk.LabelFrame(parent, text="Contrôles de Lecture", bg="#34495e", fg="white",
                                      font=('Arial', 10, 'bold'))
        playback_frame.pack(fill=tk.X, padx=10, pady=5)

        control_buttons = tk.Frame(playback_frame, bg="#34495e")
        control_buttons.pack(fill=tk.X, padx=5, pady=5)

        tk.Button(control_buttons, text="⏮️", command=self.first_image,
                  bg="#9b59b6", fg="white", width=3).pack(side=tk.LEFT, padx=1)
        tk.Button(control_buttons, text="⏪", command=self.previous_image,
                  bg="#9b59b6", fg="white", width=3).pack(side=tk.LEFT, padx=1)
        tk.Button(control_buttons, text="⏯️", command=self.toggle_play,
                  bg="#9b59b6", fg="white", width=3).pack(side=tk.LEFT, padx=1)
        tk.Button(control_buttons, text="⏩", command=self.next_image,
                  bg="#9b59b6", fg="white", width=3).pack(side=tk.LEFT, padx=1)
        tk.Button(control_buttons, text="⏭️", command=self.last_image,
                  bg="#9b59b6", fg="white", width=3).pack(side=tk.LEFT, padx=1)

        analysis_frame = tk.LabelFrame(parent, text="Analyse et Export", bg="#34495e", fg="white",
                                      font=('Arial', 10, 'bold'))
        analysis_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(analysis_frame, text="📊 Générer Rapport", command=self.generate_report,
                  bg="#f39c12", fg="white", font=('Arial', 10)).pack(fill=tk.X, padx=5, pady=2)
        tk.Button(analysis_frame, text="📝 Rapport Détaillé", command=self.generate_detailed_report,
                  bg="#16a085", fg="white", font=('Arial', 10)).pack(fill=tk.X, padx=5, pady=2)
        tk.Button(analysis_frame, text="💾 Sauvegarder Résultats", command=self.save_results,
                  bg="#f39c12", fg="white", font=('Arial', 10)).pack(fill=tk.X, padx=5, pady=2)

        tk.Button(parent, text="❌ Quitter", command=self.quit_app,
                  bg="#c0392b", fg="white", font=('Arial', 12, 'bold')).pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

    def setup_display_panel(self, parent):
        """Configuration du panneau d'affichage"""
        self.image_frame = tk.Frame(parent, bg="#2c3e50")
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.display_container = tk.Frame(self.image_frame, bg="#2c3e50")
        self.display_container.pack(fill=tk.BOTH, expand=True)

        self.fullscreen_btn = tk.Button(self.display_container, text="⛶",
                                       command=self.toggle_fullscreen,
                                       bg="#2c3e50", fg="white",
                                       font=('Arial', 12), bd=0)
        self.fullscreen_btn.pack(anchor=tk.NE, padx=5, pady=5)

        self.image_label = tk.Label(self.image_frame, bg="#2c3e50", text="Chargez des images pour commencer",
                                   fg="white", font=('Arial', 16))
        self.image_label.pack(expand=True)

        progress_frame = tk.Frame(parent, bg="#34495e")
        progress_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(progress_frame, text="Progression:", bg="#34495e", fg="white").pack(side=tk.LEFT)
        self.progress_var = tk.StringVar(value="0/0")
        tk.Label(progress_frame, textvariable=self.progress_var, bg="#34495e", fg="white").pack(side=tk.RIGHT)

        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=10)

    def setup_advanced_features(self, parent):
        """Configuration des paramètres avancés avec onglets"""
        advanced_notebook = ttk.Notebook(parent)
        advanced_notebook.pack(fill=tk.X, padx=5, pady=5)

        processing_tab = tk.Frame(advanced_notebook, bg="#34495e")
        advanced_notebook.add(processing_tab, text="Paramètres de Traitement")

        flow_frame = tk.LabelFrame(processing_tab, text="Flux Optique",
                                  bg="#34495e", fg="white", font=('Arial', 10))
        flow_frame.pack(fill=tk.X, padx=5, pady=2)

        tk.Label(flow_frame, text="Seuil de détection:", bg="#34495e", fg="white").pack()
        self.optical_flow_threshold = tk.Scale(flow_frame, from_=0, to=10, resolution=0.1,
                                              orient=tk.HORIZONTAL, bg="#2c3e50", fg="white")
        self.optical_flow_threshold.set(2.0)
        self.optical_flow_threshold.pack(fill=tk.X, padx=5, pady=2)

        style = ttk.Style()
        style.configure('TNotebook', background='#2c3e50', borderwidth=0)
        style.configure('TNotebook.Tab', background='#34495e', foreground='white',
                        padding=[10, 5], font=('Arial', 9, 'bold'))
        style.map('TNotebook.Tab', background=[('selected', '#2c3e50')])

    def setup_scientific_visualization(self, parent):
        """Configuration des visualisations scientifiques"""
        self.visualization_frame = tk.Frame(parent, bg="#2c3e50")
        self.visualization_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.visualization_notebook = ttk.Notebook(self.visualization_frame)
        self.visualization_notebook.pack(fill=tk.BOTH, expand=True)

        self.histogram_tab = tk.Frame(self.visualization_notebook, bg="#2c3e50")
        self.visualization_notebook.add(self.histogram_tab, text="Histogramme")

        

        self.histogram_fig, self.histogram_ax = plt.subplots(figsize=(5, 3))
        self.cloud_properties_fig, self.cloud_properties_ax = plt.subplots(figsize=(5, 3))

        self.histogram_canvas = FigureCanvasTkAgg(self.histogram_fig, master=self.histogram_tab)
        self.histogram_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


        

    def toggle_fullscreen(self, event=None):
        """Basculer entre le mode plein écran et normal"""
        self.fullscreen_state = not self.fullscreen_state

        if self.myvariabl == "Détection de contours Canny":
            self.apply_canny_edges()
        elif self.myvariabl == "Détection de cyclones":
            self.apply_cyclone_detection()
        elif self.myvariabl == "Opérations morphologiques":
            self.apply_morphological_ops()
        elif self.myvariabl == "Extraction de nuages":
            self.apply_cloud_extraction()
        elif self.myvariabl == "Analyse des précipitations":
            self.apply_precipitation_analysis()
        elif self.myvariabl == "Flot optique":
            self.apply_optical_flow()
        elif self.myvariabl == "Segmentation Multi-seuils":
            self.apply_multi_threshold_segmentation()
        elif self.myvariabl == "Filtrage Homomorphique":
            self.apply_homomorphic_filter()
        else:
            self.update_progress_bar()
            self.display_current_image()

    def load_images(self):
        """Charger les images depuis le dossier de données"""
        if os.path.exists(self.data_folder):
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            self.current_images = [f for f in os.listdir(self.data_folder)
                                  if any(f.lower().endswith(ext) for ext in image_extensions)]
            self.current_images.sort()

            if self.current_images:
                self.update_progress_bar()
                self.display_current_image()

    def load_local_images(self):
        """Charger des images depuis l'explorateur de fichiers"""
        files = filedialog.askopenfilenames(
            title="Sélectionner des images météorologiques",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("Tous les fichiers", "*.*")
            ]
        )

        if files:
            import shutil
            for file_path in files:
                filename = os.path.basename(file_path)
                dest_path = os.path.join(self.data_folder, filename)
                shutil.copy2(file_path, dest_path)

            self.load_images()
            messagebox.showinfo("Succès", f"{len(files)} image(s) chargée(s) avec succès")

    def display_current_image(self):
        """Afficher l'image courante avec une taille d'affichage maximale plus grande"""
        if not self.current_images:
            return

        image_path = os.path.join(self.data_folder, self.current_images[self.current_index])
        cv_image = cv2.imread(image_path)

        max_display_size = 900 if self.fullscreen_state else 1400

        height, width = cv_image.shape[:2]
        if width > height:
            new_width = max_display_size
            new_height = int(height * max_display_size / width)
        else:
            new_height = max_display_size
            new_width = int(width * max_display_size / height)

        cv_image = cv2.resize(cv_image, (new_width, new_height))
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_image_rgb)
        photo = ImageTk.PhotoImage(pil_image)

        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo
        self.update_histogram(cv_image)

    def update_histogram(self, image):
        """Met à jour l'histogramme avec l'image courante"""
        if image is None:
            return

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        self.histogram_ax.clear()
        self.histogram_ax.hist(gray.ravel(), 256, [0, 256], color='blue')
        self.histogram_ax.set_title('Distribution des Intensités')
        self.histogram_ax.set_xlabel('Valeur de Pixel')
        self.histogram_ax.set_ylabel('Fréquence')
        self.histogram_canvas.draw()

    def update_progress_bar(self):
        """Mettre à jour la barre de progression"""
        if self.current_images:
            progress = (self.current_index + 1) / len(self.current_images) * 100
            self.progress_bar['value'] = progress
            self.progress_var.set(f"{self.current_index + 1}/{len(self.current_images)}")
        else:
            self.progress_bar['value'] = 0
            self.progress_var.set("0/0")

    def first_image(self):
        self.current_index = 0
        self.update_progress_bar()
        self.display_current_image()

    def previous_image(self):
        if self.current_images and self.current_index > 0:
            self.current_index -= 1
            self.update_progress_bar()
            self.display_current_image()

    def next_image(self):
        if self.current_images and self.current_index < len(self.current_images) - 1:
            self.current_index += 1
            if self.myvariabl == "Détection de contours Canny":
                self.apply_canny_edges()
            elif self.myvariabl == "Détection de cyclones":
                self.apply_cyclone_detection()
            elif self.myvariabl == "Opérations morphologiques":
                self.apply_morphological_ops()
            elif self.myvariabl == "Extraction de nuages":
                self.apply_cloud_extraction()
            elif self.myvariabl == "Analyse des précipitations":
                self.apply_precipitation_analysis()
            elif self.myvariabl == "Flot optique":
                self.apply_optical_flow()
            elif self.myvariabl == "Segmentation Multi-seuils":
                self.apply_multi_threshold_segmentation()
            elif self.myvariabl == "Filtrage Homomorphique":
                self.apply_homomorphic_filter()
            else:
                self.update_progress_bar()
                self.display_current_image()

    def last_image(self):
        if self.current_images:
            self.current_index = len(self.current_images) - 1
            self.update_progress_bar()
            self.display_current_image()

    def toggle_play(self):
        """Basculer entre lecture et pause"""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_sequence()

    def play_sequence(self):
        """Lire la séquence d'images"""
        if self.is_playing and self.current_images:
            self.next_image()
            if self.current_index >= len(self.current_images) - 1:
                self.current_index = -1
            self.root.after(100, self.play_sequence)

    def apply_histogram_equalization(self):
        self.myvariabl = "Égalisation d'histogramme"
        apply_histogram_equalization(self)

    def apply_gaussian_filter(self):
        self.myvariabl = "Filtrage Gaussien"
        apply_gaussian_filter(self)

    def apply_canny_edges(self):
        self.myvariabl = "Détection de contours Canny"
        apply_canny_edges(self)

    def apply_morphological_ops(self):
        self.myvariabl = "Opérations morphologiques"
        apply_morphological_ops(self)

    def apply_cloud_extraction(self):
        self.myvariabl = "Extraction de nuages"
        apply_cloud_extraction(self)

    def apply_cyclone_detection(self):
        self.myvariabl = "Détection de cyclones"
        apply_cyclone_detection(self)

    def apply_precipitation_analysis(self):
        self.myvariabl = "Analyse des précipitations"
        apply_precipitation_analysis(self)

    def apply_optical_flow(self):
        self.myvariabl = "Flot optique"
        apply_optical_flow(self)

    def apply_homomorphic_filter(self):
        self.myvariabl = "Filtrage Homomorphique"
        apply_homomorphic_filter(self)

    def apply_multi_threshold_segmentation(self):
        self.myvariabl = "Segmentation Multi-seuils"
        apply_multi_threshold_segmentation(self)

    def display_processed_result(self, processed_image, technique_name):
        """Afficher le résultat du traitement"""
        height, width = processed_image.shape[:2]
        max_display_size = 900 if self.fullscreen_state else 1400

        if width > height:
            new_width = max_display_size
            new_height = int(height * max_display_size / width)
        else:
            new_height = max_display_size
            new_width = int(width * max_display_size / height)

        display_image = cv2.resize(processed_image, (new_width, new_height))

        if len(display_image.shape) == 3:
            display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        else:
            display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_GRAY2RGB)

        pil_image = Image.fromarray(display_image_rgb)
        photo = ImageTk.PhotoImage(pil_image)

        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo
        self.update_histogram(display_image)

    def generate_report(self):
        generate_report(self)

    def generate_detailed_report(self):
        generate_detailed_report(self)

    def save_results(self):
        save_results(self)

    def clear_data_folder(self):
        clear_data_folder(self)

    def quit_app(self):
        """Quitter l'application et nettoyer les données"""
        if messagebox.askokcancel("Quitter",
                                 "Voulez-vous vraiment quitter l'application?\n"
                                 "Toutes les images chargées seront définitivement supprimées."):
            try:
                self.clear_data_folder()
                if os.path.exists(self.output_folder):
                    import shutil
                    shutil.rmtree(self.output_folder)
                self.root.destroy()
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de nettoyer les données: {str(e)}")
                self.root.destroy()

    def run(self):
        """Lancer l'application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.quit_app()