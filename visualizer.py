import numpy as np
try:
    import pyvista as pv
except ImportError:
    print("[Error] The 'pyvista' package is required for 3D visualization. Please install it with 'pip install pyvista'.")
    raise

class PyVistaVisualizer:
    def __init__(self, wl, r_front, t_front, r_rear, abs_emitter, abs_bulk, abs_bsf, 
                 emitter_thick, bulk_thick, bsf_thick, texture_angle):
        """
        Gelen numpy matrislerini (wl, r_front, vb.) ve uzamsal parametreleri doğru sırada alır.
        """
        # 1D Fizik Optik Numpy Dizileri
        self.wl = wl
        self.r_front = r_front
        self.t_front = t_front
        self.r_rear = r_rear
        self.abs_emitter = abs_emitter
        self.abs_bulk = abs_bulk
        self.abs_bsf = abs_bsf
        
        # Geometrik Parametreler
        self.emitter_thick = float(emitter_thick)
        self.bulk_thick = float(bulk_thick)
        self.bsf_thick = float(bsf_thick)
        self.texture_angle = float(texture_angle)
        
        self.mesh_bsf = None
        self.mesh_bulk = None
        self.mesh_emitter = None
        self.mesh_pyramids = None
        
        self.text_actor = None
        
        # RAM kalıcı matrislerin başlangıçta hesaplanması
        self.absorption_matrix = None
        self.z_array_nm = None
        self.material_map = None
        
        self.generate_1d_matrices()
        
    def _get_data_at_wavelength(self, target_wl):
        """Dalgaboyuna en yakın optik katsayıları getirir (Locator)."""
        idx = np.argmin(np.abs(self.wl - target_wl))
        return idx, self.t_front[idx], self.r_rear[idx], self.abs_emitter[idx], self.abs_bulk[idx], self.abs_bsf[idx]

    def build_geometry(self):
        """Wafer'ın Z boyutlu bloklarını ve dokusunu mikrometre (um) cinsinden üretir."""
        print("[PyVista] Geometrik Mesh Koordinatları İnşa Ediliyor...")
        patch_size = 50.0 # 50x50 Mikrometrelik geniş wafer (plaka) görünümü
        
        # Z Ekseni Spagetti Sorunu Çözümü (Non-lineer büyüme / Karekök)
        g_thick_bsf = np.sqrt(self.bsf_thick) if self.bsf_thick > 0 else 0
        g_thick_bulk = np.sqrt(self.bulk_thick) if self.bulk_thick > 0 else 0
        g_thick_em = np.sqrt(self.emitter_thick) if self.emitter_thick > 0 else 0
        
        z_min_bsf = 0.0
        z_max_bsf = g_thick_bsf
        
        z_min_bulk = z_max_bsf
        z_max_bulk = z_min_bulk + g_thick_bulk
        
        z_min_emitter = z_max_bulk
        z_max_emitter = z_min_emitter + g_thick_em
        
        # Kutuların Çizimi (Arayüzde girilmeyen sıfır kalınlıklarda mesh bypass edilir)
        if self.bsf_thick > 0:
            self.mesh_bsf = pv.Box(bounds=(0.0, patch_size, 0.0, patch_size, z_min_bsf, z_max_bsf))
            self.mesh_bsf.point_data["Absorption"] = np.ones(self.mesh_bsf.n_points)
            
        if self.bulk_thick > 0:
            self.mesh_bulk = pv.Box(bounds=(0.0, patch_size, 0.0, patch_size, z_min_bulk, z_max_bulk))
            self.mesh_bulk.point_data["Absorption"] = np.ones(self.mesh_bulk.n_points)
            
        if self.emitter_thick > 0:
            self.mesh_emitter = pv.Box(bounds=(0.0, patch_size, 0.0, patch_size, z_min_emitter, z_max_emitter))
            self.mesh_emitter.point_data["Absorption"] = np.ones(self.mesh_emitter.n_points)
        
        # Piramit Hesabı
        if self.texture_angle > 0.0 and self.emitter_thick > 0:
            pitch = 2.0  
            x = np.linspace(0.0, patch_size, 100)
            y = np.linspace(0.0, patch_size, 100)
            x_grid, y_grid = np.meshgrid(x, y)
            
            x_mod = np.abs((x_grid % pitch) - (pitch / 2.0))
            y_mod = np.abs((y_grid % pitch) - (pitch / 2.0))
            
            z_peaks = ((pitch / 2.0) - np.maximum(x_mod, y_mod)) * np.tan(np.radians(self.texture_angle))
            z_grid = z_max_emitter + z_peaks
            
            self.mesh_pyramids = pv.StructuredGrid(x_grid, y_grid, z_grid)
            self.mesh_pyramids.point_data["Absorption"] = np.ones(self.mesh_pyramids.n_points)

    def update_heatmaps(self, plotter, current_wl):
        """Çift Yönlü İnkoherent Foton Akısı ve Süperpozisyon ile Absorpsiyon Dağılımı"""
        idx, t_front_val, r_rear_val, a_emit, a_bulk, a_bsf = self._get_data_at_wavelength(current_wl)
        
        # Katsayıların 1/m'den 1/um birimine geçirilmesi
        a_em_um = a_emit / 1e6
        a_bulk_um = a_bulk / 1e6
        a_bsf_um = a_bsf / 1e6
        
        # =========================================================
        # FAZ 1 & 2: AŞAĞI YÖNLÜ FOTON AKISI (ZİNCİRLEME)
        # =========================================================
        # Verilerin main_gui.py'den [0.0 - 1.0] aralığında (normalize) geldiği kesin olarak varsayılır.
        I_down_top = t_front_val 
        
        if self.mesh_pyramids is not None:
            z_top_pyr = np.max(self.mesh_pyramids.points[:, 2])
            z_bot_pyr = np.min(self.mesh_pyramids.points[:, 2])
            visual_pyr_height = z_top_pyr - z_bot_pyr
            
            # DİKKAT: Piramidin fiziksel yüksekliği görsel mesh'ten bağımsız hesaplanmalıdır.
            pitch = 2.0
            physical_pyr_height = (pitch / 2.0) * np.tan(np.radians(self.texture_angle))
            
            # Işık sahte görsel mesh'te değil, GERÇEK fiziksel mesafede sönümlenir!
            I_down_em_start = I_down_top * np.exp(-a_em_um * physical_pyr_height)
        else:
            I_down_em_start = I_down_top
            visual_pyr_height = 0
            physical_pyr_height = 0
            
        # Zincirin geri kalanı
        I_down_em_end = I_down_em_start * np.exp(-a_em_um * self.emitter_thick)
        I_down_bulk_end = I_down_em_end * np.exp(-a_bulk_um * self.bulk_thick)
        I_down_bsf_end = I_down_bulk_end * np.exp(-a_bsf_um * self.bsf_thick) # Arka yüze çarpan miktar
        
        # =========================================================
        # FAZ 3 & 4: IŞIK TUZAĞI VE YUKARI YÖNLÜ FOTON AKISI (I_up)
        # =========================================================
        # r_rear_val main_gui.py'den temizlenmiş kesir olarak beklenir
        I_up_bottom = I_down_bsf_end * r_rear_val
        
        # Katman sınırlarına ulaşan yukarı yönlü kümülatif enerji miktarları
        I_up_bsf_end = I_up_bottom * np.exp(-a_bsf_um * self.bsf_thick) # Bulk tabanına ulaşan
        I_up_bulk_end = I_up_bsf_end * np.exp(-a_bulk_um * self.bulk_thick) # Emitter tabanına ulaşan
        
        # Piramit tabanına alttan girmek üzere Emitter'ı yukarı doğru geçen ışık
        I_up_em_end = I_up_bulk_end * np.exp(-a_em_um * self.emitter_thick)
        
        # =========================================================
        # FAZ 5: LOKAL SOĞURMA HESAPLAMASI (Sıfıra Bölünme Bypass & Gerçek Fiziksel Normalizasyon)
        # =========================================================
        # Matematiksel olarak t_down ile normalize edilmiş fiziksel d_down hesabı
        
        # 1. BSF KATMANI
        A_bsf = 0
        if self.bsf_thick > 0 and self.mesh_bsf is not None:
            z_top_bsf = np.max(self.mesh_bsf.points[:, 2])
            z_bot_bsf = np.min(self.mesh_bsf.points[:, 2])
            z_range = z_top_bsf - z_bot_bsf
            
            t_down_bsf = (z_top_bsf - self.mesh_bsf.points[:, 2]) / z_range if z_range > 0 else 0
            t_up_bsf = (self.mesh_bsf.points[:, 2] - z_bot_bsf) / z_range if z_range > 0 else 0
            
            d_down_bsf = t_down_bsf * self.bsf_thick
            d_up_bsf = t_up_bsf * self.bsf_thick
            
            i_local_down_bsf = I_down_bulk_end * np.exp(-a_bsf_um * d_down_bsf)
            i_local_up_bsf = I_up_bottom * np.exp(-a_bsf_um * d_up_bsf)
            A_bsf = a_bsf_um * (i_local_down_bsf + i_local_up_bsf)

        # 2. BULK KATMANI
        A_bulk = 0
        if self.bulk_thick > 0 and self.mesh_bulk is not None:
            z_top_bulk = np.max(self.mesh_bulk.points[:, 2])
            z_bot_bulk = np.min(self.mesh_bulk.points[:, 2])
            z_range = z_top_bulk - z_bot_bulk
            
            t_down_bulk = (z_top_bulk - self.mesh_bulk.points[:, 2]) / z_range if z_range > 0 else 0
            t_up_bulk = (self.mesh_bulk.points[:, 2] - z_bot_bulk) / z_range if z_range > 0 else 0
            
            d_down_bulk = t_down_bulk * self.bulk_thick
            d_up_bulk = t_up_bulk * self.bulk_thick
            
            i_local_down_bulk = I_down_em_end * np.exp(-a_bulk_um * d_down_bulk)
            i_local_up_bulk = I_up_bsf_end * np.exp(-a_bulk_um * d_up_bulk)
            A_bulk = a_bulk_um * (i_local_down_bulk + i_local_up_bulk)

        # 3. EMITTER KATMANI
        A_em = 0
        if self.emitter_thick > 0 and self.mesh_emitter is not None:
            z_top_em = np.max(self.mesh_emitter.points[:, 2])
            z_bot_em = np.min(self.mesh_emitter.points[:, 2])
            z_height_em = z_top_em - z_bot_em
            
            t_down_em = (z_top_em - self.mesh_emitter.points[:, 2]) / z_height_em if z_height_em > 0 else 0
            t_up_em = (self.mesh_emitter.points[:, 2] - z_bot_em) / z_height_em if z_height_em > 0 else 0
            
            d_down_em = t_down_em * self.emitter_thick
            d_up_em = t_up_em * self.emitter_thick
            
            # DÜZELTME: Emitter ışığı tepeden değil, piramitten süzülen ışıktan (I_down_em_start) alır.
            i_local_down_em = I_down_em_start * np.exp(-a_em_um * d_down_em)
            i_local_up_em = I_up_bulk_end * np.exp(-a_em_um * d_up_em)
            A_em = a_em_um * (i_local_down_em + i_local_up_em)

        # 4. PİRAMİT DOKUSU (Varsa)
        A_pyr = np.array([])
        if self.mesh_pyramids is not None and visual_pyr_height > 0:
            # Görsel mesh üzerinden yüzde (t) konumu bulunur
            t_down_pyr = (z_top_pyr - self.mesh_pyramids.points[:, 2]) / visual_pyr_height
            t_up_pyr = (self.mesh_pyramids.points[:, 2] - z_bot_pyr) / visual_pyr_height
            
            # Fiziksel sönümlenme mesafesi GERÇEK yükseklikle çarpılarak bulunur!
            d_down_pyr = t_down_pyr * physical_pyr_height
            d_up_pyr = t_up_pyr * physical_pyr_height
            
            # Piramit en tepede olduğu için ilk ışığı (I_down_top) o alır.
            i_local_down_pyr = I_down_top * np.exp(-a_em_um * d_down_pyr)
            # Aşağıdan dönen ışık ise tüm Emitter'ı geçtikten sonra (I_up_em_end) piramide girer.
            i_local_up_pyr = I_up_em_end * np.exp(-a_em_um * d_up_pyr)
            
            A_pyr = a_em_um * (i_local_down_pyr + i_local_up_pyr)

        # =========================================================
        # FAZ 6: GÖRSELLEŞTİRME VE 600NM REFERANS NORMALİZASYONU
        # =========================================================
        A_log_em = np.log10(1 + A_em)
        A_log_bulk = np.log10(1 + A_bulk)
        A_log_bsf = np.log10(1 + A_bsf)
        
        if self.mesh_pyramids is not None and len(A_pyr) > 0:
            A_log_pyr = np.log10(1 + A_pyr)

        # --- DİNAMİK 600 NM TAVAN HESAPLAMASI ---
        _, t_front_600, _, a_emit_600, _, _ = self._get_data_at_wavelength(600.0)
        a_em_um_600 = a_emit_600 / 1e6
        
        # main_gui'den temizlenmiş olarak geldiği varsayılan veri direkt kullanılır
        I_down_top_600 = t_front_600
        
        A_max_600 = a_em_um_600 * I_down_top_600
        DYNAMIC_MAX_LOG = np.log10(1 + A_max_600)
        
        if DYNAMIC_MAX_LOG <= 0.0:
            DYNAMIC_MAX_LOG = 0.1 
        
        if self.mesh_emitter is not None:
            self.mesh_emitter.point_data["Absorption"] = np.clip(A_log_em / DYNAMIC_MAX_LOG, 0.0, 1.0)
        
        if self.mesh_bulk is not None:
            self.mesh_bulk.point_data["Absorption"] = np.clip(A_log_bulk / DYNAMIC_MAX_LOG, 0.0, 1.0)
            
        if self.mesh_bsf is not None:
            self.mesh_bsf.point_data["Absorption"] = np.clip(A_log_bsf / DYNAMIC_MAX_LOG, 0.0, 1.0)
        
        if self.mesh_pyramids is not None and len(A_pyr) > 0:
            self.mesh_pyramids.point_data["Absorption"] = np.clip(A_log_pyr / DYNAMIC_MAX_LOG, 0.0, 1.0)
        
        # --- HUD Güncellemesi ---
        if self.text_actor is not None:
            plotter.remove_actor(self.text_actor)
            
        # Gösterge ekranında UI/UX açısından değerleri anlık olarak % formatında (.2f) basıyoruz
        hud_text = (f"=== 3D VOLUMETRIC HEATMAP ===\n"
                    f"Wavelength      : {current_wl:.1f} nm\n"
                    f"Net Transmission: % {t_front_val * 100.0:.2f}\n"
                    f"Rear Reflection : % {r_rear_val * 100.0:.2f}\n"
                    f"Bulk Abs. Coeff : {a_bulk:.2e} 1/m")
        
        self.text_actor = plotter.add_text(hud_text, position="upper_right", 
                                           font_size=12, color="white", font="courier")
            
    def show_scene(self):
        """Oluşturulan geometrik blokları renklendirerek 3D uzaya basar ve Plotter'ı başlatır."""
        print("[PyVista] 3D Heatmap Görselleştirme Ekranı Başlatılıyor...")
        
        plotter = pv.Plotter(title="PV Volumetric Absorption Heatmap")
        plotter.set_background("#131313")
        
        # Blokları sisteme Heatmap mantığıyla yükleme
        # colormap: inferno (yüksek: sarı/beyaz, düşük: koyu mor/siyah)
        if self.mesh_bsf is not None:
            plotter.add_mesh(self.mesh_bsf, scalars="Absorption", cmap="inferno", clim=[0.0, 1.0], show_edges=True, edge_color="gray")
        if self.mesh_bulk is not None:
            plotter.add_mesh(self.mesh_bulk, scalars="Absorption", cmap="inferno", clim=[0.0, 1.0], show_edges=True, edge_color="gray", opacity=0.9)
        if self.mesh_emitter is not None:
            plotter.add_mesh(self.mesh_emitter, scalars="Absorption", cmap="inferno", clim=[0.0, 1.0], show_edges=True, edge_color="gray")
        if self.mesh_pyramids is not None:
            plotter.add_mesh(self.mesh_pyramids, scalars="Absorption", cmap="inferno", clim=[0.0, 1.0], show_edges=False)
            
        def slider_callback(value):
            # Isı haritalarını ve HUD ekranını arka planda yenile
            self.update_heatmaps(plotter, value)
            plotter.render()
            
        # Başlangıç Yüklemesi
        slider_callback(600.0)
        
        # Wavelength (nm) Slider Widget'ı
        plotter.add_slider_widget(slider_callback, rng=[300.0, 1200.0], value=600.0, 
                                  title="Wavelength (nm)", 
                                  pointa=(0.60, 0.08), pointb=(0.95, 0.08),
                                  style='modern', color='white')
            
        plotter.view_isometric()
        plotter.show()
        
    def verify_pipeline(self):
        """Borulama (pipeline) işleminin başarıyla çalıştığını test eden kontrolör."""
        print("\n======== 3D HEATMAP PIPELINE VERIFICATION ========")
        total_z_thick = self.emitter_thick + self.bulk_thick + self.bsf_thick
        print(f"[Geometri] Toplam Z Eksen Boyutu : {total_z_thick:.2f} um")
        print(f"[Geometri] Ön Yüzey Dokusu       : {self.texture_angle}°")
        print("----------------------------------------------------------")
        print("[Optik] Numpy Veri Dizisi Boyutları (Shapes):")
        print(f"  > Wavelength (wl) : {self.wl.shape}")
        print(f"  > r_front         : {self.r_front.shape}")
        print(f"  > abs_emitter     : {self.abs_emitter.shape}")
        print(f"  > abs_bulk        : {self.abs_bulk.shape}")
        print(f"  > abs_bsf         : {self.abs_bsf.shape}")
        print("======== PIPELINE HOOK SUCCESSFUL ========================\n")
        
        self.build_geometry()
        self.show_scene()

    def generate_1d_matrices(self):
        """Matrix boyutuzlandırma ve uzaysal soğurma oluşturucu (Senkronize ve Optimizasyonlu)."""
        print("[Visualizer] 1D Z-Spatial Absorption Matrix hesaplanıyor...")
        total_thick_um = self.emitter_thick + self.bulk_thick + self.bsf_thick
        total_thick_nm = total_thick_um * 1000.0
        
        # Yüksek performans için dengeli çözünürlük
        num_points = int(total_thick_nm // 2)
        if num_points < 100: num_points = 100
        if num_points > 10000: num_points = 10000
        
        self.z_array_nm = np.linspace(0, total_thick_nm, num_points)
        z_array_um = self.z_array_nm / 1000.0
        
        # Material map (Sadece optik simüle edilen bölgeler silisyumdur)
        self.material_map = [{'material': 'Si', 'thickness_nm': total_thick_nm}]
        
        a_em_um = self.abs_emitter / 1e6
        a_bulk_um = self.abs_bulk / 1e6
        a_bsf_um = self.abs_bsf / 1e6
        
        I_down_top = self.t_front
        
        if self.texture_angle > 0:
            pitch = 2.0
            physical_pyr_height = (pitch / 2.0) * np.tan(np.radians(self.texture_angle))
            I_down_em_start = I_down_top * np.exp(-a_em_um * physical_pyr_height)
        else:
            I_down_em_start = I_down_top
            
        I_down_em_end = I_down_em_start * np.exp(-a_em_um * self.emitter_thick)
        I_down_bulk_end = I_down_em_end * np.exp(-a_bulk_um * self.bulk_thick)
        I_down_bsf_end = I_down_bulk_end * np.exp(-a_bsf_um * self.bsf_thick)
        
        I_up_bottom = I_down_bsf_end * self.r_rear
        I_up_bsf_end = I_up_bottom * np.exp(-a_bsf_um * self.bsf_thick)
        I_up_bulk_end = I_up_bsf_end * np.exp(-a_bulk_um * self.bulk_thick)
        
        # RAM güvenliğini ve senkronizasyonu sağlayan 2D Broadcasting
        z_2d = z_array_um[:, np.newaxis] 
        
        mask_em = (z_2d <= self.emitter_thick)
        d_down_em = z_2d
        d_up_em = self.emitter_thick - z_2d
        A_em_2d = a_em_um * (I_down_em_start * np.exp(-a_em_um * d_down_em) + I_up_bulk_end * np.exp(-a_em_um * d_up_em))
        
        mask_bulk = (z_2d > self.emitter_thick) & (z_2d <= self.emitter_thick + self.bulk_thick)
        d_down_bulk = z_2d - self.emitter_thick
        d_up_bulk = self.bulk_thick - d_down_bulk
        A_bulk_2d = a_bulk_um * (I_down_em_end * np.exp(-a_bulk_um * d_down_bulk) + I_up_bsf_end * np.exp(-a_bulk_um * d_up_bulk))
        
        mask_bsf = (z_2d > self.emitter_thick + self.bulk_thick)
        d_down_bsf = z_2d - (self.emitter_thick + self.bulk_thick)
        d_up_bsf = self.bsf_thick - d_down_bsf
        A_bsf_2d = a_bsf_um * (I_down_bulk_end * np.exp(-a_bsf_um * d_down_bsf) + I_up_bottom * np.exp(-a_bsf_um * d_up_bsf))
        
        self.absorption_matrix = np.where(mask_em, A_em_2d,
                                 np.where(mask_bulk, A_bulk_2d,
                                 np.where(mask_bsf, A_bsf_2d, 0.0)))
        print("[Visualizer] 1D Matris kalıcılığı sağlandı.")

    def get_optical_results(self):
        """Jenerasyon motoru (fetcher.py) için köprü görevi gören Getter."""
        print(f"DEBUG: Getter çağrıldı. Veri durumu: {self.absorption_matrix is not None}")
        return self.z_array_nm, self.absorption_matrix, self.material_map
