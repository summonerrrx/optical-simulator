import numpy as np
try:
    import tmm
except ImportError:
    print("[Error] The 'tmm' package is required. Please install it with 'pip install tmm'.")
    raise

class FrontSurfaceOptics:
    def __init__(self, optical_data, layers, texture_angle=53.0, shading_fraction=0.05):
        self.optical_data = optical_data
        self.layers = layers
        self.texture_angle = texture_angle
        # Gölgelenme (shading) optik .ftr dosyasını bozmaması için hesaba katılmaz.
        # GUI parametreleri bozulmasın diye init içinde tutuyoruz, ancak matematikte KULLANMIYORUZ.
        self.morphology = "UprightPyramids" if texture_angle > 0 else "Planar"

    def calculate_tmm_reflection(self, wavelength_nm, wl_idx, theta_incident_deg):
        """Hava -> ARC Katmanları -> Si yığını için TMM hesaplayıcısı"""
        theta_incident_rad = np.radians(theta_incident_deg)

        # Giren ortam: Hava (n=1, k=0)
        n_list = [1.0 + 0j]
        d_list = [np.inf]

        # Arayüzden gelen tüm ARC katmanlarını yığına (stack) ekle
        for layer in self.layers:
            mat = layer['material']
            thick = layer['thickness_nm']
            n_val = self.optical_data[mat]['n'][wl_idx]
            k_val = self.optical_data[mat]['k'][wl_idx]
            n_list.append(n_val + 1j * k_val)
            d_list.append(thick)

        # Çıkan ortam (Substrat): Silisyum
        n_si = self.optical_data['Si']['n'][wl_idx]
        k_si = self.optical_data['Si']['k'][wl_idx]
        n_list.append(n_si + 1j * k_si)
        d_list.append(np.inf)

        try:
            # Doğal güneş ışığı polarize değildir, s ve p'nin ortalaması alınır
            res_s = tmm.coh_tmm('s', n_list, d_list, theta_incident_rad, wavelength_nm)
            res_p = tmm.coh_tmm('p', n_list, d_list, theta_incident_rad, wavelength_nm)
            
            R_unpolarized = (res_s['R'] + res_p['R']) / 2.0
            T_unpolarized = (res_s['T'] + res_p['T']) / 2.0 # Silisyuma geçen net ışık oranı
            return np.clip(R_unpolarized, 0.0, 1.0), np.clip(T_unpolarized, 0.0, 1.0)
        except Exception as e:
            print(f"[Error] TMM failed at {wavelength_nm} nm: {e}")
            return 0.0, 0.0

    def compute_front_reflection(self):
        wl = self.optical_data['Si']['wl']
        R_total_array = np.zeros_like(wl, dtype=float)
        T_total_array = np.zeros_like(wl, dtype=float) # Silisyuma giren NET ışık dizisi
        
        print(f"Optical Front Morphology: {self.morphology}, Angle: {self.texture_angle}°")
        print("Optical Front Layers:")
        for l in self.layers:
            print(f"  - {l['thickness_nm']} nm {l['material']}")

        for i, wavelength in enumerate(wl):
            if self.morphology == "Planar":
                # Düzlemsel yüzeyde ışık 0 derece (dik) ile çarpar
                R1, T1 = self.calculate_tmm_reflection(wavelength, i, 0.0)
                R_total_array[i] = R1
                T_total_array[i] = T1
            else:
                # Piramit Geometrisi (Bounce 1)
                alpha = self.texture_angle
                R1, T1 = self.calculate_tmm_reflection(wavelength, i, alpha)
                
                if alpha > 30.0:
                    # Karşı piramit yüzeyine vuruş açısı (Bounce 2)
                    theta_2 = 180.0 - 3.0 * alpha
                    theta_2 = np.clip(theta_2, 0.0, 89.9) 
                    
                    R2, T2 = self.calculate_tmm_reflection(wavelength, i, theta_2)
                    
                    # Kapsamlı Hibrit Fizik Motoru
                    R_total_array[i] = R1 * R2
                    T_total_array[i] = T1 + (R1 * T2) # İlk giriş + İkinci sekmeden giren
                else:
                    R_total_array[i] = R1
                    T_total_array[i] = T1

        # R ve T değerlerini yüzdeye çevir
        R_percent = R_total_array * 100.0
        T_percent = T_total_array * 100.0
        
        # Optik Slicer ve Visualizer'ın kullanması için T_percent'i de dışarı aktarıyoruz
        return np.column_stack((wl, R_percent, T_percent))

if __name__ == "__main__":
    from fetcher import OpticalDataLoader
    print("Loading data via OpticalDataLoader...")
    loader = OpticalDataLoader()
    optical_data = loader.run(["Si", "SiNx", "Al2O3"])
    
    if optical_data:
        # Artık çoklu katmanları destekliyor!
        test_layers = [{'material': 'Al2O3', 'thickness_nm': 10.0},
                       {'material': 'SiNx', 'thickness_nm': 70.0}]
        front_optics = FrontSurfaceOptics(optical_data, layers=test_layers, texture_angle=53.0)
        result = front_optics.compute_front_reflection()
        
        print("\nFront Reflection Array Shape:", result.shape)
        print("Sample Data:")
        print(f"{result[300][0]} nm (600nm - Minimum expected): {result[300][1]:.2f} %")
