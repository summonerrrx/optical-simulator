import os
import re
import numpy as np
try:
    import tmm
except ImportError:
    print("[Error] The 'tmm' package is required. Please install it with 'pip install tmm'.")
    raise

class RearSurfaceOptics:
    def __init__(self, optical_data, layers, texture_morphology="Planar"):
        self.optical_data = optical_data
        self.morphology = texture_morphology
        self.stack = [{'material': l['material'], 'thickness': l['thickness_nm']} for l in layers]

    def calculate_tmm_reflection(self, wavelength_nm, wl_idx, theta_incident_rad):
        # ÇÖZÜM 1: Incident Medium k=0 olmalıdır (R > 100% hatasını önlemek için)
        n_in = self.optical_data['Si']['n'][wl_idx] + 0j 
        
        n_list = [n_in]
        d_list = [np.inf]
        
        for layer in self.stack:
            mat = layer['material']
            thick = layer['thickness']
            
            n_val = self.optical_data[mat]['n'][wl_idx]
            k_val = self.optical_data[mat]['k'][wl_idx]
            n_complex = n_val + 1j * k_val
            
            n_list.append(n_complex)
            if layer == self.stack[-1]:
                d_list.append(np.inf)
            else:
                d_list.append(thick)

        try:
            # ÇÖZÜM 2 & 3: s ve p polarizasyonları hesaplanıp ortalaması alınır
            res_s = tmm.coh_tmm('s', n_list, d_list, theta_incident_rad, wavelength_nm)
            res_p = tmm.coh_tmm('p', n_list, d_list, theta_incident_rad, wavelength_nm)
            
            R_unpolarized = (res_s['R'] + res_p['R']) / 2.0
            
            # Fiziksel sınır güvenlik kilidi (Termodinamik koruma)
            return np.clip(R_unpolarized, 0.0, 1.0)
            
        except Exception as e:
            print(f"[Error] TMM failed at {wavelength_nm} nm: {e}")
            return 0.0

    def compute_rear_internal_reflection(self, front_texture_angle_deg=53.0):
        wl_array = self.optical_data['Si']['wl']
        R_internal = np.zeros_like(wl_array, dtype=float)
        
        for i, wl in enumerate(wl_array):
            # ÇÖZÜM 2: Snell Yasası ile Silisyum içindeki gerçek geliş açısının bulunması
            n_si_real = self.optical_data['Si']['n'][i]
            theta_in_rad = np.arcsin(np.sin(np.radians(front_texture_angle_deg)) / n_si_real)
            
            R_internal[i] = self.calculate_tmm_reflection(wl, i, theta_incident_rad=theta_in_rad)
            
        R_percent = R_internal * 100.0
        return np.column_stack((wl_array, R_percent))

if __name__ == "__main__":
    from fetcher import OpticalDataLoader
    print("Loading data via OpticalDataLoader...")
    loader = OpticalDataLoader()
    optical_data = loader.run(["Si", "Al2O3", "SiNx", "Al"])
    
    if optical_data:
        test_layers = [
            {'material': 'Al2O3', 'thickness_nm': 10.0},
            {'material': 'SiNx', 'thickness_nm': 100.0},
            {'material': 'Al', 'thickness_nm': 4000.0}
        ]
        rear_optics = RearSurfaceOptics(optical_data, layers=test_layers, texture_morphology="Planar")
        result = rear_optics.compute_rear_internal_reflection()
        
        print("\nRear Internal Reflection Array Shape:", result.shape)
        print("Sample Data (800nm, 1000nm, 1200nm):") # Arka yüzey genellikle uzun dalgaboylarında aktiftir
        print(f"{result[500][0]} nm: {result[500][1]:.2f} %")    # 800 nm
        print(f"{result[700][0]} nm: {result[700][1]:.2f} %")    # 1000 nm
        print(f"{result[-1][0]} nm: {result[-1][1]:.2f} %")      # 1200 nm
