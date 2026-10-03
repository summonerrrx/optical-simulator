import numpy as np

class OpticalSlicer:
    def __init__(self, optical_data, texture_angle_deg=53.0, doping_emitter=1e19, doping_bsf=5e18):
        """
        optical_data: Modül 1'den (fetcher) gelen data.
        Hayalet değişkenler (thickness, rear_reflection) SCAPS'ın çalışma prensibine 
        uygun olarak (intensive property mantığıyla) sistemden çıkartılmıştır.
        """
        self.optical_data = optical_data
        self.texture_angle = texture_angle_deg
        
        self.doping_emitter = doping_emitter
        self.doping_bsf = doping_bsf
        
        print(f"Optical Slicer Initialized:")
        print(f"  - Front Texture Angle : {self.texture_angle}°")
        print(f"  - Emitter Doping (Nd) : {self.doping_emitter:.2e} cm^-3")
        print(f"  - BSF Doping (Na)     : {self.doping_bsf:.2e} cm^-3")

    def _calculate_alpha_intrinsic(self, wl_nm, k_si):
        """Beer-Lambert baz soğurma katsayısını (1/m) hesaplar."""
        wl_m = wl_nm * 1e-9
        wl_m = np.maximum(wl_m, 1e-12) 
        alpha = (4.0 * np.pi * k_si) / wl_m
        return alpha

    def calculate_effective_absorption(self):
        """
        Vektörel Snell Yasası (n(λ)) ve Taşıyıcı Tipine (n/p) duyarlı FCA içerir.
        """
        wl_array = self.optical_data['Si']['wl']
        n_Si_array = self.optical_data['Si']['n']
        k_si_array = self.optical_data['Si']['k']
        
        # 1. Baz soğurma katsayısı (1/m)
        wl_m_array = np.maximum(wl_array * 1e-9, 1e-12)
        alpha_int_array = (4.0 * np.pi * k_si_array) / wl_m_array
        
        # 2. Snell Yasası ile Yol Uzaması (Path Enhancement)
        sin_theta_in = np.sin(np.radians(self.texture_angle))
        sin_theta_int_array = np.clip(1.0 * sin_theta_in / n_Si_array, -1.0, 1.0)
        theta_internal_rad_array = np.arcsin(sin_theta_int_array)
        path_factor_array = 1.0 / np.cos(theta_internal_rad_array)
        
        alpha_first_array = alpha_int_array * path_factor_array
        
        # 3. Yüksek Katkı (Doping) ve Taşıyıcı Tipine Bağlı FCA (Katı Hal Fiziği)
        # Dalga boyu mikrometre cinsine çevriliyor
        wl_um = wl_array / 1000.0 
        
        # n-tipi FCA Tesir Kesiti (Örn: Fosfor katkılı Emitter) ~ 1.0e-18 cm^2
        # p-tipi FCA Tesir Kesiti (Örn: Bor/Al katkılı BSF) ~ 2.7e-18 cm^2
        fca_n_type_cm_1 = 1.0e-18 * (wl_um ** 2) * self.doping_emitter
        fca_p_type_cm_1 = 2.7e-18 * (wl_um ** 2) * self.doping_bsf
        
        # cm^-1'den m^-1'e çevrim (* 100)
        alpha_fca_em_array = fca_n_type_cm_1 * 100.0
        alpha_fca_bs_array = fca_p_type_cm_1 * 100.0
        
        # Katmanların nihai SCAPS alpha_eff yüklemeleri
        alpha_eff_emitter = alpha_first_array + alpha_fca_em_array
        alpha_eff_bulk = alpha_first_array.copy() # Bulk optik olarak intrinsik kabul edilir
        alpha_eff_bsf = alpha_first_array + alpha_fca_bs_array
        
        print("Optical slicing (with n/p specific FCA) complete.")
        return wl_array, alpha_eff_emitter, alpha_eff_bulk, alpha_eff_bsf

if __name__ == "__main__":
    from fetcher import OpticalDataLoader
    
    print("Loading data via OpticalDataLoader...")
    loader = OpticalDataLoader()
    optical_data = loader.run(["Si"])
    
    if optical_data:
        print("\nRunning OpticalSlicer...")
        # Artık thickness veya rear_reflection gibi kullanılmayan verileri istemiyor
        slicer = OpticalSlicer(optical_data, texture_angle_deg=53.0, doping_emitter=1e19, doping_bsf=5e18)
        wl, a_emit, a_bulk, a_bsf = slicer.calculate_effective_absorption()
        
        print("\nSample Data (1200 nm - FCA'nın en güçlü olduğu bölge):")
        idx_1200 = np.where(wl == 1200.0)[0][0]
        print(f"Alpha Intrinsic Si : {slicer._calculate_alpha_intrinsic(1200.0, optical_data['Si']['k'][idx_1200]):.2e} 1/m")
        print(f"Alpha Emitter (n)  : {a_emit[idx_1200]:.2e} 1/m")
        print(f"Alpha BSF (p)      : {a_bsf[idx_1200]:.2e} 1/m")
