import os
import sys
import urllib.request
import urllib.error
import yaml
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.constants import h, c
import csv  

class OpticalDataLoader:
    def __init__(self):
        self.optical_data = {}
        
        self.MATERIAL_DB = {
            'Si': 'main/Si/nk/Green-2008.yml',
            'Al2O3': 'main/Al2O3/nk/Boidin.yml',
            'Al': 'main/Al/nk/Rakic.yml',
            'Ag': 'main/Ag/nk/Johnson.yml',        
            'MgF2': 'main/MgF2/nk/Dodge.yml',      
            'TiO2': 'main/TiO2/nk/Siefke.yml',     
            'ZnO': 'main/ZnO/nk/Bond.yml',         
            'ITO': 'other/mixed%20crystals/In2O3-SnO2/nk/Moerland.yml'
        }

    def fetch_data(self, material_list):
        base_url = "https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/main/database/data/"
        materials_to_fetch = set(material_list)
        if 'Si' not in materials_to_fetch:
            materials_to_fetch.add('Si')
            
        print(f"[Info] Fetching materials: {materials_to_fetch}")
        
        for mat in materials_to_fetch:
            # YEREL CSV DOSYALARI (SiNx, SiOx, Air, MgF2, p-Poly c-Si)
            if mat == 'SiNx':
                self._parse_csv_data(mat, "SiNx_PECVD 1.91 [Vog15].csv")
                continue
            elif mat == 'SiOx':
                # SiNx ile aynı yapıya sahip CSV okuyucuyu kullanır
                self._parse_csv_data(mat, "SiOx_PECVD [McI14].csv")
                continue
            elif mat == 'MgF2':
                # MgF2 de SiNx ile aynı yapıda olduğu için standart okuyucuyu kullanır
                self._parse_csv_data(mat, "MgF2_Evaporated [Siq88].csv")
                continue
            elif mat == 'p-Poly c-Si':
                # p-Poly c-Si de SiNx ile aynı formatta olduğu için standart okuyucu kullanılır
                self._parse_csv_data(mat, "Si_Polycrystalline p [Fon25].csv")
                continue
            elif mat == 'Air':
                # Ciddor dosyası farklı bir formatta olduğu için özel fonksiyon
                self._parse_ciddor_csv(mat, "Ciddor.csv")
                continue
                
            # İNTERNETTEN (YAML) ÇEKİLECEK DOSYALAR
            if mat not in self.MATERIAL_DB:
                raise ValueError(f"[Fatal Error] Material '{mat}' is not in the verified database or local CSV list.")
                
            yaml_url = base_url + self.MATERIAL_DB[mat]
            try:
                print(f"Fetching {mat} data from: {yaml_url}")
                req = urllib.request.Request(yaml_url, headers={'User-Agent': 'Mozilla/5.0'})
                yf = urllib.request.urlopen(req)
                yaml_data = yaml.safe_load(yf.read())
                self._parse_yaml_data(mat, yaml_data)
            except urllib.error.URLError as e:
                raise ConnectionError(f"[Fatal Error] Failed to fetch {mat} from internet. Check connection. Details: {e}")

    # YARDIMCI FONKSİYON: Akıllı dosya bulucu
    def _get_filepath(self, filename):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = parent_dir
            
        possible_paths = [
            os.path.join(current_dir, filename),             
            os.path.join(parent_dir, filename),              
            os.path.join(base_dir, filename),                
            os.path.join(base_dir, "modules", filename)      
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        aranan_yerler = "\n".join(possible_paths)
        raise FileNotFoundError(f"[HATA] '{filename}' bulunamadı!\nŞu konumlara bakıldı:\n{aranan_yerler}\nLütfen dosyayı .exe'nin yanına veya modules klasörüne koyun.")

    # SiNx, SiOx, MgF2 ve p-Poly c-Si İÇİN STANDART OKUYUCU
    def _parse_csv_data(self, material, filename):
        wl, n, k = [], [], []
        filepath = self._get_filepath(filename)
        
        print(f"[OK] CSV Loading from: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            
        for line in lines[1:]: 
            line = line.strip()
            if not line:
                continue
                
            if ';' in line:
                parts = line.split(';')
            elif '\t' in line:
                parts = line.split('\t')
            else:
                parts = line.split(',')
                
            if len(parts) >= 2:
                try:
                    w_val = float(parts[0].strip())
                    if w_val < 100.0:
                        w_val *= 1000.0
                        
                    n_val = float(parts[1].strip())
                    k_val = 0.0
                    
                    if len(parts) >= 4 and parts[3].strip() != '':
                        k_val = float(parts[3].strip())
                    elif len(parts) == 3 and parts[2].strip() != '':
                        k_val = float(parts[2].strip())
                        
                    wl.append(w_val)
                    n.append(n_val)
                    k.append(k_val)
                except ValueError:
                    pass
                        
        if len(wl) > 0:
            sort_idx = np.argsort(wl)
            self.optical_data[material] = {
                'wl': np.array(wl)[sort_idx],
                'n': np.array(n)[sort_idx],
                'k': np.array(k)[sort_idx]
            }
            print(f"[Success] {len(wl)} data points loaded for {material} directly from CSV.")
        else:
            raise ValueError(f"[Fatal Error] '{filename}' dosyasından optik veri okunamadı. Ayraç veya format hatası.")

    # AIR (HAVA) İÇİN ÖZEL SADE CSV OKUYUCU
    def _parse_ciddor_csv(self, material, filename):
        wl, n, k = [], [], []
        filepath = self._get_filepath(filename)
        
        print(f"[OK] Simple CSV Loading from: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            
        for line in lines[1:]: # Başlığı atla
            line = line.strip()
            if not line:
                continue
                
            # Ciddor virgülle ayrılmış (wl, n) formatında
            parts = line.split(',')
                
            if len(parts) >= 2:
                try:
                    w_val_um = float(parts[0].strip())
                    w_val_nm = w_val_um * 1000.0 # Mikrometreyi nm'ye çevir
                    
                    n_val = float(parts[1].strip())
                    k_val = 0.0 # Bu formatta k değeri her zaman 0 kabul ediliyor
                    
                    wl.append(w_val_nm)
                    n.append(n_val)
                    k.append(k_val)
                except ValueError:
                    pass
                        
        if len(wl) > 0:
            sort_idx = np.argsort(wl)
            self.optical_data[material] = {
                'wl': np.array(wl)[sort_idx],
                'n': np.array(n)[sort_idx],
                'k': np.array(k)[sort_idx]
            }
            print(f"[Success] {len(wl)} data points loaded for {material} directly from local CSV.")
        else:
            raise ValueError(f"[Fatal Error] '{filename}' dosyasından veri okunamadı.")

    def _parse_yaml_data(self, material, yaml_data):
        wl, n, k = [], [], []
        
        dataset = yaml_data.get('DATA', [])[0]
        data_str = dataset.get('data', '')
        lines = data_str.strip().split('\n')
        
        for line in lines:
            parts = line.split()
            if dataset.get('type') == 'tabulated nk':
                if len(parts) >= 3:
                    wl.append(float(parts[0]) * 1000) 
                    n.append(float(parts[1]))
                    k.append(float(parts[2]))
            elif dataset.get('type') == 'tabulated n':
                if len(parts) >= 2:
                    wl.append(float(parts[0]) * 1000)
                    n.append(float(parts[1]))
                    k.append(0.0) 
                    
        if wl:
            sort_idx = np.argsort(wl)
            self.optical_data[material] = {
                'wl': np.array(wl)[sort_idx],
                'n': np.array(n)[sort_idx],
                'k': np.array(k)[sort_idx]
            }

    def _parse_am15g_xls(self, filename="astmg173.xls"):
        """
        NREL astmg173.xls dosyasını okur, temizler ve 300-1200nm arasına hizalar.
        """
        try:
            filepath = self._get_filepath(filename)
            print(f"[OK] AM1.5G loading from: {filepath}")
            
            # Excel okuma (NREL formatına göre ilk satırı atla)
            df = pd.read_excel(filepath, engine='xlrd', skiprows=1)
            
            # 1. Ham veriyi string olarak al ve virgülleri noktaya çevir
            wl_col = df.iloc[:, 0].astype(str).str.replace(',', '.')
            irr_col = df.iloc[:, 2].astype(str).str.replace(',', '.') # Global Tilt
            
            # 2. Sayısal dönüşüm ve temizlik (Birim satırlarını NaN yapar ve atar)
            wl_numeric = pd.to_numeric(wl_col, errors='coerce')
            irr_numeric = pd.to_numeric(irr_col, errors='coerce')
            
            # Geçerli verileri birleştir ve NaN olanları (metin satırlarını) temizle
            valid_df = pd.DataFrame({'wl': wl_numeric, 'irr': irr_numeric}).dropna()
            
            # 3. Hedef Grid (300-1200nm) ve Güvenli İnterpolasyon
            target_wl = np.arange(300, 1201, 1)
            matched_irradiance = np.interp(
                target_wl, 
                valid_df['wl'].values, 
                valid_df['irr'].values, 
                left=0, 
                right=0
            )
            
            # Veriyi sınıfa sözlük olarak kaydet (Birim: W/m2/nm)
            self.am15g_data = {
                'wl': target_wl,
                'irradiance': matched_irradiance
            }
            print(f"[Success] AM1.5G spectrum aligned (300-1200 nm).")
            return self.am15g_data

        except Exception as e:
            print(f"[HATA] AM1.5G dosyası işlenirken sorun: {e}")
            return None

    def calculate_photon_flux(self):
        """
        W/m2/nm irradyansı photons/cm2/s/nm birimine çevirir. 
        Varsayım: IQE = 1 (Emilen her foton 1 EHP oluşturur).
        """
        if not hasattr(self, 'am15g_data'):
            print("[Hata] Önce AM1.5G verisi yüklenmeli!")
            return None
            
        wl_nm = self.am15g_data['wl']
        irr = self.am15g_data['irradiance'] # W/m2/nm
        
        # 1. Kuantum Dönüşümü (Vektörize)
        # Metre dönüşümünü (1e-9) ve cm2 dönüşümünü (1e-4) uyguluyoruz
        photon_flux = (irr * (wl_nm * 1e-9)) / (h * c) * 1e-4
        
        self.am15g_data['photon_flux'] = photon_flux
        
        # 2. İntegral Hesabı (Numpy 2.0 Uyumluluğu)
        if hasattr(np, 'trapezoid'):
            total_flux = np.trapezoid(photon_flux, wl_nm)
        else:
            # Eski sürüm Numpy kullananlar için (1.2x gibi)
            total_flux = np.trapz(photon_flux, wl_nm)
        
        # 3. Konsol Doğrulaması (Sanity Check)
        print(f"\n--- AM1.5G Foton Akısı Doğrulaması ---")
        print(f"Toplam Foton Akısı (300-1200nm): {total_flux:.4e} photons/cm2/s")
        print(f"300 nm Akı Değeri: {photon_flux[0]:.2e}")
        print(f"1200 nm Akı Değeri: {photon_flux[-1]:.2e}")
        
        if 2.5e17 < total_flux < 2.7e17:
            print("[Check] Veri AM1.5G standartlarıyla fiziksel olarak uyumlu.")
        else:
            print("[Uyarı] Toplam akı beklenen değerden saptı, birimleri kontrol edin.")
            
        return photon_flux

    def generate_generation_profile(self, z_array_nm, absorption_matrix, layers_info):
        """
        Vektörize edilmiş, Simpson tabanlı ve Mesh-korumalı SCAPS G(z) üretici.
        """
        print("[DEBUG] Dosya yazma işlemi başladı...")
        import numpy as np
        from scipy.integrate import simpson
        import traceback
        
        try:
            if not hasattr(self, 'am15g_data') or 'photon_flux' not in self.am15g_data:
                print("[Hata] Foton akısı verisi bulunamadı! Önce am15g_data yüklenmeli ve foton akısı hesaplanmalı.")
                return None, None
                
            wl_nm = self.am15g_data['wl']
            flux = self.am15g_data['photon_flux'] # Shape: (901,)
            
            print("[Info] Vektörize İntegrasyon ve Mutlak Parazitik Maskeleme başlatılıyor...")
            
            # 1. VEKTÖRİZE SİMPSON İNTEGRASYONU (Döngüsüz, yüksek hız)
            # KRİTİK BİRİM DENETİMİ:
            # flux = photons / (cm^2 * s * nm)
            # absorption_matrix şu an $\mu m^{-1}$ birimindedir. Bunu cm^{-1} birimine çevirmek için
            # 1 \mu m = 1e-4 cm => 1 / \mu m = 1e4 / cm. Dolayısıyla 10000 ile çarpıyoruz.
            absorption_matrix_cm = absorption_matrix * 10000.0
            
            spectral_gen = flux * absorption_matrix_cm 
            G_z_cm3 = simpson(y=spectral_gen, x=wl_nm, axis=-1)
            
            # 2. MUTLAK PARAZİTİK MASKE UYGULAMASI (Sadece 'Si' katmanı)
            active_materials = ['Si'] 
            mask = np.array([
                1.0 if self._get_material_at_z(z, layers_info) in active_materials else 0.0 
                for z in z_array_nm
            ])
            G_z_cm3 = G_z_cm3 * mask
                
            # 3. SCAPS-1D HİZALAMASI (ÖN YÜZEYDEN MAKSİMUM ABSORPSİYON)
            # 1e6 çarpanı kaldırıldı (SCAPS G(z) verisini cm^-3 bazında bekler)
            # Işığın giriş yönü Z=0 olduğundan, z_array_nm halihazırda x=0 noktasında Jenerasyon maksimumdur.
            # Ters çevirmeye veya aynalamaya (mirroring) fiziksel olarak ihtiyaç yoktur.
            x_orig_um = z_array_nm / 1000.0
            
            # --- YENİ EKLENEN ENTERPOLASYON (1000 Nokta) KISMI ---
            # Hücrenin sıfır noktası ve kalınlığı koruyarak fiziksel yönlü (0 -> x_max) interp
            target_x = np.linspace(0.0, np.max(x_orig_um), 1000)
            target_G = np.interp(target_x, x_orig_um, G_z_cm3)
            
            print(f"[SCAPS Limit] Veri enterpolasyon ile 1000 noktaya, cm^-3 bazında normalize edildi. Kalınlık: {target_x.max():.2f} um")
            print(f"[SCAPS Check] Maksimum Jenerasyon (Top Surface): {target_G.max():.2e} cm^-3.s^-1")
            
            # 4. DOSYA İHRACATI (BAŞLIKSIZ - DOĞRUDAN VERİ)
            filename = "astmg173_generation.gen"
            with open(filename, 'w') as f:
                for x_val, g_val in zip(target_x, target_G):
                    f.write(f"{x_val:.8f}\t{g_val:.4E}\n")
                    
            print(f"[Success] Başlıksız SCAPS jenerasyon dosyası hazır: {filename}")
            return target_x, target_G

        except Exception as e:
            print(f"[HATA] Generation Profile üretilirken hata oluştu: {e}")
            traceback.print_exc()
            return None, None

    def _get_material_at_z(self, z, layers_info):
        """
        Z derinliğine göre katmanın malzemesini döndürür.
        layers_info dict yapısında olmalıdır: [{'material': 'SiNx', 'thickness_nm': 70}, {'material': 'Si', 'thickness_nm': 150000}]
        """
        current_z = 0.0
        for layer in layers_info:
            thick = float(layer.get('thickness_nm', 0.0))
            if current_z <= z <= current_z + thick:
                return layer.get('material', '')
            current_z += thick
        return 'Unknown'

    def interpolate_data(self, material_list):
        target_wl = np.arange(300, 1201, 1) 
        materials_to_process = set(material_list)
        if 'Si' not in materials_to_process:
            materials_to_process.add('Si')
            
        self.interpolated_data = {}
        for mat in materials_to_process:
            if mat not in self.optical_data:
                raise DataError(f"[Fatal Error] Optical data for {mat} was not successfully loaded. Halting simulation.")
                
            data = self.optical_data[mat]
            
            n_interp = interp1d(data['wl'], data['n'], kind='linear', bounds_error=False, 
                                fill_value=(data['n'][0], data['n'][-1]))(target_wl)
            
            k_interp = interp1d(data['wl'], data['k'], kind='linear', bounds_error=False, 
                                fill_value=(data['k'][0], data['k'][-1]))(target_wl)
            
            k_interp = np.clip(k_interp, 0.0, None)
            
            self.interpolated_data[mat] = {
                'wl': target_wl,
                'n': n_interp,
                'k': k_interp
            }
            
    def run(self, material_list):
        self.fetch_data(material_list)
        self.interpolate_data(material_list)
        return self.interpolated_data

class DataError(Exception):
    pass