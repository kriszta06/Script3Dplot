
# Scorurile pentru clasificarea nivelului de dizabilitate
DISABILITY_THRESHOLDS = {
    'SEVERE': 0.7,      # Score > 0.7 = Dizabilitate severă
    'MODERATE': 0.5,    # Score 0.5-0.7 = Dizabilitate moderată  
    'LIGHT': 0.3,       # Score 0.3-0.5 = Dizabilitate ușoară
    'NONE': 0.2         # Score ≤ 0.2 = Fără dizabilități
}

# Cum se combină diferitele scoruri pentru scorul final
SCORE_WEIGHTS = {
    'distance': 0.5,        # 50% - Scorul de distanță (comportament anormal)
    'mahalanobis': 0.3,    # 30% - Scorul Mahalanobis (pattern-uri nestandard)
    'consistency': 0.2      # 20% - Scorul de consistență (inconsistență între scenarii)
}

# Parametrii pentru DBSCAN clustering
CLUSTERING_PARAMS = {
    'eps': 1.5,           # Distanța maximă între puncte pentru a forma un cluster
    'min_samples': 2,     # Numărul minim de puncte pentru a forma un cluster
    'metric': 'euclidean' # Metrica de distanță
}

# Parametrii pentru PCA
PCA_PARAMS = {
    'n_components': 0.95,  # Păstrează 95% din varianță
    'random_state': 42     # Pentru reproducibilitate
}

# Parametrii pentru analiza mișcării
MOVEMENT_ANALYSIS = {
    'sharp_turn_threshold': 30,  # Grade pentru a considera un viraj "brusc"
    'subsampling_factor': 10,    # Factorul de sub-eșantionare pentru analiză
    'histogram_bins': 5,         # Numărul de bin-uri pentru histograma mișcărilor
    'min_points_for_autocorr': 10  # Puncte minime pentru calculul autocorelării
}


# Ce să se includă în rapoarte
REPORTING_CONFIG = {
    'include_raw_features': True,     # Include caracteristicile brute în JSON
    'include_risk_factors': True,     # Include factorii de risc identificați
    'include_cluster_analysis': True, # Include analiza cluster-elor
    'save_visual_report': True,       # Salvează raportul vizual
    'save_individual_plots': False    # Salvează plot-urile individuale (opțional)
}

# Culorile pentru diferitele niveluri de dizabilitate
DISABILITY_COLORS = {
    'HIGH': 'red',      # Dizabilitate severă
    'MEDIUM': 'orange', # Dizabilitate moderată
    'LOW': 'yellow',    # Dizabilitate ușoară
    'NONE': 'green'     # Fără dizabilități
}

# Mesajele pentru diferitele niveluri de dizabilitate (doar titluri)
DISABILITY_MESSAGES = {
    'HIGH': '🔴 DIZABILITATE SEVERĂ',
    'MEDIUM': '🟠 DIZABILITATE MODERATĂ',
    'LOW': '🟡 DIZABILITATE UȘOARĂ',
    'NONE': '🟢 FĂRĂ DIZABILITĂȚI'
}

def validate_config():
    """Validează configurația pentru a se asigura că este corectă"""
    errors = []
    
    # Verifică threshold-urile
    if DISABILITY_THRESHOLDS['SEVERE'] <= DISABILITY_THRESHOLDS['MODERATE']:
        errors.append("SEVERE threshold trebuie să fie > MODERATE threshold")
    
    if DISABILITY_THRESHOLDS['MODERATE'] <= DISABILITY_THRESHOLDS['LIGHT']:
        errors.append("MODERATE threshold trebuie să fie > LIGHT threshold")
    
    if DISABILITY_THRESHOLDS['LIGHT'] <= DISABILITY_THRESHOLDS['NONE']:
        errors.append("LIGHT threshold trebuie să fie > NONE threshold")
    
    # Verifică ponderile
    total_weight = sum(SCORE_WEIGHTS.values())
    if abs(total_weight - 1.0) > 0.001:
        errors.append(f"Ponderile trebuie să adune 1.0, nu {total_weight}")
    
    # Verifică parametrii clustering
    if CLUSTERING_PARAMS['eps'] <= 0:
        errors.append("eps trebuie să fie > 0")
    
    if CLUSTERING_PARAMS['min_samples'] < 1:
        errors.append("min_samples trebuie să fie >= 1")
    
    # Verifică parametrii PCA
    if PCA_PARAMS['n_components'] <= 0 or PCA_PARAMS['n_components'] > 1:
        errors.append("n_components trebuie să fie între 0 și 1")
    
    if errors:
        print("❌ Erori în configurație:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    print("✅ Configurația este validă!")
    return True

def print_config_summary():
    """Afișează un rezumat al configurației"""
    print("\n📋 CONFIGURAȚIA SISTEMULUI DE DETECTARE A DIZABILITĂȚILOR")
    print("=" * 70)
    
    print(f"\n🎯 THRESHOLD-URI:")
    for level, threshold in DISABILITY_THRESHOLDS.items():
        print(f"   {level}: {threshold}")
    
    print(f"\n⚖️ PONDERI SCOR FINAL:")
    for component, weight in SCORE_WEIGHTS.items():
        print(f"   {component}: {weight*100}%")
    
    print(f"\n🔗 PARAMETRI CLUSTERING:")
    for param, value in CLUSTERING_PARAMS.items():
        print(f"   {param}: {value}")
    
    print(f"\n📊 PARAMETRI PCA:")
    for param, value in PCA_PARAMS.items():
        print(f"   {param}: {value}")
    
    print(f"\n🎨 CULORI:")
    for status, color in DISABILITY_COLORS.items():
        print(f"   {status}: {color}")
    
    print("=" * 70)

if __name__ == "__main__":
    # Validează și afișează configurația
    print_config_summary()
    validate_config()
