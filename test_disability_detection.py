
import numpy as np
import json
from pathlib import Path
from detailed_head_analysis import (
    group_files_by_person, 
    extract_behavior_features,
    analyze_person_disability,
    create_disability_report
)

def test_grouping():
    """Testează funcția de grupare a fișierelor"""
    print("🧪 Testare grupare fișiere...")
    
    # Simulează fișierele existente
    test_files = [
        Path("1-A.json"), Path("1-B.json"), Path("1-C.json"),
        Path("2-A.json"), Path("2-B.json"), Path("2-C.json"),
        Path("3-A.json"), Path("3-B.json"), Path("3-C.json")
    ]
    
    grouped = group_files_by_person(test_files)
    
    print(f"Fișiere găsite: {len(test_files)}")
    print(f"Persoane identificate: {len(grouped)}")
    
    for person, files in grouped.items():
        print(f"  {person}: {len(files)} scenarii")
        for file in files:
            print(f"    - {file.name}")
    
    print("✅ Testare grupare completă!\n")
    return grouped

def test_feature_extraction():
    """Testează extragerea de caracteristici"""
    print("🧪 Testare extragere caracteristici...")
    
    # Simulează date pentru un scenariu
    n_points = 100
    scenario_data = np.random.randn(n_points, 10)  # 10 coloane (3 pos + 3 rot + 3 forward + 1 time)
    
    try:
        features = extract_behavior_features(scenario_data)
        print(f"Caracteristici extrase: {len(features)}")
        print(f"Tip date: {type(features)}")
        print(f"Formă: {features.shape}")
        print("✅ Testare extragere caracteristici completă!\n")
        return features
    except Exception as e:
        print(f"❌ Eroare la extragerea caracteristicilor: {e}")
        return None

def test_disability_analysis():
    """Testează analiza dizabilităților"""
    print("🧪 Testare analiză dizabilități...")
    
    # Simulează caracteristici pentru 5 persoane
    n_persons = 5
    n_features = 50  # Numărul de caracteristici extrase
    
    all_person_features = {}
    for i in range(n_persons):
        person_id = f"Person_{i+1}"
        # Simulează că unele persoane au comportament normal, altele anormal
        if i < 2:  # Primele 2 persoane au comportament normal
            features = np.random.randn(n_features) * 0.1
        else:  # Ultimele 3 persoane au comportament anormal
            features = np.random.randn(n_features) * 2.0
        
        all_person_features[person_id] = features
    
    try:
        disability_assessment = analyze_person_disability(all_person_features)
        print(f"Persoane analizate: {len(disability_assessment)}")
        
        # Afișează rezultatele
        for person, result in disability_assessment.items():
            print(f"  {person}: {result['status']} (score: {result['final_score']:.3f})")
        
        print("✅ Testare analiză dizabilități completă!\n")
        return disability_assessment
    except Exception as e:
        print(f"❌ Eroare la analiza dizabilităților: {e}")
        return None

def test_report_generation():
    """Testează generarea rapoartelor"""
    print("🧪 Testare generare rapoarte...")
    
    # Folosește rezultatele de la testul anterior
    disability_assessment = test_disability_analysis()
    
    if disability_assessment:
        try:
            create_disability_report(disability_assessment)
            print("✅ Testare generare rapoarte completă!\n")
        except Exception as e:
            print(f"❌ Eroare la generarea rapoartelor: {e}")
    else:
        print("❌ Nu se pot genera rapoartele fără analiza dizabilităților")

def test_file_structure():
    """Testează structura fișierelor existente"""
    print("🧪 Testare structură fișiere...")
    
    vr_dir = Path("vr_recordings")
    if vr_dir.exists():
        json_files = list(vr_dir.glob("*.json"))
        print(f"Fișiere JSON găsite: {len(json_files)}")
        
        if json_files:
            # Testează grupare pe fișierele reale
            grouped = group_files_by_person(json_files)
            print(f"Persoane identificate: {len(grouped)}")
            
            for person, files in grouped.items():
                print(f"  {person}: {len(files)} scenarii")
                for file in files:
                    print(f"    - {file.name}")
            
            print("✅ Testare structură fișiere completă!\n")
            return True
        else:
            print("❌ Nu s-au găsit fișiere JSON în vr_recordings/")
            return False
    else:
        print("❌ Directorul vr_recordings/ nu există")
        return False

def main():
    """Funcția principală de testare"""
    print("🚀 ÎNCEPERE TESTARE SISTEM DETECTARE DIZABILITĂȚI")
    print("=" * 60)
    
    # Testează funcționalitățile de bază
    test_grouping()
    test_feature_extraction()
    test_disability_analysis()
    test_report_generation()
    
    # Testează cu fișierele reale
    print("🔍 TESTARE CU FIȘIERE REALE:")
    test_file_structure()
    
    print("🎯 TESTAREA S-A TERMINAT!")
    print("=" * 60)
    print("\n💡 Pentru a rula analiza completă:")
    print("   python detailed_head_analysis.py")
    print("\n📚 Pentru mai multe informații:")
    print("   Consultați README_DISABILITY_DETECTION.md")

if __name__ == "__main__":
    main()
