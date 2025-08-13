
import re
from pathlib import Path

def test_regex():
    """Testează regex-ul pentru diferite formate de fișiere"""
    print("🧪 TESTARE REGEX PENTRU GRUPAREA FIȘIERELOR")
    print("=" * 50)
    
    # Testează regex-ul actual
    regex_pattern = r'(\d+)-[A-Za-z]+\.json$'
    
    # Fișiere de test
    test_files = [
        '1-A.json', '1-B.json', '1-C.json',
        '2-A.json', '2-B.json', '2-C.json',
        '10-A.json', '10-B.json', '10-C.json',
        '13-A.json', '13-B.json', '13-C.json'
    ]
    
    print(f"Regex pattern: {regex_pattern}")
    print(f"\nTestare fișiere:")
    
    for filename in test_files:
        match = re.search(regex_pattern, filename)
        if match:
            person_id = match.group(1)
            print(f"  ✅ {filename} -> Person_{person_id}")
        else:
            print(f"  ❌ {filename} -> NO MATCH")
    
    # Testează cu fișierele reale
    print(f"\n🔍 TESTARE CU FIȘIERE REALE:")
    vr_dir = Path("vr_recordings")
    if vr_dir.exists():
        json_files = list(vr_dir.glob("*.json"))
        print(f"Fișiere JSON găsite: {len(json_files)}")
        
        # Testează primul fișier
        if json_files:
            first_file = json_files[0]
            print(f"Primul fișier: {first_file.name}")
            
            match = re.search(regex_pattern, first_file.name)
            if match:
                person_id = match.group(1)
                print(f"  ✅ Match: Person_{person_id}")
            else:
                print(f"  ❌ NO MATCH")
                
            # Testează câteva fișiere
            print(f"\nTestare câteva fișiere:")
            for i, file_path in enumerate(json_files[:10]):
                match = re.search(regex_pattern, file_path.name)
                if match:
                    person_id = match.group(1)
                    print(f"  ✅ {file_path.name} -> Person_{person_id}")
                else:
                    print(f"  ❌ {file_path.name} -> NO MATCH")
                    
                if i >= 9:  # Limitează la 10 fișiere
                    break
    else:
        print("❌ Directorul vr_recordings/ nu există")

if __name__ == "__main__":
    test_regex()
