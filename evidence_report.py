"""Observable file evidence and mode-specific verification guidance.
No fraud score or source claim is inferred from image quality or unsigned EXIF.
"""
import io
from PIL import Image

GUIDANCE = {
    'general': [
        ('Kaynağı karşılaştır', 'Görselin ilk yayımlandığı kaynağı ve varsa orijinal dosyasını iste.'),
        ('Bağımsız örneklerle kontrol et', 'Tek bir model skoru, görselin kaynağını kanıtlamaz.'),
    ],
    'car': [
        ('Aynı aracı doğrula', 'Satıcıdan aracın farklı açılardan güncel fotoğraflarını ve canlı görüntüsünü iste.'),
        ('Kimlik ve araç kaydı', 'Satıcının yetkisini ve araç bilgilerinin belgelerle tutarlılığını ayrıca kontrol et.'),
        ('Bağımsız inceleme', 'Fotoğraf analizi mekanik durumu, hasar geçmişini veya satıcının güvenilirliğini doğrulamaz.'),
    ],
    'realestate': [
        ('Aynı mülkü doğrula', 'Konumu ve oda düzenini yerinde ziyaret veya canlı görüntüyle karşılaştır.'),
        ('Yetkiyi doğrula', 'İlan verenin mülk sahibi veya yetkili kişi olduğunu ayrıca kontrol et.'),
        ('Belgeleri karşılaştır', 'Fotoğraf analizi mülkiyeti, adresi veya ilan fiyatının doğruluğunu doğrulamaz.'),
    ],
}

def build_report(contents, mode, breakdown, verdict, degraded=False, disagreement=False, category='not_checked'):
    with Image.open(io.BytesIO(contents)) as image:
        width, height = image.size
        file_format = image.format
        exif = image.getexif()
        # Only presence is reported; GPS, dates, serial numbers and camera values are not returned.
        metadata_present = bool(exif)
    observations = [
        {'title': 'Görsel boyutu', 'detail': f'{width} × {height} piksel · {file_format}'},
        {'title': 'Dosya metaverisi', 'detail': 'EXIF mevcut; özgünlük kanıtı değildir.' if metadata_present else 'EXIF bulunamadı; bu tek başına AI işareti değildir.'},
    ]
    if min(width, height) < 256:
        observations.append({'title': 'Küçük görsel', 'detail': 'Küçük görsel ayrıntıları sınırlar. Mümkünse orijinal dosyayı kullan.'})
    reasons = []
    if degraded:
        reasons.append('Bazı modeller cevap vermedi; mevcut skor eksik model grubuna dayanıyor.')
    if disagreement:
        reasons.append('Modellerin AI skorları arasında 40 puandan fazla fark var.')
    if verdict == 'UNCERTAIN' and not reasons:
        reasons.append('Skor karar eşiğinin belirsizlik aralığında.')
    if not reasons:
        reasons.append('Karar mevcut model skoruna ve ayarlanmış eşiğe dayanıyor.')
    if mode != 'general':
        if isinstance(category, dict):
            status = category.get('status')
            if status == 'mismatch':
                detail = 'Görsel seçili bölüme uymuyor olabilir. Genel taramayı veya uygun bölümü kullanabilirsiniz; AI skoru bu uyarıdan bağımsızdır.'
            elif status == 'compatible':
                detail = 'Araçla ilişkili görsel işaretler bulundu.' if mode == 'car' else 'Bina veya iç mekânla ilişkili nesneler bulundu; mülkün kimliği doğrulanmadı.'
            else:
                detail = 'Görsel türü güvenle belirlenemedi; fotoğraf analizi sürdürüldü.'
            observations.append({'title': 'Kategori kontrolü', 'detail': detail})
        elif category == 'not_checked':
            pass  # Mode selects guidance; it does not change the AI score.
        elif category == 'unknown':
            observations.append({'title': 'Kategori kontrolü', 'detail': 'Kategori modeli cevap vermedi; araç veya mülk içeriği doğrulanamadı.'})
        elif category != mode:
            observations.append({'title': 'Kategori kontrolü', 'detail': 'Kategori modeli seçili modla eşleşmedi; sonuçları bu sınırlamayla değerlendir.'})
        else:
            observations.append({'title': 'Kategori kontrolü', 'detail': 'Kategori modeli seçili modla eşleşti; ilanın doğruluğunu kanıtlamaz.'})
    return {
        'version': '2.0', 'mode': mode, 'observations': observations, 'decision_reasons': reasons,
        'models': breakdown,
        'category_check': category if isinstance(category, dict) else {'status': 'not_checked', 'detected': 'unknown'},
        'verification_steps': [{'title': title, 'detail': detail} for title, detail in GUIDANCE[mode]],
        'limitations': 'Model skoru kalibre edilmiş bir olasılık değildir. Gerçek fotoğraf kullanılmış olması ilanı güvenilir yapmaz.',
        'provenance': 'not_verified', 'fraud_assessment': 'not_determined',
        'learning': 'Kullanıcı geri bildirimi inceleme adayıdır; otomatik eğitim veya doğrulanmış gerçek etiket sayılmaz.',
    }
