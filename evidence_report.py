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

def _build_report(contents, mode, breakdown, verdict, degraded=False, disagreement=False, category='not_checked'):
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

GUIDANCE_EN = {
    'general': [
        ('Compare the source', 'Ask for the original file and the source where the image was first published.'),
        ('Check independent examples', 'A single model score cannot establish the origin of an image.'),
    ],
    'car': [
        ('Verify the same vehicle', 'Ask the seller for recent photos from several angles and a live view of the vehicle.'),
        ('Identity and vehicle records', 'Check the seller’s authority and compare vehicle details with the documents.'),
        ('Independent inspection', 'Photo analysis cannot establish mechanical condition, damage history, or seller trustworthiness.'),
    ],
    'realestate': [
        ('Verify the same property', 'Compare the location and room layout during a visit or live viewing.'),
        ('Verify authority', 'Separately check whether the advertiser owns the property or is authorized to represent it.'),
        ('Compare documents', 'Photo analysis cannot establish ownership, the address, or whether the asking price is accurate.'),
    ],
}

EN_TEXT = {
    'Görsel boyutu': 'Image dimensions',
    'Dosya metaverisi': 'File metadata',
    'EXIF mevcut; özgünlük kanıtı değildir.': 'EXIF is present; it does not prove authenticity.',
    'EXIF bulunamadı; bu tek başına AI işareti değildir.': 'No EXIF was found; this alone is not evidence of AI generation.',
    'Küçük görsel': 'Small image',
    'Küçük görsel ayrıntıları sınırlar. Mümkünse orijinal dosyayı kullan.': 'Small images limit visible detail. Use the original file if possible.',
    'Bazı modeller cevap vermedi; mevcut skor eksik model grubuna dayanıyor.': 'Some models did not respond; the score uses an incomplete model group.',
    'Modellerin AI skorları arasında 40 puandan fazla fark var.': 'The models’ AI scores differ by more than 40 points.',
    'Skor karar eşiğinin belirsizlik aralığında.': 'The score is within the uncertainty range around the decision threshold.',
    'Karar mevcut model skoruna ve ayarlanmış eşiğe dayanıyor.': 'The decision uses the current model score and configured threshold.',
    'Kategori kontrolü': 'Category check',
    'Görsel seçili bölüme uymuyor olabilir. Genel taramayı veya uygun bölümü kullanabilirsiniz; AI skoru bu uyarıdan bağımsızdır.': 'The image may not match the selected section. Try General Scan or the appropriate section; this warning does not affect the AI score.',
    'Araçla ilişkili görsel işaretler bulundu.': 'Vehicle-related visual cues were found.',
    'Bina veya iç mekânla ilişkili nesneler bulundu; mülkün kimliği doğrulanmadı.': 'Building or interior-related objects were found; the identity of the property was not verified.',
    'Görsel türü güvenle belirlenemedi; fotoğraf analizi sürdürüldü.': 'The image type could not be determined confidently; image analysis continued.',
    'Kategori modeli cevap vermedi; araç veya mülk içeriği doğrulanamadı.': 'The category model did not respond; vehicle or property content could not be verified.',
    'Kategori modeli seçili modla eşleşmedi; sonuçları bu sınırlamayla değerlendir.': 'The category model did not match the selected mode; consider this limitation when interpreting the result.',
    'Kategori modeli seçili modla eşleşti; ilanın doğruluğunu kanıtlamaz.': 'The category model matched the selected mode; this does not prove that an advertisement is accurate.',
}

def build_report(contents, mode, breakdown, verdict, degraded=False, disagreement=False, category='not_checked'):
    report = _build_report(contents, mode, breakdown, verdict, degraded, disagreement, category)
    english = {
        'observations': [{
            'title': EN_TEXT[item['title']],
            'detail': item['detail'].replace('piksel', 'pixels') if item['title'] == 'Görsel boyutu' else EN_TEXT[item['detail']],
        } for item in report['observations']],
        'decision_reasons': [EN_TEXT[item] for item in report['decision_reasons']],
        'verification_steps': [{'title': title, 'detail': detail} for title, detail in GUIDANCE_EN[mode]],
        'limitations': 'Model scores are not calibrated probabilities. A real photo does not establish that an advertisement is trustworthy.',
        'learning': 'User feedback is a candidate for review; it is not automatic training or a verified ground-truth label.',
    }
    report['localized'] = {'en': english}
    return report
