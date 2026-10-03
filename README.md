# AI Image Detector

Genel, Araba ve Emlak modları aynı ConvNeXt analiz motorunu kullanır. Fotoğraflar standart analizde Render sunucusunda işlenir; Hugging Face API'sine gönderilmez ve eğitim için saklanmaz. Sonuçlar model skorudur, kesin kaynak veya dolandırıcılık tespiti değildir.

## Ölçüm — 3 Ekim 2026

Model seçimi için kullanılan ilk 200 görsel: 158 doğru, 39 yanlış, 3 belirsiz (%79). Ayrı 200 görsel: 170 doğru, 26 yanlış, 4 belirsiz (%85). Toplam: 328/400 (%82). Her grup 100 AI ve 100 gerçek görsel içerir. Belirsizler başarı sayılmaz. Eğitim verisiyle örtüşme bilinmiyor; Gemini, araba ve emlak için ayrı başarı oranı ölçülmedi.

Kaynaklar: bitmind/open-images-v7-subset, bitmind/MS-COCO-unique___FLUX.1-dev, bitmind/MS-COCO-unique___stable-diffusion-xl-base-1.0. Yayıncı etiketleri kullanıldı; tüm örneklerin kaynağı bağımsız olarak denetlenmedi. Örnekler rastgele satır aralıklarından alındı. Bu test tüm yeni görsellerde aynı başarıyı garanti etmez.

## Render

Mevcut Python 3.11 ortamı ve `pip install -r requirements.txt` kurulumu uygundur. Başlatma: `uvicorn app:app --host 0.0.0.0 --port $PORT` (tek worker). Standart motor `local` olduğu için AI_MODELS ayarı kullanılmaz; değiştirmeniz gerekmez. İsteğe bağlı AI_BACKEND=hf eski API motoruna dönüş içindir.

`detector_model.part00`–`part15` dosyaları ve manifest aynı klasörde bulunmalıdır. İlk analizde bütünlükleri kontrol edilerek tek ONNX dosyasına birleştirilirler. Bozuk/eksik modelde sonuç üretilmez. Dosyaları parçalara bölme amacı GitHub tarayıcı yükleme sınırına uymaktır.

Ücretsiz sunucuda bellek için aynı anda tek analiz çalışır; diğer istekler tekrar deneme mesajı alır. Dosya sınırı 20 MB ve 20 megapikseldir. Ücretsiz Render uykudan uyanırken ve yoğunken analiz daha uzun sürebilir. Windows testinde uygulamanın tepe çalışma belleği yaklaşık 313 MB ölçüldü; bu Linux/Render bellek garantisi değildir.

Kullanıcı geri bildirimi otomatik model eğitimi veya doğrulanmış etiket değildir. `maintenance.py` aktif motorun sağlık kontrolünü ve etiketli CSV ile ölçümünü yapabilir; zamanlayıcı ve otomatik model terfisi etkin değildir.

## Model ve lisans

[xRayon/convnext-ai-images-detector](https://huggingface.co/xRayon/convnext-ai-images-detector), sürüm `1b4d270be0590cde4320fa7503d68b44bb7ee77e`. Yayımlanan checkpoint'ten ONNX opset 17 olarak dönüştürüldü; doğrusal ağırlıklara dinamik INT8 küçültme uygulandı. ONNX Runtime 1.30.0 kullanılır. Kaynak ve türetilmiş dosya SHA256 değerleri manifesttedir. Modelin MIT lisansı MODEL-LICENSE.txt dosyasındadır.
