# AI Image Detector

Genel, Araba ve Emlak modları aynı ConvNeXt analiz motorunu kullanır. Fotoğraflar standart analizde Render sunucusunda işlenir; Hugging Face API'sine gönderilmez ve eğitim için saklanmaz. Sonuçlar model skorudur, kesin kaynak veya dolandırıcılık tespiti değildir.

## Ölçüm — 3 Ekim 2026

Yerel Windows testinde model seçimi için kullanılan ilk 200 görsel: 158 doğru, 39 yanlış, 3 belirsiz (%79). Ayrı 200 görsel: 170 doğru, 26 yanlış, 4 belirsiz (%85). Toplam: 328/400 (%82). Her grup 100 AI ve 100 gerçek görsel içerir. Belirsizler başarı sayılmaz. Eğitim verisiyle örtüşme bilinmiyor; Bu genel test Gemini, araba ve emlak için ayrı başarı oranı ölçmez.

Kaynaklar: bitmind/open-images-v7-subset, bitmind/MS-COCO-unique___FLUX.1-dev, bitmind/MS-COCO-unique___stable-diffusion-xl-base-1.0. Yayıncı etiketleri kullanıldı; tüm örneklerin kaynağı bağımsız olarak denetlenmedi. Örnekler rastgele satır aralıklarından alındı. Bu test tüm yeni görsellerde aynı başarıyı garanti etmez.

Canlı Render tekrar testi: aynı ayrı 200 görselde 169 doğru, 29 yanlış, 2 belirsiz, 0 istek hatası (%84,5). AI: 83/100; gerçek: 86/100. Bu tekrar yeni bir veri grubu değildir. Yerel Windows skorlarıyla canlı Linux skorları farklılaşabildiği için canlı sonuç ayrıca ölçüldü.

## Render

Mevcut Python 3.11 ortamı ve `pip install -r requirements.txt` kurulumu uygundur. Başlatma: `uvicorn app:app --host 0.0.0.0 --port $PORT` (tek worker). Standart motor `local` olduğu için AI_MODELS ayarı kullanılmaz; değiştirmeniz gerekmez. İsteğe bağlı AI_BACKEND=hf eski API motoruna dönüş içindir.

`detector_model.part00`–`part15` dosyaları ve manifest aynı klasörde bulunmalıdır. İlk analizde bütünlükleri kontrol edilerek tek ONNX dosyasına birleştirilirler. Bozuk/eksik modelde sonuç üretilmez. Dosyaları parçalara bölme amacı GitHub tarayıcı yükleme sınırına uymaktır.

Ücretsiz sunucuda bellek için aynı anda tek analiz çalışır; diğer istekler tekrar deneme mesajı alır. Dosya sınırı 20 MB ve 20 megapikseldir. Ücretsiz Render uykudan uyanırken ve yoğunken analiz daha uzun sürebilir. Windows testinde uygulamanın tepe çalışma belleği yaklaşık 313 MB ölçüldü; bu Linux/Render bellek garantisi değildir.

Kullanıcı geri bildirimi otomatik model eğitimi veya doğrulanmış etiket değildir. `maintenance.py` aktif motorun sağlık kontrolünü ve etiketli CSV ile ölçümünü yapabilir; zamanlayıcı ve otomatik model terfisi etkin değildir.

## Model ve lisans

[xRayon/convnext-ai-images-detector](https://huggingface.co/xRayon/convnext-ai-images-detector), sürüm `1b4d270be0590cde4320fa7503d68b44bb7ee77e`. Yayımlanan checkpoint'ten ONNX opset 17 olarak dönüştürüldü; doğrusal ağırlıklara dinamik INT8 küçültme uygulandı. ONNX Runtime 1.30.0 kullanılır. Kaynak ve türetilmiş dosya SHA256 değerleri manifesttedir. Modelin MIT lisansı MODEL-LICENSE.txt dosyasındadır.


## Yerel kategori kontrolü — 7 Ekim 2026

Araba ve Emlak modlarında küçük bir MobileNetV3 modeli, araç veya bina/iç mekânla ilişkili nesne ipuçlarını kontrol eder. Fotoğraf başka bir servise gönderilmez. Model skoru kesin kategori olasılığı değildir; genel amaçlı ImageNet sınıfları kullanılır. Bina/mobilya/ev aleti ipucu mülk kimliğini kanıtlamaz. Fotoğrafın ortası ve tamamı ayrı değerlendirilir. Zayıf ipuçlarında iki görünümün anlaşması gerekir; tek başına tekerlek/yorgan yeterli değildir. Uyumsuz, belirsiz veya kullanılamayan kategori kontrolü Araba/Konut analizini durdurur; misafir deneme hakkı korunur. Genel Tarama kategori kontrolü yapmaz. Bu kontrol ana AI skorunu değiştirmez.

`category_model.part00`, `category_model.part01`, `category_model_manifest.json`, `category_labels.json` ve `category_detector.py` aynı klasörde bulunmalıdır. Model ve etiket SHA-256 değerleri doğrulanır. Dosya bozuk veya eksikse Araba/Konut analizi durur; Genel Tarama ayrıca kullanılabilir. Bu model PyTorch/Torchvision MobileNetV3 Small IMAGENET1K_V1 ağırlıklarının ONNX dönüşümüdür. Kaynak: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html . Torchvision BSD lisans bildirimi CATEGORY-MODEL-LICENSE.txt dosyasındadır. Model ağırlıklarının ve eğitim veri kaynağının kullanım koşulları ayrıca geçerlidir.

Son karar politikası `two_view_evidence_v3`. Önceki 500 örnek ve ek 250 örnekten farklı, SHA-256 çakışması olmayan son 125 yerel doğrulama örneğinde kategori kabulü: araba 34/50→37/50, yatak odası 25/50→41/50, özgün iç mekân 11/25→18/25. Son grupta araba/konut çapraz kabul çıkmadı; bu küçük örneklem genel kategori garantisi değildir. Ayrı 11 alakasız fotoğraf da iki bölümde reddedildi. Kaynaklar CNNDetection/ProGAN, LSUN-Bedroom ve genai-manipulation-detection-interior yayıncı etiketleridir; eğitim örtüşmesi ve bağımsız köken doğrulaması bilinmiyor. Son seçim tohumu 2026100703. Windows CPU, ONNX Runtime 1.30.0; canlı Linux tekrar testi değildir. Bu oranlar AI/gerçek doğruluğu değil, kategori kabulüdür.

Araba için geniş orta kesit, konut için sol/sağ kesit ayrıca aynı AI modeliyle karşılaştırılır. Sabit kesitler hasar veya dolandırıcılık tespiti değildir; ana fotoğraf kararını değiştirmez. 384 piksel altındaki kısa kenarlarda bu ek kontrol atlanır. Aşağıdaki AI oranları kategori zorunluluğu eklenmeden önceki ana model testlerine aittir; güncel tüm akışın başarı oranı gibi sunulmamalıdır.

Misafirler bir başarılı fotoğraf analizinden sonra ücretsiz hesaba giriş yapmalıdır. Başarısız ve kategori nedeniyle reddedilen analizler bu hakkı tüketmez. Fotoğraflar eğitim için saklanmaz; otomatik öğrenme etkin değildir.

## Ayrı araba AI testi

Canlı sitede 100 yayıncı etiketli gerçek + 100 ProGAN üretimi araba görseli: 173 doğru, 23 yanlış, 4 belirsiz (%86,5). AI 92/100, gerçek 81/100. Kaynak CNNDetection/ProGAN test grubudur; tek üretici, bağımsız kamera kökeni denetimi yok, eğitim örtüşmesi bilinmiyor. Bu oran diğer üreticilere veya tüm ilanlara genellenmez.


## Ayrı yatak odası AI testi

Ricker ve arkadaşlarının LSUN-Bedroom araştırma test grubundan 100 gerçek + 100 üretilmiş oda: canlı sitede 150 doğru, 46 yanlış, 4 belirsiz (%75). Gerçek 85/100, AI 65/100; ProGAN, StyleGAN, ProjectedGAN, Diff-StyleGAN2, Diff-ProjectedGAN, DDPM, IDDPM, ADM, PNDM ve LDM üreticilerinden 10'ar örnek. Kaynak https://zenodo.org/records/7528113 . Rastgele seçim tohumu 20261006. Eğitim örtüşmesi bilinmiyor; bu sonuç dış cephe veya tüm emlak ilanları için geçerli değildir.
