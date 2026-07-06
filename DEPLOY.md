# DEPLOY — sherif/baybars OCR → RunPod Serverless GPU

Bu repo, **sherif/baybars** Arapça el-yazması OCR modelini (Qwen2.5-VL-3B + LoRA)
RunPod Serverless GPU'da **istek-başına (scale-to-zero)** servis etmek için hazırlandı.
Muhakkik V2'de Google Vision yerine bu kullanılacak.

> **Halid abi:** Aşağıdaki adımları **kendi RunPod hesabından** yap. Kurulum + dosyalar
> hazır; senin yapacağın tek şey endpoint'i oluşturup krediyi yüklemek.

---

## Repo içeriği (hazır, dokunma)

| Dosya | Görev |
|---|---|
| `Dockerfile` | CUDA image; 7.5GB base'i **build'de HF'den çeker** (RunPod tarafında), adapter'ı COPY'ler |
| `handler.py` | RunPod handler; modeli 1 kez yükler, satır görüntülerini OCR'lar |
| `requirements.txt` | Sabitlenmiş bağımlılıklar (referans; Dockerfile inline kurar) |
| `adapter/` | 14MB LoRA adapter (sherif base'in üstüne biner) |

Base model **public**: `sherif1313/Arabic-English-handwritten-OCR-v3` (HF, apache-2.0).
RunPod build sırasında **kendi tarafında** indirir → yerel internet yükü yok.

---

## 0) Repo erişimi (ÖNCE bu)

Repo `enesmalik22/muhakkik-ocr-gpu` altında ve **private**. Senin RunPod'unun
build edebilmesi için repoya erişmen lazım. Enes şu ikisinden birini yapsın:
- **Collaborator ekle:** repo → Settings → Collaborators → Halid'in GitHub kullanıcı adı, **VEYA**
- Repoyu **public** yap (içinde secret YOK — sadece Dockerfile + handler + açık-lisans adapter).

Sonra sen (Halid) RunPod → Settings → **GitHub Connect** yap ve bu repoya erişim ver.

---

## 1) Billing — kredi yükle

RunPod → **Billing** → kart ekle + ~**$10** kredi. Kredisiz build/çalışma olmaz.

---

## 2) Endpoint oluştur

Serverless → **New Endpoint** → **Deploy from a GitHub repository**:

- Repo: `enesmalik22/muhakkik-ocr-gpu`
- Branch: **main** · Dockerfile path: `/` (→ "Dockerfile found" görmeli)
- Endpoint Type: **Queue** (Default) — ⚠️ **Load balancer DEĞİL**
- (Uyarı çıkarsa: "Could not find runpod.serverless.start()" → **yanlış alarm**, indexleme gecikmesi; `handler.py`'nin son satırı zaten `runpod.serverless.start(...)`. Görmezden gel.)

**Next →**

### Configure endpoint
- **Endpoint name:** `muhakkik-ocr-gpu`
- **Compute type:** GPU
- **GPU configuration** (kutucuklar — ikisini işaretle):
  - ✅ **24 GB** (~$0.00019/s) — ana
  - ✅ **16 GB** (~$0.00016/s) — yedek
  - ❌ 24GB PRO / 32GB / 48GB — gereksiz pahalı
- **Max workers:** **1** (başlangıç; sonra artırılır)
- **Advanced settings** (aç):
  - **Active / Min Workers = 0** (scale-to-zero → boşta $0) ← en önemli
  - **Execution Timeout ≈ 300** sn
  - **Idle Timeout** 5s + **FlashBoot** açık → varsayılan

**Deploy →**

---

## 3) Build'i izle (**Builds** sekmesi)

RunPod repoyu klonlar, kendi sunucusunda `docker build` yapar (7.5GB HF çekimi dahil).
~birkaç dk – 30 dk. **Logs**'ta `[startup] model ready in Ns` görürsen model yüklendi.

- ⚠️ Build **30 dk** cap'i var; HF yavaşsa fail edebilir → yeniden dene.
- ⚠️ Sonradan repoya commit **otomatik build tetiklemez** → değişiklik için GitHub'da **Release** oluştur.

---

## 4) Smoke test (build "Completed" olunca)

Endpoint ID + API key al (endpoint sayfası + Settings → API Keys), sonra:

```bash
ENDPOINT_ID=<endpoint id>
RUNPOD_API_KEY=<api key>
B64=$(base64 -i /bir/satir_kirpintisi.png)

curl -s -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync?wait=300000" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"input\": {\"images\": [\"${B64}\"]}}"
```

**Handler kontratı (ÖNEMLİ):**
- Girdi: `input.images` = base64 satır-kırpıntı **LİSTESİ** (sayfa başına 1 çağrı), opsiyonel `input.prompt`
- Çıktı: `output.texts` = aynı sırada metin **listesi**; hatalı satırlar `output.errors[]`'da
- Başarı: `{"status":"COMPLETED","output":{"texts":["...arapça..."]}}`
- `IN_QUEUE`/`IN_PROGRESS` dönerse cold-start bitmemiş → `GET /v2/{ID}/status/{id}` ile poll et
- `FAILED` → sistemik hata (Logs'a bak)

---

## 5) V2'ye bağlama (endpoint çalışınca)

Prod V2'ye env ver:
```
BAYBARS_OCR_URL=https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync
BAYBARS_OCR_API_KEY=<runpod api key>
OCR_FORCE_LANE=baybars_htr
```
> Not: V2'deki OCR lane'i (baybars_recognition.py) **RunPod runsync API**'si konuşacak şekilde
> (yeniden) yazılacak — eski OpenAI `/v1/chat/completions` formatı DEĞİL. Body `{"input":{"images":[...]}}`,
> Bearer API key, cold-start için `?wait=300000` + IN_QUEUE ise `/status/{id}` poll, FAILED→fallback.
> (Bu lane kodu Enes tarafında Claude ile hazırlanıp V2 repo'ya commit edilecek.)

---

## Maliyet (24GB tier)
- ~**$0.68/saat** ama **sadece çalışırken**; boşta **$0** (scale-to-zero)
- ~**1-2 cent/sayfa** · ~**$3 / 300-sayfa nüsha** · sabit: image diski ~$1.5/ay

## Kalite notu (dürüst)
Model CER ~**%16.5** / WER ~%47 (Baybars test seti). El yazması nüshalarda kapalı Vision'dan
iyi olabilir ama **gerçek sayfalarda doğrulanmadı**. İlk hedef: pipeline'ı GPU'da uçtan uca
çalıştırmak; kalite kıyası sonra.

## Sorun giderme (endpoint → Logs)
- "model ready" yok + "loading"da donma → VRAM OOM → 24GB tier'a çık
- "CUDA not available" → CPU worker/CPU wheel → GPU tier'ı kontrol et
- Build 30dk aştı → HF yavaş → yeniden dene
- snapshot_download 401/403/404 → repo id yanlış (doğrusu `sherif1313/Arabic-English-handwritten-OCR-v3`)

---

> Bu deploy **test edilmedi** (ödemeli hesap + gerçek GPU gerek). Model mantığı yerel
> çalışan sunucudan birebir kopya (kanıtlı); RunPod paketlemesi ilk-deploy hipotezi.
> İlk build'de takılırsa Logs paylaş.
