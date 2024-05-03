# Computer Vision

Computer vision, insanların görüntüleri ve videoları algılama ve
yorumlama biçimine benzer şekilde makinelerin görsel verileri görmesini
ve anlamasını sağlamaya odaklanan bir çalışma alanıdır. Bilgisayarların
resimler ve videolar gibi görsel girdilerden anlamlı bilgiler
çıkarmasına ve bu bilgilere dayanarak akıllı kararlar vermesine olanak
tanıyan algoritmalar ve teknikler geliştirmeyi içerir. Bilgisayarla
görme, makinelerin görsel verileri analiz etmesini ve yorumlamasını
sağlamak için bilgisayar bilimi, yapay zeka ve görüntü işleme
unsurlarını birleştirir.

Computer vision\'da makineler nesneler, şekiller, renkler, dokular ve
hareketler gibi çeşitli görsel unsurları tanımak ve anlamak üzere
eğitilir. Bu, derin öğrenme ve konvolüsyonel sinir ağları (CNN\'ler)
gibi gelişmiş algoritmalar ve makine öğrenimi tekniklerinin
kullanılmasıyla elde edilir. Bu algoritmalar makinelerin nesneleri
tespit etmesini ve sınıflandırmasını, hareketlerini izlemesini,
pozlarını tahmin etmesini ve hatta anlamsal anlamlarını anlamasını
sağlar.

Bilgisayarla görme, çeşitli sektörlerde ve alanlarda geniş bir uygulama
alanına sahiptir. Tıbbi görüntü analizi ve teşhisi için sağlık
hizmetleri, nesne algılama ve navigasyon için otonom araçlarda, güvenlik
ve izleme için gözetim sistemlerinde, nesne manipülasyonu ve navigasyon
için robotikte ve kullanıcı deneyimlerini geliştirmek için artırılmış
gerçeklik ve sanal gerçeklik gibi alanlarda kullanılır.
:::

::: {#ebfa1415-77ef-4991-9a36-5cfa1cb92932 .cell .markdown}
### OpenCv Kütüphanesi Kullanımı

OpenCV (Open Source Computer Vision Library), öncelikle gerçek zamanlı
bilgisayarla görmeyi amaçlayan açık kaynaklı bir kütüphanedir. Görüntü
ve video işleme, analiz ve manipülasyon için geniş bir araç ve algoritma
koleksiyonuna sahiptir. Bu araçlar nesne algılama, yüz tanıma, hareket
izleme ve çok daha fazlası gibi görevleri mümkün kılar. OpenCV robotik,
sürücüsüz arabalar, tıbbi görüntüleme ve artırılmış gerçeklik gibi
çeşitli alanlarda uygulama alanı bulmaktadır. Platformlar arası
uyumluluğu ve kapsamlı dokümantasyonu, onu bilgisayarla görme alanında
hem yeni başlayanlar hem de uzmanlar için popüler bir seçim haline
getirmektedir.

Bugünkü projemizde OpenCV\'ye girişi ele alacağız. İlk olarak, temel
görüntü işleme adımlarından olan görünüyü basitçe manipüle etmek gibi
basit işlemleri inceleyeceğiz. Ardından, bu işlevleri kullanarak küçük
bir OpenCV programı yazacağız.

`<b>`{=html}Sırasıyla aşağıdaki adımları izleyeceğiz:`</b>`{=html}

1.  Kullanacağımız kütüphaneleri içeri aktarma.
2.  Bir resimi okuma fonksiyonu oluşturma.
3.  Bir resimi gri tonlamaya dönüştürme fonksiyonunu oluşturma.
4.  Bir resimi gauss bulanıklaştırma yöntemiyle bulanıklaştırma
    fonksiyonu oluşturma.
5.  Bir resimde kenar algılama fonksiyonu oluşturma.
6.  Resimi pencerede görüntüleme fonksiyonu oluşturma.
7.  Programımızın algoritmasına uygun kod yapısını oluşturma.
:::

::: {#bf8597a7-af19-4875-9ea5-9fb77b63b804 .cell .markdown}
### 0- Kütüphanenin kurulumu {#0--kütüphanenin-kurulumu}
:::

::: {#e8300d50-497c-42e1-ac2b-cc458bffa579 .cell .markdown vscode="{\"languageId\":\"bat\"}"}
pip install opencv-python pip install numpy
:::

::: {#3a012d5b-7d9d-4b64-82fd-0539628cd9ab .cell .markdown}
### 1 - Kullanacağımız kütüphaneleri içeri aktarma. {#1---kullanacağımız-kütüphaneleri-içeri-aktarma}
:::

::: {#16af5290-9980-42ea-86f9-c4266926ae24 .cell .code execution_count="11"}
``` python
import cv2
import numpy as np
```
:::

::: {#474f1e1f-1f51-4097-b13b-2997d2f13464 .cell .markdown}
### 2 - Bir resimi okuma fonksiyonu oluşturma. {#2---bir-resimi-okuma-fonksiyonu-oluşturma}
:::

::: {#3be2c9b7-6708-4de3-9ab5-215ce741fef0 .cell .code execution_count="36"}
``` python
def read_image(image_path):
  """
  Belirtilen yoldan bir resmi okur ve RGB formatında döndürür.

  Args:
      image_path (str): Görüntünün dosya yolu.

  Returns:
      numpy.ndarray: RGB formatında görüntü dizisi.
  """
  image = cv2.imread(image_path)
  return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```
:::

::: {#1384bf1c-b05d-4ff2-a44b-0740b99eba31 .cell .markdown}
### 3 - Bir resimi gri tonlamaya dönüştürme fonksiyonunu oluşturma. {#3---bir-resimi-gri-tonlamaya-dönüştürme-fonksiyonunu-oluşturma}
:::

::: {#b265a851-406a-49c5-af2f-3e8cfaf6087f .cell .code execution_count="35"}
``` python
def grayscale(image):
  """
  Bir resmi gri tonlamaya dönüştürür.

  Args:
      image (numpy.ndarray): RGB formatında görüntü dizisi.

  Returns:
      numpy.ndarray: Gri tonlamalı görüntü dizisi.
  """
  return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
```
:::

::: {#038b2cde .cell .markdown}
### 4 - Bir resimi gauss bulanıklaştırma yöntemiyle bulanıklaştırma fonksiyonu oluşturma. {#4---bir-resimi-gauss-bulanıklaştırma-yöntemiyle-bulanıklaştırma-fonksiyonu-oluşturma}
:::

::: {#1f8e4f00-3502-461b-aadf-2b4f2b72b9d7 .cell .code execution_count="34"}
``` python
def blur(image, kernel_size=(3, 3)):
  """
  Bir resmi Gauss bulanıklaştırma ile bulanıklaştırır.

  Args:
      image (numpy.ndarray): RGB formatında görüntü dizisi.
      kernel_size (tuple): Bulanıklaştırma çekirdeği boyutu (genişlik, yükseklik).

  Returns:
      numpy.ndarray: Bulanıklaştırılmış görüntü dizisi.
  """
  return cv2.GaussianBlur(image, kernel_size, 0)
```
:::

::: {#3cd42710 .cell .markdown}
### 5 - Bir resimde kenar algılama fonksiyonu oluşturma. {#5---bir-resimde-kenar-algılama-fonksiyonu-oluşturma}
:::

::: {#bd78a0e9 .cell .code execution_count="55"}
``` python
def edges(image):
  """
  Bir resimde kenarları algılar ve görüntüler.

  Args:
      image (numpy.ndarray): Gri tonlamalı görüntü dizisi.

  Returns:
      numpy.ndarray: Kenarları gösteren görüntü dizisi.
  """
  return cv2.Canny(image, 10, 40)
```
:::

::: {#204221e8 .cell .markdown}
### 6 - Resimi pencerede görüntüleme fonksiyonu oluşturma. {#6---resimi-pencerede-görüntüleme-fonksiyonu-oluşturma}
:::

::: {#2c198e8a .cell .code execution_count="33"}
``` python
def show_image(image, title="Resim"):
  """
  Bir resmi pencerede görüntüler.

  Args:
      image (numpy.ndarray): Görüntü dizisi (RGB veya gri tonlamalı).
      title (str): Pencere başlığı.
  """
  cv2.imshow(title, image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
```
:::

::: {#e580219d .cell .markdown}
### 7 - Programımızın algoritmasına uygun kod yapısını oluşturma. {#7---programımızın-algoritmasına-uygun-kod-yapısını-oluşturma}
:::

::: {#343de40a .cell .code execution_count="58"}
``` python
# Bir resim oku
image = read_image("kum-tepesi.jpg")

# Resmi gri tonlamaya dönüştür
gray_image = grayscale(image)

# Resmi bulanıklaştır
blurred_image = blur(gray_image, kernel_size=(7, 7))

# Kenarları algıla
edges_image = edges(blurred_image)

# Orijinal, gri tonlamalı, bulanıklaştırılmış ve kenar algılanmış resimleri göster
show_image(image, "Orijinal")
show_image(gray_image, "Gri Tonlamalı")
show_image(blurred_image, "Bulanıklaştırılmış")
show_image(edges_image, "Kenarlar")
```
:::
