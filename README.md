# BursaDepremHackaton

Demo Linki: https://aita-demo.web.app


Demo Video Linki: https://clipchamp.com/watch/MT9YooI1CPP

Projemizde hedef depremin ilk müdahale anlarında ulaşımın sağlanması için gereken analizler, rota çizimi, insan ve araç gibi objelerin tespiti vb. hayati bilgilerin kullanıcıya sunulması (yardım ekipleri, sürücüler, depremzedeler...) ve deprem gibi afetlerde oluşan ulaşım kargaşasının azaltılmasıdır


Kullanılan Teknolojiler:

* Yolov8
* OpenCV
* Matplotlib
* Numpy
* Pickle

Görüntü İşleme Adımları

  * Kaynak Görüntü
  
  ![image](/Assets/Original.png)

  * Kenar Tespiti(Sobel)

  ![image](/Assets/Sobel.png)

  * Closing ve Opening  operasyonları



    ![image](/Assets/AfterClosingOperation.png)


    ![image](/Assets/AfterOpeningOperation.png)


* Connected Components Operasyonu Sonrasında


    ![image](/Assets/ConnectedComponent.png)

* Rota Çizimlesi


    ![image](/Assets/OptimalRoute.png)


* Objelerin Tespit Edilmesi


    ![image](/Assets/Detections.png)

* Sanal Haritanın Oluşturulması
  

    ![image](/Assets/VirtualMap.png)

